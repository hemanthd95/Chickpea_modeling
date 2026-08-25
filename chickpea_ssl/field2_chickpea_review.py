"""Frozen prediction-free NDVI rule and chickpea-only review workflow for Field 2 v2."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import rasterio
from spectral.io import envi
import yaml

from chickpea_ssl.data import load_records
from chickpea_ssl.field2_point_spectra import PredictionFreeSpectrumStore
from chickpea_ssl.field2_readiness import parse_wavelengths, sha256
from chickpea_ssl.field2_blind_review import atomic_write_bytes, atomic_write_yaml


REFERENCE_PROVENANCE = (
    "investigator_chickpea",
    "investigator_nonchickpea_vegetation",
    "investigator_uncertain",
    "rule_based_soil_ndvi_0p30",
    "investigator_area_constrained_field_weed",
    "investigator_area_constrained_outside_weed_ood",
    "nodata_invalid",
)
MANUAL_LABEL_PROVENANCE = {
    "chickpea": "investigator_chickpea",
    "weed_unspecified": "investigator_nonchickpea_vegetation",
    "chickpea_weed_mixed": "investigator_chickpea",
    "uncertain": "investigator_uncertain",
}


def csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO()
    frame.to_csv(stream, index=False, lineterminator="\n")
    return stream.getvalue().encode()


def load_main_without_reserve(project: Path, policy: dict) -> tuple[dict, dict, pd.DataFrame]:
    """Validate and read only the v2 main frame; never open the reserve table."""
    v2_path = project / policy["v2_sampling_config"]
    v2 = yaml.safe_load(v2_path.read_text())
    contract_path = project / v2["outputs"]["contract"]
    contract = yaml.safe_load(contract_path.read_text())
    if contract.get("version") != "field2_area_stratified_sampling_contract_v2" or contract.get("status") != "frozen":
        raise ValueError("Field 2 v2 sampling is not frozen")
    if contract.get("main_count") != 800 or contract.get("reserve_count") != 400:
        raise ValueError("Field 2 v2 frame counts changed")
    if contract.get("reserve_release", {}).get("status") != "locked_not_exposed":
        raise ValueError("Field 2 v2 reserve is not locked")
    main_ref = contract["frozen_outputs"]["main_frame"]
    main_path = project / main_ref["path"]
    if not main_path.is_file() or sha256(main_path) != main_ref["sha256"]:
        raise ValueError("Frozen v2 main frame changed")
    main = pd.read_csv(main_path, keep_default_na=False)
    if len(main) != 800 or set(main.sampling_frame) != {"main"}:
        raise ValueError("Expected the exact 800-point v2 main frame")
    return v2, contract, main


def ndvi_status(red: float, nir: float, threshold: float = 0.30) -> tuple[float | None, str]:
    denominator = float(nir) + float(red)
    if not np.isfinite(red) or not np.isfinite(nir) or not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return None, "invalid"
    value = (float(nir) - float(red)) / denominator
    if not np.isfinite(value):
        return None, "invalid"
    return float(value), "soil" if value < threshold else "vegetation"


def provisional_reference(domain: str, status: str, rules: dict) -> tuple[str, str, bool]:
    if status == "invalid":
        label, provenance = rules["invalid"]
        return label, provenance, False
    label, provenance = rules[domain][status]
    return label, provenance, label == "pending_investigator_chickpea_review"


def compute_field2_references(
    project: Path, policy: dict, v2: dict, main: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = yaml.safe_load((project / v2["base_config"]).read_text())
    inventory_path = project / base["inputs"]["readiness_inventory"]
    support_path = project / base["inputs"]["valid_support_manifest"]
    spectra = PredictionFreeSpectrumStore(project, inventory_path, support_path, main)
    threshold = float(policy["ndvi_rule"]["soil_threshold"])
    rules = policy["provisional_reference_rules"]
    expected_red = int(policy["ndvi_rule"]["expected_red_python_band_index"])
    expected_nir = int(policy["ndvi_rule"]["expected_nir_python_band_index"])
    rows = []
    band_rows = []
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    source_by_cube = inventory[inventory.product_type == "reflectance"].set_index("cube_id")
    for cube_id, group in main.groupby("cube_id", sort=True):
        image, wavelengths, support = spectra._cube(str(cube_id))
        red_index = int(np.argmin(np.abs(wavelengths - float(policy["ndvi_rule"]["red_target_nm"]))))
        nir_index = int(np.argmin(np.abs(wavelengths - float(policy["ndvi_rule"]["nir_target_nm"]))))
        red_nm, nir_nm = float(wavelengths[red_index]), float(wavelengths[nir_index])
        if (red_index, nir_index) != (expected_red, expected_nir) or not np.isclose(red_nm, 669.09) or not np.isclose(nir_nm, 798.91):
            raise ValueError(f"Unexpected Field 2 NDVI bands: {cube_id}")
        source = source_by_cube.loc[cube_id]
        band_rows.append({
            "cube_id": cube_id, "red_wavelength_nm": red_nm, "nir_wavelength_nm": nir_nm,
            "red_python_band_index": red_index, "nir_python_band_index": nir_index,
            "red_envi_band_number": red_index + 1, "nir_envi_band_number": nir_index + 1,
            "formula": policy["ndvi_rule"]["formula"], "threshold": threshold,
            "reflectance_binary_path": source.binary_path,
            "reflectance_binary_sha256": source.sha256,
            "support_mask_sha256": str(group.support_mask_sha256.iloc[0]),
        })
        for sample in group.itertuples(index=False):
            row, column = int(sample.row), int(sample.column)
            if not support[row, column]:
                raise ValueError(f"Frozen v2 main point outside support: {sample.sample_id}")
            red, nir = float(image[row, column, red_index]), float(image[row, column, nir_index])
            value, status = ndvi_status(red, nir, threshold)
            reference, provenance, manual = provisional_reference(str(sample.domain_name), status, rules)
            rows.append({
                "sample_id": sample.sample_id, "cube_id": cube_id,
                "cube_evaluation_role": sample.cube_evaluation_role,
                "domain_name": sample.domain_name, "row": row, "column": column,
                "red_reflectance": red, "nir_reflectance": nir,
                "raw_reflectance_ndvi": "" if value is None else value,
                "ndvi_rule_status": status, "provisional_reference": reference,
                "reference_provenance": provenance, "manual_review_required": manual,
                "design_weight": sample.design_weight,
                "overall_inclusion_probability": sample.overall_inclusion_probability,
                "sampling_frame_sha256": sha256(project / v2["outputs"]["root"] / v2["outputs"]["main_frame"]),
            })
    references = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    bands = pd.DataFrame(band_rows).sort_values("cube_id").reset_index(drop=True)
    return references, bands


def audit_field1_rule(project: Path, paths_file: Path, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = policy["field1_audit"]
    manifest_path = project / audit["authoritative_manifest"]
    contract_path = project / audit["confident_label_contract"]
    contract = yaml.safe_load(contract_path.read_text())
    if sha256(manifest_path) != contract["input_hashes"][audit["authoritative_manifest"]]:
        raise ValueError("Frozen Field 1 authoritative manifest changed")
    records = {record.cube_id: record for record in load_records(paths_file, manifest_path)}
    label_root = project / audit["canonical_label_root"]
    threshold = float(policy["ndvi_rule"]["soil_threshold"])
    counts = {}
    source_rows = []
    class_names = {int(value): name for name, value in audit["class_mapping"].items() if name != "unresolved"}
    for cube_dir in sorted(label_root.iterdir()):
        label_path = cube_dir / "confident_labels.tif"
        if not label_path.is_file():
            continue
        cube_id = cube_dir.name
        relative = str(label_path.relative_to(project))
        if sha256(label_path) != contract["output_hashes"].get(relative):
            raise ValueError(f"Frozen Field 1 label raster changed: {cube_id}")
        record = records[cube_id]
        image = envi.open(str(record.header), str(record.data))
        wavelengths = parse_wavelengths(image.metadata)
        red_index = int(np.argmin(np.abs(wavelengths - float(policy["ndvi_rule"]["red_target_nm"]))))
        nir_index = int(np.argmin(np.abs(wavelengths - float(policy["ndvi_rule"]["nir_target_nm"]))))
        if (red_index, nir_index) != (
            int(policy["ndvi_rule"]["expected_red_python_band_index"]),
            int(policy["ndvi_rule"]["expected_nir_python_band_index"]),
        ):
            raise ValueError(f"Unexpected Field 1 NDVI bands: {cube_id}")
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(label_path) as source:
                labels = source.read(1)
        if labels.shape != (image.nrows, image.ncols):
            raise ValueError(f"Field 1 label/reflectance shape mismatch: {cube_id}")
        array = image.open_memmap()
        for start in range(0, image.nrows, 128):
            stop = min(image.nrows, start + 128)
            label_chunk = labels[start:stop]
            red = np.asarray(array[start:stop, :, red_index], dtype=float)
            nir = np.asarray(array[start:stop, :, nir_index], dtype=float)
            denominator = nir + red
            finite = np.isfinite(red) & np.isfinite(nir) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
            ndvi = np.full(red.shape, np.nan, dtype=float)
            ndvi[finite] = (nir[finite] - red[finite]) / denominator[finite]
            for class_id, class_name in class_names.items():
                selected = label_chunk == class_id
                key = (class_name, "soil")
                counts[key] = counts.get(key, 0) + int((selected & finite & (ndvi < threshold)).sum())
                key = (class_name, "vegetation")
                counts[key] = counts.get(key, 0) + int((selected & finite & (ndvi >= threshold)).sum())
                key = (class_name, "invalid")
                counts[key] = counts.get(key, 0) + int((selected & ~finite).sum())
        source_rows.append({
            "cube_id": cube_id, "reflectance_header_path": str(record.header),
            "reflectance_binary_path": str(record.data), "reflectance_header_sha256": sha256(record.header),
            "label_path": relative, "label_sha256": sha256(label_path),
            "red_wavelength_nm": float(wavelengths[red_index]), "nir_wavelength_nm": float(wavelengths[nir_index]),
            "red_python_band_index": red_index, "nir_python_band_index": nir_index,
            "red_envi_band_number": red_index + 1, "nir_envi_band_number": nir_index + 1,
        })
    rows = []
    for class_name in ("soil", "chickpea", "weed"):
        total = sum(counts.get((class_name, status), 0) for status in ("soil", "vegetation", "invalid"))
        for status in ("soil", "vegetation", "invalid"):
            count = counts.get((class_name, status), 0)
            rows.append({
                "frozen_field1_label": class_name, "ndvi_rule_status": status,
                "pixel_count": count, "class_total": total,
                "within_class_fraction": count / total if total else 0.0,
                "threshold": threshold, "threshold_optimized_on_field2": False,
            })
    return pd.DataFrame(rows), pd.DataFrame(source_rows).sort_values("cube_id")


def workload_table(references: pd.DataFrame) -> pd.DataFrame:
    cubes = sorted(references.cube_id.unique())
    domains = ["research_crop_area", "unassigned_valid_support"]
    rows = []
    for cube in cubes:
        for domain in domains:
            group = references[(references.cube_id == cube) & (references.domain_name == domain)]
            rows.append({
                "cube_id": cube, "domain_name": domain, "v2_main_points": len(group),
                "rule_based_soil": int((group.ndvi_rule_status == "soil").sum()),
                "manual_vegetation_review": int(group.manual_review_required.sum()),
                "invalid": int((group.ndvi_rule_status == "invalid").sum()),
            })
    return pd.DataFrame(rows)


def output_targets(project: Path, policy: dict) -> dict[str, Path]:
    output = policy["outputs"]
    root = project / output["report_root"]
    return {
        "provisional_references": root / output["provisional_references"],
        "workload_by_cube_domain": root / output["workload_by_cube_domain"],
        "field1_ndvi_audit": root / output["field1_ndvi_audit"],
        "field1_band_source_audit": root / output["field1_band_source_audit"],
        "field2_band_source_audit": root / output["field2_band_source_audit"],
        "audit_yaml": root / output["audit_yaml"],
        "contract": project / output["contract"],
        "sha256_manifest": project / output["sha256_manifest"],
    }


def materialize_policy(project: Path, paths_file: Path, policy: dict, policy_path: Path) -> dict:
    targets = output_targets(project, policy)
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise RuntimeError(f"REFUSED: immutable chickpea-review policy output already exists: {existing}")
    v2, v2_contract, main = load_main_without_reserve(project, policy)
    annotation_path = project / v2["point_annotation"]["output_root"] / "field2_blind_main_point_annotations.json"
    annotation_hash = sha256(annotation_path)
    annotations = json.loads(annotation_path.read_text())
    if len(annotations.get("annotations", {})) != 800 or any(
        record.get("selected_label") or record.get("reviewed") for record in annotations["annotations"].values()
    ):
        raise RuntimeError("The v2 annotation store is not the preserved neutral 800-record package")
    references, field2_bands = compute_field2_references(project, policy, v2, main)
    field1_audit, field1_sources = audit_field1_rule(project, paths_file, policy)
    workload = workload_table(references)
    counts = {
        "v2_main_total": len(references),
        "rule_based_soil": int((references.ndvi_rule_status == "soil").sum()),
        "alley_field_weed": int((references.provisional_reference == "field_weed").sum()),
        "outside_weed_ood": int((references.provisional_reference == "outside_weed_ood").sum()),
        "manual_chickpea_review": int(references.manual_review_required.sum()),
        "invalid": int((references.ndvi_rule_status == "invalid").sum()),
    }
    if counts != {
        "v2_main_total": 800, "rule_based_soil": 99, "alley_field_weed": 146,
        "outside_weed_ood": 78, "manual_chickpea_review": 477, "invalid": 0,
    }:
        raise RuntimeError(f"Unexpected frozen-rule workload: {counts}")
    audit_payload = {
        "status": "field2_chickpea_review_policy_audit_passed", "counts": counts,
        "ndvi_rule": policy["ndvi_rule"],
        "manual_domains": policy["manual_review"]["eligible_domains"],
        "reserve_frame_opened": False, "checkpoint_or_model_loaded": False,
        "field2_labels_used_to_choose_threshold": False,
        "stored_scalar_used_as_ndvi": False,
        "production_annotation_sha256_before_and_after_materialization": annotation_hash,
    }
    for path in targets.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("provisional_references", references), ("workload_by_cube_domain", workload),
        ("field1_ndvi_audit", field1_audit), ("field1_band_source_audit", field1_sources),
        ("field2_band_source_audit", field2_bands),
    ):
        atomic_write_bytes(targets[name], csv_bytes(frame))
    atomic_write_yaml(targets["audit_yaml"], audit_payload)
    frozen_names = [name for name in targets if name not in {"contract", "sha256_manifest"}]
    refs = {
        name: {"path": str(targets[name].relative_to(project)), "sha256": sha256(targets[name])}
        for name in frozen_names
    }
    manifest = pd.DataFrame([{"artifact": name, **item} for name, item in refs.items()]).sort_values("artifact")
    atomic_write_bytes(targets["sha256_manifest"], csv_bytes(manifest))
    refs["sha256_manifest"] = {
        "path": str(targets["sha256_manifest"].relative_to(project)),
        "sha256": sha256(targets["sha256_manifest"]),
    }
    field1_contract_path = project / policy["field1_audit"]["confident_label_contract"]
    v2_contract_path = project / v2["outputs"]["contract"]
    v2_config_path = project / policy["v2_sampling_config"]
    spectral_config_path = project / policy["spectral_band_config"]
    contract = {
        "version": "field2_chickpea_review_policy_contract_v1", "status": "frozen_before_field2_model_use",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "materialization_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        "ndvi_rule": policy["ndvi_rule"], "provisional_reference_rules": policy["provisional_reference_rules"],
        "manual_review": policy["manual_review"], "evaluation_policy": policy["evaluation_policy"],
        "counts": counts, "allowed_reference_provenance": list(REFERENCE_PROVENANCE),
        "inputs": {
            "policy_config": {"path": str(policy_path.relative_to(project)), "sha256": sha256(policy_path)},
            "spectral_band_config": {"path": str(spectral_config_path.relative_to(project)), "sha256": sha256(spectral_config_path)},
            "v2_sampling_config": {"path": str(v2_config_path.relative_to(project)), "sha256": sha256(v2_config_path)},
            "v2_sampling_contract": {"path": str(v2_contract_path.relative_to(project)), "sha256": sha256(v2_contract_path)},
            "v2_main_frame_sha256": v2_contract["frozen_outputs"]["main_frame"]["sha256"],
            "neutral_v2_annotation_store": {"path": str(annotation_path.relative_to(project)), "sha256": annotation_hash},
            "field1_confident_label_contract": {"path": str(field1_contract_path.relative_to(project)), "sha256": sha256(field1_contract_path)},
        },
        "frozen_outputs": refs,
        "provenance": {
            "prediction_free": True, "raw_reflectance_only_for_ndvi": True,
            "stored_scalar_used_as_authoritative_ndvi": False, "checkpoint_loaded": False,
            "model_prediction_or_probability_used": False, "threshold_tuned_on_field2": False,
            "reserve_frame_opened": False, "biological_class_preselected_in_manual_queue": False,
        },
    }
    atomic_write_yaml(targets["contract"], contract)
    if sha256(annotation_path) != annotation_hash:
        raise RuntimeError("Production v2 annotation store changed during policy materialization")
    return {"contract": contract, "references": references, "workload": workload, "field1_audit": field1_audit}


def validate_policy(project: Path, policy: dict) -> dict:
    targets = output_targets(project, policy)
    contract = yaml.safe_load(targets["contract"].read_text())
    if contract.get("version") != "field2_chickpea_review_policy_contract_v1" or contract.get("status") != "frozen_before_field2_model_use":
        raise ValueError("Unknown or unfinished chickpea-review policy contract")
    provenance = contract.get("provenance", {})
    if provenance.get("prediction_free") is not True or provenance.get("raw_reflectance_only_for_ndvi") is not True:
        raise ValueError("Review policy is not raw-reflectance prediction-free")
    forbidden_true = (
        "stored_scalar_used_as_authoritative_ndvi", "checkpoint_loaded", "model_prediction_or_probability_used",
        "threshold_tuned_on_field2", "reserve_frame_opened", "biological_class_preselected_in_manual_queue",
    )
    if any(provenance.get(key) is not False for key in forbidden_true):
        raise ValueError("Review policy violates a frozen safety declaration")
    for name in ("policy_config", "spectral_band_config", "v2_sampling_config", "v2_sampling_contract", "field1_confident_label_contract"):
        item = contract["inputs"][name]
        path = project / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen review-policy input changed: {name}")
    v2, _, main = load_main_without_reserve(project, policy)
    if contract["inputs"]["v2_main_frame_sha256"] != sha256(project / v2["outputs"]["root"] / v2["outputs"]["main_frame"]):
        raise ValueError("Review policy v2 main hash changed")
    for name, item in contract["frozen_outputs"].items():
        path = project / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen review-policy output changed: {name}")
    references = pd.read_csv(targets["provisional_references"], keep_default_na=False)
    if len(references) != 800 or set(references.sample_id) != set(main.sample_id):
        raise ValueError("Provisional references do not cover exactly v2 main")
    if references.manual_review_required.astype(str).str.lower().eq("true").sum() != 477:
        raise ValueError("Manual chickpea-review workload changed")
    manual = references[references.manual_review_required.astype(str).str.lower().eq("true")]
    if not set(manual.domain_name).issubset(set(policy["manual_review"]["eligible_domains"])) or not (manual.raw_reflectance_ndvi.astype(float) >= .30).all():
        raise ValueError("Manual queue violates frozen domain/NDVI eligibility")
    pending = references.provisional_reference.eq("pending_investigator_chickpea_review")
    if not references.loc[pending, "reference_provenance"].eq("").all():
        raise ValueError("Pending manual records must not claim investigator provenance")
    completed = references.loc[~pending, "reference_provenance"]
    if not completed.isin(contract["allowed_reference_provenance"]).all():
        raise ValueError("A completed reference has unknown provenance")
    annotation_ref = contract["inputs"]["neutral_v2_annotation_store"]
    annotation_path = project / annotation_ref["path"]
    if not annotation_path.is_file():
        raise ValueError("V2 annotation store is missing")
    current = json.loads(annotation_path.read_text())
    if len(current.get("annotations", {})) != 800:
        raise ValueError("V2 annotation inventory changed")
    return {"contract": contract, "references": references, "main": main, "workload": pd.read_csv(targets["workload_by_cube_domain"], keep_default_na=False)}
