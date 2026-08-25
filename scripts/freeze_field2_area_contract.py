#!/usr/bin/env python
"""Freeze prediction-free Field 2 area domains without changing raw annotations."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.field2_area_operational import (
    PRECEDENCE_RULE, compile_operational_membership, raw_memberships_from_bits,
)
from chickpea_ssl.field2_area_review import Field2AreaAnnotationStore, ZONE_CODES, utc_now, zone_name_from_code
from chickpea_ssl.field2_blind_review import atomic_write_bytes, atomic_write_json, validate_frozen_sampling_contract
from chickpea_ssl.field2_readiness import sha256
from scripts.run_field2_area_annotator import validate_config_inputs


RAW_ARTIFACT_NAMES = {
    "annotations_json": "field2_area_annotations.json",
    "annotations_geojson": "field2_area_annotations.geojson",
    "vertices_csv": "field2_area_vertices.csv",
    "geometry_audit_csv": "field2_area_geometry_audit.csv",
    "overview_png": "field2_area_annotation_overview.png",
}
DOMAIN_CLASSIFICATION = {
    "research_crop_area": "primary_in_field",
    "alley": "supplementary_alley",
    "outside_research_field": "outside_domain_ood",
    "uncertain_boundary": "supplementary_uncertain",
    "unassigned_valid_support": "separately_reported_unassigned",
}
MEMBERSHIP_COLUMNS = [
    "sample_id", "cube_id", "cube_evaluation_role", "sampling_frame",
    "domain_code", "domain_name", "inside_valid_support", "raw_polygon_memberships",
    "raw_zone_memberships", "winning_operational_membership", "precedence_applied",
    "precedence_rule", "raw_coverage_mode", "effective_coverage_mode",
    "domain_reporting_stratum", "primary_external_validation_eligible",
    "stored_index_rank_stratum", "spatial_group_id",
]


def _csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def raw_artifact_hashes(store: Field2AreaAnnotationStore) -> dict[str, dict[str, str]]:
    result = {}
    for name, filename in RAW_ARTIFACT_NAMES.items():
        path = store.output_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw area artifact: {path}")
        result[name] = {"path": str(path.relative_to(store.project)), "sha256": sha256(path)}
    return result


def membership_rows(frame: pd.DataFrame, compiled: dict[str, dict], sampling_frame: str) -> list[dict]:
    """Assign exact frozen samples without emitting coordinates."""
    rows = []
    for sample in frame.itertuples(index=False):
        cube = compiled[str(sample.cube_id)]; row, column = int(sample.row), int(sample.column)
        code = int(cube["winning_membership"][row, column])
        if code == ZONE_CODES["invalid_outside_valid_support"]:
            raise ValueError(f"Frozen sample is outside valid support: {sample.sample_id}")
        name = zone_name_from_code(code)
        raw_bits = int(cube["raw_membership_bitmask"][row, column])
        polygon_bits = int(cube["raw_polygon_membership_bitmask"][row, column])
        rows.append({
            "sample_id": str(sample.sample_id), "cube_id": str(sample.cube_id),
            "cube_evaluation_role": str(sample.cube_evaluation_role), "sampling_frame": sampling_frame,
            "domain_code": code, "domain_name": name, "inside_valid_support": True,
            "raw_polygon_memberships": "|".join(raw_memberships_from_bits(polygon_bits)),
            "raw_zone_memberships": "|".join(raw_memberships_from_bits(raw_bits)) or "unassigned_valid_support",
            "winning_operational_membership": name,
            "precedence_applied": bool(cube["precedence_applied"][row, column]),
            "precedence_rule": PRECEDENCE_RULE,
            "raw_coverage_mode": cube["raw_coverage_mode"],
            "effective_coverage_mode": cube["effective_coverage_mode"],
            "domain_reporting_stratum": DOMAIN_CLASSIFICATION[name],
            "primary_external_validation_eligible": name == "research_crop_area",
            "stored_index_rank_stratum": str(sample.scalar_index_rank_stratum),
            "spatial_group_id": str(sample.spatial_group_id),
        })
    return rows


def validate_ready_payload(store: Field2AreaAnnotationStore, payload: dict) -> None:
    issues = store.validate(payload)
    invalid = {cube_id: values for cube_id, values in issues.items() if values}
    if invalid:
        raise ValueError("Area annotations are not freeze-ready: " + json.dumps(invalid, sort_keys=True))
    records = payload["annotations"]
    if len(records) != 40 or not all(record["reviewed"] for record in records.values()):
        raise ValueError("Area freeze requires all 40 cubes reviewed")
    if not all(record.get("confidence") == "high" for record in records.values()):
        raise ValueError("Area freeze requires the declared high confidence on all 40 cubes")
    metadata = payload.get("automatic_metadata", {})
    if metadata.get("predictions_or_probabilities_used") is not False:
        raise ValueError("Area annotations are not prediction-free")
    if metadata.get("biological_labels_assigned_automatically") is not False:
        raise ValueError("Area annotations contain automatic biological labels")


def _component_geometry(component: dict) -> dict:
    exterior = [[point["x"], point["y"]] for point in component["exterior_vertices_pixel"]]
    rings = [[*exterior, exterior[0]]]
    for hole in component.get("holes_vertices_pixel", []):
        ring = [[point["x"], point["y"]] for point in hole]
        rings.append([*ring, ring[0]])
    return {"type": "Polygon", "coordinates": rings}


def _count_rows(all_memberships: pd.DataFrame) -> list[dict]:
    rows = []
    dimensions = {
        "overall": None, "cube": "cube_id", "frozen_cube_role": "cube_evaluation_role",
        "stored_index_rank_stratum": "stored_index_rank_stratum", "spatial_block": "spatial_group_id",
    }
    for dimension_name, column in dimensions.items():
        group_columns = ["sampling_frame", "domain_code", "domain_name"] + ([column] if column else [])
        for keys, group in all_memberships.groupby(group_columns, sort=True, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rows.append({
                "grouping_dimension": dimension_name, "grouping_value": str(keys[3]) if column else "all",
                "sampling_frame": str(keys[0]), "domain_code": int(keys[1]),
                "domain_name": str(keys[2]), "sample_count": int(len(group)),
            })
    return rows


def materialize_freeze_products(
    project: Path, area_config: dict, store: Field2AreaAnnotationStore, payload: dict,
    main_frame: pd.DataFrame, reserve_frame: pd.DataFrame, output_root: Path,
) -> dict:
    """Write deterministic operational products to a new directory."""
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite area-freeze products: {output_root}")
    output_root.mkdir(parents=True)
    mask_root = output_root / area_config["freeze"]["mask_root"]; mask_root.mkdir()
    raw_mask_root = output_root / area_config["freeze"]["raw_membership_mask_root"]; raw_mask_root.mkdir()
    compiled: dict[str, dict] = {}
    mask_manifest_rows, domain_rows, precedence_rows, reconciliation_rows = [], [], [], []
    geometry_rows, geometry_features = [], []
    source_hash = sha256(store.json_path)

    for cube_id in store.cube_order:
        record = payload["annotations"][cube_id]
        support_path = project / str(store.support[cube_id].mask_path)
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(support_path, "r") as source:
                support = source.read(1) == 1; profile = source.profile.copy()
                source_grid = {
                    "width": source.width, "height": source.height, "crs": str(source.crs),
                    "transform": json.dumps(list(source.transform)), "bounds": json.dumps(list(source.bounds)),
                }
        cube = compile_operational_membership(
            record, support,
            float(area_config["operational_geometry"]["terminal_near_duplicate_distance_pixels"]),
        )
        cube["raw_coverage_mode"] = str(record.get("coverage_mode", ""))
        compiled[cube_id] = cube
        domain_mask = cube["winning_membership"]
        if np.any(domain_mask[support] == 0) or np.any(domain_mask[~support] != 0):
            raise ValueError(f"Operational domain coding violates valid support: {cube_id}")
        profile.update(dtype="uint8", count=1, nodata=0, compress="DEFLATE", predictor=2, zlevel=9)
        domain_path = mask_root / f"{cube_id}_area_domain.tif"
        raw_path = raw_mask_root / f"{cube_id}_raw_zone_membership_bits.tif"
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(domain_path, "w", **profile) as target: target.write(domain_mask, 1)
            with rasterio.open(raw_path, "w", **profile) as target: target.write(cube["raw_membership_bitmask"], 1)
        mask_manifest_rows.append({
            "cube_id": cube_id, "domain_mask_path": str(domain_path.relative_to(output_root)),
            "domain_mask_sha256": sha256(domain_path), "raw_membership_mask_path": str(raw_path.relative_to(output_root)),
            "raw_membership_mask_sha256": sha256(raw_path), "support_mask_path": str(support_path.relative_to(project)),
            "support_mask_sha256": sha256(support_path), **source_grid,
        })
        for code, name in sorted((value, name) for name, value in ZONE_CODES.items()):
            domain_rows.append({
                "cube_id": cube_id, "domain_code": code, "domain_name": name,
                "pixel_count": int(np.count_nonzero(domain_mask == code)), "inside_valid_support": code != 0,
            })
        pairs = np.stack((cube["raw_membership_bitmask"][support], cube["raw_polygon_membership_bitmask"][support]), axis=1)
        for raw_bits, polygon_bits in np.unique(pairs, axis=0):
            selected = support & (cube["raw_membership_bitmask"] == raw_bits) & (
                cube["raw_polygon_membership_bitmask"] == polygon_bits
            )
            raw = raw_memberships_from_bits(int(raw_bits)); polygon_raw = raw_memberships_from_bits(int(polygon_bits))
            winners, winner_counts = np.unique(domain_mask[selected], return_counts=True)
            for winner_code, count in zip(winners, winner_counts):
                precedence_rows.append({
                    "cube_id": cube_id, "raw_membership_bits": int(raw_bits),
                    "raw_polygon_membership_bits": int(polygon_bits),
                    "raw_zone_memberships": "|".join(raw) or "unassigned_valid_support",
                    "raw_polygon_memberships": "|".join(polygon_raw),
                    "winning_domain_code": int(winner_code),
                    "winning_operational_membership": zone_name_from_code(int(winner_code)),
                    "precedence_applied": len(raw) > 1, "precedence_rule": PRECEDENCE_RULE,
                    "pixel_count": int(count),
                })
        raw_mode = str(record.get("coverage_mode", ""))
        reconciliation_rows.append({
            "cube_id": cube_id, "raw_coverage_mode": raw_mode or "blank",
            "effective_coverage_mode": cube["effective_coverage_mode"],
            "reconciliation_reason": (
                "reviewed_cube_with_investigator_polygons"
                if not raw_mode and record.get("reviewed") and record.get("polygons")
                else "preserved_recorded_coverage_mode"
            ),
            "raw_polygon_count": len(record.get("polygons", [])),
        })
        for item in cube["polygon_audits"]:
            geometry_rows.append({
                "cube_id": cube_id, "polygon_id": item["polygon_id"], "zone_type": item["zone_type"],
                "repair_method": item["repair_method"], "raw_vertex_count": item["raw_vertex_count"],
                "operational_vertex_count": item["operational_vertex_count"], "component_count": item["component_count"],
                "first_to_last_distance_pixels": item["first_to_last_distance_pixels"],
                "raw_self_intersection": item["raw_self_intersection"],
                "operational_self_intersection": item["operational_self_intersection"],
                "raw_area_pixels2": item["raw_area_pixels2"],
                "pre_support_operational_area_pixels2": item["pre_support_operational_area_pixels2"],
                "operational_area_pixels2": item["operational_area_pixels2"],
                "absolute_area_change_pixels2": item["absolute_area_change_pixels2"],
                "percentage_area_change": item["percentage_area_change"],
                "rasterized_pixel_count": item["rasterized_pixel_count"],
                "warnings": "|".join(item["warnings"]), "source_vertices_preserved": True,
                "source_annotation_sha256": source_hash,
            })
            for component in item["components"]:
                geometry_features.append({
                    "type": "Feature", "properties": {
                        "cube_id": cube_id, "polygon_id": item["polygon_id"], "zone_type": item["zone_type"],
                        "component_index": component["component_index"], "repair_method": item["repair_method"],
                        "source_annotation_sha256": source_hash, "source_vertices_preserved": True,
                    }, "geometry": _component_geometry(component),
                })

    main_rows = membership_rows(main_frame, compiled, "main")
    reserve_rows = membership_rows(reserve_frame, compiled, "reserve")
    if len(main_rows) != 800 or len(reserve_rows) != 396:
        raise ValueError("Frozen sample membership inventory changed")
    if set(row["sample_id"] for row in main_rows) & set(row["sample_id"] for row in reserve_rows):
        raise ValueError("Main/reserve sample intersection detected")
    main_path = output_root / area_config["freeze"]["main_membership"]
    reserve_path = output_root / area_config["freeze"]["reserve_membership_locked"]
    atomic_write_bytes(main_path, _csv_bytes(main_rows, MEMBERSHIP_COLUMNS))
    atomic_write_bytes(reserve_path, _csv_bytes(reserve_rows, MEMBERSHIP_COLUMNS))
    count_rows = _count_rows(pd.DataFrame([*main_rows, *reserve_rows]))

    tables = {
        area_config["freeze"]["counts"]: (count_rows, [
            "grouping_dimension", "grouping_value", "sampling_frame", "domain_code", "domain_name", "sample_count",
        ]),
        area_config["freeze"]["precedence_summary"]: (precedence_rows, list(precedence_rows[0])),
        area_config["freeze"]["mask_manifest"]: (mask_manifest_rows, list(mask_manifest_rows[0])),
        area_config["freeze"]["domain_pixel_totals"]: (domain_rows, list(domain_rows[0])),
        area_config["freeze"]["effective_mode_reconciliation"]: (reconciliation_rows, list(reconciliation_rows[0])),
        area_config["freeze"]["geometry_repair_audit"]: (geometry_rows, list(geometry_rows[0])),
    }
    for filename, (rows, columns) in tables.items():
        atomic_write_bytes(output_root / filename, _csv_bytes(rows, columns))
    operational_path = output_root / area_config["freeze"]["operational_geometry"]
    atomic_write_bytes(operational_path, json.dumps({
        "type": "FeatureCollection", "pixel_domain": True, "clipped_to_frozen_valid_support": True,
        "source_annotation_sha256": source_hash, "precedence_rule": PRECEDENCE_RULE,
        "features": geometry_features,
    }, indent=2, sort_keys=True).encode())
    return {
        "compiled": compiled, "main_rows": main_rows, "reserve_rows": reserve_rows,
        "geometry_rows": geometry_rows, "reconciliation_rows": reconciliation_rows,
    }


def directory_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _git_commit(project: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()


def build_contract(
    project: Path, area_config: dict, field2_config: dict, store: Field2AreaAnnotationStore,
    payload: dict, output_root: Path, result: dict, raw_hashes: dict,
) -> dict:
    output_hashes = directory_hashes(output_root)
    frozen_outputs = {
        name: {"path": str((output_root / name).relative_to(project)), "sha256": digest}
        for name, digest in output_hashes.items()
    }
    repair_counts = Counter(row["repair_method"] for row in result["geometry_rows"])
    warning_counts = Counter(
        warning for row in result["geometry_rows"] for warning in str(row["warnings"]).split("|") if warning
    )
    main_counts = Counter(row["domain_name"] for row in result["main_rows"])
    reserve_counts = Counter(row["domain_name"] for row in result["reserve_rows"])
    contract_inputs = {
        name: {"path": item["path"], "sha256": sha256(project / item["path"])}
        for name, item in area_config["inputs"].items()
    }
    return {
        "version": "field2_area_zone_contract_v2", "status": "field2_area_zone_contract_frozen",
        "freeze_timestamp_utc": utc_now(), "creation_git_commit": _git_commit(project),
        "creation_environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "rasterio": rasterio.__version__,
        },
        "annotation_revision": int(payload["revision"]), "reviewed_cube_count": 40,
        "high_confidence_cube_count": 40,
        "raw_polygon_count": sum(len(record["polygons"]) for record in payload["annotations"].values()),
        "raw_vertex_count": sum(
            len(polygon["vertices_pixel"]) for record in payload["annotations"].values()
            for polygon in record["polygons"]
        ),
        "raw_annotation_inputs": raw_hashes, "frozen_input_contracts": contract_inputs,
        "field2_source_manifest_sha256": field2_config["source_manifest_sha256"],
        "frozen_outputs": frozen_outputs,
        "operational_mask_manifest_sha256": output_hashes[area_config["freeze"]["mask_manifest"]],
        "main_count": 800, "reserve_count": 396, "reserve_coordinates_exposed": False,
        "main_reserve_intersection": 0, "invalid_support_sample_memberships": 0,
        "main_domain_counts": dict(sorted(main_counts.items())),
        "locked_reserve_domain_counts": dict(sorted(reserve_counts.items())),
        "effective_mode_reconciliation": result["reconciliation_rows"],
        "operational_geometry_policy": {
            "source_vertices_immutable": True, "source_polygons_immutable": True,
            "terminal_reconciliation_scope": "operational_only",
            "validity_repair": "deterministic_raster_polygonize_all_components",
            "all_polygonal_components_retained": True, "clipped_to_frozen_valid_support": True,
            "repair_method_counts": dict(sorted(repair_counts.items())),
            "warning_counts": dict(sorted(warning_counts.items())),
            "warnings_not_blockers": ["self_intersection", "small_overlap", "boundary_extension", "terminal_artifact"],
        },
        "domain_codes": {
            "0": "invalid_outside_valid_support", "1": "research_crop_area", "2": "alley",
            "3": "outside_research_field", "4": "uncertain_boundary", "5": "unassigned_valid_support",
        },
        "precedence_rule": "outside_research_field > alley > uncertain_boundary > research_crop_area > unassigned_valid_support",
        "biological_interpretation": {
            "research_crop_area": "contextual candidate planted-row/research geometry; not a chickpea label",
            "alley": {"classification": "supplementary", "chickpea_prohibited": True, "permitted_labels": ["ordinary_weed", "tall_grass_weed", "other_weed", "soil", "weed_soil_mixed", "uncertain", "nodata_invalid"]},
            "outside_research_field": {"classification": "outside_domain_ood", "chickpea_prohibited": True, "permitted_labels": ["ordinary_weed", "tall_grass_weed", "other_weed", "soil", "weed_soil_mixed", "uncertain", "nodata_invalid"]},
            "uncertain_boundary": "excluded from headline accuracy and reported separately",
            "unassigned_valid_support": "reported separately unless later reviewed",
            "automatic_biological_label_from_area": False,
        },
        "investigator_declarations": {
            "all_40_cubes_reviewed": True, "area_confidence": "high",
            "alley_contains_soil_and_weeds_only": True,
            "outside_field_contains_soil_and_outside_domain_weeds_only": True,
            "research_crop_area_is_not_a_pure_chickpea_mask": True,
            "chickpea_visually_small_and_difficult": True,
        },
        "prediction_free_status": {
            "checkpoint_loaded_or_deserialized": False, "model_run": False,
            "predictions_generated": False, "probabilities_generated": False,
            "pseudo_labels_generated": False, "biological_labels_assigned": False,
        },
        "deterministic_byte_identical_rerun_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--area-config", default=Path("configs/field2_area_annotation.yaml"), type=Path)
    parser.add_argument("--field2-config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text()); area_config = yaml.safe_load(args.area_config.read_text())
    field2_config = yaml.safe_load(args.field2_config.read_text()); project = Path(paths["project_root"]).resolve()
    validate_config_inputs(project, area_config)
    contract_path = project / area_config["freeze"]["contract"]
    output_root = project / area_config["freeze"]["output_root"]
    if contract_path.exists() or output_root.exists():
        raise FileExistsError("Frozen area contract/output already exists; overwrite is prohibited")
    sampling_contract_path = project / field2_config["sampling"]["sampling_contract"]
    sampling = validate_frozen_sampling_contract(project, field2_config, sampling_contract_path)
    if sampling["contract"]["reserve_release_policy"]["status"] != "locked_not_released_for_annotation":
        raise RuntimeError("Reserve protection is not active")
    store = Field2AreaAnnotationStore(
        project, project / area_config["inputs"]["review_manifest"]["path"],
        project / area_config["inputs"]["natural_rgb_manifest"]["path"],
        project / area_config["inputs"]["valid_support_manifest"]["path"],
        project / area_config["annotation"]["output_root"],
    )
    payload = store.load(); validate_ready_payload(store, payload)
    before = raw_artifact_hashes(store)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    first = Path(tempfile.mkdtemp(prefix=".field2-area-freeze-first-", dir=output_root.parent))
    second = Path(tempfile.mkdtemp(prefix=".field2-area-freeze-rerun-", dir=output_root.parent))
    try:
        result = materialize_freeze_products(project, area_config, store, payload, sampling["main"], sampling["reserve"], first / "products")
        materialize_freeze_products(project, area_config, store, payload, sampling["main"], sampling["reserve"], second / "products")
        if directory_hashes(first / "products") != directory_hashes(second / "products"):
            raise ValueError("Area freeze products are not byte-identical on deterministic rerun")
        if raw_artifact_hashes(store) != before:
            raise ValueError("Raw area annotation artifacts changed during freeze")
        (first / "products").replace(output_root)
        contract = build_contract(project, area_config, field2_config, store, payload, output_root, result, before)
        atomic_write_json(contract_path, contract)
    finally:
        shutil.rmtree(first, ignore_errors=True); shutil.rmtree(second, ignore_errors=True)
    print("Field 2 area-zone contract frozen")
    print(f"Contract: {contract_path}")
    print(f"Operational mask manifest SHA-256: {contract['operational_mask_manifest_sha256']}")
    print(f"Main domain counts: {contract['main_domain_counts']}")
    print(f"Locked reserve domain counts: {contract['locked_reserve_domain_counts']}")
    print("Raw vertices modified: 0; reserve coordinates exposed: false")
    print("Checkpoint/model/prediction/probability/pseudo-label use: false")


if __name__ == "__main__":
    main()
