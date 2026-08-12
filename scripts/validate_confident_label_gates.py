#!/usr/bin/env python
"""Evaluate the predeclared gates for the Field 1 confident-label candidate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
from spectral.io import envi
import yaml

from chickpea_ssl.data import load_band_indices, load_records
from scripts.audit_chickpea_region_annotations import planter_alley_features
from scripts.build_confident_label_dataset import rasterize_alleys, raster_valid_spectra


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def add(rows, gate, passed, evidence):
    rows.append({"gate": gate, "passed": bool(passed), "evidence": str(evidence)})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/confident_labels_v1.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    parser.add_argument("--require-cuda-smoke", action="store_true"); args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text()); paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"]); local = project / "metadata/local"
    root = project / config["contract_root"]; output = project / config["output_root"]
    reports = project / config["report_root"] / "validation"; reports.mkdir(parents=True, exist_ok=True)
    label_contract_path = root / "field1_confident_label_candidate_contract.yaml"
    data_contract_path = root / "field1_confident_supervised_data_contract.yaml"
    label_contract = yaml.safe_load(label_contract_path.read_text()); data_contract = yaml.safe_load(data_contract_path.read_text())
    rows = []
    add(rows, "field2_locked", config["field2_locked"] and label_contract.get("field2_accessed") is False and data_contract.get("field2_accessed") is False, "all task contracts report Field 2 untouched")
    bad_inputs = [name for name, expected in label_contract["input_hashes"].items() if not (project/name).is_file() or sha256(project/name) != expected]
    bad_outputs = [name for name, expected in label_contract["output_hashes"].items() if not (project/name).is_file() or sha256(project/name) != expected]
    add(rows, "all_frozen_inputs_verified", not bad_inputs, bad_inputs or "all hashes match")
    add(rows, "all_label_outputs_verified", not bad_outputs, bad_outputs or "all hashes match")
    threshold = yaml.safe_load((local / "contracts/field1_investigator_probability_threshold_audit_contract.yaml").read_text())
    add(rows, "probability_threshold_gate", threshold.get("point_reference_gate_passed") is True and
        threshold["display_thresholds"] == {"weed_maximum_probability": 0.15, "chickpea_minimum_probability": 0.8}, threshold["display_thresholds"])
    data_bad = []
    for name, expected in {**data_contract["sample_hashes"], **data_contract["exhaustive_evaluation_hashes"]}.items():
        path = root/name
        if not path.is_file() or sha256(path) != expected: data_bad.append(name)
    normalization_path = root / "field1_confident_nested_normalization.csv"
    if sha256(normalization_path) != data_contract["normalization_sha256"]: data_bad.append(normalization_path.name)
    add(rows, "sample_evaluation_normalization_hashes", not data_bad, data_bad or "all hashes match")
    add(rows, "deterministic_sample_rerun", data_contract.get("deterministic_rerun_checksum_match") is True,
        data_contract.get("deterministic_rerun_checksum_match"))

    records = {record.cube_id: record for record in load_records(args.paths, local / "authoritative_manifest.csv")}
    bands = load_band_indices(args.bands, config["normalization"]["band_section"])
    alleys = planter_alley_features(local / "annotations/planter_tracks/field1_planter_track_annotations.geojson",
                                    config["geometry"]["annotation_seam_snap_m"])
    alley_chickpea = invalid_labels = cube32_dense = 0
    for cube_id in sorted(label_contract["cube_roles"]["sensitivity_only"] + label_contract["cube_roles"]["expansion_only"] +
                          [value for value in records if (output/"canonical_outer_evaluation"/value/"confident_labels.tif").exists()]):
        canonical_path = output / "canonical_outer_evaluation" / cube_id / "confident_labels.tif"
        if not canonical_path.exists(): continue
        with rasterio.open(canonical_path) as dataset:
            labels = dataset.read(1); transform = dataset.transform
        image = envi.open(str(records[cube_id].header), str(records[cube_id].data)); valid = raster_valid_spectra(image.open_memmap(), bands)
        full, _ = rasterize_alleys(alleys.get(cube_id, []), labels.shape, transform,
                                  config["geometry"]["alley_inward_buffer_m"])
        alley_chickpea += int(((labels == 1) & full).sum()); invalid_labels += int(((labels != 255) & ~valid).sum())
        if cube_id == "field1_cube32":
            with rasterio.open(output/"canonical_outer_evaluation"/cube_id/"provenance.tif") as dataset: source = dataset.read(1)
            cube32_dense += int(np.isin(source, [6, 7]).sum())
    add(rows, "zero_chickpea_inside_full_alley", alley_chickpea == 0, alley_chickpea)
    add(rows, "zero_labels_on_nodata", invalid_labels == 0, invalid_labels)
    add(rows, "cube32_no_dense_probability_labels", cube32_dense == 0, cube32_dense)

    excluded = set(config["cube_roles"]["sensitivity_only"] + config["cube_roles"]["expansion_only"] +
                   config["cube_roles"]["transfer_warning_dense_sensitivity_only"])
    leakage = duplicate_ground = missing_class = excluded_rows = 0
    for outer in config["training"]["outer_folds"]:
        for variant in ("with_alley", "without_alley"):
            frame = pd.read_csv(root / f"outer{outer}_{variant}_samples.csv")
            train_folds = set(frame.loc[frame.nested_role == "train", "fold"])
            validation_folds = set(frame.loc[frame.nested_role == "inner_validation", "fold"])
            leakage += int(bool(train_folds & validation_folds) or outer in train_folds | validation_folds)
            duplicate_ground += int(frame.duplicated(["ground_x", "ground_y"]).sum())
            excluded_rows += int(frame.cube_id.isin(excluded).sum())
            for _, group in frame.groupby("nested_role"):
                missing_class += int(set(group.class_id) != {0, 1, 2})
        evaluation = pd.read_csv(root / f"outer{outer}_exhaustive_evaluation.csv.gz")
        excluded_rows += int(evaluation.cube_id.isin(excluded).sum())
        missing_class += int(set(evaluation.class_id) != {0, 1, 2})
    add(rows, "zero_train_inner_outer_fold_overlap", leakage == 0, leakage)
    add(rows, "zero_duplicate_ground_cells_across_roles", duplicate_ground == 0, duplicate_ground)
    add(rows, "valid_class_support_every_role_and_evaluation_fold", missing_class == 0, missing_class)
    add(rows, "no_primary_excluded_cube_rows", excluded_rows == 0, excluded_rows)

    reference = pd.read_csv(root / "fold_specific_reference_use.csv")
    reference_leakage = int(((reference.reference_spatial_fold == reference.outer_fold) |
                             (reference.reference_spatial_fold == reference.inner_validation_fold)).sum())
    support = reference.groupby(["outer_fold", "kind"]).size().unstack(fill_value=0)
    add(rows, "fold_specific_reference_separation", reference_leakage == 0 and (support > 0).all().all(),
        f"leakage={reference_leakage}; minimum class support={int(support.min().min())}")
    patch_qc = pd.read_csv(project / config["report_root"] / "sampling/patch_qc.csv")
    add(rows, "patch_qc_all_classes_roles", patch_qc.gate_passed.astype(str).str.lower().eq("true").all() and
        set(patch_qc.class_id) == {0, 1, 2}, f"rows={len(patch_qc)}")

    smoke_contract_path = root / "field1_confident_cuda_smoke_contract.yaml"
    smoke_passed = False
    if smoke_contract_path.exists():
        smoke_contract = yaml.safe_load(smoke_contract_path.read_text())
        smoke_passed = smoke_contract.get("status") == "cuda_smoke_passed" and smoke_contract.get("checkpoint_reload_passed") is True
    add(rows, "cuda_real_data_smoke", smoke_passed, smoke_contract_path if smoke_passed else "not yet passed")
    result = pd.DataFrame(rows); report_path = reports / "validation_gates.csv"; result.to_csv(report_path, index=False)
    pre_cuda_pass = bool(result[result.gate != "cuda_real_data_smoke"].passed.all())
    all_pass = bool(result.passed.all())
    contract = {"status": "all_validation_gates_passed" if all_pass else "validation_gates_pending",
                "field": "Field 1", "field2_accessed": False, "pre_cuda_gates_passed": pre_cuda_pass,
                "cuda_smoke_passed": smoke_passed, "all_gates_passed": all_pass,
                "report_sha256": sha256(report_path), "failed_gates": result.loc[~result.passed, "gate"].tolist()}
    contract_path = root / "field1_confident_validation_gate_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(result.to_string(index=False)); print(f"Pre-CUDA gates passed: {pre_cuda_pass}; all gates passed: {all_pass}")
    if args.require_cuda_smoke and not all_pass: raise SystemExit(2)
    if not pre_cuda_pass: raise SystemExit(1)


if __name__ == "__main__": main()
