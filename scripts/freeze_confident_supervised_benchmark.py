#!/usr/bin/env python
"""Verify and freeze the immutable da95fd5 Field 1 supervised benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from spectral.io import envi
import yaml

from chickpea_ssl.data import load_band_indices, load_records
from chickpea_ssl.interpretability import sha256


EXPECTED_SHA = "da95fd59b7dc5d8ba4add96ed43b398bb7c93749"
EXPECTED_METRICS = {
    "confident_with_alley": {"balanced_accuracy": 0.932545, "macro_f1": 0.876089},
    "confident_without_alley": {"balanced_accuracy": 0.884095, "macro_f1": 0.835364},
    "legacy_historical_labels": {"balanced_accuracy": 0.785344, "macro_f1": 0.648250},
}


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=PROJECT_ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/confident_labels_v1.yaml"), type=Path)
    parser.add_argument("--finalization", default=Path("configs/supervised_finalization_v1.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    args = parser.parse_args()
    finalization = yaml.safe_load(args.finalization.read_text())
    if finalization["benchmark_commit"] != EXPECTED_SHA or not finalization["field2_locked"]:
        raise ValueError("Finalization configuration does not freeze da95fd5 with Field 2 locked")
    if git("rev-parse", "supervised-confident-v1-da95fd5^{}") != EXPECTED_SHA:
        raise ValueError("Benchmark tag does not resolve to da95fd5")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_SHA, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("Current work does not descend from the immutable benchmark")

    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    confident = yaml.safe_load(args.config.read_text())
    contract_root = project / confident["contract_root"]
    local_contracts = project / "metadata/local/contracts"
    gate = yaml.safe_load((contract_root / "field1_confident_validation_gate_contract.yaml").read_text())
    label = yaml.safe_load((contract_root / "field1_confident_label_candidate_contract.yaml").read_text())
    checkpoints = yaml.safe_load((contract_root / "field1_confident_supervised_checkpoints_contract.yaml").read_text())
    evaluation = yaml.safe_load((contract_root / "field1_confident_paired_evaluation_contract.yaml").read_text())
    data = yaml.safe_load((contract_root / "field1_confident_supervised_data_contract.yaml").read_text())
    if not gate.get("all_gates_passed") or any(c.get("field2_accessed") is not False for c in (label, checkpoints, evaluation, data)):
        raise ValueError("Frozen validation or Field 2 lock gate failed")
    if label.get("code_commit_at_generation") != EXPECTED_SHA:
        raise ValueError("Label contract was not generated from da95fd5")
    if checkpoints.get("primary_architecture") != "center_context_fusion" or checkpoints.get("primary_checkpoint_count") != 15:
        raise ValueError("Unexpected primary architecture or checkpoint count")

    metric_path = project / confident["report_root"] / "evaluation/primary_equal_spatial_group_metrics.csv"
    metrics = pd.read_csv(metric_path)
    ensemble = metrics[metrics.seed.astype(str) == "ensemble"].set_index("model")
    for model, expected in EXPECTED_METRICS.items():
        for metric, value in expected.items():
            observed = float(ensemble.loc[model, metric])
            if not np.isclose(observed, value, atol=5e-7):
                raise ValueError(f"Reported benchmark mismatch: {model} {metric}: {observed} != {value}")

    records = load_records(args.paths, project / "metadata/local/authoritative_manifest.csv")
    primary_cubes = set(data["primary_cubes"])
    selected_records = [record for record in records if record.cube_id in primary_cubes]
    band_indices = load_band_indices(args.bands, confident["normalization"]["band_section"])
    wavelength_vectors = []
    for record in selected_records:
        metadata = envi.open(str(record.header), str(record.data)).metadata
        wavelength_vectors.append(np.asarray(metadata["wavelength"], dtype=float)[band_indices])
    if not wavelength_vectors or any(not np.array_equal(wavelength_vectors[0], vector) for vector in wavelength_vectors[1:]):
        raise ValueError("Primary Field 1 wavelength vectors are not identical")
    wavelengths = wavelength_vectors[0]
    if len(wavelengths) != 111 or not np.isclose(wavelengths[0], 401.84, atol=.01) or not np.isclose(wavelengths[-1], 869.34, atol=.01):
        raise ValueError("Frozen 111-band wavelength interval mismatch")

    training_path = project / confident["report_root"] / "evaluation/training_run_summary.csv"
    training = pd.read_csv(training_path)
    primary = training[training.variant == "with_alley"].copy()
    if len(primary) != 15 or primary.groupby(["outer_fold", "seed"]).size().ne(1).any():
        raise ValueError("Expected one primary checkpoint per fold and seed")
    manifest_rows = []
    for row in primary.itertuples():
        checkpoint_path = Path(row.checkpoint)
        actual = sha256(checkpoint_path)
        key = f"outer{int(row.outer_fold)}:with_alley:seed{int(row.seed)}"
        if actual != row.checkpoint_sha256 or actual != checkpoints["checkpoint_hashes"][key]:
            raise ValueError(f"Checkpoint mismatch: {checkpoint_path}")
        manifest_rows.append({
            "benchmark_commit": EXPECTED_SHA,
            "variant": "with_alley",
            "architecture": "center_context_fusion",
            "outer_fold": int(row.outer_fold),
            "seed": int(row.seed),
            "checkpoint_path": str(checkpoint_path.relative_to(project)),
            "checkpoint_sha256": actual,
        })

    output_root = project / "metadata/local/contracts/supervised_finalization_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "field1_confident_supervised_checkpoint_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    source_paths = [
        args.config, args.finalization, args.bands, Path("configs/data_decisions.yaml"),
        Path("chickpea_ssl/model.py"),
        contract_root / "field1_confident_label_candidate_contract.yaml",
        contract_root / "field1_confident_supervised_data_contract.yaml",
        contract_root / "field1_confident_supervised_checkpoints_contract.yaml",
        contract_root / "field1_confident_paired_evaluation_contract.yaml",
        contract_root / "field1_confident_validation_gate_contract.yaml",
        local_contracts / "field1_spatial_fold_contract.yaml",
        local_contracts / "field1_supervised_primary_evaluation_protocol.yaml",
    ]
    contract = {
        "status": "field1_confident_supervised_benchmark_frozen",
        "benchmark_commit": EXPECTED_SHA,
        "benchmark_tag": "supervised-confident-v1-da95fd5",
        "field": "Field 1",
        "field2_accessed": False,
        "field2_locked_during_development": True,
        "architecture": "center_context_fusion",
        "model_input_channels": "reflectance_only_no_geometry_or_index_channels",
        "band_count": 111,
        "wavelength_min_nm": float(wavelengths[0]),
        "wavelength_max_nm": float(wavelengths[-1]),
        "patch_size_pixels": int(confident["sampling"]["patch_size_pixels"]),
        "class_mapping": confident["class_mapping"],
        "confident_label_thresholds": confident["probability"],
        "alley_precedence": label["precedence"],
        "outer_folds": checkpoints["outer_folds"],
        "seeds": checkpoints["seeds"],
        "normalization_scope": data["normalization_scope"],
        "evaluation_aggregation": "equal_weight_frozen_5m_spatial_groups",
        "reported_metrics": EXPECTED_METRICS,
        "source_sha256": {str(path): sha256(project / path if not path.is_absolute() else path) for path in source_paths},
        "checkpoint_manifest": str(manifest_path.relative_to(project)),
        "checkpoint_manifest_sha256": sha256(manifest_path),
        "evaluation_metrics_sha256": sha256(metric_path),
    }
    contract_path = output_root / "field1_confident_supervised_benchmark_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"Frozen benchmark verified at {EXPECTED_SHA}")
    print(f"Primary checkpoints verified: {len(manifest_rows)}")
    print(f"Contract: {contract_path}")


if __name__ == "__main__":
    main()
