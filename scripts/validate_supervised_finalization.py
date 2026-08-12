#!/usr/bin/env python
"""Validate the frozen Field 1 finalization chain without opening Field 2."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from chickpea_ssl.interpretability import sha256


BENCHMARK_SHA = "da95fd59b7dc5d8ba4add96ed43b398bb7c93749"
CONTRACT_NAMES = (
    "field1_confident_supervised_benchmark_contract.yaml",
    "field1_grouped_importance_contract.yaml",
    "field1_deployment_data_contract.yaml",
    "field1_deployment_gpu_preflight_contract.yaml",
    "field1_deployment_ensemble_contract.yaml",
    "field2_compatibility_stop_contract.yaml",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_hashes(project: Path, mapping: dict[str, str], label: str) -> None:
    for relative, expected in mapping.items():
        path = project / relative
        require(path.is_file(), f"Missing {label}: {relative}")
        require(sha256(path) == expected, f"SHA-256 mismatch for {label}: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--finalization",
        default=Path("configs/supervised_finalization_v1.yaml"),
        type=Path,
    )
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    finalization = yaml.safe_load(args.finalization.read_text())
    project = Path(paths["project_root"])
    root = project / "metadata/local/contracts/supervised_finalization_v1"

    require(finalization["benchmark_commit"] == BENCHMARK_SHA, "Finalization SHA drift")
    require(finalization["field2_locked"] is True, "Field 2 finalization lock is off")
    require(paths.get("field2", {}).get("locked") is True, "Field 2 path lock is off")
    require(not str(paths.get("field2", {}).get("imagery", "")).strip(), "Field 2 imagery path is populated")
    require(not str(paths.get("field2", {}).get("tall_grass_rtk", "")).strip(), "Field 2 label path is populated")

    tag_target = subprocess.check_output(
        ["git", "rev-list", "-n", "1", finalization["benchmark_tag"]],
        cwd=project,
        text=True,
    ).strip()
    require(tag_target == BENCHMARK_SHA, "Frozen benchmark tag target drift")

    contracts: dict[str, dict] = {}
    for name in CONTRACT_NAMES:
        path = root / name
        require(path.is_file(), f"Missing finalization contract: {name}")
        contract = yaml.safe_load(path.read_text())
        require(contract.get("benchmark_commit") == BENCHMARK_SHA, f"Benchmark SHA mismatch: {name}")
        require(contract.get("field2_accessed") is False, f"Field 2 access flag is not false: {name}")
        contracts[name] = contract

    benchmark = contracts[CONTRACT_NAMES[0]]
    require(
        benchmark["status"] == "field1_confident_supervised_benchmark_frozen",
        "Benchmark is not frozen and verified",
    )
    verify_hashes(project, benchmark["source_sha256"], "benchmark source")
    benchmark_manifest = root / "field1_confident_supervised_checkpoint_manifest.csv"
    require(sha256(benchmark_manifest) == benchmark["checkpoint_manifest_sha256"], "Benchmark checkpoint manifest drift")
    benchmark_rows = pd.read_csv(benchmark_manifest)
    require(len(benchmark_rows) == 15, "Expected exactly 15 frozen nested checkpoints")
    for row in benchmark_rows.itertuples(index=False):
        path = project / row.checkpoint_path
        require(path.is_file() and sha256(path) == row.checkpoint_sha256, f"Nested checkpoint drift: {row.checkpoint_path}")

    importance = contracts[CONTRACT_NAMES[1]]
    require(importance["status"] == "field1_grouped_importance_audit_complete", "Importance audit incomplete")
    require(importance["models_retrained"] is False, "Importance audit claims retraining")
    require(importance["bands_selected_from_audit"] is False, "Importance audit altered band selection")
    verify_hashes(project, importance["output_sha256"], "importance output")

    deployment_data = contracts[CONTRACT_NAMES[2]]
    require(deployment_data["status"] == "field1_deployment_data_frozen", "Deployment data not frozen")
    for filename, expected in deployment_data["sample_sha256"].items():
        path = root / filename
        require(path.is_file() and sha256(path) == expected, f"Deployment sample drift: {filename}")
    normalization = root / "field1_deployment_normalization.csv"
    require(sha256(normalization) == deployment_data["normalization_sha256"], "Deployment normalization drift")

    preflight = contracts[CONTRACT_NAMES[3]]
    require(preflight["status"] == "two_gpu_real_patch_preflight_passed", "GPU preflight did not pass")
    require(preflight["gpu_count"] == 2, "GPU preflight did not use exactly two GPUs")
    require(preflight["device_names"] == ["NVIDIA RTX 5000 Ada Generation"], "Unexpected GPU model")

    ensemble = contracts[CONTRACT_NAMES[4]]
    require(
        ensemble["status"] == "field1_deployment_ensembles_frozen_before_field2_compatibility",
        "Deployment ensemble not frozen",
    )
    require(ensemble["field2_predictions_generated"] is False, "Field 2 predictions were generated")
    require(ensemble["cubes24_28_used_as_blind_data"] is False, "Cubes 24/28 were treated as blind data")
    manifest_path = project / ensemble["checkpoint_manifest"]
    require(sha256(manifest_path) == ensemble["checkpoint_manifest_sha256"], "Deployment checkpoint manifest drift")
    checkpoint_rows = pd.read_csv(manifest_path)
    require(len(checkpoint_rows) == 6, "Expected exactly six deployment checkpoints")
    require(set(checkpoint_rows["variant"]) == {"with_alley", "without_alley"}, "Deployment variant drift")
    require(set(checkpoint_rows["seed"]) == {42, 43, 44}, "Deployment seed drift")
    for row in checkpoint_rows.itertuples(index=False):
        path = project / row.checkpoint_path
        require(path.is_file() and sha256(path) == row.checkpoint_sha256, f"Deployment checkpoint drift: {row.checkpoint_path}")
    curve_path = project / ensemble["training_curves"]
    require(sha256(curve_path) == ensemble["training_curves_sha256"], "Deployment training-curve drift")

    stop = contracts[CONTRACT_NAMES[5]]
    require(stop["status"] == "blocked_not_georectified_or_authorized", "Field 2 stop is not active")
    for key in (
        "field2_scientific_arrays_opened",
        "field2_headers_opened",
        "field2_coordinates_or_labels_opened",
        "field2_predictions_generated",
    ):
        require(stop[key] is False, f"Unsafe Field 2 flag: {key}")
    checklist = project / stop["required_product_checklist"]
    require(sha256(checklist) == stop["required_product_checklist_sha256"], "Field 2 checklist drift")
    require(len(pd.read_csv(checklist)) == 13, "Field 2 checklist inventory is incomplete")

    print("Supervised finalization validation passed")
    print(f"Benchmark: {BENCHMARK_SHA}")
    print("Nested checkpoints: 15; deployment checkpoints: 6")
    print("Field 2: locked, unopened, unpredicted; compatibility gate blocked")


if __name__ == "__main__":
    main()
