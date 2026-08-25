#!/usr/bin/env python
"""Validate and freeze the accepted Field 1 spatial-fold contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/spatial_splits.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    configuration = yaml.safe_load(args.config.read_text())
    candidate = configuration["candidate_split"]
    if candidate["status"] != "accepted_pending_contract_materialization":
        raise ValueError("Candidate split has not been explicitly accepted")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    reports = local / "reports" / "spatial_grouping"
    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest": local / "authoritative_manifest.csv",
        "membership": reports / "spatial_group_membership.csv",
        "class_summary": reports / "spatial_group_class_summary.csv",
        "candidate_assignments": reports / "candidate_spatial_fold_assignments.csv",
        "candidate_summary": reports / "candidate_spatial_fold_summary.csv",
        "candidate_metadata": reports / "candidate_spatial_fold_metadata.csv",
        "candidate_boundaries": reports / "candidate_spatial_fold_boundaries.csv",
        "candidate_preview": reports / "candidate_spatial_folds_overview.png",
        "configuration": args.config,
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Contract inputs are missing: {missing}")

    assignments = pd.read_csv(files["candidate_assignments"])
    summary = pd.read_csv(files["candidate_summary"])
    membership = pd.read_csv(files["membership"])
    expected_groups = set(membership["spatial_group_id"])
    assigned_groups = set(assignments["spatial_group_id"])
    if assignments["spatial_group_id"].duplicated().any():
        raise ValueError("Candidate contains duplicate spatial group assignments")
    if assigned_groups != expected_groups:
        raise ValueError("Candidate does not assign every indexed spatial group exactly once")
    if sorted(assignments["fold"].unique()) != list(range(1, int(candidate["folds"]) + 1)):
        raise ValueError("Candidate fold IDs are incomplete")

    acceptance = candidate["acceptance"]
    checks = {
        "connected_components": (
            not acceptance["require_one_connected_component_per_fold"]
            or bool((summary["connected_components"] == 1).all())
        ),
        "chickpea_share": bool(summary["chickpea_pixel_observations_share"].between(
            acceptance["minimum_chickpea_share"], acceptance["maximum_chickpea_share"]
        ).all()),
        "labeled_observation_share": bool(summary["labeled_pixel_observations_share"].between(
            acceptance["minimum_labeled_observation_share"],
            acceptance["maximum_labeled_observation_share"],
        ).all()),
        "spatial_group_share": bool(summary["spatial_groups_share"].between(
            acceptance["minimum_spatial_group_share"],
            acceptance["maximum_spatial_group_share"],
        ).all()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Candidate failed acceptance gates: {failed}")

    frozen_assignments = contracts / "field1_spatial_folds.csv"
    frozen_boundaries = contracts / "field1_spatial_fold_boundaries.csv"
    frozen_preview = contracts / "field1_spatial_folds_overview.png"
    shutil.copy2(files["candidate_assignments"], frozen_assignments)
    shutil.copy2(files["candidate_boundaries"], frozen_boundaries)
    shutil.copy2(files["candidate_preview"], frozen_preview)
    contract = {
        "status": "frozen_field1_development_contract",
        "field": "Field 1",
        "field2_accessed": False,
        "folds": int(candidate["folds"]),
        "block_size_m": float(configuration["grouping"]["block_size_m"]),
        "boundary_exclusion_m": float(candidate["boundary_exclusion_m"]),
        "acceptance_checks": checks,
        "assignment_sha256": sha256(frozen_assignments),
        "boundary_sha256": sha256(frozen_boundaries),
        "input_sha256": {name: sha256(path) for name, path in files.items()},
        "class_precedence": "weed > chickpea > soil > unlabeled",
        "notes": [
            "All observations of one global map group remain in one fold.",
            "Repeated cube observations are retained within their shared fold.",
            "The 0.30 m boundary exclusion is applied later during sample extraction.",
        ],
    }
    contract_path = contracts / "field1_spatial_fold_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"Frozen groups: {len(assignments)}")
    print(f"Folds: {assignments['fold'].nunique()}")
    print(f"Assignment SHA-256: {contract['assignment_sha256']}")
    print(f"Contract: {contract_path}")
    print(f"Visual QC: {frozen_preview}")
    print("Field 2 was not accessed; no model training occurred.")


if __name__ == "__main__":
    main()
