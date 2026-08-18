#!/usr/bin/env python
"""Read-only validation of frozen Field 2 source/readiness and valid support."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from chickpea_ssl.field2_readiness import ReadOnlySourceGuard, sha256, source_snapshot


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--readiness-config", default=Path("configs/field2_readiness.yaml"), type=Path)
    parser.add_argument("--support-config", default=Path("configs/field2_valid_support.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text()); readiness_config = yaml.safe_load(args.readiness_config.read_text())
    support_config = yaml.safe_load(args.support_config.read_text()); project = Path(paths["project_root"]).resolve()
    readiness_path = project / readiness_config["contract_path"]
    readiness = yaml.safe_load(readiness_path.read_text())
    require(readiness["status"] == "field2_georectified_inventory_complete_annotation_blocked", "Readiness status changed")
    require(readiness["source_file_count"] == 240 and readiness["cube_count"] == 40, "Readiness inventory changed")
    require(readiness["source_snapshot_comparison"] == "pass_identical", "Frozen readiness snapshot did not pass")
    require(sha256(project / readiness["configuration_path"]) == readiness["configuration_sha256"], "Readiness configuration hash changed")
    for relative, expected in readiness["output_sha256"].items():
        require(sha256(project / relative) == expected, f"Readiness output changed: {relative}")
    roots = tuple(Path(value) for value in paths["field2"]["readiness_roots"])
    current = {str(item["relative_path"]): str(item["sha256"]) for item in source_snapshot(ReadOnlySourceGuard(roots), project)}
    frozen_frame = pd.read_csv(project / "metadata/local/reports/field2_readiness/field2_source_snapshot_before.csv")
    frozen = dict(zip(frozen_frame.relative_path.astype(str), frozen_frame.sha256.astype(str)))
    require(current == frozen, "Current Field 2 source snapshot differs from frozen readiness snapshot")
    support_path = project / support_config["contract_path"]
    support = yaml.safe_load(support_path.read_text())
    require(support["status"] == "field2_valid_support_frozen_all_cubes_annotation_ready", "Valid-support status changed")
    require(support["annotation_ready_cube_count"] == 40 and support["annotation_blocked_cube_count"] == 0, "Valid-support cube counts changed")
    require(support["source_snapshot_comparison"] == "pass_identical", "Valid-support source snapshot did not pass")
    require(support["class_information_used"] is False and support["model_inference_used"] is False, "Valid-support provenance changed")
    require(sha256(project / support["configuration_path"]) == support["configuration_sha256"], "Valid-support configuration hash changed")
    require(sha256(readiness_path) == support["readiness_contract_sha256"], "Readiness contract hash changed")
    for relative, expected in support["output_sha256"].items():
        require(sha256(project / relative) == expected, f"Valid-support output changed: {relative}")
    manifest_path = project / support_config["output_root"] / "field2_valid_support_mask_manifest.csv"
    require(sha256(manifest_path) == "1f48ffe67d7ad2cbf95cca5c2e6e98d15245843435c519bbb8a67841c2bbf3e3", "Valid-support manifest hash changed")
    manifest = pd.read_csv(manifest_path)
    require(len(manifest) == 40 and manifest.validation_status.eq("pass").all(), "Valid-support manifest status changed")
    for row in manifest.itertuples(index=False):
        require(sha256(project / row.mask_path) == row.mask_sha256, f"Valid-support mask changed: {row.cube_id}")
    print("Field 2 source/readiness contract validation passed: 240 source files, 40 cubes")
    print("Field 2 valid-support contract validation passed: 40 masks, all hashes exact")
    print("Source snapshot current == frozen before snapshot")


if __name__ == "__main__": main()
