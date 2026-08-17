#!/usr/bin/env python
"""Explicitly freeze completed investigator Field 2 cube-role decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_blind_review import (
    CubeRoleReviewStore,
    atomic_write_yaml,
    reject_prediction_provenance,
    validate_review_payload,
    verify_preview_hashes,
)
from chickpea_ssl.field2_readiness import sha256


def git_output(project: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=project, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze all 40 completed cube-role reviews")
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve()
    expected = list(config["expected_cube_ids"])
    manifest_path = project / config["review"]["package_manifest"]
    output_root = project / config["review"]["annotations_root"]
    store = CubeRoleReviewStore(project, manifest_path, output_root)
    payload = store.load()
    issues = validate_review_payload(payload, expected)
    invalid = {cube_id: values for cube_id, values in issues.items() if values}
    if invalid:
        raise SystemExit("REFUSED: logical review validation failed: " + str(invalid))
    incomplete = [
        cube_id for cube_id, record in payload["reviews"].items()
        if not record["reviewed"] or record["primary_role"] == "unreviewed"
    ]
    if incomplete:
        raise SystemExit(f"REFUSED: all 40 cubes must be explicitly reviewed; incomplete={incomplete}")
    preview_issues = verify_preview_hashes(store.manifest, project)
    if preview_issues:
        raise SystemExit("REFUSED: preview checksums changed: " + str(preview_issues))
    for cube_id, record in payload["reviews"].items():
        expected_hashes = store.by_cube[cube_id]
        if record["source_preview_checksums"] != json.loads(str(expected_hashes.preview_sha256_json)):
            raise SystemExit(f"REFUSED: {cube_id} saved preview checksums do not match the package")

    support_manifest_path = project / config["inputs"]["valid_support_manifest"]
    if sha256(support_manifest_path) != config["valid_support_manifest_sha256"]:
        raise SystemExit("REFUSED: frozen valid-support manifest checksum changed")
    valid_contract = yaml.safe_load((project / config["inputs"]["valid_support_contract"]).read_text())
    if valid_contract.get("source_manifest_sha256") != config["source_manifest_sha256"]:
        raise SystemExit("REFUSED: frozen Field 2 source-manifest checksum changed")
    package_contract_path = project / config["review"]["package_contract"]
    package_contract = yaml.safe_load(package_contract_path.read_text())
    if package_contract.get("package_manifest_sha256") != sha256(manifest_path):
        raise SystemExit("REFUSED: review-package manifest checksum changed")
    provenance = {
        "allowed_inputs": [
            "investigator_cube_role_reviews", "prediction_free_review_previews",
            "frozen_reflectance_valid_support", "automatic_non_biological_metadata",
        ],
        "review_json": str(store.json_path.relative_to(project)),
        "review_csv": str(store.csv_path.relative_to(project)),
        "review_audit_csv": str(store.audit_path.relative_to(project)),
        "review_package_manifest": str(manifest_path.relative_to(project)),
        "supervised_checkpoint_opened": False,
        "prediction_path": "",
        "probability_path": "",
        "embedding_path": "",
        "field1_label_source": "",
    }
    reject_prediction_provenance(provenance)
    role_rows = []
    for cube_id in expected:
        record = payload["reviews"][cube_id]
        role_rows.append({
            "cube_id": cube_id, "primary_role": record["primary_role"],
            "flags": record["flags"], "confidence": record["confidence"],
            "reviewed": True, "review_timestamp": record["review_timestamp"],
            "reviewer_identifier": record["reviewer_identifier"],
            "investigator_notes": record["investigator_notes"],
            "exclusion_reason": record["exclusion_reason"],
        })
    contract = {
        "status": "field2_cube_roles_frozen",
        "explicit_freeze_command_required": True,
        "script_git_commit": git_output(project, "rev-parse", "HEAD"),
        "benchmark_commit": config["benchmark_commit"],
        "readiness_commit": config["readiness_commit"],
        "valid_support_commit": config["valid_support_commit"],
        "cube_count": len(role_rows), "all_cubes_reviewed": True,
        "cube_roles": role_rows,
        "review_json_sha256": sha256(store.json_path),
        "review_csv_sha256": sha256(store.csv_path),
        "review_audit_sha256": sha256(store.audit_path),
        "review_overview_sha256": sha256(store.overview_path),
        "review_package_manifest_sha256": sha256(manifest_path),
        "valid_support_manifest_sha256": sha256(support_manifest_path),
        "source_manifest_sha256": valid_contract["source_manifest_sha256"],
        "proposed_sampling_counts": config["sampling"]["proposed_points_per_cube_by_role"],
        "proposed_sampling_counts_status": config["sampling"]["proposed_counts_status"],
        "provenance": provenance,
        "automatic_biological_roles_assigned": False,
        "supervised_predictions_or_probabilities_used": False,
    }
    target = project / config["review"]["role_contract"]
    atomic_write_yaml(target, contract)
    print("FROZEN: 40 investigator-reviewed Field 2 cube roles", flush=True)
    print(f"File: {target}", flush=True)


if __name__ == "__main__":
    main()
