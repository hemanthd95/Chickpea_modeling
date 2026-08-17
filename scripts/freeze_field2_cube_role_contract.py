#!/usr/bin/env python
"""Explicitly freeze completed investigator Field 2 cube-role decisions."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_blind_review import (
    CubeRoleReviewStore,
    atomic_write_bytes,
    atomic_write_yaml,
    frozen_role_table_bytes,
    reject_prediction_provenance,
    role_summary_csv_bytes,
    validate_saved_review_products,
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
    target = project / config["review"]["role_contract"]
    role_table_path = project / config["freeze"]["role_table"]
    summary_path = project / config["freeze"]["summary_table"]
    frozen_overview_path = project / config["freeze"]["frozen_overview"]
    existing_outputs = [path for path in (target, role_table_path, summary_path, frozen_overview_path) if path.exists()]
    if existing_outputs:
        raise SystemExit(f"REFUSED: immutable freeze output already exists: {existing_outputs}")
    forbidden_sampling_outputs = [
        project / config["sampling"]["sampling_frame"],
        project / config["sampling"]["sampling_contract"],
    ]
    if any(path.exists() for path in forbidden_sampling_outputs):
        raise SystemExit("REFUSED: a blind point-sampling output already exists before role freeze")
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

    totals = validate_saved_review_products(payload, store.csv_path, store.audit_path, expected)
    for name, expected_count in config["freeze"]["expected_role_totals"].items():
        if totals["primary_roles"].get(name) != int(expected_count):
            raise SystemExit(f"REFUSED: unexpected primary-role total for {name}")
    for name, expected_count in config["freeze"]["expected_flag_totals"].items():
        if totals["biological_flags"].get(name) != int(expected_count):
            raise SystemExit(f"REFUSED: unexpected biological-flag total for {name}")

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
    review_inputs = {
        "review_json": store.json_path,
        "review_csv": store.csv_path,
        "review_audit": store.audit_path,
        "review_overview": store.overview_path,
        "review_package_manifest": manifest_path,
    }
    input_hashes = {name: sha256(path) for name, path in review_inputs.items()}
    provenance = {
        "allowed_inputs": [
            "investigator_cube_role_reviews", "prediction_free_review_previews",
            "frozen_reflectance_valid_support", "automatic_non_biological_metadata",
        ],
        "prediction_free_review": True,
        "supervised_checkpoint_loaded": False,
        "biological_roles_assigned_automatically": False,
        "prediction_path": "",
        "probability_path": "",
        "embedding_path": "",
        "field1_label_source": "",
    }
    reject_prediction_provenance(provenance)
    role_rows = [deepcopy(payload["reviews"][cube_id]) for cube_id in expected]

    atomic_write_bytes(role_table_path, frozen_role_table_bytes(payload))
    atomic_write_bytes(summary_path, role_summary_csv_bytes(totals))
    store.write_overview(payload, frozen_overview_path)
    if any(sha256(path) != input_hashes[name] for name, path in review_inputs.items()):
        raise SystemExit("REFUSED: a saved review input changed while freeze outputs were being generated")
    contract = {
        "version": "field2_cube_role_contract_v1",
        "status": "field2_cube_roles_frozen",
        "explicit_freeze_command_required": True,
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_git_commit": git_output(project, "rev-parse", "HEAD"),
        "benchmark_commit": config["benchmark_commit"],
        "readiness_commit": config["readiness_commit"],
        "valid_support_commit": config["valid_support_commit"],
        "review_schema_version": payload["version"],
        "review_revision": payload["revision"],
        "review_updated_utc": payload.get("updated_utc", ""),
        "cube_count": len(role_rows), "all_cubes_reviewed": True,
        "cube_roles": role_rows,
        "role_totals": totals["primary_roles"],
        "biological_flag_totals": totals["biological_flags"],
        "confidence_totals": totals["confidence"],
        "frozen_input_products": {
            name: {"path": str(path.relative_to(project)), "sha256": input_hashes[name]}
            for name, path in review_inputs.items()
        },
        "frozen_output_products": {
            "role_table": {"path": str(role_table_path.relative_to(project)), "sha256": sha256(role_table_path)},
            "role_summary": {"path": str(summary_path.relative_to(project)), "sha256": sha256(summary_path)},
            "frozen_overview": {"path": str(frozen_overview_path.relative_to(project)), "sha256": sha256(frozen_overview_path)},
        },
        "field2_valid_support_manifest_sha256": sha256(support_manifest_path),
        "field2_source_manifest_sha256": valid_contract["source_manifest_sha256"],
        "proposed_sampling_counts": config["sampling"]["proposed_points_per_cube_by_role"],
        "proposed_sampling_counts_status": config["sampling"]["proposed_counts_status"],
        "provenance": provenance,
        "blind_point_sampling_frame_generated": False,
        "field2_categorical_labels_generated": False,
        "models_trained": False,
    }
    atomic_write_yaml(target, contract)
    print("FROZEN: 40 investigator-reviewed Field 2 cube roles", flush=True)
    print(f"File: {target}", flush=True)


if __name__ == "__main__":
    main()
