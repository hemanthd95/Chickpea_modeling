#!/usr/bin/env python
"""Freeze accepted chickpea-only positive support for Field 1 cubes 12/14/15.

The output is deliberately not a complete binary crop mask: value 1 is frozen
chickpea support and value 0 is unknown/unlabeled. No negative or weed label is
inferred from omitted polygons, unresolved pixels, or the surrounding image.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import rasterio
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def true_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def positive_support(candidate: np.ndarray, accepted_code: int = 4) -> np.ndarray:
    """Return 1 for accepted chickpea evidence and 0 for unknown/unlabeled."""
    result = np.zeros(candidate.shape, dtype=np.uint8)
    result[candidate == accepted_code] = 1
    return result


def validate_output_hash(contract: dict, project: Path, path: Path) -> None:
    relative = str(path.relative_to(project))
    expected = contract.get("output_hashes", {}).get(relative)
    if expected != sha256(path):
        raise ValueError(f"Frozen artifact hash mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--config", default=Path("configs/chickpea_mask_refinement.yaml"), type=Path
    )
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    policy = config["trusted_expansion_chickpea_freeze"]
    if policy.get("status") != "investigator_accepted_after_imagery_review":
        raise ValueError("Trusted expansion support has not been accepted")
    if bool(policy.get("infer_weed_labels", False)):
        raise ValueError("Weed labels cannot be inferred for expansion cubes")
    if bool(policy.get("replace_authoritative_masks", False)):
        raise ValueError("Existing authoritative masks cannot be replaced here")
    if bool(policy.get("retrain_models", False)):
        raise ValueError("Model retraining is outside this freeze stage")
    if policy.get("outside_polygon_action") != "unknown_unlabeled":
        raise ValueError("Outside-polygon pixels must remain unknown/unlabeled")
    if policy.get("unresolved_pixel_action") != "unknown_unlabeled":
        raise ValueError("Unresolved pixels must remain unknown/unlabeled")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    acceptance_path = contracts / str(policy["acceptance_contract_filename"])
    candidate_contract_path = contracts / str(policy["candidate_contract_filename"])
    acceptance = yaml.safe_load(acceptance_path.read_text())
    candidates = yaml.safe_load(candidate_contract_path.read_text())
    if acceptance.get("status") != "field1_investigator_mask_acceptance_review_ready":
        raise ValueError("Imagery-backed acceptance review is not ready")
    if candidates.get("status") != "field1_investigator_mask_review_candidates_materialized":
        raise ValueError("Review candidates are not finalized")
    if acceptance.get("candidate_contract_sha256") != sha256(candidate_contract_path):
        raise ValueError("Candidate contract differs from the reviewed contract")
    for contract in (acceptance, candidates):
        if bool(contract.get("field2_accessed", True)):
            raise ValueError("A source contract does not confirm that Field 2 stayed locked")
        if bool(contract.get("authoritative_masks_modified", True)):
            raise ValueError("A source contract reports authoritative-mask modification")
        if bool(contract.get("models_retrained", True)):
            raise ValueError("A source contract reports model retraining")

    reports = (
        local / "reports" / "mask_refinement" / "investigator_mask_review_candidates"
    )
    acceptance_reports = reports / "acceptance_review"
    inventory_path = acceptance_reports / "investigator_mask_acceptance_review_inventory.csv"
    summary_path = reports / "investigator_mask_review_candidate_summary.csv"
    overview_path = acceptance_reports / "trusted_expansion_mask_acceptance_review.png"
    validate_output_hash(acceptance, project, inventory_path)
    validate_output_hash(acceptance, project, overview_path)
    validate_output_hash(candidates, project, summary_path)
    inventory = pd.read_csv(inventory_path)
    summary = pd.read_csv(summary_path)
    accepted = [str(value) for value in policy["accepted_cubes"]]
    if sorted(accepted) != sorted(acceptance.get("trusted_expansion_cubes", [])):
        raise ValueError("Accepted cubes differ from the imagery-reviewed trusted set")
    if len(accepted) != len(set(accepted)):
        raise ValueError("Accepted cube list contains duplicates")

    output_root = project / str(policy["output_root_relative"])
    output_root.mkdir(parents=True, exist_ok=True)
    accepted_code = int(policy["accepted_review_code"])
    positive_value = int(policy["positive_support_value"])
    unknown_value = int(policy["unknown_unlabeled_value"])
    if accepted_code != 4 or positive_value != 1 or unknown_value != 0:
        raise ValueError("Frozen support semantics must remain code 4 -> 1; all else -> 0")

    output_paths: list[Path] = []
    rows: list[dict] = []
    for cube_id in accepted:
        inventory_row = inventory[inventory["cube_id"] == cube_id]
        summary_row = summary[summary["cube_id"] == cube_id]
        if len(inventory_row) != 1 or len(summary_row) != 1:
            raise ValueError(f"Expected one inventory and summary row for {cube_id}")
        inventory_row = inventory_row.iloc[0]
        summary_row = summary_row.iloc[0]
        if not true_value(inventory_row["decision_eligible"]):
            raise ValueError(f"{cube_id} is not decision eligible")
        if not true_value(inventory_row["trusted_expansion_cube"]):
            raise ValueError(f"{cube_id} is not a trusted expansion cube")
        if int(inventory_row["high_confidence_weed_pixels"]) != 0:
            raise ValueError(f"{cube_id} unexpectedly contains proposed weed labels")

        candidate_path = Path(str(summary_row["review_candidate_path"]))
        validate_output_hash(candidates, project, candidate_path)
        with rasterio.open(candidate_path) as source:
            candidate = source.read(1)
            profile = source.profile.copy()
        support = positive_support(candidate, accepted_code)
        positive_count = int(support.sum())
        expected_count = int(inventory_row["high_confidence_chickpea_pixels"])
        if positive_count != expected_count:
            raise ValueError(
                f"{cube_id} support count {positive_count} differs from review {expected_count}"
            )
        output_path = output_root / f"{cube_id}_trusted_chickpea_positive_support.tif"
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(support, 1)
            destination.update_tags(
                layer_semantics="positive_support_only",
                value_1="investigator_accepted_high_confidence_chickpea",
                value_0="unknown_unlabeled_not_negative",
                weed_labels_inferred="false",
                complete_mask="false",
            )
        output_paths.append(output_path)
        rows.append({
            "cube_id": cube_id,
            "accepted_chickpea_positive_pixels": positive_count,
            "unknown_unlabeled_pixels": int(support.size - positive_count),
            "accepted_source": "investigator_polygon_plus_probability_plus_imagery_review",
            "complete_binary_mask": False,
            "weed_labels_inferred": False,
            "authoritative_existing_mask_replaced": False,
            "support_path": str(output_path),
        })
        print(
            f"Frozen {cube_id}: {positive_count:,} chickpea-positive pixels; "
            f"{support.size - positive_count:,} unknown/unlabeled",
            flush=True,
        )

    frozen_summary = pd.DataFrame(rows)
    summary_output = contracts / "field1_trusted_expansion_chickpea_support_summary.csv"
    frozen_summary.to_csv(summary_output, index=False)
    output_paths.append(summary_output)
    frozen_overview = contracts / "field1_trusted_expansion_chickpea_support_overview.png"
    shutil.copy2(overview_path, frozen_overview)
    output_paths.append(frozen_overview)

    frozen_contract = {
        "status": "field1_trusted_expansion_chickpea_positive_support_frozen",
        "field": "Field 1",
        "accepted_cubes": accepted,
        "accepted_content": "high_confidence_chickpea_positive_support_only",
        "total_accepted_chickpea_positive_pixels": int(
            frozen_summary["accepted_chickpea_positive_pixels"].sum()
        ),
        "value_semantics": {
            1: "investigator_accepted_high_confidence_chickpea",
            0: "unknown_unlabeled_not_negative",
        },
        "support_is_complete_binary_mask": False,
        "outside_polygon_pixels_are_unknown_unlabeled": True,
        "unresolved_pixels_are_unknown_unlabeled": True,
        "weed_labels_inferred": False,
        "existing_authoritative_masks_replaced": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {
            "configuration": sha256(args.config),
            "candidate_contract": sha256(candidate_contract_path),
            "acceptance_contract": sha256(acceptance_path),
            "acceptance_inventory": sha256(inventory_path),
            "imagery_backed_visual_review": sha256(overview_path),
        },
        "output_hashes": {
            str(path.relative_to(project)): sha256(path) for path in output_paths
        },
    }
    contract_path = contracts / "field1_trusted_expansion_chickpea_support_contract.yaml"
    contract_path.write_text(yaml.safe_dump(frozen_contract, sort_keys=False))

    print(f"Frozen positive-support pixels: {frozen_contract['total_accepted_chickpea_positive_pixels']:,}")
    print(f"Summary: {summary_output}")
    print(f"Frozen visual QC: {frozen_overview}")
    print(f"Contract: {contract_path}")
    print(
        "Positive support only: zero remains unknown/unlabeled, no weed or negative "
        "labels were inferred, no existing authoritative mask was replaced, no model "
        "was retrained, and Field 2 remained locked."
    )


if __name__ == "__main__":
    main()
