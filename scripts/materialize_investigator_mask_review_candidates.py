#!/usr/bin/env python
"""Materialize conservative, review-only chickpea/weed candidate layers.

This stage combines the frozen polygon/soil support with continuous
investigator-guided probabilities. It deliberately does not create an
authoritative training mask. Polygon edges, transfer-warning cubes, and the
middle probability interval stay unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records


# Review codes are intentionally distinct from the authoritative class IDs.
CLASS_NAMES = {
    0: "outside_polygon_prior",
    1: "polygon_core_soil",
    2: "high_confidence_weed_review",
    3: "unresolved_polygon_core_vegetation",
    4: "high_confidence_chickpea_review",
    5: "unresolved_polygon_edge",
    6: "explicit_alley_excluded",
    255: "nodata_or_missing_source",
}
DISPLAY_COLORS = [
    "#000000", "#8B5A2B", "#6D28D9", "#D1D5DB",
    "#16A34A", "#FDE68A", "#F59E0B", "#DC2626",
]
CMAP = ListedColormap(DISPLAY_COLORS)
NORM = BoundaryNorm(np.arange(-0.5, 8.5, 1), CMAP.N)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_review_candidate(
    support: np.ndarray,
    probability: np.ndarray,
    weed_threshold: float,
    chickpea_threshold: float,
    decision_eligible: bool,
) -> np.ndarray:
    """Combine frozen support and probability evidence into review codes."""
    if support.shape != probability.shape:
        raise ValueError("Support and probability rasters must have equal shapes")
    if not 0 <= weed_threshold < chickpea_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= weed < chickpea <= 1")

    result = np.zeros(support.shape, dtype=np.uint8)
    result[support == 1] = 1
    result[np.isin(support, [3, 4])] = 5
    result[support == 5] = 6
    result[support == 255] = 255

    core = support == 2
    result[core] = 3
    valid_probability = np.isfinite(probability) & (probability >= 0) & (probability <= 1)
    if decision_eligible:
        result[core & valid_probability & (probability <= weed_threshold)] = 2
        result[core & valid_probability & (probability >= chickpea_threshold)] = 4
    return result


def display_classes(candidate: np.ndarray) -> np.ndarray:
    result = candidate.copy()
    result[result == 255] = 7
    return result


def validate_output_hash(contract: dict, project: Path, path: Path) -> None:
    hashes = contract.get("output_hashes", {})
    relative = str(path.relative_to(project))
    expected = hashes.get(relative, hashes.get(path.name))
    if expected != sha256(path):
        raise ValueError(f"Frozen artifact hash mismatch: {path}")


def true_value(value: object) -> bool:
    """Interpret bool-like CSV values without treating 'False' as true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


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
    policy = config["investigator_mask_review_candidate"]
    if policy.get("status") != "enabled_after_threshold_audit_gate":
        raise ValueError("Investigator mask review candidate stage is not enabled")
    for forbidden in (
        "automatically_accept_review_candidates",
        "automatically_replace_authoritative_masks",
        "automatically_retrain_models",
    ):
        if bool(policy.get(forbidden, False)):
            raise ValueError(f"{forbidden} must remain false")
    if policy.get("polygon_edge_action") != "unresolved":
        raise ValueError("Polygon edges must remain unresolved")
    if policy.get("transfer_warning_action") != "unresolved":
        raise ValueError("Transfer-warning cubes must remain unresolved")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    threshold_contract_path = contracts / str(policy["threshold_contract_filename"])
    probability_contract_path = contracts / str(policy["probability_contract_filename"])
    candidate_contract_path = contracts / str(policy["candidate_contract_filename"])
    threshold_contract = yaml.safe_load(threshold_contract_path.read_text())
    probability_contract = yaml.safe_load(probability_contract_path.read_text())
    candidate_contract = yaml.safe_load(candidate_contract_path.read_text())
    if threshold_contract.get("status") != "field1_investigator_probability_threshold_audit_complete":
        raise ValueError("Probability threshold audit is not complete")
    if not bool(threshold_contract.get("point_reference_gate_passed", False)):
        raise ValueError("Point-reference threshold gate did not pass")
    if probability_contract.get("status") != "field1_investigator_chickpea_probability_review_materialized":
        raise ValueError("Probability review contract is not finalized")
    if candidate_contract.get("status") != "review_only_polygon_guided_candidates_materialized":
        raise ValueError("Polygon-guided candidate contract is not finalized")

    weed_threshold = float(policy["weed_maximum_probability"])
    chickpea_threshold = float(policy["chickpea_minimum_probability"])
    frozen_thresholds = threshold_contract["display_thresholds"]
    if not np.isclose(weed_threshold, float(frozen_thresholds["weed_maximum_probability"])):
        raise ValueError("Configured weed threshold differs from the audited threshold")
    if not np.isclose(chickpea_threshold, float(frozen_thresholds["chickpea_minimum_probability"])):
        raise ValueError("Configured chickpea threshold differs from the audited threshold")

    probability_reports = (
        local / "reports" / "mask_refinement"
        / "investigator_chickpea_probability_review"
    )
    probability_summary_path = probability_reports / "investigator_chickpea_probability_summary.csv"
    validate_output_hash(probability_contract, project, probability_summary_path)
    probability_summary = pd.read_csv(probability_summary_path)
    if len(probability_summary) != 19 or probability_summary["cube_id"].duplicated().any():
        raise ValueError("Probability summary must contain 19 unique cubes")

    support_root = project / str(
        config["polygon_guided_candidate_materialization"]["output_root_relative"]
    )
    output_root = project / str(policy["output_root_relative"])
    reports = (
        local / "reports" / "mask_refinement"
        / "investigator_mask_review_candidates"
    )
    individual = reports / "individual"
    output_root.mkdir(parents=True, exist_ok=True)
    individual.mkdir(parents=True, exist_ok=True)

    records = {
        record.cube_id: record
        for record in load_records(args.paths, local / "authoritative_manifest.csv")
    }
    warning_cubes = {
        str(value) for value in threshold_contract.get("transfer_warning_cubes", [])
    }
    eligible_roles = {str(value) for value in policy["decision_eligible_roles"]}
    summary_rows: list[dict] = []
    output_paths: list[Path] = []
    source_hashes: dict[str, str] = {}
    overview = []

    ordered = probability_summary.assign(
        cube_number=lambda frame: frame["cube_id"].str.extract(r"(\d+)$")[0].astype(int)
    ).sort_values("cube_number")
    for number, row in enumerate(ordered.itertuples(index=False), 1):
        cube_id = str(row.cube_id)
        role = str(row.analysis_role)
        warning = cube_id in warning_cubes or true_value(row.transfer_warning)
        decision_eligible = role in eligible_roles and not warning
        probability_path = Path(str(row.probability_path))
        validate_output_hash(probability_contract, project, probability_path)
        support_path = support_root / cube_id / "polygon_guided_candidate_support.tif"
        if not support_path.exists():
            raise FileNotFoundError(support_path)

        with rasterio.open(probability_path) as probability_dataset:
            probability = probability_dataset.read(1).astype(np.float32)
            profile = probability_dataset.profile.copy()
            probability_transform = probability_dataset.transform
            probability_crs = probability_dataset.crs
        with rasterio.open(support_path) as support_dataset:
            support = support_dataset.read(1)
            if support_dataset.transform != probability_transform or support_dataset.crs != probability_crs:
                raise ValueError(f"Probability/support georeferencing mismatch for {cube_id}")

        review = build_review_candidate(
            support, probability, weed_threshold, chickpea_threshold, decision_eligible
        )
        cube_root = output_root / cube_id
        cube_root.mkdir(parents=True, exist_ok=True)
        review_path = cube_root / "investigator_mask_review_candidate.tif"
        profile.update(dtype="uint8", count=1, nodata=255, compress="deflate", predictor=2)
        with rasterio.open(review_path, "w", **profile) as dataset:
            dataset.write(review, 1)
            dataset.update_tags(
                status="review_only_not_authoritative",
                decision_eligible=str(decision_eligible).lower(),
                analysis_role=role,
                transfer_warning=str(warning).lower(),
                weed_maximum_probability=f"{weed_threshold:.6f}",
                chickpea_minimum_probability=f"{chickpea_threshold:.6f}",
                class_legend=json.dumps(CLASS_NAMES, sort_keys=True),
            )
        output_paths.append(review_path)
        source_hashes[f"probability::{cube_id}"] = sha256(probability_path)
        source_hashes[f"support::{cube_id}"] = sha256(support_path)

        counts = {value: int((review == value).sum()) for value in CLASS_NAMES}
        core_total = counts[2] + counts[3] + counts[4]
        old_chickpea_to_weed = np.nan
        old_weed_to_chickpea = np.nan
        record = records.get(cube_id)
        if record is not None and record.chickpea_mask is not None and record.weed_mask is not None:
            historical = authoritative_class_map(record)
            if historical.shape != review.shape:
                raise ValueError(f"Historical-label shape mismatch for {cube_id}")
            old_chickpea_to_weed = int(((historical == 1) & (review == 2)).sum())
            old_weed_to_chickpea = int(((historical == 2) & (review == 4)).sum())

        summary_rows.append({
            "cube_id": cube_id,
            "analysis_role": role,
            "transfer_warning": warning,
            "decision_eligible": decision_eligible,
            "candidate_disposition": (
                "primary_visual_review_candidate" if decision_eligible
                else "diagnostic_only_or_forced_unresolved"
            ),
            "core_soil_pixels": counts[1],
            "high_confidence_weed_review_pixels": counts[2],
            "unresolved_core_vegetation_pixels": counts[3],
            "high_confidence_chickpea_review_pixels": counts[4],
            "unresolved_edge_pixels": counts[5],
            "alley_excluded_pixels": counts[6],
            "nodata_or_missing_source_pixels": counts[255],
            "core_vegetation_pixels": core_total,
            "high_confidence_weed_fraction_of_core": counts[2] / core_total if core_total else np.nan,
            "unresolved_fraction_of_core": counts[3] / core_total if core_total else np.nan,
            "high_confidence_chickpea_fraction_of_core": counts[4] / core_total if core_total else np.nan,
            "historical_authoritative_chickpea_to_high_confidence_weed": old_chickpea_to_weed,
            "historical_authoritative_weed_to_high_confidence_chickpea": old_weed_to_chickpea,
            "review_candidate_path": str(review_path),
            "authoritative_mask_modified": False,
        })

        step = max(1, math.ceil(max(review.shape) / 650))
        tile = display_classes(review[::step, ::step])
        preview_path = individual / f"{cube_id}_investigator_mask_review_candidate.png"
        figure, axis = plt.subplots(figsize=(6, 8), constrained_layout=True)
        axis.imshow(tile, cmap=CMAP, norm=NORM, interpolation="nearest")
        axis.axis("off")
        axis.set_title(
            f"{cube_id} | {role} | decision eligible={decision_eligible}\n"
            f"weed={counts[2]:,}; unresolved={counts[3]:,}; chickpea={counts[4]:,}",
            fontsize=10, color="#B91C1C" if warning else "black",
        )
        figure.savefig(preview_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(preview_path)
        overview.append((cube_id, role, warning, decision_eligible, tile, counts))
        print(
            f"Prepared {number}/{len(ordered)} {cube_id}: eligible={decision_eligible}; "
            f"weed={counts[2]:,}; unresolved={counts[3]:,}; chickpea={counts[4]:,}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = reports / "investigator_mask_review_candidate_summary.csv"
    summary.to_csv(summary_path, index=False)
    output_paths.append(summary_path)

    role_summary = (
        summary.groupby(["analysis_role", "decision_eligible"], dropna=False)
        [[
            "core_soil_pixels", "high_confidence_weed_review_pixels",
            "unresolved_core_vegetation_pixels",
            "high_confidence_chickpea_review_pixels", "unresolved_edge_pixels",
        ]]
        .sum().reset_index()
    )
    role_summary_path = reports / "investigator_mask_review_candidate_by_role.csv"
    role_summary.to_csv(role_summary_path, index=False)
    output_paths.append(role_summary_path)

    columns = 4
    rows = math.ceil(len(overview) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows), constrained_layout=True)
    flat = np.asarray(axes).reshape(-1)
    for axis in flat:
        axis.axis("off")
        axis.set_facecolor("black")
    for axis, (cube_id, role, warning, eligible, tile, counts) in zip(flat, overview):
        axis.imshow(tile, cmap=CMAP, norm=NORM, interpolation="nearest")
        qualifier = "PRIMARY REVIEW" if eligible else ("TRANSFER WARNING" if warning else "DIAGNOSTIC ONLY")
        axis.set_title(
            f"{cube_id} | {qualifier}\nW {counts[2]:,} | U {counts[3]:,} | C {counts[4]:,}",
            color="#B91C1C" if warning else "black", fontsize=8,
        )
    figure.suptitle(
        "Field 1 conservative investigator mask review candidates\n"
        f"purple=weed P≤{weed_threshold:.2f}; gray=unresolved; green=chickpea P≥{chickpea_threshold:.2f}; "
        "brown=soil; pale yellow=edge unresolved; orange=alley",
        fontsize=13,
    )
    overview_path = reports / "investigator_mask_review_candidates_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    output_paths.append(overview_path)

    primary = summary[summary["decision_eligible"]]
    contract = {
        "status": "field1_investigator_mask_review_candidates_materialized",
        "field": "Field 1",
        "class_legend": CLASS_NAMES,
        "weed_maximum_probability": weed_threshold,
        "chickpea_minimum_probability": chickpea_threshold,
        "decision_eligible_roles": sorted(eligible_roles),
        "decision_eligible_cubes": sorted(primary["cube_id"].astype(str).tolist()),
        "transfer_warning_cubes_forced_unresolved": sorted(warning_cubes),
        "nonprimary_cubes_are_diagnostic_only": True,
        "polygon_edges_are_unresolved": True,
        "candidate_geotiffs_are_authoritative": False,
        "visual_acceptance_completed": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {
            "configuration": sha256(args.config),
            "threshold_contract": sha256(threshold_contract_path),
            "probability_contract": sha256(probability_contract_path),
            "polygon_candidate_contract": sha256(candidate_contract_path),
            **source_hashes,
        },
        "output_hashes": {
            str(path.relative_to(project)): sha256(path) for path in output_paths
        },
    }
    contract_path = contracts / "field1_investigator_mask_review_candidate_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print("\nReview-candidate support by role:")
    print(role_summary.to_string(index=False))
    print(f"Decision-eligible primary cubes: {len(primary)}")
    print(f"Review-only GeoTIFF root: {output_root}")
    print(f"Summary: {summary_path}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print(
        "Review only: no candidate was accepted as an authoritative mask, "
        "no model was retrained, and Field 2 remained locked."
    )


if __name__ == "__main__":
    main()
