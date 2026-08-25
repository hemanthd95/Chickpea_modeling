#!/usr/bin/env python
"""Audit conservative probability thresholds without creating class masks."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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


def wilson_lower(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    """Return the lower Wilson bound for a binomial proportion."""
    if trials <= 0:
        return float("nan")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return float((centre - margin) / denominator)


def threshold_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    class_name: str,
    threshold: float,
) -> dict:
    """Compute one-vs-rest threshold metrics from held-out probabilities."""
    if class_name == "chickpea":
        positive = truth == 1
        predicted = probability >= threshold
    elif class_name == "weed":
        positive = truth == 0
        predicted = probability <= threshold
    else:
        raise ValueError(f"Unsupported class: {class_name}")
    true_positive = int((positive & predicted).sum())
    false_positive = int((~positive & predicted).sum())
    false_negative = int((positive & ~predicted).sum())
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = true_positive / predicted_positive if predicted_positive else float("nan")
    recall = true_positive / actual_positive if actual_positive else float("nan")
    return {
        "class_name": class_name,
        "threshold": float(threshold),
        "predicted_positive": predicted_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "precision_wilson_lower_95": wilson_lower(true_positive, predicted_positive),
        "recall": float(recall),
    }


def triage(probability: np.ndarray, weed_threshold: float, chickpea_threshold: float) -> np.ndarray:
    """Encode unscored=0, weed-review=1, unresolved=2, chickpea-review=3."""
    if not 0 <= weed_threshold < chickpea_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= weed < chickpea <= 1")
    result = np.zeros(probability.shape, dtype=np.uint8)
    valid = np.isfinite(probability) & (probability >= 0) & (probability <= 1)
    result[valid] = 2
    result[valid & (probability <= weed_threshold)] = 1
    result[valid & (probability >= chickpea_threshold)] = 3
    return result


def validate_hash(contract: dict, project: Path, path: Path) -> None:
    key = str(path.relative_to(project))
    expected = contract.get("output_hashes", {}).get(key)
    if expected != sha256(path):
        raise ValueError(f"Probability-review artifact hash mismatch: {path}")


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
    policy = config["investigator_probability_threshold_audit"]
    if policy["status"] != "enabled_after_probability_review_contract":
        raise ValueError("Probability threshold audit is not enabled")
    for forbidden in (
        "write_categorical_geotiffs", "automatically_materialize_masks",
        "automatically_replace_authoritative_masks", "automatically_retrain_models",
    ):
        if bool(policy.get(forbidden, False)):
            raise ValueError(f"{forbidden} must remain false")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    probability_contract_path = contracts / str(policy["probability_contract_filename"])
    reference_contract_path = contracts / str(policy["reference_contract_filename"])
    probability_contract = yaml.safe_load(probability_contract_path.read_text())
    reference_contract = yaml.safe_load(reference_contract_path.read_text())
    if probability_contract.get("status") != "field1_investigator_chickpea_probability_review_materialized":
        raise ValueError("Continuous probability review is not finalized")
    if reference_contract.get("status") != "field1_investigator_vegetation_references_frozen":
        raise ValueError("Investigator references are not frozen")
    if bool(probability_contract.get("categorical_masks_written", True)):
        raise ValueError("Probability contract unexpectedly reports categorical masks")

    probability_reports = (
        local / "reports" / "mask_refinement"
        / "investigator_chickpea_probability_review"
    )
    summary_path = probability_reports / "investigator_chickpea_probability_summary.csv"
    validate_hash(probability_contract, project, summary_path)
    summary = pd.read_csv(summary_path)
    if len(summary) != 19 or summary["cube_id"].duplicated().any():
        raise ValueError("Probability summary must contain 19 unique cubes")

    reference_source = (
        local / str(config["investigator_vegetation_reference_freeze"]["source_report_relative"])
        / "investigator_vegetation_reference_predictions.csv"
    )
    expected_prediction_hash = reference_contract.get("source_hashes", {}).get("predictions")
    if expected_prediction_hash != sha256(reference_source):
        raise ValueError("Held-out point predictions differ from the frozen source hash")
    predictions = pd.read_csv(reference_source)
    warnings = {
        str(item["heldout_cube"])
        for item in reference_contract.get("per_cube_transfer_warnings", [])
    }

    validation_rows = []
    scopes = {
        "all_reference_cubes": np.ones(len(predictions), dtype=bool),
        "excluding_transfer_warning_cubes": ~predictions["heldout_cube"].astype(str).isin(warnings).to_numpy(),
    }
    for scope_name, selected in scopes.items():
        truth = predictions.loc[selected, "true_class_id"].to_numpy(dtype=int)
        probabilities = predictions.loc[selected, "chickpea_probability"].to_numpy(dtype=float)
        for threshold in policy["chickpea_threshold_candidates"]:
            validation_rows.append({
                "validation_scope": scope_name,
                **threshold_metrics(truth, probabilities, "chickpea", float(threshold)),
            })
        for threshold in policy["weed_threshold_candidates"]:
            validation_rows.append({
                "validation_scope": scope_name,
                **threshold_metrics(truth, probabilities, "weed", float(threshold)),
            })
    validation = pd.DataFrame(validation_rows)

    chickpea_display = float(policy["display_chickpea_threshold"])
    weed_display = float(policy["display_weed_threshold"])
    primary_point_rows = validation[
        (validation["validation_scope"] == "all_reference_cubes")
        & (((validation["class_name"] == "chickpea") & np.isclose(validation["threshold"], chickpea_display))
           | ((validation["class_name"] == "weed") & np.isclose(validation["threshold"], weed_display)))
    ]
    point_gate = bool(
        len(primary_point_rows) == 2
        and (primary_point_rows["precision"] >= float(policy["minimum_point_precision_for_display_threshold"])).all()
        and (primary_point_rows["precision_wilson_lower_95"] >= float(policy["minimum_point_precision_wilson_lower_95"])).all()
    )

    candidate_root = project / str(
        config["polygon_guided_candidate_materialization"]["output_root_relative"]
    )
    sensitivity_rows = []
    preview_tiles = []
    chickpea_thresholds = [float(v) for v in policy["chickpea_threshold_candidates"]]
    weed_thresholds = [float(v) for v in policy["weed_threshold_candidates"]]
    for row in summary.itertuples(index=False):
        cube_id = str(row.cube_id)
        probability_path = Path(str(row.probability_path))
        validate_hash(probability_contract, project, probability_path)
        candidate_path = candidate_root / cube_id / "polygon_guided_candidate_support.tif"
        with rasterio.open(probability_path) as dataset:
            probability = dataset.read(1).astype(np.float32)
        with rasterio.open(candidate_path) as dataset:
            support = dataset.read(1)
        if probability.shape != support.shape:
            raise ValueError(f"Probability/support shape mismatch for {cube_id}")
        probability[~np.isfinite(probability) | (probability < 0) | (probability > 1)] = np.nan
        for chickpea_threshold in chickpea_thresholds:
            for weed_threshold in weed_thresholds:
                encoded = triage(probability, weed_threshold, chickpea_threshold)
                for zone_name, zone in {
                    "all_scored_support": encoded > 0,
                    "polygon_core": (encoded > 0) & (support == 2),
                    "uncertain_edge": (encoded > 0) & (support == 4),
                }.items():
                    total = int(zone.sum())
                    weed_count = int(((encoded == 1) & zone).sum())
                    unresolved_count = int(((encoded == 2) & zone).sum())
                    chickpea_count = int(((encoded == 3) & zone).sum())
                    sensitivity_rows.append({
                        "cube_id": cube_id,
                        "analysis_role": str(row.analysis_role),
                        "transfer_warning": bool(row.transfer_warning),
                        "zone": zone_name,
                        "weed_threshold": weed_threshold,
                        "chickpea_threshold": chickpea_threshold,
                        "scored_centres": total,
                        "high_confidence_weed": weed_count,
                        "unresolved": unresolved_count,
                        "high_confidence_chickpea": chickpea_count,
                        "high_confidence_weed_fraction": weed_count / total if total else np.nan,
                        "unresolved_fraction": unresolved_count / total if total else np.nan,
                        "high_confidence_chickpea_fraction": chickpea_count / total if total else np.nan,
                    })
        display = triage(probability, weed_display, chickpea_display)
        step = max(1, math.ceil(max(display.shape) / 650))
        preview_tiles.append((cube_id, str(row.analysis_role), cube_id in warnings, display[::step, ::step]))

    reports = probability_reports / "threshold_audit"
    reports.mkdir(parents=True, exist_ok=True)
    validation_path = reports / "investigator_probability_threshold_point_validation.csv"
    sensitivity_path = reports / "investigator_probability_threshold_sensitivity_by_cube.csv"
    validation.to_csv(validation_path, index=False)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(sensitivity_path, index=False)

    primary_support = sensitivity[
        (sensitivity["zone"] == "all_scored_support")
        & np.isclose(sensitivity["weed_threshold"], weed_display)
        & np.isclose(sensitivity["chickpea_threshold"], chickpea_display)
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    for class_name, marker in (("chickpea", "o"), ("weed", "s")):
        subset = validation[
            (validation["validation_scope"] == "all_reference_cubes")
            & (validation["class_name"] == class_name)
        ].sort_values("threshold")
        axes[0].plot(subset["threshold"], subset["precision"], marker=marker, label=f"{class_name.title()} precision")
        axes[0].plot(subset["threshold"], subset["recall"], marker=marker, linestyle="--", label=f"{class_name.title()} recall")
    axes[0].axhline(float(policy["minimum_point_precision_for_display_threshold"]), color="black", linestyle=":", label="Precision gate")
    axes[0].set(xlabel="Class-specific probability threshold", ylabel="Held-out point metric", ylim=(0, 1.02), title="Point-level LOCO threshold evidence")
    axes[0].legend(fontsize=8)
    x = np.arange(len(primary_support))
    axes[1].bar(x, primary_support["high_confidence_weed_fraction"], label=f"P≤{weed_display:.2f}: weed review", color="#6D28D9")
    axes[1].bar(x, primary_support["unresolved_fraction"], bottom=primary_support["high_confidence_weed_fraction"], label="Unresolved", color="#D1D5DB")
    axes[1].bar(x, primary_support["high_confidence_chickpea_fraction"], bottom=primary_support["high_confidence_weed_fraction"] + primary_support["unresolved_fraction"], label=f"P≥{chickpea_display:.2f}: chickpea review", color="#16A34A")
    axes[1].set_xticks(x, primary_support["cube_id"].str.replace("field1_cube", "", regex=False), rotation=0)
    axes[1].set(xlabel="Field 1 cube", ylabel="Fraction of scored candidate centres", ylim=(0, 1), title="Dense candidate triage (review only)")
    axes[1].legend(fontsize=8)
    figure.suptitle(f"Investigator probability threshold audit — point gate passed={point_gate}\nDense probabilities remain review evidence, not categorical labels", fontsize=14)
    audit_overview_path = reports / "investigator_probability_threshold_audit_overview.png"
    figure.savefig(audit_overview_path, dpi=200, facecolor="white")
    plt.close(figure)

    columns = 4
    rows = math.ceil(len(preview_tiles) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.4 * rows), constrained_layout=True)
    flat = np.asarray(axes).reshape(-1)
    cmap = ListedColormap(["black", "#6D28D9", "#D1D5DB", "#16A34A"])
    for axis in flat:
        axis.axis("off")
        axis.set_facecolor("black")
    for axis, (cube_id, role, warning, tile) in zip(flat, preview_tiles):
        axis.imshow(tile, vmin=0, vmax=3, cmap=cmap, interpolation="nearest")
        axis.set_title(f"{cube_id} | {role}" + (" | WARNING" if warning else ""), color="#B91C1C" if warning else "black", fontsize=9)
    figure.suptitle(f"Field 1 three-way probability triage — purple weed P≤{weed_display:.2f}; gray unresolved; green chickpea P≥{chickpea_display:.2f}\nReview visualization only; no categorical GeoTIFF was written", fontsize=13)
    triage_overview_path = reports / "investigator_probability_triage_maps_overview.png"
    figure.savefig(triage_overview_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    outputs = [validation_path, sensitivity_path, audit_overview_path, triage_overview_path]
    contract = {
        "status": "field1_investigator_probability_threshold_audit_complete",
        "field": "Field 1",
        "display_thresholds": {"weed_maximum_probability": weed_display, "chickpea_minimum_probability": chickpea_display},
        "point_reference_gate_passed": point_gate,
        "point_reference_gate_is_dense_pixel_validation": False,
        "transfer_warning_cubes": sorted(warnings),
        "dense_output_semantics": "three_way_review_triage_only",
        "categorical_geotiffs_written": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {"probability_review_contract": sha256(probability_contract_path), "reference_contract": sha256(reference_contract_path), "heldout_point_predictions": sha256(reference_source), "configuration": sha256(args.config)},
        "output_hashes": {path.name: sha256(path) for path in outputs},
    }
    contract_path = contracts / "field1_investigator_probability_threshold_audit_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print("Held-out point threshold evidence:")
    print(primary_point_rows[["class_name", "threshold", "predicted_positive", "precision", "precision_wilson_lower_95", "recall"]].to_string(index=False))
    print(f"Point-reference threshold gate passed: {point_gate}")
    print(f"Threshold validation: {validation_path}")
    print(f"Dense sensitivity: {sensitivity_path}")
    print(f"Visual QC: {triage_overview_path}")
    print(f"Contract: {contract_path}")
    print("Audit only: no categorical GeoTIFF or authoritative mask changed, no model was retrained, and Field 2 remained locked.")


if __name__ == "__main__":
    main()
