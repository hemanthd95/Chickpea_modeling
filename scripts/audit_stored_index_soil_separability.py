#!/usr/bin/env python
"""Test whether stored Spectronon index products can reproducibly identify soil.

The files are named NDVI/NVDI, but their physical pixels are all above the
documented 0.30 soil cutoff.  This audit therefore treats the scalar value as an
uninterpreted observed index.  It uses existing provenance-tracked soil masks
only to test separability and performs leave-one-cube-out threshold validation.
No mask is written or modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.metrics import roc_auc_score, roc_curve
import yaml

from chickpea_ssl.data import load_records
from scripts.audit_chickpea_region_annotations import (
    normalized_ring,
    observed_support,
    planter_alley_features,
    reconcile_exports,
)
from scripts.audit_polygon_soil_sources import product_array, source_root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    union = tp + fp + fn
    return {
        "true_soil": int(y_true.sum()),
        "true_nonsoil": int((~y_true).sum()),
        "predicted_soil": int(y_pred.sum()),
        "balanced_accuracy": float((recall + specificity) / 2),
        "soil_recall": float(recall),
        "soil_specificity": float(specificity),
        "soil_precision": float(precision) if np.isfinite(precision) else np.nan,
        "soil_iou": float(tp / union) if union else np.nan,
    }


def fit_rule(values: np.ndarray, labels: np.ndarray) -> tuple[str, float, float]:
    """Fit direction and Youden-J threshold using balanced cube samples."""
    auc_high = float(roc_auc_score(labels, values))
    if auc_high >= 0.5:
        direction = "higher_values_indicate_soil"
        scores = values
        auc = auc_high
    else:
        direction = "lower_values_indicate_soil"
        scores = -values
        auc = 1.0 - auc_high
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("No finite threshold available for stored-index audit")
    candidates = np.flatnonzero(finite)
    selected = candidates[int(np.argmax((tpr - fpr)[finite]))]
    score_threshold = float(thresholds[selected])
    value_threshold = score_threshold if direction.startswith("higher") else -score_threshold
    return direction, value_threshold, auc


def predict(values: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    if direction.startswith("higher"):
        return values >= threshold
    return values <= threshold


def balanced_sample(
    values: np.ndarray,
    labels: np.ndarray,
    maximum_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in (False, True):
        indices = np.flatnonzero(labels == label)
        if len(indices) > maximum_per_class:
            indices = np.sort(rng.choice(indices, maximum_per_class, replace=False))
        selected.append(indices)
    indices = np.concatenate(selected)
    return values[indices], labels[indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--annotations-root", type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    policy = config.get("stored_index_soil_separability_audit", {})
    maximum_per_class = int(policy.get("maximum_sampled_pixels_per_class_per_cube", 20000))
    seed = int(policy.get("seed", 42))
    minimum_auc = float(policy.get("minimum_loco_auc", 0.75))
    minimum_ba = float(policy.get("minimum_loco_balanced_accuracy", 0.70))
    seam_snap = float(config["chickpea_region_polygon_audit"].get("seam_snap_maximum_m", 0.30))

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    annotation_root = args.annotations_root or local / "annotations" / "chickpea_regions"
    json_path = annotation_root / "field1_chickpea_region_annotations.json"
    csv_path = annotation_root / "field1_chickpea_region_vertices.csv"
    geojson_path = annotation_root / "field1_chickpea_region_annotations.geojson"
    _, geojson, _ = reconcile_exports(json_path, csv_path, geojson_path)

    polygons: dict[str, list[dict]] = defaultdict(list)
    for feature in geojson["features"]:
        cube_id = str(feature["properties"]["cube_id"])
        coordinates = feature["geometry"]["coordinates"][0]
        if coordinates and coordinates[0] == coordinates[-1]:
            coordinates = coordinates[:-1]
        ring, _, _ = normalized_ring(coordinates, seam_snap)
        polygons[cube_id].append({"type": "Polygon", "coordinates": [ring]})

    alley_path = local / "annotations" / "planter_tracks" / "field1_planter_track_annotations.geojson"
    alleys = planter_alley_features(alley_path, seam_snap)
    records = {
        record.cube_id: record for record in load_records(
            args.paths, local / "authoritative_manifest.csv"
        ) if record.cube_id in polygons
    }
    catalog = pd.read_csv(local / "project_catalog.csv", dtype=str).fillna("")
    ndvi_catalog = catalog[
        catalog["file_role"].eq("ndvi")
        & catalog["extension"].isin([".bip", ".bil", ".bsq", ".tif", ".tiff"])
    ].copy()
    sensitivity = set(config.get("investigator_guidance", {})
                      .get("phenology_exclusions_from_refined_primary_supervised_analysis", {}))
    expansion = set(config.get("investigator_guidance", {})
                    .get("optional_mask_expansion", {}).get("cubes", []))

    per_cube: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    distribution_rows: list[dict] = []
    expansion_arrays: dict[str, np.ndarray] = {}

    for number, cube_id in enumerate(sorted(records, key=lambda item: int(item.split("cube")[-1])), 1):
        record = records[cube_id]
        candidates = ndvi_catalog[ndvi_catalog["cube_id_inferred"].eq(cube_id)]
        if len(candidates) != 1:
            print(f"Skipped {cube_id}: expected one stored index, found {len(candidates)}")
            continue
        row = candidates.iloc[0]
        path = source_root(paths, row) / row["relative_path"]
        with rasterio.open(record.data) as reference:
            shape = (reference.height, reference.width)
            transform = reference.transform
            observed = observed_support(reference)
        array, metadata = product_array(path, shape)
        if array is None or not metadata["plausible_ndvi_scale"]:
            print(f"Skipped {cube_id}: {metadata['read_status']}")
            continue
        core = rasterize(
            [(geometry, 1) for geometry in polygons[cube_id]], out_shape=shape,
            transform=transform, fill=0, dtype="uint8", all_touched=False,
        ).astype(bool) & observed
        alley_shapes = [(geometry, 1) for geometry in alleys.get(cube_id, [])]
        alley = (
            rasterize(alley_shapes, out_shape=shape, transform=transform, fill=0,
                      dtype="uint8", all_touched=True).astype(bool)
            if alley_shapes else np.zeros(shape, dtype=bool)
        )
        eligible = core & ~alley & np.isfinite(array)
        values = array[eligible].astype(np.float32)
        if record.soil_mask is None:
            expansion_arrays[cube_id] = values
            print(f"Prepared {number}/{len(records)} {cube_id}: unlabeled expansion values={len(values):,}")
            continue
        with rasterio.open(record.soil_mask) as dataset:
            if (dataset.height, dataset.width) != shape:
                raise ValueError(f"Soil-mask shape mismatch for {cube_id}")
            labels = (dataset.read(1) > 0)[eligible]
        sampled_values, sampled_labels = balanced_sample(
            values, labels, maximum_per_class, seed + int(cube_id.split("cube")[-1])
        )
        per_cube[cube_id] = (sampled_values, sampled_labels)
        soil_values = values[labels]
        nonsoil_values = values[~labels]
        auc_high = float(roc_auc_score(sampled_labels, sampled_values))
        distribution_rows.append({
            "cube_id": cube_id,
            "analysis_role": "sensitivity_only" if cube_id in sensitivity else "primary_reference",
            "eligible_pixels": int(len(values)),
            "soil_pixels": int(labels.sum()),
            "nonsoil_pixels": int((~labels).sum()),
            "soil_q10": float(np.quantile(soil_values, 0.10)),
            "soil_median": float(np.median(soil_values)),
            "soil_q90": float(np.quantile(soil_values, 0.90)),
            "nonsoil_q10": float(np.quantile(nonsoil_values, 0.10)),
            "nonsoil_median": float(np.median(nonsoil_values)),
            "nonsoil_q90": float(np.quantile(nonsoil_values, 0.90)),
            "auc_higher_values_indicate_soil": auc_high,
            "best_direction_auc": max(auc_high, 1.0 - auc_high),
        })
        print(f"Prepared {number}/{len(records)} {cube_id}: soil={labels.sum():,}; nonsoil={(~labels).sum():,}")

    primary_cubes = sorted(cube for cube in per_cube if cube not in sensitivity)
    if len(primary_cubes) < 3:
        raise ValueError("Too few primary soil-reference cubes for leave-one-cube-out validation")

    loco_rows: list[dict] = []
    for heldout in primary_cubes:
        train_values = np.concatenate([per_cube[cube][0] for cube in primary_cubes if cube != heldout])
        train_labels = np.concatenate([per_cube[cube][1] for cube in primary_cubes if cube != heldout])
        direction, threshold, train_auc = fit_rule(train_values, train_labels)
        test_values, test_labels = per_cube[heldout]
        test_auc_high = float(roc_auc_score(test_labels, test_values))
        test_auc = test_auc_high if direction.startswith("higher") else 1.0 - test_auc_high
        result = metrics(test_labels, predict(test_values, direction, threshold))
        loco_rows.append({
            "heldout_cube": heldout, "direction": direction,
            "training_threshold": threshold, "training_auc": train_auc,
            "heldout_auc": test_auc, **result,
        })

    all_values = np.concatenate([per_cube[cube][0] for cube in primary_cubes])
    all_labels = np.concatenate([per_cube[cube][1] for cube in primary_cubes])
    final_direction, final_threshold, final_auc = fit_rule(all_values, all_labels)
    loco = pd.DataFrame(loco_rows)
    gate_passed = bool(
        loco["heldout_auc"].mean() >= minimum_auc
        and loco["balanced_accuracy"].mean() >= minimum_ba
        and (loco["direction"] == final_direction).all()
    )

    expansion_rows: list[dict] = []
    for cube_id in sorted(expansion_arrays):
        values = expansion_arrays[cube_id]
        predicted = predict(values, final_direction, final_threshold)
        expansion_rows.append({
            "cube_id": cube_id, "eligible_index_pixels": int(len(values)),
            "candidate_soil_pixels": int(predicted.sum()),
            "candidate_soil_fraction": float(predicted.mean()) if len(predicted) else np.nan,
            "frozen_direction": final_direction,
            "frozen_threshold": final_threshold,
            "validation_gate_passed": gate_passed,
            "candidate_status": (
                "review_only_candidate" if gate_passed
                else "rejected_unvalidated_index_source"
            ),
        })

    reports = local / "reports" / "mask_refinement" / "stored_index_soil_separability"
    reports.mkdir(parents=True, exist_ok=True)
    distribution = pd.DataFrame(distribution_rows)
    expansion_df = pd.DataFrame(expansion_rows)
    distribution_path = reports / "stored_index_soil_distribution_by_cube.csv"
    loco_path = reports / "stored_index_soil_loco_validation.csv"
    expansion_path = reports / "stored_index_soil_expansion_candidates.csv"
    distribution.to_csv(distribution_path, index=False)
    loco.to_csv(loco_path, index=False)
    expansion_df.to_csv(expansion_path, index=False)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    positions = np.arange(len(distribution))
    axes[0].errorbar(
        positions - 0.08, distribution["soil_median"],
        yerr=[distribution["soil_median"] - distribution["soil_q10"],
              distribution["soil_q90"] - distribution["soil_median"]],
        fmt="o", capsize=2, label="Existing soil mask",
    )
    axes[0].errorbar(
        positions + 0.08, distribution["nonsoil_median"],
        yerr=[distribution["nonsoil_median"] - distribution["nonsoil_q10"],
              distribution["nonsoil_q90"] - distribution["nonsoil_median"]],
        fmt="o", capsize=2, label="Existing non-soil",
    )
    axes[0].set_xticks(positions, distribution["cube_id"].str.replace("field1_cube", ""), rotation=90)
    axes[0].set_ylabel("Stored scalar product value (median and 10–90%)")
    axes[0].set_title("Observed class distributions")
    axes[0].legend()

    axes[1].bar(np.arange(len(loco)), loco["balanced_accuracy"], color="#4C78A8")
    axes[1].axhline(minimum_ba, color="black", linestyle="--", label=f"Gate {minimum_ba:.2f}")
    axes[1].set_xticks(np.arange(len(loco)), loco["heldout_cube"].str.replace("field1_cube", ""), rotation=90)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Leave-one-cube-out threshold validation")
    axes[1].legend()

    if not expansion_df.empty:
        axes[2].bar(expansion_df["cube_id"], expansion_df["candidate_soil_fraction"], color="#B279A2")
        axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Provisional soil fraction")
    axes[2].set_title("Expansion cubes 12/14/15\nshown only if validation is defensible")
    figure.suptitle(
        "Stored-index soil separability audit — no mask changes\n"
        f"final rule: {final_direction}, threshold={final_threshold:.4f}; gate passed={gate_passed}",
        fontsize=14,
    )
    figure.tight_layout()
    overview_path = reports / "stored_index_soil_separability_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "stored_index_soil_separability_audit_complete",
        "field": "Field 1",
        "source_interpretation": "uninterpreted_scalar_index_not_assumed_to_be_documented_ndvi",
        "primary_reference_cubes": primary_cubes,
        "sensitivity_cubes_excluded_from_threshold_fitting": sorted(sensitivity),
        "final_direction": final_direction,
        "final_threshold": float(final_threshold),
        "final_training_auc": float(final_auc),
        "mean_loco_auc": float(loco["heldout_auc"].mean()),
        "mean_loco_balanced_accuracy": float(loco["balanced_accuracy"].mean()),
        "validation_gate_passed": gate_passed,
        "gate": {"minimum_mean_loco_auc": minimum_auc,
                 "minimum_mean_loco_balanced_accuracy": minimum_ba,
                 "direction_must_be_stable": True},
        "expansion_cubes_can_be_materialized": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {
            "annotations_json": sha256(json_path),
            "vertices_csv": sha256(csv_path),
            "annotations_geojson": sha256(geojson_path),
            "planter_annotations_geojson": sha256(alley_path),
        },
        "output_hashes": {
            str(path.relative_to(reports)): sha256(path) for path in
            [distribution_path, loco_path, expansion_path, overview_path]
        },
    }
    contract_path = contracts / "field1_stored_index_soil_separability_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Primary reference cubes: {len(primary_cubes)}")
    print(f"Final direction: {final_direction}; threshold={final_threshold:.6f}")
    print(f"Mean LOCO AUC: {loco['heldout_auc'].mean():.4f}")
    print(f"Mean LOCO balanced accuracy: {loco['balanced_accuracy'].mean():.4f}")
    print(f"Validation gate passed: {gate_passed}")
    print(f"Reports: {reports}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print("Audit only: no mask changed, no model retrained, and Field 2 remained locked.")


if __name__ == "__main__":
    main()
