#!/usr/bin/env python
"""Audit chickpea/weed spectra inside polygon-guided vegetation candidates.

Historical labels are used only where they agree with polygon-core non-soil
support, and only as weak diagnostic references. The linear separator is
validated by leaving out a whole primary cube. Scores are review evidence, not
labels; this script cannot write categorical masks or retrain a model.
"""

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
import numpy as np
import pandas as pd
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from spectral.io import envi
import yaml

from chickpea_ssl.data import authoritative_class_map, load_band_indices, load_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_cube_ids(cubes) -> list[str]:
    return sorted(cubes, key=lambda value: int(str(value).split("cube")[-1]))


def sample_coordinates(
    mask: np.ndarray, maximum: int, rng: np.random.Generator
) -> np.ndarray:
    """Return deterministic observed row/column coordinates."""
    flat = np.flatnonzero(mask)
    if len(flat) > maximum:
        flat = np.sort(rng.choice(flat, maximum, replace=False))
    return np.column_stack(np.unravel_index(flat, mask.shape)).astype(np.int32)


def spectra_at(
    cube: np.ndarray, coordinates: np.ndarray, band_indices: np.ndarray
) -> np.ndarray:
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape (n, 2)")
    if not len(coordinates):
        return np.empty((0, len(band_indices)), dtype=np.float32)
    values = cube[coordinates[:, 0], coordinates[:, 1], :]
    return np.asarray(values[:, band_indices], dtype=np.float32)


def fit_linear_separator(
    features: np.ndarray,
    labels: np.ndarray,
    regularization_c: float,
    maximum_iterations: int,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=maximum_iterations,
        random_state=seed,
        solver="liblinear",
    ).fit(scaler.transform(features), labels)
    return scaler, model


def predict_probability(
    features: np.ndarray,
    scaler: StandardScaler,
    model: LogisticRegression,
    chunk_size: int,
) -> np.ndarray:
    result = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        result[start:stop] = model.predict_proba(
            scaler.transform(features[start:stop])
        )[:, 1]
    return result


def classification_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability >= 0.5
    return {
        "reference_pixels": int(len(labels)),
        "weak_reference_chickpea_pixels": int(labels.sum()),
        "weak_reference_weed_pixels": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "chickpea_precision": float(
            precision_score(labels, prediction, zero_division=0)
        ),
        "chickpea_recall": float(recall_score(labels, prediction, zero_division=0)),
    }


def valid_spectra(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values).all(axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--bands", required=True, type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    policy = config.get("polygon_guided_vegetation_separability_audit", {})
    if bool(policy.get("automatically_materialize_chickpea_or_weed_masks", False)):
        raise ValueError("This audit cannot materialize chickpea or weed masks")
    if bool(policy.get("write_probability_geotiffs", False)):
        raise ValueError("Probability GeoTIFF output is disabled for this review stage")
    if not bool(policy.get("sensitivity_cubes_are_reference_excluded", True)):
        raise ValueError("Phenology sensitivity cubes must be reference-excluded")
    if not bool(policy.get("expansion_cubes_are_projection_only", True)):
        raise ValueError("Expansion cubes must remain projection-only")
    maximum_reference = int(
        policy.get("maximum_reference_pixels_per_class_per_cube", 5000)
    )
    maximum_projection = int(
        policy.get("maximum_projection_pixels_per_cube_for_summary", 50000)
    )
    chunk_size = int(policy.get("prediction_chunk_size", 50000))
    seed = int(policy.get("seed", 42))
    regularization_c = float(policy.get("logistic_regularization_c", 1.0))
    maximum_iterations = int(policy.get("maximum_iterations", 500))
    minimum_auc = float(policy.get("minimum_mean_loco_auc", 0.75))
    minimum_ba = float(policy.get("minimum_mean_loco_balanced_accuracy", 0.70))
    minimum_passing_fraction = float(
        policy.get("minimum_fraction_heldout_cubes_passing_balanced_accuracy", 0.70)
    )
    band_section = str(policy.get("spectral_band_section", "primary"))
    band_indices = load_band_indices(args.bands, band_section)

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    manifest_path = local / "authoritative_manifest.csv"
    materialization_contract = (
        local / "contracts"
        / "field1_polygon_guided_candidate_materialization_contract.yaml"
    )
    if not materialization_contract.exists():
        raise FileNotFoundError("Polygon-guided candidate contract is missing")
    frozen = yaml.safe_load(materialization_contract.read_text())
    if bool(frozen.get("candidate_geotiffs_are_authoritative", True)):
        raise ValueError("Review candidates were unexpectedly marked authoritative")

    roles_path = (
        local / "reports" / "chickpea_region_annotation"
        / "chickpea_region_annotation_layer_manifest.csv"
    )
    roles = pd.read_csv(roles_path).set_index("cube_id")["analysis_role"].astype(str)
    expansion = set(
        config["investigator_guidance"]["optional_mask_expansion"]["cubes"]
    )
    sensitivity = set(
        config["investigator_guidance"]
        ["phenology_exclusions_from_refined_primary_supervised_analysis"]
    )
    candidate_root = project / str(
        config["polygon_guided_candidate_materialization"].get(
            "output_root_relative", "data/processed/polygon_guided_mask_candidates"
        )
    )
    records = {
        record.cube_id: record for record in load_records(args.paths, manifest_path)
    }
    candidate_ids = ordered_cube_ids(
        cube for cube in records
        if (candidate_root / cube / "polygon_guided_candidate_support.tif").exists()
    )
    if not candidate_ids:
        raise ValueError("No polygon-guided candidate rasters were found")

    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    support_rows: list[dict] = []
    candidate_masks: dict[str, np.ndarray] = {}
    cube_arrays: dict[str, np.ndarray] = {}
    wavelengths: np.ndarray | None = None

    for cube_id in candidate_ids:
        record = records[cube_id]
        candidate_path = (
            candidate_root / cube_id / "polygon_guided_candidate_support.tif"
        )
        with rasterio.open(candidate_path) as dataset:
            candidate = dataset.read(1)
        candidate_masks[cube_id] = candidate
        weak_chickpea_count = weak_weed_count = 0

        if record.chickpea_mask is not None and record.weed_mask is not None:
            historical = authoritative_class_map(record)
            weak_chickpea = (candidate == 2) & (historical == 1)
            weak_weed = (candidate == 2) & (historical == 2)
            weak_chickpea_count = int(weak_chickpea.sum())
            weak_weed_count = int(weak_weed.sum())
            if cube_id not in sensitivity and cube_id not in expansion:
                rng = np.random.default_rng(
                    seed + int(cube_id.split("cube")[-1])
                )
                chickpea_coordinates = sample_coordinates(
                    weak_chickpea, maximum_reference, rng
                )
                weed_coordinates = sample_coordinates(weak_weed, maximum_reference, rng)
                if not len(chickpea_coordinates) or not len(weed_coordinates):
                    raise ValueError(f"Missing one weak reference class in {cube_id}")
                image = envi.open(str(record.header), str(record.data))
                cube = image.open_memmap()
                cube_arrays[cube_id] = cube
                if wavelengths is None:
                    wavelengths = np.asarray(
                        image.metadata["wavelength"], dtype=np.float32
                    )[band_indices]
                chickpea_values = valid_spectra(
                    spectra_at(cube, chickpea_coordinates, band_indices)
                )
                weed_values = valid_spectra(
                    spectra_at(cube, weed_coordinates, band_indices)
                )
                features = np.concatenate([weed_values, chickpea_values])
                reference_labels = np.concatenate([
                    np.zeros(len(weed_values), dtype=np.uint8),
                    np.ones(len(chickpea_values), dtype=np.uint8),
                ])
                references[cube_id] = (features, reference_labels)

        support_rows.append({
            "cube_id": cube_id,
            "analysis_role": str(roles.loc[cube_id]),
            "reference_use": (
                "projection_only" if cube_id in expansion else
                "sensitivity_only_reference_excluded" if cube_id in sensitivity else
                "primary_weak_reference"
            ),
            "core_nonsoil_candidate_pixels": int((candidate == 2).sum()),
            "weak_reference_chickpea_pixels_available": weak_chickpea_count,
            "weak_reference_weed_pixels_available": weak_weed_count,
            "weak_reference_chickpea_pixels_sampled": (
                int(references[cube_id][1].sum()) if cube_id in references else 0
            ),
            "weak_reference_weed_pixels_sampled": (
                int((references[cube_id][1] == 0).sum())
                if cube_id in references else 0
            ),
        })
        print(
            f"Prepared {cube_id}: core non-soil={(candidate == 2).sum():,}; "
            f"weak chickpea={weak_chickpea_count:,}; weak weed={weak_weed_count:,}",
            flush=True,
        )

    primary = ordered_cube_ids(references)
    if len(primary) < 3 or wavelengths is None:
        raise ValueError("Too few primary cubes for leave-one-cube-out validation")

    loco_rows: list[dict] = []
    for heldout in primary:
        training_features = np.concatenate([
            references[cube][0] for cube in primary if cube != heldout
        ])
        training_labels = np.concatenate([
            references[cube][1] for cube in primary if cube != heldout
        ])
        scaler, model = fit_linear_separator(
            training_features, training_labels, regularization_c,
            maximum_iterations, seed,
        )
        test_features, test_labels = references[heldout]
        probability = predict_probability(test_features, scaler, model, chunk_size)
        result = {"heldout_cube": heldout, **classification_metrics(test_labels, probability)}
        loco_rows.append(result)
        print(
            f"LOCO {heldout}: AUC={result['roc_auc']:.3f}; "
            f"balanced accuracy={result['balanced_accuracy']:.3f}", flush=True,
        )

    all_features = np.concatenate([references[cube][0] for cube in primary])
    all_labels = np.concatenate([references[cube][1] for cube in primary])
    scaler, model = fit_linear_separator(
        all_features, all_labels, regularization_c, maximum_iterations, seed
    )
    loco = pd.DataFrame(loco_rows)
    passing_fraction = float((loco["balanced_accuracy"] >= minimum_ba).mean())
    gate_passed = bool(
        loco["roc_auc"].mean() >= minimum_auc
        and loco["balanced_accuracy"].mean() >= minimum_ba
        and passing_fraction >= minimum_passing_fraction
    )

    reports = (
        local / "reports" / "mask_refinement"
        / "polygon_guided_vegetation_separability"
    )
    individual = reports / "individual_score_maps"
    individual.mkdir(parents=True, exist_ok=True)
    projection_rows: list[dict] = []
    score_tiles: list[tuple[str, str, np.ndarray]] = []
    output_paths: list[Path] = []

    for projection_number, cube_id in enumerate(candidate_ids, start=1):
        record = records[cube_id]
        if cube_id in cube_arrays:
            cube = cube_arrays[cube_id]
        else:
            cube = envi.open(str(record.header), str(record.data)).open_memmap()
        mask = candidate_masks[cube_id] == 2
        coordinates = np.column_stack(np.nonzero(mask)).astype(np.int32)
        probability_map = np.full(mask.shape, np.nan, dtype=np.float32)
        for start in range(0, len(coordinates), chunk_size):
            batch_coordinates = coordinates[start:start + chunk_size]
            values = spectra_at(cube, batch_coordinates, band_indices)
            valid = np.isfinite(values).all(axis=1)
            batch_probability = np.full(len(values), np.nan, dtype=np.float32)
            if valid.any():
                batch_probability[valid] = predict_probability(
                    values[valid], scaler, model, chunk_size
                )
            probability_map[
                batch_coordinates[:, 0], batch_coordinates[:, 1]
            ] = batch_probability
        probability = probability_map[np.isfinite(probability_map)]
        if len(probability) > maximum_projection:
            rng = np.random.default_rng(
                seed + 1000 + int(cube_id.split("cube")[-1])
            )
            summary_values = probability[
                rng.choice(len(probability), maximum_projection, replace=False)
            ]
        else:
            summary_values = probability
        quantiles = (
            np.quantile(summary_values, [0.10, 0.25, 0.50, 0.75, 0.90])
            if len(summary_values) else [np.nan] * 5
        )
        projection_rows.append({
            "cube_id": cube_id,
            "analysis_role": str(roles.loc[cube_id]),
            "core_nonsoil_candidate_pixels": int(mask.sum()),
            "scored_pixels": int(len(probability)),
            "summary_sampled_pixels": int(len(summary_values)),
            "probability_q10": float(quantiles[0]),
            "probability_q25": float(quantiles[1]),
            "probability_median": float(quantiles[2]),
            "probability_q75": float(quantiles[3]),
            "probability_q90": float(quantiles[4]),
            "probability_at_or_above_0_5_fraction": (
                float((summary_values >= 0.5).mean())
                if len(summary_values) else np.nan
            ),
            "status": "review_score_not_a_class_label",
        })
        step = max(1, math.ceil(max(mask.shape) / 600))
        tile = probability_map[::step, ::step]
        score_tiles.append((cube_id, str(roles.loc[cube_id]), tile))
        preview_path = individual / f"{cube_id}_polygon_guided_chickpea_score.png"
        figure, axis = plt.subplots(figsize=(6, 8), constrained_layout=True)
        handle = axis.imshow(
            tile, vmin=0, vmax=1, cmap="viridis", interpolation="nearest"
        )
        axis.axis("off")
        axis.set_title(
            f"{cube_id} — {roles.loc[cube_id]}\n"
            f"median review score={quantiles[2]:.3f}", fontsize=11,
        )
        figure.colorbar(handle, ax=axis, shrink=0.7, label="Chickpea review score")
        figure.savefig(preview_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(preview_path)
        print(
            f"Scored {projection_number}/{len(candidate_ids)} {cube_id}: "
            f"{len(probability):,} core non-soil pixels; "
            f"median review score={quantiles[2]:.3f}"
        )

    support = pd.DataFrame(support_rows)
    projection = pd.DataFrame(projection_rows)
    support_path = reports / "vegetation_reference_support_by_cube.csv"
    loco_path = reports / "vegetation_separability_loco_validation.csv"
    projection_path = reports / "polygon_core_chickpea_score_summary.csv"
    support.to_csv(support_path, index=False)
    loco.to_csv(loco_path, index=False)
    projection.to_csv(projection_path, index=False)

    spectral_rows: list[dict] = []
    for label, class_name in (
        (0, "weak_reference_weed"), (1, "weak_reference_chickpea")
    ):
        class_values = all_features[all_labels == label]
        q25, median, q75 = np.quantile(class_values, [0.25, 0.50, 0.75], axis=0)
        for band_index, wavelength, low, centre, high in zip(
            band_indices, wavelengths, q25, median, q75
        ):
            spectral_rows.append({
                "class_name": class_name,
                "band_index": int(band_index),
                "wavelength_nm": float(wavelength),
                "q25_stored_value": float(low),
                "median_stored_value": float(centre),
                "q75_stored_value": float(high),
            })
    spectral = pd.DataFrame(spectral_rows)
    spectral_path = reports / "vegetation_reference_spectral_profile.csv"
    spectral.to_csv(spectral_path, index=False)
    output_paths.extend([support_path, loco_path, projection_path, spectral_path])

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    colors = {
        "weak_reference_weed": "#7C3AED",
        "weak_reference_chickpea": "#16A34A",
    }
    names = {
        "weak_reference_weed": "Weak weed reference",
        "weak_reference_chickpea": "Weak chickpea reference",
    }
    for class_name, group in spectral.groupby("class_name"):
        axes[0].plot(
            group["wavelength_nm"], group["median_stored_value"],
            color=colors[class_name], label=names[class_name],
        )
        axes[0].fill_between(
            group["wavelength_nm"], group["q25_stored_value"],
            group["q75_stored_value"], color=colors[class_name], alpha=0.18,
        )
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Stored reflectance value")
    axes[0].set_title("Cube-balanced weak-reference spectra")
    axes[0].legend()

    x = np.arange(len(loco))
    axes[1].plot(x, loco["roc_auc"], marker="o", label="ROC AUC")
    axes[1].plot(
        x, loco["balanced_accuracy"], marker="o", label="Balanced accuracy"
    )
    axes[1].axhline(
        minimum_ba, color="black", linestyle="--", label=f"BA gate {minimum_ba:.2f}"
    )
    axes[1].set_xticks(
        x, loco["heldout_cube"].str.replace("field1_cube", ""), rotation=90
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Held-out cube")
    axes[1].set_title("Leave-one-cube-out transfer")
    axes[1].legend()

    x = np.arange(len(projection))
    lower = projection["probability_median"] - projection["probability_q10"]
    upper = projection["probability_q90"] - projection["probability_median"]
    axes[2].errorbar(
        x, projection["probability_median"], yerr=[lower, upper],
        fmt="o", capsize=2, color="#2563EB",
    )
    axes[2].axhline(
        0.5, color="black", linestyle="--", label="Review midpoint only"
    )
    axes[2].set_xticks(
        x, projection["cube_id"].str.replace("field1_cube", ""), rotation=90
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Cube")
    axes[2].set_ylabel("Chickpea review score (median, 10–90%)")
    axes[2].set_title("Polygon-core non-soil candidates")
    axes[2].legend()
    figure.suptitle(
        "Field 1 polygon-guided chickpea/weed separability — no label changes\n"
        f"mean LOCO AUC={loco['roc_auc'].mean():.3f}; mean balanced accuracy="
        f"{loco['balanced_accuracy'].mean():.3f}; review gate passed={gate_passed}",
        fontsize=14,
    )
    overview_path = reports / "polygon_guided_vegetation_separability_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    output_paths.append(overview_path)

    columns = 4
    rows = math.ceil(len(score_tiles) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(16, 4.5 * rows), constrained_layout=True
    )
    flat = np.asarray(axes).reshape(-1)
    for axis in flat:
        axis.axis("off")
        axis.set_facecolor("black")
    handle = None
    for axis, (cube_id, role, tile) in zip(flat, score_tiles):
        handle = axis.imshow(
            tile, vmin=0, vmax=1, cmap="viridis", interpolation="nearest"
        )
        axis.set_title(f"{cube_id} | {role}", fontsize=9)
    if handle is not None:
        figure.colorbar(
            handle, ax=flat.tolist(), shrink=0.35,
            label="Chickpea review score (not a label)",
        )
    figure.suptitle(
        "Field 1 polygon-core non-soil spectral review scores\n"
        "black=outside eligible core; colors are diagnostic probabilities, not masks",
        fontsize=14,
    )
    score_overview_path = reports / "polygon_guided_chickpea_score_maps_overview.png"
    figure.savefig(
        score_overview_path, dpi=200, facecolor="white", bbox_inches="tight"
    )
    plt.close(figure)
    output_paths.append(score_overview_path)

    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "weak_reference_vegetation_separability_audit_complete",
        "field": "Field 1",
        "reference_interpretation": (
            "historical_labels_intersected_with_polygon_core_nonsoil_audit_only"
        ),
        "reference_cubes": primary,
        "sensitivity_cubes_reference_excluded": ordered_cube_ids(sensitivity),
        "expansion_cubes_projection_only": ordered_cube_ids(expansion),
        "spectral_band_section": band_section,
        "band_count": int(len(band_indices)),
        "mean_loco_auc": float(loco["roc_auc"].mean()),
        "mean_loco_balanced_accuracy": float(loco["balanced_accuracy"].mean()),
        "heldout_cubes_passing_balanced_accuracy_fraction": passing_fraction,
        "review_gate_passed": gate_passed,
        "gate": {
            "minimum_mean_loco_auc": minimum_auc,
            "minimum_mean_loco_balanced_accuracy": minimum_ba,
            "minimum_fraction_heldout_cubes_passing_balanced_accuracy": (
                minimum_passing_fraction
            ),
        },
        "score_midpoint_is_an_authoritative_threshold": False,
        "probability_geotiffs_written": False,
        "categorical_masks_written": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {
            "authoritative_manifest": sha256(manifest_path),
            "annotation_layer_manifest": sha256(roles_path),
            "candidate_materialization_contract": sha256(materialization_contract),
            "spectral_band_config": sha256(args.bands),
        },
        "candidate_support_hashes": {
            cube_id: sha256(
                candidate_root / cube_id / "polygon_guided_candidate_support.tif"
            )
            for cube_id in candidate_ids
        },
        "weak_reference_mask_hashes": {
            cube_id: {
                "chickpea_mask": sha256(records[cube_id].chickpea_mask),
                "weed_mask": sha256(records[cube_id].weed_mask),
            }
            for cube_id in primary
        },
        "output_hashes": {
            str(path.relative_to(reports)): sha256(path)
            for path in output_paths if reports in path.parents
        },
    }
    contract_path = (
        contracts / "field1_polygon_guided_vegetation_separability_contract.yaml"
    )
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Primary weak-reference cubes: {len(primary)}")
    print(f"Mean LOCO AUC: {loco['roc_auc'].mean():.4f}")
    print(f"Mean LOCO balanced accuracy: {loco['balanced_accuracy'].mean():.4f}")
    print(f"Review gate passed: {gate_passed}")
    print(f"Reports: {reports}")
    print(f"Visual QC: {overview_path}")
    print(f"Spatial score QC: {score_overview_path}")
    print(f"Contract: {contract_path}")
    print(
        "Audit only: scores are not labels; no authoritative mask changed, "
        "no model was retrained, and Field 2 remained locked."
    )


if __name__ == "__main__":
    main()
