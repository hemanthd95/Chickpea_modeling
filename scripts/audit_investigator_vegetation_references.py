#!/usr/bin/env python
"""Audit investigator-confirmed chickpea/weed point spectra.

Each marked point is one independent observation. Pixels in its map-space
sampling circle are screened for soil and NoData, then reduced to a bandwise
median. Leave-one-cube-out validation therefore cannot gain artificial sample
size from neighbouring pixels or leak spectra from the held-out cube.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from spectral.io import envi
import yaml

from chickpea_ssl.data import load_band_indices, load_records


LABELS = {"confirmed_weed": 0, "confirmed_chickpea": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_cube_ids(values) -> list[str]:
    return sorted(values, key=lambda value: int(str(value).split("cube")[-1]))


def disk_indices(
    shape: tuple[int, int], transform: Affine, map_x: float, map_y: float,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel centres inside a map-space circle (or nearest pixel)."""
    inverse = ~transform
    column_float, row_float = inverse * (map_x, map_y)
    nearest_row = int(np.floor(row_float))
    nearest_column = int(np.floor(column_float))
    if radius_m <= 0:
        rows = np.asarray([nearest_row], dtype=np.int32)
        columns = np.asarray([nearest_column], dtype=np.int32)
    else:
        determinant = abs(transform.a * transform.e - transform.b * transform.d)
        equivalent_gsd = np.sqrt(determinant)
        reach = int(np.ceil(radius_m / max(equivalent_gsd, 1e-12))) + 2
        row_values = np.arange(nearest_row - reach, nearest_row + reach + 1)
        column_values = np.arange(nearest_column - reach, nearest_column + reach + 1)
        row_grid, column_grid = np.meshgrid(
            row_values, column_values, indexing="ij"
        )
        centre_x = (
            transform.a * (column_grid + 0.5)
            + transform.b * (row_grid + 0.5) + transform.c
        )
        centre_y = (
            transform.d * (column_grid + 0.5)
            + transform.e * (row_grid + 0.5) + transform.f
        )
        inside = np.hypot(centre_x - map_x, centre_y - map_y) <= radius_m
        rows = row_grid[inside].astype(np.int32)
        columns = column_grid[inside].astype(np.int32)
    bounded = (
        (rows >= 0) & (rows < shape[0])
        & (columns >= 0) & (columns < shape[1])
    )
    return rows[bounded], columns[bounded]


def aggregate_reference_spectrum(
    values: np.ndarray, soil: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """Filter a point neighbourhood and return one median spectrum."""
    observed = np.isfinite(values).all(axis=1) & np.any(values > 0, axis=1)
    usable = observed & ~soil
    diagnostics = {
        "sampled_pixels": int(len(values)),
        "observed_pixels": int(observed.sum()),
        "soil_pixels": int((observed & soil).sum()),
        "nodata_or_nonfinite_pixels": int((~observed).sum()),
        "usable_nonsoil_pixels": int(usable.sum()),
        "saturated_usable_pixels": int(
            np.any(values[usable] >= 65535, axis=1).sum()
        ),
    }
    if not usable.any():
        return None, diagnostics
    return np.median(values[usable], axis=0).astype(np.float32), diagnostics


def fit_separator(features, labels, regularization_c, iterations, seed):
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=regularization_c, class_weight="balanced", max_iter=iterations,
        random_state=seed, solver="liblinear",
    ).fit(scaler.transform(features), labels)
    return scaler, model


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    predicted = probability >= 0.5
    return {
        "reference_points": int(len(labels)),
        "chickpea_points": int(labels.sum()),
        "weed_points": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
        "chickpea_precision": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "chickpea_recall": float(
            recall_score(labels, predicted, zero_division=0)
        ),
    }


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
    policy = config["investigator_vegetation_reference_audit"]
    for forbidden in (
        "write_probability_geotiffs", "automatically_materialize_masks",
        "automatically_retrain_models",
    ):
        if bool(policy.get(forbidden, False)):
            raise ValueError(f"{forbidden} must remain false for this audit")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    annotation_path = (
        local / "annotations" / "vegetation_references"
        / "field1_vegetation_reference_points.csv"
    )
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Investigator reference points are missing: {annotation_path}"
        )
    annotations = pd.read_csv(annotation_path).fillna("")
    annotations = annotations[annotations["kind"].isin(LABELS)].copy()
    if annotations.empty:
        raise ValueError("No confirmed chickpea or weed reference points found")
    if annotations["annotation_id"].duplicated().any():
        raise ValueError("Duplicate annotation IDs found")
    if set(annotations["analysis_role"]) != {"primary_candidate"}:
        raise ValueError("Confirmed references must be on primary cubes only")

    minimum_points = int(policy["minimum_points_per_class_per_primary_cube"])
    counts = annotations.groupby(["cube_id", "kind"]).size().unstack(fill_value=0)
    if (counts.reindex(columns=list(LABELS), fill_value=0) < minimum_points).any().any():
        raise ValueError("At least one cube/class has too few investigator points")

    manifest_path = local / "authoritative_manifest.csv"
    records = {r.cube_id: r for r in load_records(args.paths, manifest_path)}
    primary = ordered_cube_ids(annotations["cube_id"].unique())
    role_manifest_path = (
        local / "reports" / "chickpea_region_annotation"
        / "chickpea_region_annotation_layer_manifest.csv"
    )
    roles = pd.read_csv(role_manifest_path).fillna("")
    expected_primary = set(
        roles.loc[roles["analysis_role"] == "primary_candidate", "cube_id"]
    )
    if set(primary) != expected_primary:
        raise ValueError(
            "Confirmed-reference cube set differs from the frozen primary set: "
            f"missing={sorted(expected_primary - set(primary))}; "
            f"unexpected={sorted(set(primary) - expected_primary)}"
        )
    sensitivity = set(
        config["investigator_guidance"]
        ["phenology_exclusions_from_refined_primary_supervised_analysis"]
    )
    expansion = set(config["investigator_guidance"]["optional_mask_expansion"]["cubes"])
    if set(primary) & (sensitivity | expansion):
        raise ValueError("Sensitivity/projection-only cubes cannot fit the separator")

    candidate_root = project / config["polygon_guided_candidate_materialization"][
        "output_root_relative"
    ]
    band_section = str(policy.get("spectral_band_section", "primary"))
    band_indices = load_band_indices(args.bands, band_section)
    minimum_usable = int(policy.get("minimum_usable_pixels_per_reference", 3))
    point_rows: list[dict] = []
    spectra: list[np.ndarray] = []
    labels: list[int] = []
    cube_ids: list[str] = []
    wavelengths = None

    for number, cube_id in enumerate(primary, start=1):
        if cube_id not in records:
            raise ValueError(f"Cube is absent from authoritative manifest: {cube_id}")
        record = records[cube_id]
        if record.soil_mask is None:
            raise ValueError(f"Primary cube lacks a soil mask: {cube_id}")
        candidate_path = candidate_root / cube_id / "polygon_guided_candidate_support.tif"
        with rasterio.open(candidate_path) as dataset:
            candidate = dataset.read(1)
            transform = dataset.transform
            candidate_crs = str(dataset.crs)
        with rasterio.open(record.soil_mask) as dataset:
            soil = dataset.read(1) > 0
        image = envi.open(str(record.header), str(record.data))
        cube = image.open_memmap()
        if candidate.shape != cube.shape[:2] or soil.shape != cube.shape[:2]:
            raise ValueError(f"Raster alignment mismatch for {cube_id}")
        if wavelengths is None:
            wavelengths = np.asarray(
                image.metadata["wavelength"], dtype=np.float32
            )[band_indices]
        cube_points = annotations[annotations["cube_id"] == cube_id]
        for point in cube_points.itertuples(index=False):
            if str(point.crs) != candidate_crs:
                raise ValueError(
                    f"Annotation/raster CRS mismatch for {point.annotation_id}: "
                    f"{point.crs} versus {candidate_crs}"
                )
            rows, columns = disk_indices(
                candidate.shape, transform, float(point.map_x), float(point.map_y),
                float(point.radius_m),
            )
            values = np.asarray(cube[rows, columns, :][:, band_indices], dtype=np.float32)
            spectrum, diagnostic = aggregate_reference_spectrum(
                values, soil[rows, columns]
            )
            centre_column_float, centre_row_float = (~transform) * (
                float(point.map_x), float(point.map_y)
            )
            centre_row = int(np.clip(
                np.floor(centre_row_float), 0, candidate.shape[0] - 1
            ))
            centre_column = int(np.clip(
                np.floor(centre_column_float), 0, candidate.shape[1] - 1
            ))
            centre_support = int(candidate[centre_row, centre_column])
            centre_soil = bool(soil[centre_row, centre_column])
            centre_values = np.asarray(
                cube[centre_row, centre_column, :][band_indices], dtype=np.float32
            )
            centre_observed = bool(
                np.isfinite(centre_values).all() and np.any(centre_values > 0)
            )
            expected_position = (
                centre_support == 2 if point.kind == "confirmed_chickpea"
                else centre_support != 2
            )
            usable = bool(
                spectrum is not None
                and diagnostic["usable_nonsoil_pixels"] >= minimum_usable
                and centre_observed and not centre_soil and expected_position
            )
            row = {
                "cube_id": cube_id,
                "annotation_id": point.annotation_id,
                "kind": point.kind,
                "class_id": LABELS[point.kind],
                "layer": point.layer,
                "radius_m": float(point.radius_m),
                "map_x": float(point.map_x), "map_y": float(point.map_y),
                "crs": point.crs, "candidate_crs": candidate_crs,
                "centre_row": centre_row, "centre_column": centre_column,
                "centre_candidate_support": centre_support,
                "centre_is_soil": centre_soil,
                "centre_is_observed": centre_observed,
                "centre_position_matches_reference": expected_position,
                **diagnostic,
                "usable_reference": usable,
            }
            point_rows.append(row)
            if usable:
                spectra.append(spectrum)
                labels.append(LABELS[point.kind])
                cube_ids.append(cube_id)
        usable_count = sum(
            bool(row["usable_reference"]) for row in point_rows
            if row["cube_id"] == cube_id
        )
        print(
            f"Audited {number}/{len(primary)} {cube_id}: "
            f"{len(cube_points)} points; {usable_count} usable", flush=True,
        )

    point_qc = pd.DataFrame(point_rows)
    usable_counts = (
        point_qc[point_qc["usable_reference"]]
        .groupby(["cube_id", "kind"]).size().unstack(fill_value=0)
        .reindex(index=primary, columns=list(LABELS), fill_value=0)
    )
    if (usable_counts < minimum_points).any().any():
        bad = point_qc.loc[
            ~point_qc["usable_reference"], "annotation_id"
        ].tolist()
        raise ValueError(
            "Too few usable references remain in at least one cube/class after "
            f"geometry/soil/NoData QC; rejected IDs begin: {bad[:5]}"
        )
    features = np.stack(spectra)
    target = np.asarray(labels, dtype=np.uint8)
    cube_vector = np.asarray(cube_ids)

    regularization = float(policy.get("logistic_regularization_c", 1.0))
    iterations = int(policy.get("maximum_iterations", 1000))
    seed = int(policy.get("seed", 42))
    prediction_rows: list[dict] = []
    loco_rows: list[dict] = []
    for heldout in primary:
        train = cube_vector != heldout
        test = ~train
        scaler, model = fit_separator(
            features[train], target[train], regularization, iterations, seed
        )
        probability = model.predict_proba(scaler.transform(features[test]))[:, 1]
        result = {"heldout_cube": heldout, **metrics(target[test], probability)}
        loco_rows.append(result)
        heldout_qc = point_qc.loc[
            (point_qc["cube_id"] == heldout) & point_qc["usable_reference"]
        ].reset_index(drop=True)
        for point, truth, score in zip(
            heldout_qc.itertuples(index=False), target[test], probability
        ):
            prediction_rows.append({
                "heldout_cube": heldout, "annotation_id": point.annotation_id,
                "kind": point.kind, "true_class_id": int(truth),
                "chickpea_probability": float(score),
                "predicted_class_id": int(score >= 0.5),
                "correct": bool((score >= 0.5) == truth),
            })
        print(
            f"LOCO {heldout}: AUC={result['roc_auc']:.3f}; "
            f"balanced accuracy={result['balanced_accuracy']:.3f}", flush=True,
        )

    loco = pd.DataFrame(loco_rows)
    predictions = pd.DataFrame(prediction_rows)
    minimum_auc = float(policy["minimum_mean_loco_auc"])
    minimum_ba = float(policy["minimum_mean_loco_balanced_accuracy"])
    minimum_fraction = float(
        policy["minimum_fraction_heldout_cubes_passing_balanced_accuracy"]
    )
    passing_fraction = float((loco["balanced_accuracy"] >= minimum_ba).mean())
    gate_passed = bool(
        loco["roc_auc"].mean() >= minimum_auc
        and loco["balanced_accuracy"].mean() >= minimum_ba
        and passing_fraction >= minimum_fraction
    )

    reports = (
        local / "reports" / "mask_refinement"
        / "investigator_vegetation_reference_audit"
    )
    reports.mkdir(parents=True, exist_ok=True)
    support = (
        point_qc.groupby(["cube_id", "kind"], as_index=False)
        .agg(
            reference_points=("annotation_id", "size"),
            usable_references=("usable_reference", "sum"),
            sampled_pixels=("sampled_pixels", "sum"),
            usable_nonsoil_pixels=("usable_nonsoil_pixels", "sum"),
            soil_pixels_excluded=("soil_pixels", "sum"),
            nodata_or_nonfinite_pixels_excluded=("nodata_or_nonfinite_pixels", "sum"),
        )
    )
    point_path = reports / "investigator_vegetation_reference_qc.csv"
    support_path = reports / "investigator_vegetation_reference_support_by_cube.csv"
    loco_path = reports / "investigator_vegetation_reference_loco_validation.csv"
    prediction_path = reports / "investigator_vegetation_reference_predictions.csv"
    point_qc.to_csv(point_path, index=False)
    support.to_csv(support_path, index=False)
    loco.to_csv(loco_path, index=False)
    predictions.to_csv(prediction_path, index=False)

    spectral_rows = []
    for class_id, class_name in ((0, "confirmed_weed"), (1, "confirmed_chickpea")):
        class_features = features[target == class_id]
        q25, median, q75 = np.quantile(class_features, [0.25, 0.5, 0.75], axis=0)
        for band, wavelength, low, centre, high in zip(
            band_indices, wavelengths, q25, median, q75
        ):
            spectral_rows.append({
                "class_name": class_name, "band_index": int(band),
                "wavelength_nm": float(wavelength),
                "q25_stored_value": float(low),
                "median_stored_value": float(centre),
                "q75_stored_value": float(high),
            })
    spectral = pd.DataFrame(spectral_rows)
    spectral_path = reports / "investigator_vegetation_reference_spectral_profile.csv"
    spectral.to_csv(spectral_path, index=False)

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    colors = {"confirmed_weed": "#7C3AED", "confirmed_chickpea": "#16A34A"}
    for class_name, group in spectral.groupby("class_name"):
        axes[0].plot(
            group["wavelength_nm"], group["median_stored_value"],
            color=colors[class_name], label=class_name.replace("confirmed_", "").title(),
        )
        axes[0].fill_between(
            group["wavelength_nm"], group["q25_stored_value"],
            group["q75_stored_value"], color=colors[class_name], alpha=0.18,
        )
    axes[0].set_title("Investigator-confirmed point spectra")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Stored reflectance value")
    axes[0].legend()

    x = np.arange(len(loco))
    axes[1].plot(x, loco["roc_auc"], marker="o", label="ROC AUC")
    axes[1].plot(x, loco["balanced_accuracy"], marker="o", label="Balanced accuracy")
    axes[1].axhline(
        minimum_ba, color="black", linestyle="--", label=f"Gate {minimum_ba:.2f}"
    )
    axes[1].set_xticks(
        x, loco["heldout_cube"].str.replace("field1_cube", ""), rotation=90
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Leave-one-cube-out transfer")
    axes[1].set_xlabel("Held-out cube")
    axes[1].legend()

    matrix = confusion_matrix(
        predictions["true_class_id"], predictions["predicted_class_id"],
        labels=[0, 1], normalize="true",
    )
    handle = axes[2].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[2].text(
                column, row, f"{matrix[row, column]:.2f}", ha="center", va="center"
            )
    axes[2].set_xticks([0, 1], ["Weed", "Chickpea"])
    axes[2].set_yticks([0, 1], ["Weed", "Chickpea"])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Investigator reference")
    axes[2].set_title("Cross-cube point confusion")
    figure.colorbar(handle, ax=axes[2], shrink=0.75)
    figure.suptitle(
        "Field 1 investigator vegetation-reference audit — no mask changes\n"
        f"{len(point_qc)} points; mean LOCO AUC={loco['roc_auc'].mean():.3f}; "
        f"balanced accuracy={loco['balanced_accuracy'].mean():.3f}; gate passed={gate_passed}",
        fontsize=14,
    )
    overview_path = reports / "investigator_vegetation_reference_audit_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    output_paths = [
        point_path, support_path, loco_path, prediction_path, spectral_path,
        overview_path,
    ]
    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "investigator_vegetation_reference_audit_complete",
        "field": "Field 1",
        "independent_observation_unit": "one_investigator_annotation_point",
        "neighbourhood_aggregation": (
            "bandwise_median_after_soil_and_nodata_exclusion"
        ),
        "reference_cubes": primary,
        "reference_points": int(len(point_qc)),
        "confirmed_chickpea_points": int(
            (point_qc["kind"] == "confirmed_chickpea").sum()
        ),
        "confirmed_weed_points": int(
            (point_qc["kind"] == "confirmed_weed").sum()
        ),
        "all_reference_points_passed_qc": bool(
            point_qc["usable_reference"].all()
        ),
        "mean_loco_auc": float(loco["roc_auc"].mean()),
        "mean_loco_balanced_accuracy": float(
            loco["balanced_accuracy"].mean()
        ),
        "heldout_cubes_passing_balanced_accuracy_fraction": passing_fraction,
        "review_gate_passed": gate_passed,
        "gate": {
            "minimum_mean_loco_auc": minimum_auc,
            "minimum_mean_loco_balanced_accuracy": minimum_ba,
            "minimum_fraction_heldout_cubes_passing_balanced_accuracy": minimum_fraction,
        },
        "sensitivity_cubes_reference_excluded": ordered_cube_ids(sensitivity),
        "expansion_cubes_projection_only": ordered_cube_ids(expansion),
        "historical_chickpea_and_weed_masks_used_as_truth": False,
        "probability_geotiffs_written": False,
        "categorical_masks_written": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {
            "reference_points": sha256(annotation_path),
            "authoritative_manifest": sha256(manifest_path),
            "annotation_role_manifest": sha256(role_manifest_path),
            "spectral_band_config": sha256(args.bands),
        },
        "output_hashes": {path.name: sha256(path) for path in output_paths},
    }
    contract_path = (
        contracts / "field1_investigator_vegetation_reference_audit_contract.yaml"
    )
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Investigator reference points audited: {len(point_qc)}")
    print(f"Usable point-level spectra: {len(features)}")
    print(f"Mean LOCO AUC: {loco['roc_auc'].mean():.4f}")
    print(f"Mean LOCO balanced accuracy: {loco['balanced_accuracy'].mean():.4f}")
    print(f"Review gate passed: {gate_passed}")
    print(f"Reports: {reports}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print(
        "Audit only: no authoritative mask changed, no probability raster was "
        "written, no model was retrained, and Field 2 remained locked."
    )


if __name__ == "__main__":
    main()
