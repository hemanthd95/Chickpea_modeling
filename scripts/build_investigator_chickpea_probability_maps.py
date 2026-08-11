#!/usr/bin/env python
"""Build continuous, review-only chickpea probability maps.

The separator is fitted from the frozen investigator-confirmed vegetation
references. Candidate centres are limited to polygon-guided non-soil support,
and their features use the same bandwise-median neighbourhood representation
as the reference audit. Outputs are diagnostic probabilities, never categorical
labels or authoritative masks.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from spectral.io import envi
import yaml

from chickpea_ssl.data import load_band_indices, load_records
from scripts.audit_investigator_vegetation_references import (
    LABELS,
    aggregate_reference_spectrum,
    disk_indices,
    fit_separator,
    ordered_cube_ids,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if (~normalized.isin({"true", "false"})).any():
        raise ValueError("Frozen usable_reference values are not boolean")
    return normalized.eq("true")


def map_disk_offsets(transform: Affine, radius_m: float) -> np.ndarray:
    """Return integer offsets whose pixel-centre displacement is in a circle."""
    if radius_m < 0:
        raise ValueError("Neighbourhood radius cannot be negative")
    determinant = abs(transform.a * transform.e - transform.b * transform.d)
    equivalent_gsd = math.sqrt(determinant)
    reach = int(math.ceil(radius_m / max(equivalent_gsd, 1e-12))) + 2
    offsets = []
    for delta_row in range(-reach, reach + 1):
        for delta_column in range(-reach, reach + 1):
            delta_x = transform.a * delta_column + transform.b * delta_row
            delta_y = transform.d * delta_column + transform.e * delta_row
            if math.hypot(delta_x, delta_y) <= radius_m + 1e-12:
                offsets.append((delta_row, delta_column))
    if not offsets:
        offsets = [(0, 0)]
    return np.asarray(offsets, dtype=np.int32)


def dense_neighbourhood_features(
    cube: np.ndarray,
    coordinates: np.ndarray,
    band_indices: np.ndarray,
    soil: np.ndarray,
    spatially_allowed: np.ndarray,
    offsets: np.ndarray,
    minimum_usable_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate candidate-centred neighbourhoods into median spectra."""
    if len(coordinates) == 0:
        return (
            np.empty((0, len(band_indices)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    height, width = cube.shape[:2]
    values = np.full(
        (len(coordinates), len(offsets), len(band_indices)),
        np.nan,
        dtype=np.float32,
    )
    for offset_number, (delta_row, delta_column) in enumerate(offsets):
        rows = coordinates[:, 0] + int(delta_row)
        columns = coordinates[:, 1] + int(delta_column)
        inside = (
            (rows >= 0) & (rows < height)
            & (columns >= 0) & (columns < width)
        )
        if not inside.any():
            continue
        selected = np.flatnonzero(inside)
        selected_rows = rows[selected]
        selected_columns = columns[selected]
        usable = (
            spatially_allowed[selected_rows, selected_columns]
            & ~soil[selected_rows, selected_columns]
        )
        selected = selected[usable]
        if not len(selected):
            continue
        selected_rows = rows[selected]
        selected_columns = columns[selected]
        spectra = np.asarray(
            cube[selected_rows, selected_columns, :][:, band_indices],
            dtype=np.float32,
        )
        observed = np.isfinite(spectra).all(axis=1) & np.any(spectra > 0, axis=1)
        selected = selected[observed]
        if len(selected):
            values[selected, offset_number, :] = spectra[observed]
    usable_counts = np.isfinite(values).all(axis=2).sum(axis=1).astype(np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        features = np.nanmedian(values, axis=1).astype(np.float32)
    valid = (
        (usable_counts >= minimum_usable_pixels)
        & np.isfinite(features).all(axis=1)
    )
    features[~valid] = np.nan
    return features, usable_counts


def reconstruct_frozen_features(
    records: dict,
    points: pd.DataFrame,
    point_qc: pd.DataFrame,
    band_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the exact accepted point-level median spectra."""
    usable_ids = set(
        point_qc.loc[boolean_series(point_qc["usable_reference"]), "annotation_id"]
    )
    features: list[np.ndarray] = []
    labels: list[int] = []
    cubes: list[str] = []
    for cube_id in ordered_cube_ids(points["cube_id"].unique()):
        record = records[cube_id]
        if record.soil_mask is None:
            raise ValueError(f"Frozen reference cube lacks a soil mask: {cube_id}")
        with rasterio.open(record.data) as reference:
            transform = reference.transform
            shape = (reference.height, reference.width)
        with rasterio.open(record.soil_mask) as dataset:
            soil = dataset.read(1) > 0
        image = envi.open(str(record.header), str(record.data))
        cube = image.open_memmap()
        for point in points.loc[points["cube_id"] == cube_id].itertuples(index=False):
            if point.annotation_id not in usable_ids:
                continue
            rows, columns = disk_indices(
                shape, transform, float(point.map_x), float(point.map_y),
                float(point.radius_m),
            )
            raw = np.asarray(
                cube[rows, columns, :][:, band_indices], dtype=np.float32
            )
            spectrum, _ = aggregate_reference_spectrum(raw, soil[rows, columns])
            if spectrum is None:
                raise ValueError(
                    f"Frozen usable reference no longer reconstructs: {point.annotation_id}"
                )
            features.append(spectrum)
            labels.append(LABELS[str(point.kind)])
            cubes.append(cube_id)
    if len(features) != len(usable_ids):
        raise ValueError(
            f"Reconstructed {len(features)} of {len(usable_ids)} frozen usable references"
        )
    return (
        np.stack(features), np.asarray(labels, dtype=np.uint8), np.asarray(cubes)
    )


def role_note(role: str, transfer_warning: bool) -> str:
    if transfer_warning:
        return "TRANSFER WARNING"
    if role == "label_expansion_candidate":
        return "PROJECTION ONLY"
    if role == "sensitivity_only":
        return "SENSITIVITY ONLY"
    return "PRIMARY REVIEW"


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
    policy = config["investigator_vegetation_probability_review"]
    if policy["status"] != "enabled_after_reference_contract_freeze":
        raise ValueError("Probability review has not been enabled")
    for forbidden in (
        "probability_threshold_is_a_class_label",
        "automatically_materialize_categorical_masks",
        "automatically_replace_authoritative_masks",
        "automatically_retrain_supervised_models",
    ):
        if bool(policy.get(forbidden, False)):
            raise ValueError(f"{forbidden} must remain false")
    if not bool(policy.get("write_continuous_probability_geotiffs", False)):
        raise ValueError("Continuous review output is disabled")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reference_contract_path = contracts / str(policy["reference_contract_filename"])
    candidate_contract_path = contracts / str(policy["candidate_contract_filename"])
    reference_contract = yaml.safe_load(reference_contract_path.read_text())
    candidate_contract = yaml.safe_load(candidate_contract_path.read_text())
    if reference_contract.get("status") != "field1_investigator_vegetation_references_frozen":
        raise ValueError("Investigator reference contract is not frozen")
    if not bool(reference_contract.get("acceptance", {}).get("gate_passed", False)):
        raise ValueError("Frozen investigator reference gate did not pass")
    if reference_contract.get("downstream_use") != "polygon_constrained_probability_review_only":
        raise ValueError("Frozen references do not permit this downstream use")
    if bool(reference_contract.get("field2_accessed", True)):
        raise ValueError("Frozen reference contract does not preserve the Field 2 lock")
    if candidate_contract.get("status") != "review_only_polygon_guided_candidates_materialized":
        raise ValueError("Polygon-guided candidate contract is not ready")

    frozen_files = {
        "reference_points": contracts / "field1_investigator_vegetation_reference_points.csv",
        "point_qc": contracts / "field1_investigator_vegetation_reference_qc.csv",
        "loco": contracts / "field1_investigator_vegetation_reference_loco_validation.csv",
        "spectral_profile": contracts / "field1_investigator_vegetation_reference_spectral_profile.csv",
        "visual_qc": contracts / "field1_investigator_vegetation_reference_overview.png",
    }
    for name, path in frozen_files.items():
        expected = reference_contract.get("frozen_artifact_hashes", {}).get(name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen reference artifact failed hash verification: {name}")

    points = pd.read_csv(frozen_files["reference_points"])
    point_qc = pd.read_csv(frozen_files["point_qc"])
    usable_points = int(boolean_series(point_qc["usable_reference"]).sum())
    if usable_points != int(reference_contract["usable_reference_points"]):
        raise ValueError("Frozen usable-reference count changed")

    manifest_path = local / "authoritative_manifest.csv"
    records = {record.cube_id: record for record in load_records(args.paths, manifest_path)}
    band_section = str(policy.get("spectral_band_section", "primary"))
    band_indices = load_band_indices(args.bands, band_section)
    features, target, reference_cubes = reconstruct_frozen_features(
        records, points, point_qc, band_indices
    )
    scaler, model = fit_separator(
        features, target,
        float(policy.get("logistic_regularization_c", 1.0)),
        int(policy.get("maximum_iterations", 1000)),
        int(policy.get("seed", 42)),
    )

    first_record = records[str(points.iloc[0]["cube_id"])]
    first_image = envi.open(str(first_record.header), str(first_record.data))
    wavelengths = np.asarray(
        first_image.metadata["wavelength"], dtype=np.float32
    )[band_indices]

    layer_manifest_path = (
        local / "reports" / "chickpea_region_annotation"
        / "chickpea_region_annotation_layer_manifest.csv"
    )
    roles = pd.read_csv(layer_manifest_path).set_index("cube_id")["analysis_role"]
    candidate_root = project / str(
        config["polygon_guided_candidate_materialization"]["output_root_relative"]
    )
    output_root = project / str(policy["output_root_relative"])
    reports = (
        local / "reports" / "mask_refinement"
        / "investigator_chickpea_probability_review"
    )
    individual = reports / "individual"
    output_root.mkdir(parents=True, exist_ok=True)
    individual.mkdir(parents=True, exist_ok=True)

    warning_cubes = {
        str(item["heldout_cube"])
        for item in reference_contract.get("per_cube_transfer_warnings", [])
    }
    support_codes = np.asarray(policy["score_candidate_support_codes"], dtype=np.uint8)
    radius_m = float(policy["dense_neighbourhood_radius_m"])
    minimum_pixels = int(policy["minimum_usable_pixels_per_neighbourhood"])
    chunk_size = int(policy["prediction_chunk_size"])
    probability_nodata = float(policy["probability_nodata_value"])
    summary_rows: list[dict] = []
    output_paths: list[Path] = []
    overview_tiles = []

    cube_ids = ordered_cube_ids(roles.index)
    for cube_number, cube_id in enumerate(cube_ids, 1):
        record = records[cube_id]
        role = str(roles.loc[cube_id])
        candidate_path = candidate_root / cube_id / "polygon_guided_candidate_support.tif"
        with rasterio.open(candidate_path) as dataset:
            candidate = dataset.read(1)
            profile = dataset.profile.copy()
            transform = dataset.transform
        image = envi.open(str(record.header), str(record.data))
        cube = image.open_memmap()
        if cube.shape[:2] != candidate.shape:
            raise ValueError(f"Candidate/cube shape mismatch for {cube_id}")

        if record.soil_mask is not None:
            with rasterio.open(record.soil_mask) as dataset:
                soil = dataset.read(1) > 0
            spatially_allowed = np.ones(candidate.shape, dtype=bool)
            neighbourhood_scope = "full_observed_cube_with_authoritative_soil_exclusion"
        else:
            soil = np.isin(candidate, [1, 3])
            spatially_allowed = np.isin(candidate, [1, 2, 3, 4])
            neighbourhood_scope = "polygon_prior_with_validated_index_soil_exclusion"

        eligible = np.isin(candidate, support_codes)
        coordinates = np.argwhere(eligible).astype(np.int32)
        offsets = map_disk_offsets(transform, radius_m)
        probability = np.full(candidate.shape, probability_nodata, dtype=np.float32)
        usable_count_map = np.zeros(candidate.shape, dtype=np.uint8)
        progress_every = max(1, math.ceil(100000 / chunk_size))
        for chunk_number, start in enumerate(
            range(0, len(coordinates), chunk_size), start=1
        ):
            chunk = coordinates[start:start + chunk_size]
            chunk_features, usable_counts = dense_neighbourhood_features(
                cube, chunk, band_indices, soil, spatially_allowed, offsets,
                minimum_pixels,
            )
            valid = np.isfinite(chunk_features).all(axis=1)
            scores = np.full(len(chunk), probability_nodata, dtype=np.float32)
            if valid.any():
                scores[valid] = model.predict_proba(
                    scaler.transform(chunk_features[valid])
                )[:, 1].astype(np.float32)
            probability[chunk[:, 0], chunk[:, 1]] = scores
            usable_count_map[chunk[:, 0], chunk[:, 1]] = np.minimum(
                usable_counts, 255
            ).astype(np.uint8)
            completed = min(start + len(chunk), len(coordinates))
            if (
                chunk_number == 1
                or chunk_number % progress_every == 0
                or completed == len(coordinates)
            ):
                print(
                    f"  [{cube_id}] {completed:,}/{len(coordinates):,} "
                    "candidate centres aggregated",
                    flush=True,
                )

        cube_output = output_root / cube_id
        cube_output.mkdir(parents=True, exist_ok=True)
        probability_path = cube_output / "chickpea_review_probability.tif"
        probability_profile = profile.copy()
        probability_profile.update(
            dtype="float32", count=1, nodata=probability_nodata,
            compress="deflate", predictor=3,
        )
        with rasterio.open(probability_path, "w", **probability_profile) as dataset:
            dataset.write(probability, 1)
            dataset.update_tags(
                status="continuous_probability_for_investigator_review_only",
                probability_class="chickpea",
                categorical_label="false",
                reference_authority="frozen_investigator_confirmed_points",
                neighbourhood_radius_m=f"{radius_m:.6f}",
                transfer_warning=str(cube_id in warning_cubes).lower(),
                analysis_role=role,
            )
        output_paths.append(probability_path)

        valid_scores = probability != probability_nodata
        core_scores = probability[(candidate == 2) & valid_scores]
        edge_scores = probability[(candidate == 4) & valid_scores]
        all_scores = probability[valid_scores]
        quantiles = (
            np.quantile(all_scores, [0.10, 0.25, 0.50, 0.75, 0.90])
            if len(all_scores) else [np.nan] * 5
        )
        summary_rows.append({
            "cube_id": cube_id,
            "analysis_role": role,
            "transfer_warning": cube_id in warning_cubes,
            "neighbourhood_scope": neighbourhood_scope,
            "candidate_centres": int(len(coordinates)),
            "scored_centres": int(valid_scores.sum()),
            "unscored_insufficient_neighbourhood": int(
                len(coordinates) - valid_scores.sum()
            ),
            "core_scored_centres": int(len(core_scores)),
            "edge_scored_centres": int(len(edge_scores)),
            "probability_q10": float(quantiles[0]),
            "probability_q25": float(quantiles[1]),
            "probability_median": float(quantiles[2]),
            "probability_q75": float(quantiles[3]),
            "probability_q90": float(quantiles[4]),
            "probability_path": str(probability_path),
        })

        display = probability.copy()
        display[~valid_scores] = np.nan
        step = max(1, math.ceil(max(candidate.shape) / 700))
        tile = display[::step, ::step]
        preview_path = individual / f"{cube_id}_chickpea_probability_review.png"
        figure, axis = plt.subplots(figsize=(6, 8), constrained_layout=True)
        handle = axis.imshow(tile, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        axis.axis("off")
        axis.set_title(
            f"{cube_id} | {role_note(role, cube_id in warning_cubes)}\n"
            f"continuous P(chickpea); scored {valid_scores.sum():,}/{len(coordinates):,}",
            color="#B91C1C" if cube_id in warning_cubes else "black", fontsize=10,
        )
        figure.colorbar(handle, ax=axis, shrink=0.72)
        figure.savefig(preview_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(preview_path)
        overview_tiles.append((cube_id, role, cube_id in warning_cubes, tile))
        print(
            f"Scored {cube_number}/{len(cube_ids)} {cube_id}: "
            f"{valid_scores.sum():,}/{len(coordinates):,} candidate centres; "
            f"median P(chickpea)={np.nanmedian(tile):.3f}", flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = reports / "investigator_chickpea_probability_summary.csv"
    summary.to_csv(summary_path, index=False)
    output_paths.append(summary_path)

    coefficient_rows = []
    coefficients = model.coef_[0]
    for band, wavelength, mean, scale, coefficient in zip(
        band_indices, wavelengths, scaler.mean_, scaler.scale_, coefficients
    ):
        coefficient_rows.append({
            "band_index": int(band), "wavelength_nm": float(wavelength),
            "standardization_mean": float(mean),
            "standardization_scale": float(scale),
            "logistic_coefficient": float(coefficient),
        })
    coefficient_path = reports / "investigator_review_separator_coefficients.csv"
    pd.DataFrame(coefficient_rows).to_csv(coefficient_path, index=False)
    output_paths.append(coefficient_path)

    columns = 4
    rows = math.ceil(len(overview_tiles) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(16, 4.4 * rows), constrained_layout=True
    )
    flat = np.asarray(axes).reshape(-1)
    for axis in flat:
        axis.axis("off")
        axis.set_facecolor("black")
    handle = None
    for axis, (cube_id, role, transfer_warning, tile) in zip(flat, overview_tiles):
        handle = axis.imshow(tile, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        axis.set_title(
            f"{cube_id} | {role_note(role, transfer_warning)}",
            color="#B91C1C" if transfer_warning else "black", fontsize=9,
        )
    if handle is not None:
        figure.colorbar(handle, ax=list(flat), shrink=0.55, label="P(chickpea), review only")
    figure.suptitle(
        "Field 1 investigator-guided chickpea probability review\n"
        "black=outside eligible non-soil support; probabilities are not categorical masks",
        fontsize=14,
    )
    overview_path = reports / "investigator_chickpea_probability_review_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    output_paths.append(overview_path)

    contract = {
        "status": "field1_investigator_chickpea_probability_review_materialized",
        "field": "Field 1",
        "output_semantics": "continuous_chickpea_probability_for_review_only",
        "reference_authority": "frozen_investigator_confirmed_points",
        "usable_reference_points": int(len(features)),
        "reference_cubes": ordered_cube_ids(np.unique(reference_cubes)),
        "separator": {
            "type": "standardized_balanced_logistic_regression",
            "spectral_band_section": band_section,
            "neighbourhood_aggregation": "bandwise_median_after_soil_and_nodata_exclusion",
            "dense_neighbourhood_radius_m": radius_m,
            "minimum_usable_pixels": minimum_pixels,
            "intercept": float(model.intercept_[0]),
        },
        "transfer_warning_cubes": ordered_cube_ids(warning_cubes),
        "candidate_support_codes_scored": [int(value) for value in support_codes],
        "probability_threshold_is_a_class_label": False,
        "categorical_masks_written": False,
        "authoritative_masks_modified": False,
        "supervised_models_retrained": False,
        "review_separator_fitted": True,
        "field2_accessed": False,
        "source_hashes": {
            "frozen_reference_contract": sha256(reference_contract_path),
            "candidate_materialization_contract": sha256(candidate_contract_path),
            "configuration": sha256(args.config),
            "spectral_band_config": sha256(args.bands),
            "authoritative_manifest": sha256(manifest_path),
            "annotation_layer_manifest": sha256(layer_manifest_path),
            **{
                f"candidate_support:{cube_id}": sha256(
                    candidate_root / cube_id / "polygon_guided_candidate_support.tif"
                )
                for cube_id in cube_ids
            },
        },
        "output_hashes": {
            str(path.relative_to(project)): sha256(path) for path in output_paths
        },
    }
    contract_path = contracts / "field1_investigator_chickpea_probability_review_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Review probability GeoTIFF root: {output_root}")
    print(f"Summary: {summary_path}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print(
        "Continuous probability review complete; no categorical or authoritative "
        "mask changed, no supervised model was retrained, and Field 2 remained locked."
    )


if __name__ == "__main__":
    main()
