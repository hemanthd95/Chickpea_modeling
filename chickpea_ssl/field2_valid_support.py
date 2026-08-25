"""Reflectance-only, windowed valid-support derivation and mask validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from affine import Affine
from scipy import ndimage

from chickpea_ssl.field2_readiness import ReadOnlySourceGuard


def threshold_token(value: float) -> str:
    """Return a stable rule-name token for a numeric stored-unit threshold."""
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number).replace(".", "p")


def model_band_indices(first: int, last_inclusive: int, band_count: int) -> np.ndarray:
    """Resolve an explicitly zero-based inclusive Python band range."""
    if first < 0 or last_inclusive < first or last_inclusive >= band_count:
        raise ValueError(
            f"Invalid zero-based band range {first}:{last_inclusive} for {band_count} bands"
        )
    return np.arange(first, last_inclusive + 1, dtype=np.int64)


def support_masks_windowed(
    array: np.ndarray,
    band_indices: Iterable[int],
    thresholds: Iterable[float],
    window_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Derive requested masks without materializing a complete spectral cube."""
    if array.ndim != 3:
        raise ValueError("Reflectance array must have (rows, columns, bands)")
    if window_rows <= 0:
        raise ValueError("window_rows must be positive")
    indices = np.asarray(list(band_indices), dtype=np.int64)
    if not len(indices) or indices.min() < 0 or indices.max() >= array.shape[2]:
        raise ValueError("Model band indices are empty or outside the cube")
    threshold_values = sorted(set(float(value) for value in thresholds))
    height, width = array.shape[:2]
    names = ["finite_any_nonzero", "finite_all_nonzero", "full_spectrum_any_nonzero"]
    names += [f"finite_positive_norm_gt_{threshold_token(value)}" for value in threshold_values]
    masks = {name: np.zeros((height, width), dtype=bool) for name in names}
    diagnostics = {
        "total_pixels": int(height * width),
        "finite_failure_count": 0,
        "full_spectrum_finite_failure_count": 0,
        "exact_all_zero_spectrum_count": 0,
        "partial_zero_spectrum_count": 0,
        "negative_value_spectrum_count": 0,
        "window_count": 0,
    }
    for start in range(0, height, window_rows):
        stop = min(start + window_rows, height)
        block = np.asarray(array[start:stop, :, :])
        model = block[..., indices]
        finite = np.isfinite(model).all(axis=2)
        any_nonzero = np.any(model != 0, axis=2)
        all_nonzero = np.all(model != 0, axis=2)
        model_float = np.asarray(model, dtype=np.float32)
        norm = np.max(np.abs(model_float), axis=2)
        full_finite = np.isfinite(block).all(axis=2)
        masks["finite_any_nonzero"][start:stop] = finite & any_nonzero
        masks["finite_all_nonzero"][start:stop] = finite & all_nonzero
        masks["full_spectrum_any_nonzero"][start:stop] = full_finite & np.any(block != 0, axis=2)
        for threshold in threshold_values:
            name = f"finite_positive_norm_gt_{threshold_token(threshold)}"
            masks[name][start:stop] = finite & (norm > threshold)
        diagnostics["finite_failure_count"] += int(np.count_nonzero(~finite))
        diagnostics["full_spectrum_finite_failure_count"] += int(np.count_nonzero(~full_finite))
        diagnostics["exact_all_zero_spectrum_count"] += int(np.count_nonzero(finite & ~any_nonzero))
        diagnostics["partial_zero_spectrum_count"] += int(
            np.count_nonzero(finite & any_nonzero & ~all_nonzero)
        )
        diagnostics["negative_value_spectrum_count"] += int(
            np.count_nonzero(finite & np.any(model_float < 0, axis=2))
        )
        diagnostics["window_count"] += 1
    return masks, diagnostics


def component_metrics(
    mask: np.ndarray,
    transform: Affine | None = None,
    connectivity: int = 8,
    small_component_max_pixels: int = 25,
) -> dict[str, object]:
    """Report components, holes, boundary, pixel extents, and map bounds."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Support mask must be two-dimensional")
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, count = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]
    valid_count = int(mask.sum())
    largest = int(sizes.max()) if len(sizes) else 0
    small = sizes <= int(small_component_max_pixels)
    small_pixels = int(sizes[small].sum()) if len(sizes) else 0
    component_ids = np.arange(1, count + 1, dtype=np.int64)
    non_small_ids = set(component_ids[~small].tolist())
    boundary_component_ids = set(
        np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist()
    ) - {0}
    non_small_interior_ids = non_small_ids - boundary_component_ids
    invalid_labels, _ = ndimage.label(~mask, structure=structure)
    border_labels = np.unique(
        np.concatenate(
            [invalid_labels[0], invalid_labels[-1], invalid_labels[:, 0], invalid_labels[:, -1]]
        )
    )
    hole_labels = set(np.unique(invalid_labels)) - set(border_labels) - {0}
    invalid_sizes = np.bincount(invalid_labels.ravel())
    boundary = mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)
    rows, columns = np.nonzero(mask)
    result: dict[str, object] = {
        "connected_component_count": int(count),
        "largest_component_pixels": largest,
        "largest_component_fraction": float(largest / valid_count) if valid_count else 0.0,
        "small_component_count": int(np.count_nonzero(small)),
        "small_component_pixels": small_pixels,
        "small_component_pixel_fraction": float(small_pixels / valid_count) if valid_count else 0.0,
        "raster_boundary_component_count": int(len(boundary_component_ids)),
        "non_small_component_count": int(len(non_small_ids)),
        "non_small_boundary_component_count": int(len(non_small_ids & boundary_component_ids)),
        "non_small_interior_component_count": int(len(non_small_interior_ids)),
        "interior_hole_count": int(len(hole_labels)),
        "interior_hole_pixels": int(sum(invalid_sizes[item] for item in hole_labels)),
        "boundary_pixel_count": int(boundary.sum()),
        "minimum_row": int(rows.min()) if len(rows) else -1,
        "maximum_row": int(rows.max()) if len(rows) else -1,
        "minimum_column": int(columns.min()) if len(columns) else -1,
        "maximum_column": int(columns.max()) if len(columns) else -1,
    }
    if len(rows) and transform is not None:
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(columns.min()), int(columns.max()) + 1
        corners = [transform * point for point in ((c0, r0), (c1, r0), (c0, r1), (c1, r1))]
        xs, ys = zip(*corners)
        result.update(
            geographic_left=float(min(xs)), geographic_bottom=float(min(ys)),
            geographic_right=float(max(xs)), geographic_top=float(max(ys)),
        )
    else:
        result.update(
            geographic_left=np.nan, geographic_bottom=np.nan,
            geographic_right=np.nan, geographic_top=np.nan,
        )
    return result


def disagreement_metrics(primary: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    primary = np.asarray(primary, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if primary.shape != candidate.shape:
        raise ValueError("Masks must have the same shape")
    disagreement = primary ^ candidate
    primary_valid = int(primary.sum())
    return {
        "disagreement_count": int(disagreement.sum()),
        "disagreement_fraction_total": float(disagreement.mean()),
        "disagreement_fraction_primary_valid": (
            float(disagreement.sum() / primary_valid) if primary_valid else 0.0
        ),
        "primary_only_count": int(np.count_nonzero(primary & ~candidate)),
        "candidate_only_count": int(np.count_nonzero(candidate & ~primary)),
    }


def grid_identity(reference: dict, candidate: dict, tolerance: float = 0.0) -> tuple[bool, list[str]]:
    """Compare CRS, dimensions, affine, bounds, and pixel-center grid."""
    issues: list[str] = []
    for key in ("width", "height", "count"):
        if key in candidate and key in reference and int(reference[key]) != int(candidate[key]):
            issues.append(f"{key}_mismatch")
    if str(reference.get("crs")) != str(candidate.get("crs")):
        issues.append("crs_mismatch")
    first = np.asarray(tuple(reference["transform"]), dtype=float)
    second = np.asarray(tuple(candidate["transform"]), dtype=float)
    if float(np.max(np.abs(first - second))) > tolerance:
        issues.append("transform_mismatch")
    if "bounds" in reference and "bounds" in candidate:
        first_bounds = np.asarray(tuple(reference["bounds"]), dtype=float)
        second_bounds = np.asarray(tuple(candidate["bounds"]), dtype=float)
        if float(np.max(np.abs(first_bounds - second_bounds))) > tolerance:
            issues.append("bounds_mismatch")
    return not issues, issues


def write_mask_geotiff(
    path: Path,
    mask: np.ndarray,
    reference_profile: dict,
    guard: ReadOnlySourceGuard,
    source_sha256: str,
    rule_name: str,
    configuration_sha256: str,
    band_indices: Iterable[int],
    compression: str = "DEFLATE",
    predictor: int = 2,
    zlevel: int = 9,
) -> None:
    """Write a deterministic uint8 support mask outside all source roots."""
    target = guard.output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": int(mask.shape[0]), "width": int(mask.shape[1]),
        "count": 1, "dtype": "uint8", "crs": reference_profile["crs"],
        "transform": reference_profile["transform"], "nodata": 0,
        "compress": compression, "predictor": predictor, "zlevel": zlevel,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(target, "w", **profile) as dataset:
            dataset.write(np.asarray(mask, dtype=np.uint8), 1)
            dataset.update_tags(
                valid_support_code="1", invalid_support_code="0",
                creation_purpose="Field 2 reflectance valid-support mask",
                creation_version="field2_valid_support_v1",
                source_reflectance_sha256=source_sha256,
                rule_name=rule_name, configuration_sha256=configuration_sha256,
                python_zero_based_band_indices=json.dumps([int(value) for value in band_indices]),
                class_information_used="false", model_inference_used="false",
            )


def verify_mask_geotiff(path: Path, reference: dict) -> tuple[bool, list[str], set[int]]:
    issues: list[str] = []
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(path, "r") as dataset:
            observed = {
                "width": dataset.width, "height": dataset.height, "count": dataset.count,
                "crs": dataset.crs, "transform": dataset.transform, "bounds": dataset.bounds,
            }
            _, grid_issues = grid_identity(reference, observed, tolerance=0.0)
            issues.extend(grid_issues)
            codes = {int(value) for value in np.unique(dataset.read(1))}
            if dataset.dtypes != ("uint8",):
                issues.append("dtype_not_uint8")
            if not codes.issubset({0, 1}):
                issues.append("unexpected_mask_codes")
    return not issues, issues, codes
