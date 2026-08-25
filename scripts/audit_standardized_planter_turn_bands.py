#!/usr/bin/env python
"""Build a standardized reflectance-PCA audit of transverse planter evidence.

This investigator-review script uses only observed Field 1 reflectance. It
does not alter masks, assign class labels, train a model, or access Field 2.
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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from sklearn.decomposition import PCA
from spectral.io import envi
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metres_per_pixel(transform) -> float:
    x = math.hypot(transform.a, transform.d)
    y = math.hypot(transform.b, transform.e)
    if x <= 0 or y <= 0:
        raise ValueError("Raster transform does not contain a positive pixel size")
    return float(math.sqrt(x * y))


def circular_distance_degrees(values: np.ndarray, center: float) -> np.ndarray:
    return np.abs((values - center + 90.0) % 180.0 - 90.0)


def dominant_row_angle(mask: np.ndarray, gsd: float, settings: dict) -> tuple[float, int]:
    lines = cv2.HoughLinesP(
        mask.astype(np.uint8) * 255,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(settings["hough_threshold"]),
        minLineLength=max(5, round(float(settings["minimum_line_length_m"]) / gsd)),
        maxLineGap=max(1, round(float(settings["maximum_line_gap_m"]) / gsd)),
    )
    if lines is None:
        return float("nan"), 0
    segments = np.asarray(lines).reshape(-1, 4).astype(float)
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    lengths = np.hypot(dx, dy)
    angles = np.degrees(np.arctan2(dy, dx)) % 180.0
    width = float(settings["orientation_bin_degrees"])
    edges = np.arange(0.0, 180.0 + width, width)
    histogram, _ = np.histogram(angles, bins=edges, weights=lengths)
    dominant = float((edges[int(histogram.argmax())] + width / 2.0) % 180.0)
    tolerance = float(settings["orientation_tolerance_degrees"])
    return dominant, int((circular_distance_degrees(angles, dominant) <= tolerance).sum())


def robust_scale(channel: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if valid is None:
        valid = np.isfinite(channel)
    else:
        valid = valid & np.isfinite(channel)
    result = np.zeros(channel.shape, dtype=np.float32)
    values = channel[valid]
    if not len(values):
        return result
    low, high = np.percentile(values, (2, 98))
    result[valid] = np.clip((channel[valid] - low) / max(float(high - low), 1e-6), 0, 1)
    return result


def standardized_preview_pca(
    memory: np.ndarray, step: int, maximum_fit_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    sampled = np.asarray(memory[::step, ::step, :], dtype=np.float32)
    finite = np.all(np.isfinite(sampled), axis=2)
    # Match the frozen project NoData contract: a spatial pixel is observed
    # when at least one spectral band is positive. A value of 65535 in an
    # individual band is not automatically NoData in the stored Pika-L cubes.
    observed = np.any(sampled > 0, axis=2)
    valid = finite & observed
    spectra = sampled[valid]
    if len(spectra) < 1000:
        raise ValueError(
            "Too few valid reflectance spectra for standardized PCA: "
            f"valid={len(spectra):,}, finite_pixels={int(finite.sum()):,}, "
            f"observed_pixels={int(observed.sum()):,}, total_pixels={valid.size:,}"
        )
    rng = np.random.default_rng(seed)
    if len(spectra) > maximum_fit_samples:
        spectra = spectra[rng.choice(len(spectra), maximum_fit_samples, replace=False)]
    mean = spectra.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = spectra.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=seed)
    pca.fit((spectra - mean) / std)
    flat = sampled.reshape(-1, sampled.shape[2])
    flat_valid = valid.reshape(-1)
    scores = np.zeros((len(flat), 3), dtype=np.float32)
    chunk = 100_000
    indices = np.flatnonzero(flat_valid)
    for start in range(0, len(indices), chunk):
        chosen = indices[start:start + chunk]
        scores[chosen] = pca.transform((flat[chosen] - mean) / std).astype(np.float32)
    scores = scores.reshape(sampled.shape[:2] + (3,))
    diagnostics = {
        "preview_total_pixels": int(valid.size),
        "preview_finite_pixels": int(finite.sum()),
        "preview_observed_pixels": int(observed.sum()),
        "preview_valid_pca_pixels": int(valid.sum()),
        "pca_fit_spectra": int(len(spectra)),
    }
    return scores, valid, pca.explained_variance_ratio_, mean, diagnostics


def pca_rgb(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    channels = [robust_scale(scores[..., index], valid) for index in range(3)]
    rgb = np.stack(channels, axis=-1)
    rgb[~valid] = np.array([0.0, 0.45, 0.45])
    return rgb


def spatial_derivatives(scores: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_layers = []
    second_layers = []
    for index in range(scores.shape[2]):
        channel = robust_scale(scores[..., index], valid)
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        first_layers.append(cv2.magnitude(gx, gy))
        second_layers.append(np.abs(cv2.Laplacian(channel, cv2.CV_32F, ksize=3)))
    first = robust_scale(np.mean(first_layers, axis=0), valid)
    second = robust_scale(np.mean(second_layers, axis=0), valid)
    structure = robust_scale(0.55 * first + 0.45 * second, valid)
    return first, second, structure


def rotation_matrices(shape: tuple[int, int], row_angle: float):
    height, width = shape
    rotation = 90.0 - row_angle
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotation, 1.0)
    inverse = cv2.invertAffineTransform(matrix)
    return matrix, inverse


def rotate(array: np.ndarray, matrix: np.ndarray, interpolation: int) -> np.ndarray:
    height, width = array.shape[:2]
    return cv2.warpAffine(
        array, matrix, (width, height), flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def transverse_profile(
    structure: np.ndarray, valid: np.ndarray, reduced_gsd: float, settings: dict
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    threshold = float(np.percentile(structure[valid], float(settings["structure_percentile"])))
    active = (structure >= threshold) & valid
    valid_count = valid.sum(axis=1)
    energy = np.divide(
        (structure * active).sum(axis=1), valid_count,
        out=np.zeros(structure.shape[0], dtype=np.float32), where=valid_count > 0,
    )
    coverage = np.divide(
        active.sum(axis=1), valid_count,
        out=np.zeros(structure.shape[0], dtype=np.float32), where=valid_count > 0,
    )
    sigma = max(1.0, float(settings["profile_smoothing_m"]) / reduced_gsd)
    combined = gaussian_filter1d(energy, sigma=sigma) + gaussian_filter1d(coverage, sigma=sigma)
    eligible = valid_count >= max(5, int(float(settings["minimum_valid_width_fraction"]) * valid.shape[1]))
    values = combined[eligible]
    median = float(np.median(values)) if len(values) else 0.0
    mad = float(np.median(np.abs(values - median))) if len(values) else 0.0
    score = (combined - median) / max(1.4826 * mad, 1e-6)
    margin = max(1, round(float(settings["edge_exclusion_m"]) / reduced_gsd))
    score[:margin] = -np.inf
    score[-margin:] = -np.inf
    score[~eligible] = -np.inf
    minimum_distance = max(1, round(float(settings["minimum_candidate_separation_m"]) / reduced_gsd))
    finite_score = np.where(np.isfinite(score), score, -1e6)
    peaks, properties = find_peaks(
        finite_score,
        height=float(settings["minimum_robust_z"]),
        prominence=float(settings["minimum_prominence_z"]),
        distance=minimum_distance,
    )
    candidates: list[dict[str, float]] = []
    if len(peaks):
        widths, _, left, right = peak_widths(finite_score, peaks, rel_height=0.5)
        for peak, height_value, prominence, width, left_edge, right_edge in zip(
            peaks, properties["peak_heights"], properties["prominences"], widths, left, right
        ):
            width_m = float(width * reduced_gsd)
            if not (float(settings["minimum_band_width_m"]) <= width_m <= float(settings["maximum_band_width_m"])):
                continue
            candidates.append({
                "center_row_rotated": float(peak),
                "left_row_rotated": float(left_edge),
                "right_row_rotated": float(right_edge),
                "width_m": width_m,
                "robust_z": float(height_value),
                "prominence_z": float(prominence),
                "active_coverage_fraction": float(coverage[peak]),
            })
    return score, coverage, candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    settings = config["planter_pass_reconstruction"]["standardized_turn_band_audit"]
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    report_root = local / "reports" / "standardized_planter_turn_audit"
    details = report_root / "cube_previews"
    layers_root = report_root / "annotation_layers"
    contracts = local / "contracts"
    details.mkdir(parents=True, exist_ok=True)
    layers_root.mkdir(parents=True, exist_ok=True)
    contracts.mkdir(parents=True, exist_ok=True)

    excluded = set(config["investigator_guidance"]["phenology_exclusions_from_refined_primary_supervised_analysis"])
    records = sorted(
        [record for record in load_records(args.paths, local / "authoritative_manifest.csv") if record.chickpea_mask],
        key=lambda record: record.cube_id,
    )
    inventory_rows = []
    candidate_rows = []
    layer_rows = []
    overview_tiles = []
    input_hashes = []
    max_preview = int(settings["maximum_preview_dimension_pixels"])
    seed = int(settings["seed"])

    for number, record in enumerate(records, start=1):
        image = envi.open(str(record.header), str(record.data))
        memory = image.open_memmap()
        height, width = memory.shape[:2]
        with rasterio.open(record.data) as reference:
            transform = reference.transform
            gsd = metres_per_pixel(transform)
            crs = str(reference.crs or "")
        labels = authoritative_class_map(record)
        chickpea = labels == 1
        angle, inlier_lines = dominant_row_angle(chickpea, gsd, config["row_detection"])
        if not np.isfinite(angle):
            inventory_rows.append({"cube_id": record.cube_id, "audit_status": "row_angle_unresolved"})
            print(f"Skipped {record.cube_id}: row angle unresolved")
            continue
        step = max(1, math.ceil(max(height, width) / max_preview))
        scores, valid, explained, _, pca_diagnostics = standardized_preview_pca(
            memory, step, int(settings["maximum_pca_fit_samples"]), seed
        )
        rgb = pca_rgb(scores, valid)
        first, second, structure = spatial_derivatives(scores, valid)
        cube_layers = layers_root / record.cube_id
        cube_layers.mkdir(parents=True, exist_ok=True)
        layer_paths = {
            "pca": cube_layers / "pca_rgb.png",
            "first_difference": cube_layers / "first_difference_magnitude.png",
            "second_difference": cube_layers / "second_difference_magnitude.png",
        }
        plt.imsave(layer_paths["pca"], rgb)
        plt.imsave(layer_paths["first_difference"], first, cmap="gray", vmin=0, vmax=1)
        plt.imsave(layer_paths["second_difference"], second, cmap="gray", vmin=0, vmax=1)
        layer_rows.append({
            "cube_id": record.cube_id,
            "preview_height": scores.shape[0],
            "preview_width": scores.shape[1],
            "preview_step": step,
            "original_height": height,
            "original_width": width,
            "transform_a": transform.a,
            "transform_b": transform.b,
            "transform_c": transform.c,
            "transform_d": transform.d,
            "transform_e": transform.e,
            "transform_f": transform.f,
            "crs": crs,
            **{f"{name}_path": str(path) for name, path in layer_paths.items()},
        })
        matrix, inverse = rotation_matrices(structure.shape, angle)
        rotated_structure = rotate(structure, matrix, cv2.INTER_LINEAR)
        rotated_valid = rotate(valid.astype(np.uint8), matrix, cv2.INTER_NEAREST) > 0
        reduced_gsd = gsd * step
        profile, coverage, candidates = transverse_profile(
            rotated_structure, rotated_valid, reduced_gsd, settings
        )

        overlay_rotated = np.dstack([rotated_structure] * 3)
        overlay_original = rgb.copy()
        for candidate_id, candidate in enumerate(candidates, start=1):
            top = int(round(candidate["left_row_rotated"]))
            bottom = int(round(candidate["right_row_rotated"]))
            top = max(0, min(top, rotated_structure.shape[0] - 1))
            bottom = max(top + 1, min(bottom, rotated_structure.shape[0] - 1))
            cv2.rectangle(
                overlay_rotated, (0, top), (rotated_structure.shape[1] - 1, bottom),
                color=(1.0, 0.8, 0.0), thickness=2,
            )
            rotated_center = np.array([[[rotated_structure.shape[1] / 2.0, candidate["center_row_rotated"]]]], dtype=np.float32)
            original_center = cv2.transform(rotated_center, inverse)[0, 0]
            column_reduced = float(original_center[0])
            row_reduced = float(original_center[1])
            full_row = row_reduced * step
            full_column = column_reduced * step
            map_x, map_y = xy(transform, full_row, full_column, offset="center")
            candidate.update({
                "candidate_id": candidate_id,
                "cube_id": record.cube_id,
                "center_row_reduced": row_reduced,
                "center_column_reduced": column_reduced,
                "map_x": float(map_x), "map_y": float(map_y), "crs": crs,
                "status": "investigator_review_required",
            })
            candidate_rows.append(candidate)
            cv2.circle(overlay_original, (round(column_reduced), round(row_reduced)), 8, (1.0, 0.8, 0.0), 2)

        chickpea_small = chickpea[::step, ::step]
        mask_overlay = rgb.copy()
        mask_overlay[chickpea_small] = 0.25 * mask_overlay[chickpea_small] + 0.75 * np.array([0.1, 1.0, 0.1])
        figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
        panels = [
            (rgb, "Standardized reflectance PCA 1–3"),
            (first, "Spatial first-derivative magnitude"),
            (second, "Spatial second-derivative magnitude"),
            (mask_overlay, "Current chickpea mask — green"),
            (overlay_rotated, "Rows upright; candidate bands — yellow"),
        ]
        for axis, (panel, title) in zip(axes.reshape(-1)[:5], panels):
            axis.imshow(panel, cmap=None if panel.ndim == 3 else "gray")
            axis.set_title(title)
            axis.axis("off")
        profile_axis = axes.reshape(-1)[5]
        y = np.arange(len(profile)) * reduced_gsd
        profile_axis.plot(profile, y, color="black", label="robust transverse score")
        profile_axis.axvline(float(settings["minimum_robust_z"]), color="red", linestyle="--", label="candidate threshold")
        for candidate in candidates:
            profile_axis.axhspan(
                candidate["left_row_rotated"] * reduced_gsd,
                candidate["right_row_rotated"] * reduced_gsd,
                color="gold", alpha=0.35,
            )
        profile_axis.invert_yaxis()
        profile_axis.set_xlabel("Robust transverse-structure score")
        profile_axis.set_ylabel("Rotated cross-row position (m)")
        profile_axis.set_title("Transverse-band evidence profile")
        profile_axis.legend(fontsize=8)
        analysis_role = "sensitivity_only" if record.cube_id in excluded else "primary_candidate"
        figure.suptitle(
            f"{record.cube_id} standardized planter-turn audit | row angle {angle:.1f}° | "
            f"{len(candidates)} candidate bands | {analysis_role}", fontsize=15,
        )
        preview_path = details / f"{record.cube_id}_standardized_planter_turn_audit.png"
        figure.savefig(preview_path, dpi=180, facecolor="white")
        plt.close(figure)

        inventory_rows.append({
            "cube_id": record.cube_id,
            "audit_status": "review_ready",
            "analysis_role": analysis_role,
            "height": height, "width": width, "preview_step": step,
            "equivalent_gsd_m": gsd,
            "dominant_row_angle_degrees": angle,
            "row_hough_inlier_lines": inlier_lines,
            "pca_explained_variance_1": float(explained[0]),
            "pca_explained_variance_2": float(explained[1]),
            "pca_explained_variance_3": float(explained[2]),
            **pca_diagnostics,
            "candidate_bands": len(candidates),
            "preview_path": str(preview_path),
        })
        input_hashes.append({
            "cube_id": record.cube_id,
            "reflectance_header": str(record.header),
            "header_sha256": sha256(record.header),
        })
        overview_step = max(1, math.ceil(max(overlay_original.shape[:2]) / 320))
        overview_tiles.append((record.cube_id, overlay_original[::overview_step, ::overview_step], len(candidates), analysis_role))
        print(
            f"Prepared {number}/{len(records)} {record.cube_id}: "
            f"{pca_diagnostics['preview_valid_pca_pixels']:,} valid preview spectra; "
            f"{len(candidates)} transverse candidate bands"
        )

    inventory = pd.DataFrame(inventory_rows)
    candidate_frame = pd.DataFrame(candidate_rows)
    layer_frame = pd.DataFrame(layer_rows)
    inventory_path = report_root / "standardized_planter_turn_inventory.csv"
    candidate_path = report_root / "standardized_planter_turn_candidates.csv"
    inventory.to_csv(inventory_path, index=False)
    candidate_frame.to_csv(candidate_path, index=False)
    layer_manifest_path = report_root / "annotation_layer_manifest.csv"
    layer_frame.to_csv(layer_manifest_path, index=False)

    columns = 4
    rows = math.ceil(len(overview_tiles) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (cube_id, tile, count, role) in zip(axes, overview_tiles):
        axis.imshow(tile)
        axis.set_title(f"{cube_id} | {count} bands | {role}")
    figure.suptitle(
        "Field 1 standardized reflectance-PCA transverse-band audit\n"
        "yellow circles mark review candidates; no mask has been changed",
        fontsize=16,
    )
    overview_path = report_root / "standardized_planter_turn_overview.png"
    figure.savefig(overview_path, dpi=180, facecolor="white")
    plt.close(figure)

    contract = {
        "status": "investigator_review_only",
        "field": "Field 1",
        "input": "original_observed_150_band_reflectance",
        "standardized_processing": "per_cube_standardized_pca_plus_spatial_derivatives",
        "review_ready_cubes": len(inventory),
        "candidate_bands": len(candidate_frame),
        "annotation_layers": len(layer_frame),
        "phenology_sensitivity_only": sorted(excluded),
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "input_headers": input_hashes,
    }
    contract_path = contracts / "field1_standardized_planter_turn_audit_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"Review-ready cubes: {len(inventory)}")
    print(f"Transverse candidate bands: {len(candidate_frame)}")
    print(f"Reports: {report_root}")
    print(f"Visual QC: {overview_path}")
    print(f"Annotation layers: {layer_manifest_path}")
    print(f"Contract: {contract_path}")
    print("Audit only: no mask changed, no model retrained, and Field 2 remained locked.")


if __name__ == "__main__":
    main()
