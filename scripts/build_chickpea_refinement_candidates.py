#!/usr/bin/env python
"""Build reversible plot- and row-constrained chickpea-mask candidates.

This is an investigator-review stage. It never edits source or authoritative
masks and never chooses a refinement threshold from model performance.
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

try:
    import cv2
except ModuleNotFoundError as error:
    raise SystemExit(
        "OpenCV is required for the declared Hough-row refinement. Install it in "
        "the active environment with: conda install -c conda-forge opencv"
    ) from error
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records
from chickpea_ssl.shapefile import read_polygon_geometries


MASK_CMAP = ListedColormap(["#000000", "#22C55E", "#F59E0B", "#EF4444"])


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


def ellipse_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def circular_distance_degrees(values: np.ndarray, center: float) -> np.ndarray:
    return np.abs((values - center + 90.0) % 180.0 - 90.0)


def detect_row_lines(mask: np.ndarray, gsd: float, settings: dict) -> tuple[np.ndarray, float, int]:
    image = mask.astype(np.uint8) * 255
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(settings["hough_threshold"]),
        minLineLength=max(5, round(float(settings["minimum_line_length_m"]) / gsd)),
        maxLineGap=max(1, round(float(settings["maximum_line_gap_m"]) / gsd)),
    )
    if lines is None:
        raise ValueError("Hough transform found no candidate row lines")
    # OpenCV packages HoughLinesP output as either (N, 1, 4) or (N, 4),
    # depending on the build. Normalize both representations explicitly.
    line_array = np.asarray(lines)
    if line_array.size == 0 or line_array.size % 4:
        raise ValueError(f"Unexpected HoughLinesP output shape: {line_array.shape}")
    segments = line_array.reshape(-1, 4).astype(float)
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    lengths = np.hypot(dx, dy)
    angles = np.degrees(np.arctan2(dy, dx)) % 180.0
    bin_width = float(settings["orientation_bin_degrees"])
    edges = np.arange(0.0, 180.0 + bin_width, bin_width)
    histogram, _ = np.histogram(angles, bins=edges, weights=lengths)
    dominant = float((edges[int(histogram.argmax())] + bin_width / 2.0) % 180.0)
    tolerance = float(settings["orientation_tolerance_degrees"])
    inliers = circular_distance_degrees(angles, dominant) <= tolerance
    if int(inliers.sum()) < int(settings["minimum_inlier_lines"]):
        raise ValueError(
            f"Only {int(inliers.sum())} Hough lines support dominant orientation {dominant:.1f} degrees"
        )

    # Candidate row support follows the dominant inlier lines. Extending them
    # across the raster reconstructs complete straight rows from fragmented
    # plant detections; plot/alleys remain a separate review overlay.
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    if settings.get("extend_inlier_lines_across_raster", False):
        extension = 2 * max(mask.shape)
    else:
        extension = max(1, round(float(settings["maximum_line_gap_m"]) / gsd))
    for segment, length in zip(segments[inliers], lengths[inliers]):
        x1, y1, x2, y2 = segment
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        p1 = (round(x1 - ux * extension), round(y1 - uy * extension))
        p2 = (round(x2 + ux * extension), round(y2 + uy * extension))
        cv2.line(canvas, p1, p2, color=1, thickness=1)
    return canvas.astype(bool), dominant, int(inliers.sum())


def preferred_plot_geometries(project: Path, local: Path) -> list[dict[str, object]]:
    archive = project / "data" / "OneDrive_2026-07-31_raw"
    exact = pd.read_csv(local / "plot_exact_geometry_qc.csv")
    exact = exact[exact["eligible_as_chickpea_spatial_prior"].astype(bool)]
    if exact.empty:
        raise ValueError("No verified Field 1 plot polygons are eligible as a chickpea prior")
    geometries: list[dict[str, object]] = []
    for relative_path in exact["relative_path"].drop_duplicates():
        geometries.extend(read_polygon_geometries(archive / relative_path))
    return geometries


def plot_support(
    geometries: list[dict[str, object]], shape: tuple[int, int], transform, buffer_m: float, gsd: float
) -> np.ndarray:
    support = rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    radius = int(math.ceil(buffer_m / gsd))
    if radius:
        support = cv2.dilate(support, ellipse_kernel(radius))
    return support.astype(bool)


def write_mask(path: Path, mask: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(path, "w", **output_profile) as dataset:
        dataset.write(mask.astype(np.uint8), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/chickpea_mask_refinement.yaml"), type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked during Field 1 mask refinement")
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    manifest = local / "authoritative_manifest.csv"
    reports = local / "reports" / "mask_refinement"
    comparisons = reports / "candidate_comparisons"
    candidates_root = project / "data" / "processed" / "chickpea_mask_refinement_candidates"
    comparisons.mkdir(parents=True, exist_ok=True)
    plot_geometries = preferred_plot_geometries(project, local)
    records = sorted(
        [record for record in load_records(args.paths, manifest) if record.chickpea_mask],
        key=lambda record: record.cube_id,
    )

    summary_rows: list[dict[str, object]] = []
    review_tiles: list[tuple[str, np.ndarray, float]] = []
    output_paths: list[Path] = []
    plot_buffers = [float(value) for value in config["plot_buffer_candidates_m"]]
    corridor_widths = [float(value) for value in config["row_corridor_half_width_candidates_m"]]
    review_corridor = max(corridor_widths)

    for number, record in enumerate(records, start=1):
        labels = authoritative_class_map(record)
        original = labels == 1
        # GDAL/Rasterio opens an ENVI dataset through its binary payload; the
        # .hdr file is metadata-only and cannot be selected as the dataset.
        with rasterio.open(record.data) as reference:
            transform = reference.transform
            profile = reference.profile
            shape = (reference.height, reference.width)
        if original.shape != shape:
            raise ValueError(f"Mask/reference shape mismatch for {record.cube_id}")
        gsd = metres_per_pixel(transform)
        line_seed, angle, inlier_lines = detect_row_lines(original, gsd, config["row_detection"])

        plot_masks = {
            buffer_m: plot_support(plot_geometries, shape, transform, buffer_m, gsd)
            for buffer_m in plot_buffers
        }
        exact_plot = plot_masks[min(plot_buffers)]
        plot_overlap_pixels = int((original & exact_plot).sum())
        plot_overlap_fraction = plot_overlap_pixels / int(original.sum())
        plot_status = "overlap_review_only" if plot_overlap_pixels else "no_plot_overlap"
        row_candidates: dict[float, np.ndarray] = {}
        combined_candidates: dict[tuple[float, float], np.ndarray] = {}
        for corridor_m in corridor_widths:
            radius = max(1, int(math.ceil(corridor_m / gsd)))
            row_corridor = cv2.dilate(line_seed.astype(np.uint8), ellipse_kernel(radius)).astype(bool)
            row_candidate = original & row_corridor
            row_candidates[corridor_m] = row_candidate
            row_variant = f"row_only_halfwidth_{corridor_m:.2f}m"
            row_path = candidates_root / row_variant / f"{record.cube_id}_chickpea_candidate.tif"
            if config["review"]["write_candidate_geotiffs"]:
                write_mask(row_path, row_candidate, profile)
            summary_rows.append({
                "cube_id": record.cube_id,
                "candidate_type": "row_only",
                "plot_status": plot_status,
                "plot_buffer_m": np.nan,
                "row_corridor_half_width_m": corridor_m,
                "source_chickpea_pixels": int(original.sum()),
                "candidate_chickpea_pixels": int(row_candidate.sum()),
                "retained_fraction": float(row_candidate.sum() / original.sum()),
                "source_pixels_intersecting_exact_plots": plot_overlap_pixels,
                "source_fraction_intersecting_exact_plots": plot_overlap_fraction,
                "removed_outside_plot_pixels": np.nan,
                "removed_off_row_pixels": int((original & ~row_corridor).sum()),
                "dominant_image_row_angle_degrees": angle,
                "hough_inlier_lines": inlier_lines,
                "equivalent_gsd_m": gsd,
                "candidate_path": str(row_path),
            })
            for buffer_m in plot_buffers:
                candidate = row_candidate & plot_masks[buffer_m]
                combined_candidates[(buffer_m, corridor_m)] = candidate
                variant = f"plot_buffer_{buffer_m:.2f}m_row_halfwidth_{corridor_m:.2f}m"
                candidate_path = candidates_root / variant / f"{record.cube_id}_chickpea_candidate.tif"
                if config["review"]["write_candidate_geotiffs"]:
                    write_mask(candidate_path, candidate, profile)
                summary_rows.append({
                    "cube_id": record.cube_id,
                    "candidate_type": "plot_and_row_review_only",
                    "plot_status": plot_status,
                    "plot_buffer_m": buffer_m,
                    "row_corridor_half_width_m": corridor_m,
                    "source_chickpea_pixels": int(original.sum()),
                    "candidate_chickpea_pixels": int(candidate.sum()),
                    "retained_fraction": float(candidate.sum() / original.sum()) if original.any() else 0.0,
                    "removed_outside_plot_pixels": int((original & ~plot_masks[buffer_m]).sum()),
                    "source_pixels_intersecting_exact_plots": plot_overlap_pixels,
                    "source_fraction_intersecting_exact_plots": plot_overlap_fraction,
                    "removed_off_row_pixels": int((original & ~row_corridor).sum()),
                    "dominant_image_row_angle_degrees": angle,
                    "hough_inlier_lines": inlier_lines,
                    "equivalent_gsd_m": gsd,
                    "candidate_path": str(candidate_path),
                })

        review_candidate = row_candidates[review_corridor]
        categories = np.zeros(shape, dtype=np.uint8)
        categories[review_candidate] = 1
        categories[original & ~review_candidate] = 3
        review_tiles.append((record.cube_id, categories, float(review_candidate.sum() / original.sum())))

        figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
        axes = axes.reshape(-1)
        axes[0].imshow(original, cmap=ListedColormap(["black", "#22C55E"]), vmin=0, vmax=1)
        axes[0].set_title(f"Current authoritative\n{int(original.sum()):,} pixels")
        axes[1].imshow(original & exact_plot, cmap=ListedColormap(["black", "#F59E0B"]), vmin=0, vmax=1)
        axes[1].set_title(
            f"Exact plot intersection — review only\n"
            f"{plot_overlap_pixels:,} ({plot_overlap_fraction:.1%}); {plot_status}"
        )
        for axis, corridor_m in zip(axes[2:5], corridor_widths):
            candidate = row_candidates[corridor_m]
            axis.imshow(candidate, cmap=ListedColormap(["black", "#22C55E"]), vmin=0, vmax=1)
            axis.set_title(
                f"Row-only {corridor_m:.2f} m half-width\n"
                f"{int(candidate.sum()):,} ({candidate.sum() / original.sum():.1%})"
            )
        combined = combined_candidates[(max(plot_buffers), review_corridor)]
        axes[5].imshow(combined, cmap=ListedColormap(["black", "#22C55E"]), vmin=0, vmax=1)
        axes[5].set_title(
            f"Plot + row intersection — review only\n"
            f"{int(combined.sum()):,} ({combined.sum() / original.sum():.1%})"
        )
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{record.cube_id} chickpea refinement candidates | row angle {angle:.1f}° | "
            f"{inlier_lines} Hough lines",
            fontsize=15,
        )
        comparison_path = comparisons / f"{record.cube_id}_refinement_candidates.png"
        figure.savefig(comparison_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(comparison_path)
        print(
            f"Prepared {number}/{len(records)} {record.cube_id}: angle={angle:.1f}°, "
            f"row retention={review_candidate.sum() / original.sum():.1%}, "
            f"plot overlap={plot_overlap_fraction:.1%}",
            flush=True,
        )

    columns = 4
    rows = math.ceil(len(review_tiles) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.8 * rows), constrained_layout=True)
    flat_axes = np.asarray(axes).reshape(-1)
    for axis in flat_axes:
        axis.axis("off")
    for axis, (cube_id, categories, retention) in zip(flat_axes, review_tiles):
        step = max(1, math.ceil(max(categories.shape) / 700))
        axis.imshow(categories[::step, ::step], cmap=MASK_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axis.set_title(f"{cube_id} | retained {retention:.1%}")
    figure.suptitle(
        f"Field 1 row-only chickpea refinement review — {review_corridor:.2f} m row half-width\n"
        "green = retained chickpea; red = current chickpea outside detected row corridors",
        fontsize=16,
    )
    overview_path = reports / "chickpea_refinement_candidate_overview.png"
    figure.savefig(overview_path, dpi=220, facecolor="white")
    plt.close(figure)
    output_paths.append(overview_path)

    summary = pd.DataFrame(summary_rows)
    summary_path = reports / "chickpea_refinement_sensitivity.csv"
    summary.to_csv(summary_path, index=False)
    output_paths.append(summary_path)
    contract = {
        "status": "investigator_review_candidates_only",
        "field": "Field 1",
        "field2_accessed": False,
        "source_masks_modified": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "candidate_reassignment_if_accepted": "removed chickpea -> weed",
        "plot_constraint_mode": "review_only_due_to_incomplete_cube_coverage",
        "configuration_sha256": sha256(args.config),
        "authoritative_manifest_sha256": sha256(manifest),
        "plot_geometry_qc_sha256": sha256(local / "plot_exact_geometry_qc.csv"),
        "outputs": {str(path.relative_to(reports)): sha256(path) for path in output_paths},
        "review_rule": (
            "No candidate becomes authoritative until visual review. Selection cannot use supervised "
            "or SSL performance on held-out labels."
        ),
    }
    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract_path = contracts / "field1_chickpea_refinement_candidate_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    variants_per_cube = len(corridor_widths) * (1 + len(plot_buffers))
    print(f"\nCandidate variants per cube: {variants_per_cube}")
    print(f"Sensitivity table: {summary_path}")
    print(f"Individual visual QC: {comparisons}")
    print(f"Overview: {overview_path}")
    print(f"Candidate GeoTIFF root: {candidates_root}")
    print(f"Contract: {contract_path}")
    print("Review only: no authoritative mask changed, no model retrained, and Field 2 remained locked.")


if __name__ == "__main__":
    main()
