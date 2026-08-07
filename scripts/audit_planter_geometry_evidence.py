#!/usr/bin/env python
"""Audit PCA-derived evidence for six-row planter passes and headland turns.

This is a read-only, investigator-review stage. PCA products provide geometric
evidence only; this script never changes a class mask, estimates model metrics,
or accesses Field 2.
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
        "OpenCV is required. Install it with: conda install -c conda-forge opencv"
    ) from error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from spectral.io import envi
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records


DERIVED_ROLES = {
    "pca": "pca",
    "deriv1": "legacy_pca_first_difference",
    "deriv2": "legacy_pca_second_difference",
}


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


def binary_path(header: Path) -> Path:
    if header.suffix.lower() != ".hdr":
        raise ValueError(f"Expected ENVI .hdr path, found: {header}")
    return Path(str(header)[:-4])


def choose_derived_product(
    catalog: pd.DataFrame, cube_id: str, role: str, root: Path
) -> tuple[Path | None, str]:
    rows = catalog[
        (catalog["cube_id_inferred"] == cube_id)
        & (catalog["source"] == "field1.derived_ssl")
        & (catalog["file_role"] == role)
        & (catalog["extension"] == ".hdr")
    ]
    valid: list[Path] = []
    for relative in rows["relative_path"].drop_duplicates():
        header = root / relative
        if header.is_file() and binary_path(header).is_file():
            valid.append(header)
    if len(valid) == 1:
        return valid[0], "unique"
    if not valid:
        return None, "missing"
    return None, f"ambiguous_{len(valid)}"


def load_product(header: Path, step: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    image = envi.open(str(header), str(binary_path(header)))
    memory = image.open_memmap()
    if memory.ndim != 3:
        raise ValueError(f"Expected a three-dimensional ENVI cube: {header}")
    sampled = np.asarray(memory[::step, ::step, :], dtype=np.float32)
    return sampled, tuple(int(value) for value in memory.shape)


def robust_scale(channel: np.ndarray, signed: bool = False) -> np.ndarray:
    result = np.zeros(channel.shape, dtype=np.float32)
    valid = np.isfinite(channel) & (channel != 0)
    values = channel[valid]
    if not len(values):
        return result
    if signed:
        limit = float(np.percentile(np.abs(values), 98))
        result[valid] = np.clip(channel[valid] / max(limit, 1e-6), -1, 1) * 0.5 + 0.5
    else:
        low, high = np.percentile(values, (2, 98))
        result[valid] = np.clip((channel[valid] - low) / max(float(high - low), 1e-6), 0, 1)
    return result


def pca_rgb(array: np.ndarray) -> np.ndarray:
    indices = list(range(min(3, array.shape[2])))
    while len(indices) < 3:
        indices.append(indices[-1])
    return np.stack([robust_scale(array[..., index], signed=True) for index in indices], axis=-1)


def derivative_magnitude(array: np.ndarray) -> np.ndarray:
    layers = []
    for index in range(array.shape[2]):
        channel = array[..., index]
        valid = np.isfinite(channel) & (channel != 0)
        values = channel[valid]
        if not len(values):
            continue
        median = float(np.median(values))
        scale = float(np.percentile(np.abs(values - median), 75))
        layers.append(np.clip(np.abs(channel - median) / max(scale, 1e-6), 0, 6))
    if not layers:
        return np.zeros(array.shape[:2], dtype=np.float32)
    magnitude = np.mean(layers, axis=0)
    return robust_scale(magnitude)


def structure_segments(
    structure: np.ndarray,
    valid: np.ndarray,
    row_angle: float,
    reduced_gsd: float,
    settings: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    image = np.clip(structure * 255, 0, 255).astype(np.uint8)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    low = int(settings.get("canny_low", 60))
    high = int(settings.get("canny_high", 150))
    edges = cv2.Canny(image, low, high)
    edges[~valid] = 0
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(settings.get("hough_threshold", 35)),
        minLineLength=max(5, round(float(settings.get("minimum_segment_length_m", 0.50)) / reduced_gsd)),
        maxLineGap=max(1, round(float(settings.get("maximum_segment_gap_m", 0.20)) / reduced_gsd)),
    )
    parallel = np.zeros_like(edges)
    nonparallel = np.zeros_like(edges)
    rows: list[dict[str, float]] = []
    if lines is None or not np.isfinite(row_angle):
        return parallel.astype(bool), nonparallel.astype(bool), rows
    tolerance = float(settings.get("parallel_tolerance_degrees", 12.0))
    for index, segment in enumerate(np.asarray(lines).reshape(-1, 4), start=1):
        x1, y1, x2, y2 = [int(value) for value in segment]
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
        difference = float(circular_distance_degrees(np.asarray([angle]), row_angle)[0])
        length_m = float(math.hypot(x2 - x1, y2 - y1) * reduced_gsd)
        relation = "row_parallel" if difference <= tolerance else "nonparallel"
        target = parallel if relation == "row_parallel" else nonparallel
        cv2.line(target, (x1, y1), (x2, y2), color=255, thickness=2)
        rows.append({
            "segment_id": index,
            "x1_reduced": x1, "y1_reduced": y1,
            "x2_reduced": x2, "y2_reduced": y2,
            "angle_degrees": angle,
            "difference_from_row_degrees": difference,
            "length_m": length_m,
            "relation": relation,
        })
    return parallel > 0, nonparallel > 0, rows


def grouped_candidates(mask: np.ndarray, reduced_gsd: float, settings: dict) -> tuple[np.ndarray, list[dict[str, float]]]:
    radius = max(1, round(float(settings.get("candidate_grouping_radius_m", 0.20)) / reduced_gsd))
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    grouped = cv2.dilate(mask.astype(np.uint8), kernel)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    minimum_area = max(4, round(float(settings.get("minimum_candidate_area_m2", 0.05)) / (reduced_gsd ** 2)))
    candidates = []
    accepted = np.zeros_like(labels, dtype=np.int32)
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        components.append((area, label, centers[label]))
    for candidate_id, (area, label, center) in enumerate(sorted(components, reverse=True), start=1):
        accepted[labels == label] = candidate_id
        candidates.append({
            "candidate_id": candidate_id,
            "center_column_reduced": float(center[0]),
            "center_row_reduced": float(center[1]),
            "grouped_area_m2": float(area * reduced_gsd ** 2),
        })
    return accepted, candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/chickpea_mask_refinement.yaml"), type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    review = config["review"]
    reconstruction = config["planter_pass_reconstruction"]
    if not review["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    if not reconstruction["geometry_evidence_is_not_a_class_label"]:
        raise ValueError("Geometry evidence must never be treated as a class label")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    report_root = local / "reports" / "planter_geometry_evidence"
    details = report_root / "cube_previews"
    contracts = local / "contracts"
    details.mkdir(parents=True, exist_ok=True)
    contracts.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(local / "project_catalog.csv", dtype=str).fillna("")
    records = sorted(
        [record for record in load_records(args.paths, local / "authoritative_manifest.csv") if record.chickpea_mask],
        key=lambda record: record.cube_id,
    )
    derived_root = Path(paths["field1"]["derived_ssl"])
    evidence_settings = reconstruction.get("evidence_audit", {})
    max_preview = int(evidence_settings.get("maximum_preview_dimension_pixels", 900))

    inventory_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    overview_tiles: list[tuple[str, np.ndarray, int]] = []
    hashed_inputs: list[dict[str, str]] = []

    for number, record in enumerate(records, start=1):
        products = {}
        statuses = {}
        for name, role in DERIVED_ROLES.items():
            products[name], statuses[name] = choose_derived_product(catalog, record.cube_id, role, derived_root)
        if any(path is None for path in products.values()):
            inventory_rows.append({"cube_id": record.cube_id, **{f"{k}_status": v for k, v in statuses.items()}, "audit_status": "skipped"})
            print(f"Skipped {record.cube_id}: {statuses}")
            continue

        labels = authoritative_class_map(record)
        chickpea = labels == 1
        with rasterio.open(record.data) as reference:
            transform = reference.transform
            gsd = metres_per_pixel(transform)
            height, width = reference.height, reference.width
            crs = str(reference.crs or "")
        if chickpea.shape != (height, width):
            raise ValueError(f"Mask/reference shape mismatch: {record.cube_id}")
        step = max(1, math.ceil(max(height, width) / max_preview))
        arrays = {}
        shapes = {}
        for name, header in products.items():
            arrays[name], shapes[name] = load_product(header, step)
        if any(shape[:2] != (height, width) for shape in shapes.values()):
            inventory_rows.append({
                "cube_id": record.cube_id,
                **{f"{name}_status": f"shape_{shapes[name]}" for name in products},
                "audit_status": "shape_mismatch_skipped",
            })
            print(f"Skipped {record.cube_id}: derivative/reference shape mismatch")
            continue

        angle, inlier_lines = dominant_row_angle(chickpea, gsd, config["row_detection"])
        pca_view = pca_rgb(arrays["pca"])
        deriv1 = derivative_magnitude(arrays["deriv1"])
        deriv2 = derivative_magnitude(arrays["deriv2"])
        structure = robust_scale(0.5 * deriv1 + 0.5 * deriv2)
        valid = np.any(np.isfinite(arrays["pca"]) & (arrays["pca"] != 0), axis=2)
        reduced_gsd = gsd * step
        parallel, nonparallel, segments = structure_segments(
            structure, valid, angle, reduced_gsd, evidence_settings
        )
        candidate_labels, candidates = grouped_candidates(nonparallel, reduced_gsd, evidence_settings)

        overlay = pca_view.copy()
        overlay[parallel] = 0.25 * overlay[parallel] + 0.75 * np.array([0.1, 0.9, 0.2])
        overlay[nonparallel] = 0.20 * overlay[nonparallel] + 0.80 * np.array([1.0, 0.1, 0.7])
        chickpea_small = chickpea[::step, ::step]
        chickpea_overlay = pca_view.copy()
        chickpea_overlay[chickpea_small] = 0.25 * chickpea_overlay[chickpea_small] + 0.75 * np.array([0.1, 1.0, 0.1])

        figure, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        panels = [
            (pca_view, "PCA components 1–3 (local stretch)"),
            (deriv1, "PCA first-difference magnitude"),
            (deriv2, "PCA second-difference magnitude"),
            (chickpea_overlay, "Current chickpea mask — green"),
            (overlay, "Structure: row-parallel green; nonparallel magenta"),
            (structure, "Numbered nonparallel candidate groups"),
        ]
        for axis, (panel, title) in zip(axes.reshape(-1), panels):
            axis.imshow(panel, cmap=None if panel.ndim == 3 else "gray")
            axis.set_title(title)
            axis.axis("off")
        candidate_axis = axes.reshape(-1)[5]
        for candidate in candidates:
            x = candidate["center_column_reduced"]
            y = candidate["center_row_reduced"]
            candidate_axis.text(x, y, str(candidate["candidate_id"]), color="yellow", fontsize=9, weight="bold", ha="center", va="center")
        figure.suptitle(
            f"{record.cube_id} planter-geometry evidence | row angle {angle:.1f}° | "
            f"{len(candidates)} nonparallel candidate groups",
            fontsize=15,
        )
        preview_path = details / f"{record.cube_id}_planter_geometry_evidence.png"
        figure.savefig(preview_path, dpi=180, facecolor="white")
        plt.close(figure)

        for row in segments:
            row.update({"cube_id": record.cube_id, "dominant_row_angle_degrees": angle})
            segment_rows.append(row)
        for candidate in candidates:
            full_row = candidate["center_row_reduced"] * step
            full_column = candidate["center_column_reduced"] * step
            map_x, map_y = xy(transform, full_row, full_column, offset="center")
            candidate.update({
                "cube_id": record.cube_id,
                "map_x": float(map_x), "map_y": float(map_y), "crs": crs,
                "status": "investigator_review_required",
            })
            candidate_rows.append(candidate)
        inventory_rows.append({
            "cube_id": record.cube_id,
            **{f"{name}_header": str(path) for name, path in products.items()},
            **{f"{name}_status": statuses[name] for name in products},
            "height": height, "width": width, "preview_step": step,
            "equivalent_gsd_m": gsd, "dominant_row_angle_degrees": angle,
            "row_hough_inlier_lines": inlier_lines,
            "row_parallel_segments": sum(row["relation"] == "row_parallel" for row in segments),
            "nonparallel_segments": sum(row["relation"] == "nonparallel" for row in segments),
            "nonparallel_candidate_groups": len(candidates),
            "audit_status": "review_ready",
            "preview_path": str(preview_path),
        })
        for name, header in products.items():
            hashed_inputs.append({"cube_id": record.cube_id, "role": name, "header": str(header), "header_sha256": sha256(header)})
        overview_step = max(1, math.ceil(max(overlay.shape[:2]) / 320))
        overview_tiles.append((record.cube_id, overlay[::overview_step, ::overview_step], len(candidates)))
        print(f"Prepared {number}/{len(records)} {record.cube_id}: {len(candidates)} candidate groups")

    inventory = pd.DataFrame(inventory_rows)
    segments_frame = pd.DataFrame(segment_rows)
    candidates_frame = pd.DataFrame(candidate_rows)
    inventory.to_csv(report_root / "planter_geometry_inventory.csv", index=False)
    segments_frame.to_csv(report_root / "planter_geometry_segments.csv", index=False)
    candidates_frame.to_csv(report_root / "planter_turn_candidates.csv", index=False)

    if overview_tiles:
        columns = 4
        rows = math.ceil(len(overview_tiles) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)
        for axis in axes:
            axis.axis("off")
        for axis, (cube_id, tile, count) in zip(axes, overview_tiles):
            axis.imshow(tile)
            axis.set_title(f"{cube_id} | {count} candidates")
        figure.suptitle(
            "Field 1 PCA-derived planter geometry audit\n"
            "green = row-parallel structure; magenta = nonparallel turn/wheel candidates",
            fontsize=16,
        )
        overview_path = report_root / "planter_geometry_evidence_overview.png"
        figure.savefig(overview_path, dpi=180, facecolor="white")
        plt.close(figure)
    else:
        overview_path = report_root / "planter_geometry_evidence_overview.png"

    contract = {
        "status": "investigator_review_only",
        "field": "Field 1",
        "planter_rows_per_pass": int(reconstruction["planter_rows_per_pass"]),
        "row_spacing_m": float(reconstruction["row_spacing_m"]),
        "review_ready_cubes": int((inventory.get("audit_status", pd.Series(dtype=str)) == "review_ready").sum()),
        "candidate_groups": len(candidates_frame),
        "geometry_evidence_is_not_a_class_label": True,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "input_headers": hashed_inputs,
    }
    contract_path = contracts / "field1_planter_geometry_evidence_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Review-ready cubes: {contract['review_ready_cubes']}")
    print(f"Numbered nonparallel candidate groups: {len(candidates_frame)}")
    print(f"Reports: {report_root}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print("Audit only: no mask changed, no model retrained, and Field 2 remained locked.")


if __name__ == "__main__":
    main()
