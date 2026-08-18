"""Prediction-free Field 2 area annotations and deterministic zone geometry."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import tempfile
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
import yaml

from chickpea_ssl.field2_blind_review import atomic_write_bytes, atomic_write_json, verify_preview_hashes
from chickpea_ssl.field2_readiness import sha256


COVERAGE_MODES = (
    "entire_support_research_field",
    "entire_support_outside_research_field",
    "mixed_manual_boundaries",
    "uncertain_requires_review",
)

ZONE_TYPES = (
    "research_crop_area",
    "alley",
    "outside_research_field",
    "uncertain_boundary",
)

ZONE_DEFINITIONS = {
    "research_crop_area": "investigator-identified planted-row/research-plot domain; not a biological chickpea label.",
    "alley": "investigator-confirmed plot or tractor alley; only soil and weeds occur by investigator rule.",
    "outside_research_field": "valid imagery outside the intended experiment; it may contain soil or varied vegetation.",
    "uncertain_boundary": "geometry that cannot be placed confidently.",
}

ZONE_CODES = {
    "unassigned_valid_support": 0,
    "research_crop_area": 1,
    "alley": 2,
    "outside_research_field": 3,
    "uncertain_boundary": 4,
}

CONFIDENCE_VALUES = ("high", "medium", "low")

ALLEY_POINT_LABELS = (
    "soil", "ordinary_weed", "tall_grass_weed", "weed_soil_mixed",
    "uncertain", "nodata_invalid",
)

THREE_CLASS_MAPPING = {
    "chickpea": "chickpea",
    "ordinary_weed": "weed",
    "tall_grass_weed": "weed",
    "soil": "soil",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def affine_from_json(value: str | list[float]) -> Affine:
    values = json.loads(value) if isinstance(value, str) else value
    if len(values) < 6:
        raise ValueError("Raster transform must contain at least six coefficients")
    return Affine(*[float(item) for item in values[:6]])


def normalize_vertices(vertices: list[dict | list | tuple]) -> list[tuple[float, float]]:
    result = []
    for vertex in vertices:
        if isinstance(vertex, dict):
            x, y = float(vertex["x"]), float(vertex["y"])
        else:
            x, y = float(vertex[0]), float(vertex[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Polygon vertices must be finite")
        result.append((x, y))
    return result


def distinct_vertex_count(vertices: list[dict | list | tuple], epsilon: float = 1e-9) -> int:
    """Count distinct submitted points without changing or deduplicating the source list."""
    distinct: list[tuple[float, float]] = []
    for point in normalize_vertices(vertices):
        if not any(abs(point[0] - seen[0]) <= epsilon and abs(point[1] - seen[1]) <= epsilon for seen in distinct):
            distinct.append(point)
    return len(distinct)


def polygon_area(vertices: list[tuple[float, float]]) -> float:
    if len(vertices) < 3:
        return 0.0
    return abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1])
    )) / 2.0


def _orientation(a, b, c, epsilon=1e-9) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    return 0 if abs(value) <= epsilon else (1 if value > 0 else 2)


def _on_segment(a, b, c, epsilon=1e-9) -> bool:
    return (
        min(a[0], c[0]) - epsilon <= b[0] <= max(a[0], c[0]) + epsilon
        and min(a[1], c[1]) - epsilon <= b[1] <= max(a[1], c[1]) + epsilon
    )


def segments_intersect(first_a, first_b, second_a, second_b) -> bool:
    orientations = (
        _orientation(first_a, first_b, second_a),
        _orientation(first_a, first_b, second_b),
        _orientation(second_a, second_b, first_a),
        _orientation(second_a, second_b, first_b),
    )
    if orientations[0] != orientations[1] and orientations[2] != orientations[3]:
        return True
    return any((
        orientations[0] == 0 and _on_segment(first_a, second_a, first_b),
        orientations[1] == 0 and _on_segment(first_a, second_b, first_b),
        orientations[2] == 0 and _on_segment(second_a, first_a, second_b),
        orientations[3] == 0 and _on_segment(second_a, first_b, second_b),
    ))


def polygon_self_intersects(vertices: list[dict | list | tuple]) -> bool:
    points = normalize_vertices(vertices)
    if len(points) < 4:
        return False
    edges = list(zip(points, points[1:] + points[:1]))
    for first_index, first in enumerate(edges):
        for second_index in range(first_index + 1, len(edges)):
            if second_index in {first_index, first_index + 1}:
                continue
            if first_index == 0 and second_index == len(edges) - 1:
                continue
            if segments_intersect(first[0], first[1], edges[second_index][0], edges[second_index][1]):
                return True
    return False


def clip_polygon_to_bounds(
    vertices: list[dict | list | tuple], width: int, height: int,
) -> list[tuple[float, float]]:
    points = normalize_vertices(vertices)

    def clip(points_in, inside, intersect):
        if not points_in:
            return []
        output = []
        previous = points_in[-1]
        for current in points_in:
            current_inside, previous_inside = inside(current), inside(previous)
            if current_inside:
                if not previous_inside:
                    output.append(intersect(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous = current
        return output

    def vertical(x_value):
        return lambda a, b: (x_value, a[1] + (b[1] - a[1]) * (x_value - a[0]) / (b[0] - a[0]))

    def horizontal(y_value):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y_value - a[1]) / (b[1] - a[1]), y_value)

    points = clip(points, lambda p: p[0] >= 0, vertical(0))
    points = clip(points, lambda p: p[0] <= width, vertical(width))
    points = clip(points, lambda p: p[1] >= 0, horizontal(0))
    return clip(points, lambda p: p[1] <= height, horizontal(height))


def polygon_stage_counts(vertices, width: int, height: int) -> dict[str, int]:
    """Report validation stages while leaving every submitted vertex untouched."""
    submitted = normalize_vertices(vertices)
    clipped = clip_polygon_to_bounds(submitted, width, height) if len(submitted) >= 3 else list(submitted)
    return {
        "displayed_vertex_count": len(submitted),
        "submitted_vertex_count": len(submitted),
        "distinct_pixel_vertex_count": distinct_vertex_count(submitted),
        "clipped_vertex_count": len(clipped),
    }


def pixel_vertices_to_map(vertices, transform: Affine) -> list[dict[str, float]]:
    result = []
    for x, y in normalize_vertices(vertices):
        map_x, map_y = transform * (x, y)
        result.append({"x": float(map_x), "y": float(map_y)})
    return result


def map_vertices_to_pixel(vertices, transform: Affine) -> list[dict[str, float]]:
    inverse = ~transform
    result = []
    for x, y in normalize_vertices(vertices):
        pixel_x, pixel_y = inverse * (x, y)
        result.append({"x": float(pixel_x), "y": float(pixel_y)})
    return result


def polygon_raster_mask(vertices, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    points = clip_polygon_to_bounds(vertices, width, height)
    if len(points) < 3 or polygon_area(points) <= 0:
        return np.zeros(shape, dtype=bool)
    geometry = {"type": "Polygon", "coordinates": [[*points, points[0]]]}
    return rasterize(
        [(geometry, 1)], out_shape=shape, transform=Affine.identity(),
        fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)


def polygon_audit(polygon: dict, support: np.ndarray) -> dict:
    vertices = normalize_vertices(polygon.get("vertices_pixel", []))
    height, width = support.shape
    original_area = polygon_area(vertices)
    clipped = clip_polygon_to_bounds(vertices, width, height)
    clipped_area = polygon_area(clipped)
    inside_raster_fraction = clipped_area / original_area if original_area else 0.0
    raster_mask = polygon_raster_mask(vertices, support.shape)
    raster_pixels = int(raster_mask.sum())
    operational_pixels = int(np.count_nonzero(raster_mask & support))
    return {
        "polygon_id": str(polygon.get("polygon_id", "")),
        "zone_type": str(polygon.get("zone_type", "")),
        "vertex_count": len(vertices),
        "self_intersection": polygon_self_intersects(vertices),
        "original_area_pixels2": original_area,
        "inside_raster_fraction": inside_raster_fraction,
        "outside_raster_fraction": max(0.0, 1.0 - inside_raster_fraction),
        "rasterized_pixel_count": raster_pixels,
        "operational_support_pixel_count": operational_pixels,
        "outside_valid_support_fraction": (
            1.0 - operational_pixels / raster_pixels if raster_pixels else 0.0
        ),
    }


def incompatible_overlap_rows(polygons: list[dict], support: np.ndarray) -> list[dict]:
    rows = []
    masks = [polygon_raster_mask(item["vertices_pixel"], support.shape) & support for item in polygons]
    for first_index, first in enumerate(polygons):
        for second_index in range(first_index + 1, len(polygons)):
            second = polygons[second_index]
            if first["zone_type"] == second["zone_type"]:
                continue
            count = int(np.count_nonzero(masks[first_index] & masks[second_index]))
            if count:
                rows.append({
                    "polygon_id_a": first["polygon_id"], "zone_type_a": first["zone_type"],
                    "polygon_id_b": second["polygon_id"], "zone_type_b": second["zone_type"],
                    "overlap_support_pixels": count,
                })
    return rows


def compile_zone_mask(record: dict, support: np.ndarray) -> np.ndarray:
    result = np.zeros(support.shape, dtype=np.uint8)
    mode = record["coverage_mode"]
    if mode == "entire_support_research_field":
        result[support] = ZONE_CODES["research_crop_area"]
    elif mode == "entire_support_outside_research_field":
        result[support] = ZONE_CODES["outside_research_field"]
    elif mode == "uncertain_requires_review":
        result[support] = ZONE_CODES["uncertain_boundary"]
    elif mode == "mixed_manual_boundaries":
        for polygon in record["polygons"]:
            mask = polygon_raster_mask(polygon["vertices_pixel"], support.shape) & support
            result[mask] = ZONE_CODES[polygon["zone_type"]]
        if record.get("treat_unassigned_valid_support_as_outside", False):
            result[support & (result == 0)] = ZONE_CODES["outside_research_field"]
    else:
        raise ValueError(f"Unknown coverage mode: {mode}")
    result[~support] = 0
    return result


def zone_name_from_code(code: int) -> str:
    return next(name for name, value in ZONE_CODES.items() if value == int(code))


class Field2AreaAnnotationStore:
    """Atomic, resumable, prediction-free area-annotation state."""

    def __init__(
        self, project: Path, review_manifest_path: Path, rgb_manifest_path: Path,
        support_manifest_path: Path, output_root: Path,
    ):
        self.project = project.resolve()
        self.review_frame = pd.read_csv(review_manifest_path).fillna("").sort_values("cube_id")
        self.rgb_frame = pd.read_csv(rgb_manifest_path).fillna("").sort_values("cube_id")
        self.support_frame = pd.read_csv(support_manifest_path).fillna("").sort_values("cube_id")
        cube_order = self.review_frame.cube_id.astype(str).tolist()
        if len(cube_order) != 40 or cube_order != self.rgb_frame.cube_id.astype(str).tolist() or cube_order != self.support_frame.cube_id.astype(str).tolist():
            raise ValueError("Area annotator requires the exact aligned frozen 40-cube inventory")
        preview_issues = verify_preview_hashes(self.review_frame, self.project)
        if preview_issues:
            raise ValueError(f"Prediction-free preview checks failed: {preview_issues}")
        for row in self.rgb_frame.itertuples(index=False):
            if sha256(self.project / str(row.natural_rgb_path)) != str(row.natural_rgb_sha256):
                raise ValueError(f"Natural-RGB hash mismatch: {row.cube_id}")
        for row in self.support_frame.itertuples(index=False):
            if sha256(self.project / str(row.mask_path)) != str(row.mask_sha256):
                raise ValueError(f"Valid-support hash mismatch: {row.cube_id}")
        self.cube_order = cube_order
        self.review = {str(row.cube_id): row for row in self.review_frame.itertuples(index=False)}
        self.rgb = {str(row.cube_id): row for row in self.rgb_frame.itertuples(index=False)}
        self.support = {str(row.cube_id): row for row in self.support_frame.itertuples(index=False)}
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_root / "field2_area_annotations.json"
        self.geojson_path = self.output_root / "field2_area_annotations.geojson"
        self.vertex_csv_path = self.output_root / "field2_area_vertices.csv"
        self.audit_csv_path = self.output_root / "field2_area_geometry_audit.csv"
        self.overview_path = self.output_root / "field2_area_annotation_overview.png"
        self.lock = threading.Lock()

    def _record(self, cube_id: str) -> dict:
        review, rgb, support = self.review[cube_id], self.rgb[cube_id], self.support[cube_id]
        return {
            "cube_id": cube_id, "coverage_mode": "", "polygons": [],
            "treat_unassigned_valid_support_as_outside": False,
            "confidence": "", "investigator_notes": "", "reviewed": False,
            "review_timestamp": "", "reviewer_identifier": "",
            "source_preview_hashes": json.loads(str(review.preview_sha256_json)),
            "valid_support_sha256": str(support.mask_sha256),
            "rgb_display_sha256": str(rgb.natural_rgb_sha256),
            "source_transform": json.loads(str(support.transform)), "source_crs": str(support.crs),
        }

    def initial_payload(self) -> dict:
        return {
            "version": "field2_area_annotations_v1", "revision": 0,
            "cube_order": self.cube_order,
            "automatic_metadata": {
                "biological_labels_assigned_automatically": False,
                "predictions_or_probabilities_used": False,
                "geometry_inferred_automatically": False,
                "reserve_frame_exposed": False,
                "area_geometry_frozen": False,
            },
            "annotations": {cube_id: self._record(cube_id) for cube_id in self.cube_order},
        }

    def load(self) -> dict:
        payload = json.loads(self.json_path.read_text()) if self.json_path.exists() else self.initial_payload()
        self.validate(payload)
        return payload

    def public_manifest(self) -> list[dict]:
        result = []
        for cube_id in self.cube_order:
            review, rgb = self.review[cube_id], self.rgb[cube_id]
            result.append({
                "cube_id": cube_id, "width": int(rgb.width), "height": int(rgb.height),
                "preview_step": int(review.preview_step), "natural_rgb_preview_step": 1,
                "crs": str(review.crs), "transform": json.loads(str(review.transform)),
            })
        return result

    def layer_path(self, cube_id: str, layer: str) -> Path:
        if cube_id not in self.review:
            raise KeyError(cube_id)
        if layer == "natural_rgb":
            return self.project / str(self.rgb[cube_id].natural_rgb_path)
        mapping = {
            "false_colour": "false_colour_path", "pca": "pca_path",
            "stored_index": "stored_index_path", "valid_support": "valid_support_path",
            "support_outline": "support_outline_path",
        }
        if layer not in mapping:
            raise KeyError(layer)
        return self.project / str(getattr(self.review[cube_id], mapping[layer]))

    def support_mask(self, cube_id: str) -> np.ndarray:
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(self.project / str(self.support[cube_id].mask_path), "r") as dataset:
                return dataset.read(1) == 1

    def normalize_payload(self, payload: dict) -> dict:
        result = json.loads(json.dumps(payload))
        for cube_id in self.cube_order:
            record = result["annotations"][cube_id]
            transform = affine_from_json(record.get("source_transform", self.support[cube_id].transform))
            for polygon in record.get("polygons", []):
                vertices = normalize_vertices(polygon.get("vertices_pixel", []))
                # Existing source vertices, ordering, and geospatial values are immutable.
                # New or edited polygons deliberately omit map vertices so only those values
                # are derived from the original submitted pixel coordinates.
                if "vertices_geospatial" not in polygon:
                    polygon["vertices_geospatial"] = pixel_vertices_to_map(vertices, transform)
        return result

    def validate(self, payload: dict) -> dict[str, list[str]]:
        if payload.get("version") != "field2_area_annotations_v1":
            raise ValueError("Unknown Field 2 area-annotation schema")
        if payload.get("cube_order") != self.cube_order:
            raise ValueError("Area annotations differ from the frozen 40-cube inventory")
        annotations = payload.get("annotations")
        if not isinstance(annotations, dict) or list(annotations) != self.cube_order:
            raise ValueError("Area annotation inventory is missing, reordered, or duplicated")
        issues = {}
        for cube_id in self.cube_order:
            record = annotations[cube_id]
            item_issues = []
            expected = self._record(cube_id)
            if record.get("cube_id") != cube_id:
                item_issues.append("cube_id_mismatch")
            for key in ("source_preview_hashes", "valid_support_sha256", "rgb_display_sha256", "source_transform", "source_crs"):
                if record.get(key) != expected[key]:
                    item_issues.append(f"{key}_mismatch")
            mode = record.get("coverage_mode", "")
            if mode and mode not in COVERAGE_MODES:
                item_issues.append("unknown_coverage_mode")
            confidence = record.get("confidence", "")
            if confidence and confidence not in CONFIDENCE_VALUES:
                item_issues.append("unknown_confidence")
            if not isinstance(record.get("reviewed"), bool):
                item_issues.append("reviewed_must_be_boolean")
            if not isinstance(record.get("treat_unassigned_valid_support_as_outside"), bool):
                item_issues.append("unassigned_outside_action_must_be_boolean")
            if record.get("treat_unassigned_valid_support_as_outside") and mode != "mixed_manual_boundaries":
                item_issues.append("unassigned_outside_action_requires_mixed_mode")
            polygons = record.get("polygons", [])
            if not isinstance(polygons, list):
                raise ValueError(f"Polygons must be a list: {cube_id}")
            if mode and mode != "mixed_manual_boundaries" and polygons:
                item_issues.append("polygons_require_mixed_manual_boundaries")
            seen_ids = set()
            width, height = int(self.support[cube_id].width), int(self.support[cube_id].height)
            transform = affine_from_json(expected["source_transform"])
            extension = 0.1 * max(width, height)
            for polygon in polygons:
                polygon_id = str(polygon.get("polygon_id", ""))
                if not polygon_id or polygon_id in seen_ids:
                    item_issues.append("polygon_id_missing_or_duplicate")
                seen_ids.add(polygon_id)
                if polygon.get("zone_type") not in ZONE_TYPES:
                    item_issues.append("unknown_zone_type")
                try:
                    vertices = normalize_vertices(polygon.get("vertices_pixel", []))
                except (KeyError, TypeError, ValueError):
                    item_issues.append("invalid_polygon_vertices")
                    continue
                counts = polygon_stage_counts(vertices, width, height)
                if counts["distinct_pixel_vertex_count"] < 3 or polygon_area(vertices) <= 0:
                    detail = ",".join(f"{key}={value}" for key, value in counts.items())
                    item_issues.append(f"polygon_too_small:{polygon_id}:stage=submitted_geometry_validation:{detail}")
                if any(x < -extension or x > width + extension or y < -extension or y > height + extension for x, y in vertices):
                    item_issues.append(f"polygon_exceeds_allowed_raster_margin:{polygon_id}")
                if record.get("reviewed") and polygon_self_intersects(vertices):
                    item_issues.append(f"self_intersection:{polygon_id}")
                expected_geo = pixel_vertices_to_map(vertices, transform)
                observed_geo = polygon.get("vertices_geospatial", [])
                if len(observed_geo) != len(expected_geo) or any(
                    abs(float(observed[axis]) - expected_vertex[axis]) > 1e-6
                    for observed, expected_vertex in zip(observed_geo, expected_geo)
                    for axis in ("x", "y")
                ):
                    item_issues.append(f"geospatial_vertices_mismatch:{polygon_id}")
            if polygons and record.get("reviewed"):
                overlaps = incompatible_overlap_rows(polygons, self.support_mask(cube_id))
                if overlaps:
                    item_issues.append("incompatible_zone_overlap")
            if record.get("reviewed"):
                if not mode:
                    item_issues.append("reviewed_requires_coverage_mode")
                if confidence not in CONFIDENCE_VALUES:
                    item_issues.append("reviewed_requires_confidence")
                if not str(record.get("review_timestamp", "")).strip():
                    item_issues.append("reviewed_requires_timestamp")
                if mode == "mixed_manual_boundaries" and not polygons:
                    item_issues.append("mixed_review_requires_polygons")
            for key in ("investigator_notes", "review_timestamp", "reviewer_identifier"):
                if not isinstance(record.get(key, ""), str):
                    item_issues.append(f"{key}_must_be_string")
            issues[cube_id] = sorted(set(item_issues))
        return issues

    def _export_bytes(self, payload: dict) -> tuple[bytes, bytes, bytes]:
        vertices, features, audits = [], [], []
        for cube_id in self.cube_order:
            record = payload["annotations"][cube_id]
            support = self.support_mask(cube_id)
            for polygon in record["polygons"]:
                pixel = normalize_vertices(polygon["vertices_pixel"])
                geo = normalize_vertices(polygon["vertices_geospatial"])
                audit = polygon_audit(polygon, support)
                audits.append({"cube_id": cube_id, "audit_type": "polygon", **audit,
                               "polygon_id_a": "", "zone_type_a": "", "polygon_id_b": "", "zone_type_b": "", "overlap_support_pixels": 0})
                for index, ((px, py), (map_x, map_y)) in enumerate(zip(pixel, geo)):
                    vertices.append({
                        "cube_id": cube_id, "polygon_id": polygon["polygon_id"],
                        "zone_type": polygon["zone_type"], "vertex_index": index,
                        "pixel_x": px, "pixel_y": py, "map_x": map_x, "map_y": map_y,
                        "crs": record["source_crs"],
                    })
                ring = [[x, y] for x, y in geo] + [[geo[0][0], geo[0][1]]]
                features.append({
                    "type": "Feature",
                    "properties": {"cube_id": cube_id, "polygon_id": polygon["polygon_id"], "zone_type": polygon["zone_type"], "crs": record["source_crs"]},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                })
            for overlap in incompatible_overlap_rows(record["polygons"], support):
                audits.append({
                    "cube_id": cube_id, "audit_type": "incompatible_overlap",
                    "polygon_id": "", "zone_type": "", "vertex_count": 0,
                    "self_intersection": False, "original_area_pixels2": 0,
                    "inside_raster_fraction": 1, "outside_raster_fraction": 0,
                    "rasterized_pixel_count": 0, "operational_support_pixel_count": 0,
                    "outside_valid_support_fraction": 0, **overlap,
                })

        def csv_data(rows, columns):
            stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows); return stream.getvalue().encode()

        vertex_columns = ["cube_id", "polygon_id", "zone_type", "vertex_index", "pixel_x", "pixel_y", "map_x", "map_y", "crs"]
        audit_columns = [
            "cube_id", "audit_type", "polygon_id", "zone_type", "vertex_count",
            "self_intersection", "original_area_pixels2", "inside_raster_fraction",
            "outside_raster_fraction", "rasterized_pixel_count", "operational_support_pixel_count",
            "outside_valid_support_fraction", "polygon_id_a", "zone_type_a", "polygon_id_b",
            "zone_type_b", "overlap_support_pixels",
        ]
        geojson = json.dumps({"type": "FeatureCollection", "features": features}, indent=2).encode()
        return csv_data(vertices, vertex_columns), csv_data(audits, audit_columns), geojson

    def _write_overview(self, payload: dict) -> None:
        colors = {"research_crop_area": "#22c55e", "alley": "#f97316", "outside_research_field": "#3b82f6", "uncertain_boundary": "#a855f7"}
        fig, axes = plt.subplots(5, 8, figsize=(20, 13), constrained_layout=True)
        for axis, cube_id in zip(axes.flat, self.cube_order):
            axis.imshow(plt.imread(self.project / str(self.rgb[cube_id].natural_rgb_path)))
            record = payload["annotations"][cube_id]
            for polygon in record["polygons"]:
                points = np.asarray(normalize_vertices(polygon["vertices_pixel"] + polygon["vertices_pixel"][:1]))
                axis.fill(points[:, 0], points[:, 1], color=colors[polygon["zone_type"]], alpha=.22)
                axis.plot(points[:, 0], points[:, 1], color=colors[polygon["zone_type"]], linewidth=1)
            axis.set_title(f"{cube_id}\n{record['coverage_mode'] or 'unassigned'}", fontsize=7)
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#22c55e" if record["reviewed"] else "#6b7280"); spine.set_linewidth(3)
        fig.suptitle("Field 2 prediction-free area annotation progress")
        descriptor, name = tempfile.mkstemp(prefix=f".{self.overview_path.name}.", suffix=".png", dir=self.output_root)
        os.close(descriptor)
        Path(name).unlink(missing_ok=True)
        try:
            fig.savefig(name, dpi=140); plt.close(fig); Path(name).replace(self.overview_path)
        except Exception:
            plt.close(fig); Path(name).unlink(missing_ok=True); raise

    def save(self, payload: dict) -> tuple[int, int, int, int]:
        with self.lock:
            current = self.load()
            if int(payload.get("revision", -1)) != int(current["revision"]):
                raise ValueError("Area annotation revision conflict; reload before saving")
            if payload == current:
                reviewed = sum(item["reviewed"] for item in current["annotations"].values())
                polygons = sum(len(item["polygons"]) for item in current["annotations"].values())
                vertices = sum(
                    len(polygon["vertices_pixel"])
                    for item in current["annotations"].values()
                    for polygon in item["polygons"]
                )
                return reviewed, polygons, vertices, current["revision"]
            normalized = self.normalize_payload(payload)
            issues = self.validate(normalized)
            invalid = {cube_id: value for cube_id, value in issues.items() if value}
            if invalid:
                raise ValueError("Area annotation validation failed: " + json.dumps(invalid, sort_keys=True))
            normalized["revision"] = current["revision"] + 1
            normalized["updated_utc"] = utc_now()
            normalized["automatic_metadata"] = {
                "biological_labels_assigned_automatically": False,
                "predictions_or_probabilities_used": False,
                "geometry_inferred_automatically": False,
                "reserve_frame_exposed": False,
                "area_geometry_frozen": False,
            }
            vertex_csv, audit_csv, geojson = self._export_bytes(normalized)
            atomic_write_json(self.json_path, normalized)
            atomic_write_bytes(self.vertex_csv_path, vertex_csv)
            atomic_write_bytes(self.audit_csv_path, audit_csv)
            atomic_write_bytes(self.geojson_path, geojson)
            self._write_overview(normalized)
            reviewed = sum(item["reviewed"] for item in normalized["annotations"].values())
            polygons = sum(len(item["polygons"]) for item in normalized["annotations"].values())
            vertices = sum(len(polygon["vertices_pixel"]) for item in normalized["annotations"].values() for polygon in item["polygons"])
            return reviewed, polygons, vertices, normalized["revision"]


def require_frozen_area_contract(project: Path, config: dict) -> tuple[dict, pd.DataFrame]:
    """Validate a frozen area contract and return only the main-frame membership."""
    contract_path = project / config["freeze"]["contract"]
    if not contract_path.is_file():
        raise RuntimeError("Field 2 area geometry is not frozen; point annotation remains blocked")
    contract = json.loads(contract_path.read_text()) if contract_path.suffix == ".json" else yaml.safe_load(contract_path.read_text())
    if contract.get("status") != "field2_area_zone_contract_frozen":
        raise RuntimeError("Field 2 area contract is not frozen")
    if contract.get("biological_labels_assigned") is not False or contract.get("predictions_or_probabilities_used") is not False:
        raise RuntimeError("Area contract violates prediction-free provenance")
    for name, item in contract.get("frozen_outputs", {}).items():
        path = project / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen area output mismatch: {name}")
    main_item = contract["frozen_outputs"]["main_membership"]
    main = pd.read_csv(project / main_item["path"]).fillna("")
    if len(main) != 800 or "row" in main.columns or "column" in main.columns:
        raise RuntimeError("Frozen main area membership is invalid or exposes coordinates")
    if contract.get("main_count") != 800 or contract.get("reserve_count") != 396:
        raise RuntimeError("Frozen area membership counts differ from sampling contracts")
    return contract, main
