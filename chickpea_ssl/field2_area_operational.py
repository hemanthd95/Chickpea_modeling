"""Derived, deterministic operational geometry for raw Field 2 area annotations."""

from __future__ import annotations

import json
import math

import numpy as np
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine

from chickpea_ssl.field2_area_review import (
    ZONE_CODES,
    ZONE_TYPES,
    clip_polygon_to_bounds,
    distinct_vertex_count,
    normalize_vertices,
    polygon_area,
    polygon_self_intersects,
)


# Low to high so later assignments win deterministically.
ZONE_PRECEDENCE = (
    "research_crop_area",
    "uncertain_boundary",
    "alley",
    "outside_research_field",
)
PRECEDENCE_RULE = "outside_research_field>alley>uncertain_boundary>research_crop_area"
RAW_ZONE_BITS = {zone: 1 << index for index, zone in enumerate(ZONE_PRECEDENCE)}
DEFAULT_TERMINAL_DISTANCE_PIXELS = 12.0


class OperationalGeometryError(ValueError):
    """Raised only when no usable operational polygon can be recovered."""


def effective_coverage_mode(record: dict) -> str:
    """Polygons always imply mixed operational coverage, including legacy blanks."""
    if record.get("polygons"):
        return "mixed_manual_boundaries"
    return str(record.get("coverage_mode", ""))


def _component_area(component: dict) -> float:
    exterior = polygon_area(normalize_vertices(component["exterior_vertices_pixel"]))
    holes = sum(polygon_area(normalize_vertices(hole)) for hole in component.get("holes_vertices_pixel", []))
    return max(0.0, exterior - holes)


def _component_geometry(component: dict) -> dict:
    exterior = normalize_vertices(component["exterior_vertices_pixel"])
    rings = [[*exterior, exterior[0]]]
    for hole in component.get("holes_vertices_pixel", []):
        points = normalize_vertices(hole)
        rings.append([*points, points[0]])
    return {"type": "Polygon", "coordinates": rings}


def rasterize_operational_components(components: list[dict], shape: tuple[int, int]) -> np.ndarray:
    geometries = [(_component_geometry(component), 1) for component in components]
    if not geometries:
        return np.zeros(shape, dtype=bool)
    return rasterize(
        geometries, out_shape=shape, transform=Affine.identity(), fill=0,
        dtype="uint8", all_touched=False,
    ).astype(bool)


def _polygonized_components(mask: np.ndarray) -> list[dict]:
    components = []
    for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=Affine.identity(), connectivity=4):
        if int(value) != 1 or geometry["type"] != "Polygon":
            continue
        rings = geometry["coordinates"]
        exterior = [{"x": float(x), "y": float(y)} for x, y in rings[0][:-1]]
        holes = [[{"x": float(x), "y": float(y)} for x, y in ring[:-1]] for ring in rings[1:]]
        if distinct_vertex_count(exterior) < 3 or polygon_self_intersects(exterior):
            continue
        components.append({"exterior_vertices_pixel": exterior, "holes_vertices_pixel": holes})
    components.sort(key=lambda item: (
        min(point["y"] for point in item["exterior_vertices_pixel"]),
        min(point["x"] for point in item["exterior_vertices_pixel"]),
        -_component_area(item),
        json.dumps(item, sort_keys=True),
    ))
    for index, component in enumerate(components, 1):
        component["component_index"] = index
    return components


def operationalize_polygon(
    polygon: dict, shape: tuple[int, int],
    terminal_distance_pixels: float = DEFAULT_TERMINAL_DISTANCE_PIXELS,
) -> dict:
    """Derive valid operational components while preserving the raw polygon exactly."""
    polygon_id = str(polygon.get("polygon_id", ""))
    zone_type = str(polygon.get("zone_type", ""))
    if not polygon_id:
        raise OperationalGeometryError("missing_polygon_identity")
    if zone_type not in ZONE_TYPES:
        raise OperationalGeometryError(f"unknown_zone_type:{zone_type}")
    try:
        raw = normalize_vertices(polygon.get("vertices_pixel", []))
    except (KeyError, TypeError, ValueError) as error:
        raise OperationalGeometryError(f"nonfinite_or_corrupt_vertices:{polygon_id}") from error
    if distinct_vertex_count(raw) < 3:
        raise OperationalGeometryError(f"fewer_than_three_distinct_original_vertices:{polygon_id}")

    height, width = shape
    raw_area = polygon_area(raw)
    raw_self_intersection = polygon_self_intersects(raw)
    if raw_area <= 0 and not raw_self_intersection:
        raise OperationalGeometryError(f"no_recoverable_polygonal_operational_geometry:{polygon_id}")
    first_last_distance = math.dist(raw[0], raw[-1])
    working = list(raw)
    repair_method = "none"
    warnings = []
    if raw_self_intersection:
        warnings.append("raw_self_intersection")
        terminal_candidate = raw[:-1]
        if (
            first_last_distance <= terminal_distance_pixels
            and distinct_vertex_count(terminal_candidate) >= 3
            and polygon_area(terminal_candidate) > 0
            and not polygon_self_intersects(terminal_candidate)
        ):
            working = terminal_candidate
            repair_method = "terminal_vertex_removed_operational_only"
            warnings.append("near_duplicate_terminal_vertex")

    clipped = clip_polygon_to_bounds(working, width, height)
    if len(clipped) != len(working):
        warnings.append("raster_edge_clipping")
    components = []
    if distinct_vertex_count(clipped) >= 3 and polygon_area(clipped) > 0 and not polygon_self_intersects(clipped):
        components = [{
            "component_index": 1,
            "exterior_vertices_pixel": [{"x": float(x), "y": float(y)} for x, y in clipped],
            "holes_vertices_pixel": [],
        }]
    else:
        # Deterministic make-valid on the frozen pixel grid. Rasterization resolves
        # ring topology; polygonization retains every recoverable polygon component.
        if distinct_vertex_count(clipped) < 3 or (
            polygon_area(clipped) <= 0 and not polygon_self_intersects(clipped)
        ):
            raise OperationalGeometryError(f"no_recoverable_polygonal_operational_geometry:{polygon_id}")
        geometry = {"type": "Polygon", "coordinates": [[*clipped, clipped[0]]]}
        repaired_mask = rasterize(
            [(geometry, 1)], out_shape=shape, transform=Affine.identity(),
            fill=0, dtype="uint8", all_touched=False,
        ).astype(bool)
        components = _polygonized_components(repaired_mask)
        repair_method = "raster_polygonize_make_valid"
        warnings.append("deterministic_validity_repair")

    if not components:
        raise OperationalGeometryError(f"no_recoverable_polygonal_operational_geometry:{polygon_id}")
    operational_mask = rasterize_operational_components(components, shape)
    if not operational_mask.any():
        raise OperationalGeometryError(f"operational_geometry_not_rasterizable:{polygon_id}")
    operational_area = sum(_component_area(component) for component in components)
    return {
        "polygon_id": polygon_id,
        "zone_type": zone_type,
        "repair_method": repair_method,
        "raw_vertex_count": len(raw),
        "operational_vertex_count": sum(len(component["exterior_vertices_pixel"]) for component in components),
        "component_count": len(components),
        "first_to_last_distance_pixels": first_last_distance,
        "raw_self_intersection": raw_self_intersection,
        "operational_self_intersection": any(
            polygon_self_intersects(component["exterior_vertices_pixel"]) for component in components
        ),
        "raw_area_pixels2": raw_area,
        "operational_area_pixels2": operational_area,
        "absolute_area_change_pixels2": abs(operational_area - raw_area),
        "percentage_area_change": 100.0 * abs(operational_area - raw_area) / raw_area if raw_area else 0.0,
        "rasterized_pixel_count": int(operational_mask.sum()),
        "warnings": sorted(set(warnings)),
        "source_vertices_preserved": True,
        "components": components,
    }


def compile_operational_membership(
    record: dict, support: np.ndarray,
    terminal_distance_pixels: float = DEFAULT_TERMINAL_DISTANCE_PIXELS,
) -> dict:
    """Compile raw claims and a deterministic winning membership mask."""
    effective_mode = effective_coverage_mode(record)
    raw_zone_masks = {zone: np.zeros(support.shape, dtype=bool) for zone in ZONE_PRECEDENCE}
    polygon_audits = []
    if effective_mode == "entire_support_research_field":
        raw_zone_masks["research_crop_area"] = support.copy()
    elif effective_mode == "entire_support_outside_research_field":
        raw_zone_masks["outside_research_field"] = support.copy()
    elif effective_mode == "uncertain_requires_review":
        raw_zone_masks["uncertain_boundary"] = support.copy()
    elif effective_mode == "mixed_manual_boundaries":
        for polygon in record.get("polygons", []):
            operational = operationalize_polygon(polygon, support.shape, terminal_distance_pixels)
            mask = rasterize_operational_components(operational["components"], support.shape) & support
            raw_zone_masks[operational["zone_type"]] |= mask
            polygon_audits.append(operational)
        if record.get("treat_unassigned_valid_support_as_outside", False):
            claimed = np.logical_or.reduce(list(raw_zone_masks.values()))
            raw_zone_masks["outside_research_field"] |= support & ~claimed
    elif not effective_mode:
        # Incomplete, unreviewed cubes remain explicitly unassigned while work continues.
        pass
    else:
        raise OperationalGeometryError(f"missing_or_unknown_effective_coverage_mode:{effective_mode}")

    raw_bitmask = np.zeros(support.shape, dtype=np.uint8)
    winning = np.zeros(support.shape, dtype=np.uint8)
    membership_count = np.zeros(support.shape, dtype=np.uint8)
    for zone in ZONE_PRECEDENCE:
        mask = raw_zone_masks[zone] & support
        raw_bitmask[mask] |= RAW_ZONE_BITS[zone]
        membership_count[mask] += 1
        winning[mask] = ZONE_CODES[zone]
    winning[~support] = 0
    raw_bitmask[~support] = 0
    return {
        "effective_coverage_mode": effective_mode,
        "raw_zone_masks": raw_zone_masks,
        "raw_membership_bitmask": raw_bitmask,
        "winning_membership": winning,
        "precedence_applied": membership_count > 1,
        "precedence_rule": PRECEDENCE_RULE,
        "polygon_audits": polygon_audits,
    }


def raw_memberships_from_bits(bits: int) -> list[str]:
    return [zone for zone in ZONE_PRECEDENCE if int(bits) & RAW_ZONE_BITS[zone]]
