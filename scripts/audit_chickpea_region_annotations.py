#!/usr/bin/env python
"""Audit investigator-drawn chickpea row polygons without changing labels.

The polygons are treated as uncertain spatial eligibility priors, not as pixel
classes.  This stage reconciles all three browser exports, normalizes tiny
closing-seam artifacts, clips polygons to observed raster support, measures
overlap with investigator-drawn alleys, and reports how sensitive the current
chickpea mask would be to several outward edge tolerances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy.ndimage import binary_dilation
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records


QC_CMAP = ListedColormap([
    "#000000",  # background
    "#15803D",  # polygon core
    "#86EFAC",  # outward uncertainty tolerance
    "#F59E0B",  # explicit alley precedence
    "#DC2626",  # polygon over internal NoData
    "#D946EF",  # current chickpea beyond tolerance
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metres_per_pixel(transform) -> float:
    x = math.hypot(transform.a, transform.d)
    y = math.hypot(transform.b, transform.e)
    if x <= 0 or y <= 0:
        raise ValueError("Raster transform does not contain a positive pixel size")
    return float(math.sqrt(x * y))


def normalized_ring(
    coordinates: list[list[float]], seam_snap_maximum_m: float
) -> tuple[list[tuple[float, float]], bool, float]:
    if len(coordinates) < 3:
        raise ValueError("A chickpea region needs at least three vertices")
    ring = [(float(x), float(y)) for x, y in coordinates]
    closure_distance = math.dist(ring[0], ring[-1])
    snapped = False
    if ring[0] != ring[-1] and closure_distance <= seam_snap_maximum_m:
        ring[-1] = ring[0]
        snapped = True
    elif ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring, snapped, closure_distance


def polygon_area(ring: list[tuple[float, float]]) -> float:
    return abs(sum(
        ring[index][0] * ring[index + 1][1]
        - ring[index + 1][0] * ring[index][1]
        for index in range(len(ring) - 1)
    )) / 2.0


def segment_intersects(a, b, c, d) -> bool:
    def orientation(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    first, second = orientation(a, b, c), orientation(a, b, d)
    third, fourth = orientation(c, d, a), orientation(c, d, b)
    return (first > 0) != (second > 0) and (third > 0) != (fourth > 0)


def self_intersections(ring: list[tuple[float, float]]) -> int:
    count = 0
    segment_count = len(ring) - 1
    for first in range(segment_count):
        for second in range(first + 1, segment_count):
            if abs(first - second) <= 1 or (first == 0 and second == segment_count - 1):
                continue
            if segment_intersects(
                ring[first], ring[first + 1], ring[second], ring[second + 1]
            ):
                count += 1
    return count


def reconcile_exports(json_path: Path, csv_path: Path, geojson_path: Path):
    annotations = json.loads(json_path.read_text())
    geojson = json.loads(geojson_path.read_text())
    with csv_path.open(newline="") as stream:
        vertices = list(csv.DictReader(stream))
    json_items = {
        item["id"]: (cube_id, item)
        for cube_id, items in annotations.get("cubes", {}).items()
        for item in items
    }
    geo_items = {
        feature["properties"]["annotation_id"]: feature
        for feature in geojson.get("features", [])
    }
    vertex_items: dict[str, list[dict]] = defaultdict(list)
    for row in vertices:
        vertex_items[row["annotation_id"]].append(row)
    if not (set(json_items) == set(geo_items) == set(vertex_items)):
        raise ValueError("JSON, CSV, and GeoJSON annotation IDs do not agree")
    for annotation_id, (cube_id, item) in json_items.items():
        rows = sorted(vertex_items[annotation_id], key=lambda row: int(row["vertex_index"]))
        feature = geo_items[annotation_id]
        coordinates = feature["geometry"]["coordinates"][0]
        exported = coordinates[:-1] if coordinates and coordinates[0] == coordinates[-1] else coordinates
        if len(rows) != len(item["points"]) or len(rows) != len(exported):
            raise ValueError(f"Vertex-count mismatch for {annotation_id}")
        if feature["properties"]["cube_id"] != cube_id:
            raise ValueError(f"Cube-ID mismatch for {annotation_id}")
        for row, coordinate in zip(rows, exported):
            if max(
                abs(float(row["map_x"]) - float(coordinate[0])),
                abs(float(row["map_y"]) - float(coordinate[1])),
            ) > 1e-7:
                raise ValueError(f"Coordinate mismatch for {annotation_id}")
    return annotations, geojson, vertices


def observed_support(dataset, chunk_rows: int = 256) -> np.ndarray:
    observed = np.zeros((dataset.height, dataset.width), dtype=bool)
    for row_off in range(0, dataset.height, chunk_rows):
        height = min(chunk_rows, dataset.height - row_off)
        window = Window(0, row_off, dataset.width, height)
        block = dataset.read(window=window)
        observed[row_off:row_off + height] = np.any(block > 0, axis=0)
    return observed


def disk(radius_pixels: int) -> np.ndarray:
    if radius_pixels <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius_pixels:radius_pixels + 1, -radius_pixels:radius_pixels + 1]
    return x * x + y * y <= radius_pixels * radius_pixels


def planter_alley_features(path: Path, seam_snap_maximum_m: float):
    if not path.exists():
        raise FileNotFoundError(f"Planter annotation GeoJSON is missing: {path}")
    features = json.loads(path.read_text()).get("features", [])
    result: dict[str, list[dict]] = defaultdict(list)
    for feature in features:
        properties = feature["properties"]
        if properties.get("kind") != "alley_boundary":
            continue
        coordinates = feature["geometry"]["coordinates"]
        if coordinates and isinstance(coordinates[0][0], (list, tuple)):
            coordinates = coordinates[0]
        ring, _, _ = normalized_ring(coordinates, max(0.75, seam_snap_maximum_m))
        result[str(properties["cube_id"])].append({"type": "Polygon", "coordinates": [ring]})
    return result


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
    audit = config.get("chickpea_region_polygon_audit", {})
    seam_snap = float(audit.get("seam_snap_maximum_m", 0.30))
    tolerances = sorted({float(value) for value in audit.get(
        "outward_edge_tolerance_candidates_m", [0.00, 0.05, 0.10, 0.15]
    )})
    visual_tolerance = float(audit.get("visual_review_tolerance_m", max(tolerances)))
    if visual_tolerance not in tolerances:
        tolerances.append(visual_tolerance)
        tolerances.sort()

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    annotation_root = args.annotations_root or local / "annotations" / "chickpea_regions"
    json_path = annotation_root / "field1_chickpea_region_annotations.json"
    csv_path = annotation_root / "field1_chickpea_region_vertices.csv"
    geojson_path = annotation_root / "field1_chickpea_region_annotations.geojson"
    for path in (json_path, csv_path, geojson_path):
        if not path.exists():
            raise FileNotFoundError(f"Required annotation export is missing: {path}")

    _, geojson, vertices = reconcile_exports(json_path, csv_path, geojson_path)
    features_by_cube: dict[str, list[dict]] = defaultdict(list)
    geometry_rows = []
    for feature in geojson["features"]:
        properties = feature["properties"]
        if properties.get("crs") != "EPSG:32617":
            raise ValueError(f"Unexpected annotation CRS: {properties.get('crs')}")
        coordinates = feature["geometry"]["coordinates"][0]
        if coordinates and coordinates[0] == coordinates[-1]:
            coordinates = coordinates[:-1]
        ring, snapped, closure_distance = normalized_ring(coordinates, seam_snap)
        intersections = self_intersections(ring)
        if intersections:
            raise ValueError(
                f"Normalized polygon {properties['annotation_id']} still has "
                f"{intersections} self-intersections; redraw only this polygon"
            )
        area = polygon_area(ring)
        if area <= 0:
            raise ValueError(f"Zero-area polygon: {properties['annotation_id']}")
        normalized = {"type": "Polygon", "coordinates": [ring]}
        features_by_cube[str(properties["cube_id"])].append(normalized)
        geometry_rows.append({
            "cube_id": properties["cube_id"],
            "annotation_id": properties["annotation_id"],
            "analysis_role": properties.get("analysis_role", ""),
            "vertices": len(coordinates),
            "closure_distance_m": closure_distance,
            "closing_endpoint_snapped_for_audit": snapped,
            "normalized_self_intersections": intersections,
            "polygon_area_m2": area,
        })

    layer_manifest_path = (
        local / "reports" / "chickpea_region_annotation"
        / "chickpea_region_annotation_layer_manifest.csv"
    )
    layer_manifest = pd.read_csv(layer_manifest_path).set_index("cube_id")
    records = {
        record.cube_id: record for record in load_records(
            args.paths, local / "authoritative_manifest.csv"
        ) if record.cube_id in features_by_cube
    }
    missing = sorted(set(features_by_cube) - set(records))
    if missing:
        raise ValueError(f"Annotated cubes are absent from the manifest: {missing}")

    alley_geojson = (
        local / "annotations" / "planter_tracks"
        / "field1_planter_track_annotations.geojson"
    )
    alleys_by_cube = planter_alley_features(alley_geojson, seam_snap)
    reports = local / "reports" / "mask_refinement" / "chickpea_region_annotation_qc"
    individual = reports / "individual_cubes"
    individual.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    sensitivity_rows = []
    overview = []
    output_paths: list[Path] = []

    for number, cube_id in enumerate(
        sorted(records, key=lambda item: int(item.split("cube")[-1])), start=1
    ):
        record = records[cube_id]
        with rasterio.open(record.data) as reference:
            shape = (reference.height, reference.width)
            transform = reference.transform
            gsd = metres_per_pixel(transform)
            observed = observed_support(reference)
        polygons = [(geometry, 1) for geometry in features_by_cube[cube_id]]
        core_raw = rasterize(
            polygons, out_shape=shape, transform=transform, fill=0,
            dtype="uint8", all_touched=False,
        ).astype(bool)
        core = core_raw & observed
        alley_shapes = [(geometry, 1) for geometry in alleys_by_cube.get(cube_id, [])]
        alley = (
            rasterize(
                alley_shapes, out_shape=shape, transform=transform, fill=0,
                dtype="uint8", all_touched=True,
            ).astype(bool)
            if alley_shapes else np.zeros(shape, dtype=bool)
        )
        labels = authoritative_class_map(record) if record.chickpea_mask else None
        current = labels == 1 if labels is not None else np.zeros(shape, dtype=bool)
        tolerant_masks = {}
        for tolerance in tolerances:
            radius = int(math.ceil(tolerance / gsd))
            tolerant = binary_dilation(core_raw, structure=disk(radius)) & observed
            tolerant_masks[tolerance] = tolerant
            sensitivity_rows.append({
                "cube_id": cube_id,
                "analysis_role": layer_manifest.loc[cube_id, "analysis_role"],
                "outward_edge_tolerance_m": tolerance,
                "tolerance_radius_pixels": radius,
                "eligible_observed_pixels": int(tolerant.sum()),
                "eligible_pixels_inside_alley": int((tolerant & alley).sum()),
                "current_chickpea_pixels": int(current.sum()),
                "current_chickpea_inside_prior": int((current & tolerant).sum()),
                "current_chickpea_outside_prior": int((current & ~tolerant).sum()),
                "current_chickpea_retained_fraction": (
                    float((current & tolerant).sum() / current.sum()) if current.any() else np.nan
                ),
            })
        tolerant = tolerant_masks[visual_tolerance]
        qc = np.zeros(shape, dtype=np.uint8)
        qc[core] = 1
        qc[tolerant & ~core] = 2
        qc[tolerant & alley] = 3
        qc[core_raw & ~observed] = 4
        qc[current & ~tolerant] = 5
        role = str(layer_manifest.loc[cube_id, "analysis_role"])
        outside_vertex_count = 0
        cube_vertices = [row for row in vertices if row["cube_id"] == cube_id]
        for row in cube_vertices:
            column, raster_row = float(row["original_column"]), float(row["original_row"])
            if not (0 <= column < shape[1] and 0 <= raster_row < shape[0]):
                outside_vertex_count += 1
        summary_rows.append({
            "cube_id": cube_id,
            "analysis_role": role,
            "polygons": len(features_by_cube[cube_id]),
            "vertices": len(cube_vertices),
            "vertices_outside_raster_grid": outside_vertex_count,
            "polygon_core_raw_pixels": int(core_raw.sum()),
            "polygon_core_observed_pixels": int(core.sum()),
            "polygon_core_nodata_pixels": int((core_raw & ~observed).sum()),
            "alley_pixels_inside_polygon_core": int((core & alley).sum()),
            "visual_tolerance_m": visual_tolerance,
            "tolerant_observed_pixels": int(tolerant.sum()),
            "tolerant_pixels_inside_alley": int((tolerant & alley).sum()),
            "current_chickpea_pixels": int(current.sum()),
            "current_chickpea_outside_tolerant_prior": int((current & ~tolerant).sum()),
            "current_chickpea_retained_fraction": (
                float((current & tolerant).sum() / current.sum()) if current.any() else np.nan
            ),
        })
        step = max(1, math.ceil(max(shape) / 500))
        tile = qc[::step, ::step]
        overview.append((cube_id, role, tile, int((current & ~tolerant).sum())))
        image_path = individual / f"{cube_id}_chickpea_region_annotation_qc.png"
        figure, axes = plt.subplots(1, 2, figsize=(10, 7), constrained_layout=True)
        axes[0].imshow(tile, cmap=QC_CMAP, vmin=0, vmax=5, interpolation="nearest")
        axes[0].set_title("Polygon prior QC")
        axes[1].imshow(current[::step, ::step], cmap=ListedColormap(["#000000", "#20C76F"]))
        axes[1].imshow(
            (current & ~tolerant)[::step, ::step],
            cmap=ListedColormap([(0, 0, 0, 0), "#D946EF"]), vmin=0, vmax=1,
        )
        axes[1].set_title("Current chickpea; magenta = beyond tolerance")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{cube_id} — {role}\n{visual_tolerance:.2f} m outward uncertainty; "
            "audit only",
            fontsize=14,
        )
        figure.savefig(image_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(image_path)
        print(
            f"Audited {number}/{len(records)} {cube_id}: "
            f"{len(features_by_cube[cube_id])} polygons; "
            f"current-mask retention at {visual_tolerance:.2f} m="
            f"{summary_rows[-1]['current_chickpea_retained_fraction']}",
            flush=True,
        )

    summary_path = reports / "chickpea_region_annotation_summary.csv"
    sensitivity_path = reports / "chickpea_region_edge_tolerance_sensitivity.csv"
    geometry_path = reports / "chickpea_region_geometry_qc.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    pd.DataFrame(geometry_rows).to_csv(geometry_path, index=False)
    output_paths.extend([summary_path, sensitivity_path, geometry_path])

    columns = 4
    rows = math.ceil(len(overview) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows), constrained_layout=True)
    flat = np.asarray(axes).reshape(-1)
    for axis in flat:
        axis.axis("off")
        axis.set_facecolor("black")
    for axis, (cube_id, role, tile, beyond) in zip(flat, overview):
        axis.imshow(tile, cmap=QC_CMAP, vmin=0, vmax=5, interpolation="nearest")
        axis.set_title(f"{cube_id} | {role}\ncurrent chickpea beyond prior: {beyond:,}", fontsize=9)
    figure.suptitle(
        "Field 1 investigator chickpea-region polygon QC\n"
        "dark green=core; light green=edge tolerance; orange=alley; "
        "red=NoData; magenta=current chickpea beyond prior",
        fontsize=15,
    )
    overview_path = reports / "chickpea_region_annotation_qc_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white")
    plt.close(figure)
    output_paths.append(overview_path)

    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "investigator_chickpea_region_polygons_audited_no_mask_change",
        "field": "Field 1",
        "polygon_interpretation": "uncertain spatial eligibility prior, never a pixel class",
        "edge_tolerance_candidates_m": tolerances,
        "visual_review_tolerance_m": visual_tolerance,
        "seam_snap_maximum_m": seam_snap,
        "explicit_alley_precedence_audited": True,
        "source_masks_modified": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "cubes_audited": len(records),
        "annotations_reconciled": len(geojson["features"]),
        "seams_snapped_for_audit": int(sum(row["closing_endpoint_snapped_for_audit"] for row in geometry_rows)),
        "source_hashes": {
            "annotations_json": sha256(json_path),
            "vertices_csv": sha256(csv_path),
            "annotations_geojson": sha256(geojson_path),
            "layer_manifest": sha256(layer_manifest_path),
            "planter_annotations_geojson": sha256(alley_geojson),
        },
        "output_hashes": {
            str(path.relative_to(reports)): sha256(path) for path in output_paths
        },
    }
    contract_path = contracts / "field1_chickpea_region_annotation_qc_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"Annotations reconciled: {len(geojson['features'])}")
    print(f"Geometry QC: {geometry_path}")
    print(f"Edge sensitivity: {sensitivity_path}")
    print(f"Summary: {summary_path}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print("Audit only: no mask changed, no model retrained, and Field 2 remained locked.")


if __name__ == "__main__":
    main()
