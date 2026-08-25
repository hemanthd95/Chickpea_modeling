#!/usr/bin/env python
"""Audit investigator-drawn planter-track polygons against chickpea masks.

This stage is deliberately non-destructive.  It interprets the nearly closed
browser traces as observed feature footprints, measures their intersection
with the current authoritative chickpea class, and renders review figures.  It
does not write or replace any mask.
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
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records


IMPACT_CMAP = ListedColormap(["#000000", "#20C76F", "#F59E0B", "#E83E8C"])


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


def closed_ring(
    coordinates: list[list[float]], snap_distance_m: float = 0.75
) -> tuple[list[tuple[float, float]], bool]:
    if len(coordinates) < 3:
        raise ValueError("A footprint needs at least three vertices")
    ring = [(float(x), float(y)) for x, y in coordinates]
    snapped = False
    if ring[0] != ring[-1] and math.dist(ring[0], ring[-1]) <= snap_distance_m:
        # Browser traces normally return to the starting end of the footprint.
        # Replacing the final near-duplicate with the first point prevents a
        # centimetre-scale seam crossover without moving any other vertex.
        ring[-1] = ring[0]
        snapped = True
    elif ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring, snapped


def polygon_geometry(coordinates: list[list[float]]) -> dict[str, object]:
    ring, _ = closed_ring(coordinates)
    return {"type": "Polygon", "coordinates": [ring]}


def polygon_area(coordinates: list[list[float]]) -> float:
    ring, _ = closed_ring(coordinates)
    return abs(sum(
        ring[index][0] * ring[index + 1][1]
        - ring[index + 1][0] * ring[index][1]
        for index in range(len(ring) - 1)
    )) / 2.0


def segment_intersects(a, b, c, d) -> bool:
    def orientation(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    first = orientation(a, b, c)
    second = orientation(a, b, d)
    third = orientation(c, d, a)
    fourth = orientation(c, d, b)
    return (first > 0) != (second > 0) and (third > 0) != (fourth > 0)


def self_intersections(coordinates: list[list[float]]) -> int:
    ring, _ = closed_ring(coordinates)
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


def reconcile_exports(json_path: Path, csv_path: Path, geojson_path: Path) -> tuple[dict, dict]:
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
        coordinates = feature["geometry"]["coordinates"]
        if len(rows) != len(item["points"]) or len(rows) != len(coordinates):
            raise ValueError(f"Vertex-count mismatch for {annotation_id}")
        if feature["properties"]["cube_id"] != cube_id:
            raise ValueError(f"Cube-ID mismatch for {annotation_id}")
        for row, coordinate in zip(rows, coordinates):
            if max(
                abs(float(row["map_x"]) - float(coordinate[0])),
                abs(float(row["map_y"]) - float(coordinate[1])),
            ) > 1e-7:
                raise ValueError(f"Coordinate mismatch for {annotation_id}")
    return annotations, geojson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--decisions", default=Path("configs/data_decisions.yaml"), type=Path)
    parser.add_argument("--annotations-root", type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    decisions = yaml.safe_load(args.decisions.read_text())
    if not decisions["rules"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked during Field 1 mask refinement")
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    annotation_root = args.annotations_root or local / "annotations" / "planter_tracks"
    json_path = annotation_root / "field1_planter_track_annotations.json"
    csv_path = annotation_root / "field1_planter_track_vertices.csv"
    geojson_path = annotation_root / "field1_planter_track_annotations.geojson"
    for path in (json_path, csv_path, geojson_path):
        if not path.exists():
            raise FileNotFoundError(f"Required annotation export is missing: {path}")

    annotations, geojson = reconcile_exports(json_path, csv_path, geojson_path)
    features_by_cube: dict[str, list[dict]] = defaultdict(list)
    geometry_rows: list[dict[str, object]] = []
    for feature in geojson["features"]:
        properties = feature["properties"]
        if properties.get("crs") != "EPSG:32617":
            raise ValueError(f"Unexpected annotation CRS: {properties.get('crs')}")
        coordinates = feature["geometry"]["coordinates"]
        intersections = self_intersections(coordinates)
        if intersections:
            raise ValueError(
                f"Self-intersecting annotation {properties['annotation_id']}: "
                f"{intersections} segment crossings"
            )
        feature = dict(feature)
        feature["polygon"] = polygon_geometry(coordinates)
        features_by_cube[str(properties["cube_id"])].append(feature)
        closure_m = math.dist(coordinates[0], coordinates[-1])
        _, closure_snapped = closed_ring(coordinates)
        geometry_rows.append({
            "cube_id": properties["cube_id"],
            "annotation_id": properties["annotation_id"],
            "kind": properties["kind"],
            "layer": properties.get("layer", ""),
            "vertices": len(coordinates),
            "closure_distance_m": closure_m,
            "closure_endpoint_snapped_for_audit": closure_snapped,
            "polygon_area_m2": polygon_area(coordinates),
            "self_intersections": intersections,
        })

    manifest = local / "authoritative_manifest.csv"
    records = {
        record.cube_id: record
        for record in load_records(args.paths, manifest)
        if record.chickpea_mask and record.cube_id in features_by_cube
    }
    missing = sorted(set(features_by_cube) - set(records))
    if missing:
        raise ValueError(f"Annotated cubes lack authoritative chickpea masks: {missing}")

    reports = local / "reports" / "mask_refinement" / "planter_annotation_qc"
    individual = reports / "individual_cubes"
    individual.mkdir(parents=True, exist_ok=True)
    phenology_exclusions = set(decisions.get("investigator_phenology_exclusions", {}))
    summary_rows: list[dict[str, object]] = []
    overview_tiles: list[tuple[str, np.ndarray, int, str]] = []
    output_paths: list[Path] = []

    for number, cube_id in enumerate(sorted(records, key=lambda item: int(item.split("cube")[-1])), start=1):
        record = records[cube_id]
        labels = authoritative_class_map(record)
        chickpea = labels == 1
        with rasterio.open(record.data) as reference:
            transform = reference.transform
            shape = (reference.height, reference.width)
        if chickpea.shape != shape:
            raise ValueError(f"Mask/reference shape mismatch for {cube_id}")
        gsd = metres_per_pixel(transform)
        kinds = Counter(feature["properties"]["kind"] for feature in features_by_cube[cube_id])

        masks: dict[str, np.ndarray] = {}
        for kind in ("alley_boundary", "tyre_track", "uncertain_structure"):
            polygons = [
                (feature["polygon"], 1)
                for feature in features_by_cube[cube_id]
                if feature["properties"]["kind"] == kind
            ]
            masks[kind] = (
                rasterize(polygons, out_shape=shape, transform=transform, fill=0,
                          dtype="uint8", all_touched=True).astype(bool)
                if polygons else np.zeros(shape, dtype=bool)
            )

        alley_chickpea = chickpea & masks["alley_boundary"]
        tyre_chickpea = chickpea & masks["tyre_track"]
        proposed = np.zeros(shape, dtype=np.uint8)
        proposed[chickpea] = 1
        proposed[tyre_chickpea] = 3
        proposed[alley_chickpea] = 2
        role = "sensitivity_only" if cube_id in phenology_exclusions else "primary_candidate"

        image_path = individual / f"{cube_id}_planter_annotation_qc.png"
        figure, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
        axes[0].imshow(chickpea, cmap=ListedColormap(["#000000", "#20C76F"]), vmin=0, vmax=1)
        axes[0].set_title(f"Current chickpea\n{int(chickpea.sum()):,} pixels")
        axes[1].imshow(proposed, cmap=IMPACT_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axes[1].set_title(
            f"Investigator footprints\norange alley={int(alley_chickpea.sum()):,}; "
            f"magenta tyre={int(tyre_chickpea.sum()):,}"
        )
        axes[2].imshow(masks["alley_boundary"] | masks["tyre_track"], cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(
            f"All marked footprints\n{kinds['alley_boundary']} alleys; "
            f"{kinds['tyre_track']} tyre tracks"
        )
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{cube_id} planter-annotation impact review — {role}\n"
            "No pixels are changed by this audit",
            fontsize=15,
        )
        figure.savefig(image_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(image_path)

        summary_rows.append({
            "cube_id": cube_id,
            "analysis_role": role,
            "alley_annotations": kinds["alley_boundary"],
            "tyre_track_annotations": kinds["tyre_track"],
            "uncertain_annotations": kinds["uncertain_structure"],
            "authoritative_chickpea_pixels": int(chickpea.sum()),
            "chickpea_pixels_inside_alley_footprints": int(alley_chickpea.sum()),
            "chickpea_fraction_inside_alley_footprints": (
                float(alley_chickpea.sum() / chickpea.sum()) if chickpea.any() else 0.0
            ),
            "chickpea_pixels_inside_tyre_footprints": int(tyre_chickpea.sum()),
            "chickpea_fraction_inside_tyre_footprints": (
                float(tyre_chickpea.sum() / chickpea.sum()) if chickpea.any() else 0.0
            ),
            "alley_footprint_area_m2": float(masks["alley_boundary"].sum() * gsd * gsd),
            "tyre_footprint_area_m2": float(masks["tyre_track"].sum() * gsd * gsd),
        })
        step = max(1, math.ceil(max(shape) / 500))
        overview_tiles.append((cube_id, proposed[::step, ::step], int(alley_chickpea.sum()), role))
        print(
            f"Audited {number}/{len(records)} {cube_id}: "
            f"{kinds['alley_boundary']} alley footprints, "
            f"{int(alley_chickpea.sum()):,} chickpea pixels inside alleys",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = reports / "planter_annotation_mask_impact.csv"
    summary.to_csv(summary_path, index=False)
    geometry_path = reports / "planter_annotation_geometry_qc.csv"
    pd.DataFrame(geometry_rows).to_csv(geometry_path, index=False)
    output_paths.extend([summary_path, geometry_path])

    columns = 4
    rows = math.ceil(len(overview_tiles) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.8 * rows), constrained_layout=True)
    flat = np.asarray(axes).reshape(-1)
    for axis in flat:
        axis.axis("off")
        axis.set_facecolor("black")
    for axis, (cube_id, tile, alley_pixels, role) in zip(flat, overview_tiles):
        axis.imshow(tile, cmap=IMPACT_CMAP, vmin=0, vmax=3, interpolation="nearest")
        axis.set_title(f"{cube_id} | alley overlap {alley_pixels:,}\n{role}", fontsize=10)
    figure.suptitle(
        "Field 1 investigator-drawn planter-footprint QC\n"
        "green=current chickpea; orange=chickpea in alley; magenta=chickpea in tyre footprint",
        fontsize=16,
    )
    overview_path = reports / "planter_annotation_mask_impact_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white")
    plt.close(figure)
    output_paths.append(overview_path)

    contract = {
        "status": "investigator_planter_annotations_audited_no_mask_change",
        "field": "Field 1",
        "field2_accessed": False,
        "source_masks_modified": False,
        "authoritative_masks_modified": False,
        "annotations_interpreted_as": "closed investigator-drawn feature footprints",
        "alley_policy": "review chickpea intersections as proposed weed relabel candidates",
        "tyre_track_policy": "geometry evidence only; do not automatically relabel",
        "annotation_counts": dict(Counter(
            feature["properties"]["kind"] for feature in geojson["features"]
        )),
        "cubes_audited": len(records),
        "phenology_sensitivity_only": sorted(phenology_exclusions),
        "source_hashes": {
            "annotations_json": sha256(json_path),
            "vertices_csv": sha256(csv_path),
            "annotations_geojson": sha256(geojson_path),
            "authoritative_manifest": sha256(manifest),
        },
        "output_hashes": {
            str(path.relative_to(reports)): sha256(path) for path in output_paths
        },
    }
    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract_path = contracts / "field1_planter_annotation_qc_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Annotations reconciled: {len(geojson['features'])}")
    print(f"Geometry QC: {geometry_path}")
    print(f"Mask-impact summary: {summary_path}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print("Audit only: no mask changed, no model retrained, and Field 2 remained locked.")


if __name__ == "__main__":
    main()
