#!/usr/bin/env python
"""Count authoritative observed labels within globally anchored spatial groups."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records
from chickpea_ssl.spatial import map_block_indices, spatial_group_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", default=Path("configs/spatial_splits.yaml"), type=Path)
    parser.add_argument("--chunk-rows", default=512, type=int)
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")

    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    reports = local / "reports" / "spatial_grouping"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or local / "authoritative_manifest.csv"
    grouping = yaml.safe_load(args.config.read_text())["grouping"]
    block_size = float(grouping["block_size_m"])
    origin_x = float(grouping.get("origin_x_m", 0.0))
    origin_y = float(grouping.get("origin_y_m", 0.0))

    membership_path = reports / "spatial_group_membership.csv"
    if not membership_path.is_file():
        legacy_path = local / "spatial_group_membership.csv"
        if legacy_path.is_file():
            membership_path = legacy_path
        else:
            raise FileNotFoundError("Run build_spatial_group_index.py first")
    indexed_groups = set(pd.read_csv(membership_path)["spatial_group_id"])

    counts: defaultdict[tuple[str, str, int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(3, dtype=np.int64)
    )
    cubes_with_labels = 0
    for record in load_records(args.paths, manifest):
        if not any((record.soil_mask, record.chickpea_mask, record.weed_mask)):
            continue
        labels = authoritative_class_map(record)
        cubes_with_labels += 1
        with rasterio.open(record.data) as dataset:
            transform = dataset.transform
            crs = dataset.crs.to_string()
        for row_start in range(0, labels.shape[0], args.chunk_rows):
            chunk = labels[row_start:row_start + args.chunk_rows]
            local_rows, columns = np.nonzero(chunk >= 0)
            if len(local_rows) == 0:
                continue
            rows = local_rows + row_start
            values = chunk[local_rows, columns]
            x = (transform.a * (columns + 0.5) + transform.b * (rows + 0.5)
                 + transform.c)
            y = (transform.d * (columns + 0.5) + transform.e * (rows + 0.5)
                 + transform.f)
            block_x, block_y = map_block_indices(
                x, y, block_size, origin_x, origin_y
            )
            unique_blocks, inverse = np.unique(
                np.column_stack((block_x, block_y)), axis=0, return_inverse=True
            )
            for index, (one_x, one_y) in enumerate(unique_blocks):
                group = spatial_group_id(crs, int(one_x), int(one_y))
                if group not in indexed_groups:
                    raise ValueError(f"Label found outside indexed groups: {group}")
                selected = inverse == index
                key = (group, record.cube_id, int(one_x), int(one_y))
                counts[key] += np.bincount(values[selected], minlength=3)

    rows = []
    for (group, cube, block_x, block_y), values in counts.items():
        rows.append({
            "spatial_group_id": group, "block_x": block_x, "block_y": block_y,
            "cube_id": cube, "soil_pixels": int(values[0]),
            "chickpea_pixels": int(values[1]), "weed_pixels": int(values[2]),
            "labeled_pixels": int(values.sum()),
        })
    by_cube = pd.DataFrame(rows).sort_values(["spatial_group_id", "cube_id"])
    summary = by_cube.groupby(
        ["spatial_group_id", "block_x", "block_y"], as_index=False
    ).agg(
        cubes_with_labels=("cube_id", "nunique"),
        soil_pixel_observations=("soil_pixels", "sum"),
        chickpea_pixel_observations=("chickpea_pixels", "sum"),
        weed_pixel_observations=("weed_pixels", "sum"),
        labeled_pixel_observations=("labeled_pixels", "sum"),
    )
    by_cube.to_csv(reports / "spatial_group_class_counts_by_cube.csv", index=False)
    summary.to_csv(reports / "spatial_group_class_summary.csv", index=False)

    totals = summary[[
        "soil_pixel_observations", "chickpea_pixel_observations",
        "weed_pixel_observations"
    ]].sum()
    print(f"Cubes with authoritative labels: {cubes_with_labels}")
    print(f"Spatial groups containing labels: {len(summary)}")
    print(f"Soil pixel observations: {int(totals.iloc[0]):,}")
    print(f"Chickpea pixel observations: {int(totals.iloc[1]):,}")
    print(f"Weed pixel observations: {int(totals.iloc[2]):,}")
    print(f"Reports written to: {reports}")
    print("Counts include repeated observations in overlapping cubes; no fold was assigned.")


if __name__ == "__main__":
    main()

