#!/usr/bin/env python
"""Index globally anchored map blocks shared across overlapping Field 1 cubes."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

# Support direct execution as documented: `python scripts/build_spatial_group_index.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.data import load_records
from chickpea_ssl.spatial import spatial_group_id


def blocks_inside_dataset(dataset: rasterio.io.DatasetReader, block_size: float,
                          origin_x: float, origin_y: float) -> list[tuple[int, int]]:
    bounds = dataset.bounds
    first_x = int(np.floor((bounds.left - origin_x) / block_size))
    last_x = int(np.floor((bounds.right - origin_x) / block_size))
    first_y = int(np.floor((bounds.bottom - origin_y) / block_size))
    last_y = int(np.floor((bounds.top - origin_y) / block_size))
    inverse = ~dataset.transform
    selected: list[tuple[int, int]] = []
    for block_x in range(first_x, last_x + 1):
        x = origin_x + (block_x + 0.5) * block_size
        for block_y in range(first_y, last_y + 1):
            y = origin_y + (block_y + 0.5) * block_size
            column, row = inverse * (x, y)
            if 0 <= row < dataset.height and 0 <= column < dataset.width:
                selected.append((block_x, block_y))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", default=Path("configs/spatial_splits.yaml"), type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    manifest = args.manifest or local / "authoritative_manifest.csv"
    configuration = yaml.safe_load(args.config.read_text())["grouping"]
    block_size = float(configuration["block_size_m"])
    origin_x = float(configuration.get("origin_x_m", 0.0))
    origin_y = float(configuration.get("origin_y_m", 0.0))

    footprints: list[dict[str, object]] = []
    membership: list[dict[str, object]] = []
    cube_groups: dict[str, set[str]] = {}
    expected_crs: str | None = None
    for record in load_records(args.paths, manifest):
        with rasterio.open(record.data) as dataset:
            if dataset.crs is None or not dataset.crs.is_projected:
                raise ValueError(f"{record.cube_id} lacks a projected CRS")
            crs = dataset.crs.to_string()
            if expected_crs is None:
                expected_crs = crs
            elif crs != expected_crs:
                raise ValueError(f"Mixed CRS: {record.cube_id} uses {crs}, expected {expected_crs}")
            if dataset.crs.linear_units.lower() not in {"metre", "meter"}:
                raise ValueError(f"{record.cube_id} CRS units are not metres")
            blocks = blocks_inside_dataset(dataset, block_size, origin_x, origin_y)
            groups = {
                spatial_group_id(crs, block_x, block_y)
                for block_x, block_y in blocks
            }
            cube_groups[record.cube_id] = groups
            footprints.append({
                "cube_id": record.cube_id, "crs": crs,
                "height": dataset.height, "width": dataset.width,
                "pixel_size_x_m": float(np.hypot(dataset.transform.a, dataset.transform.d)),
                "pixel_size_y_m": float(np.hypot(dataset.transform.b, dataset.transform.e)),
                "left": dataset.bounds.left, "bottom": dataset.bounds.bottom,
                "right": dataset.bounds.right, "top": dataset.bounds.top,
                "block_size_m": block_size, "groups_covered": len(groups),
            })
            for block_x, block_y in blocks:
                membership.append({
                    "spatial_group_id": spatial_group_id(crs, block_x, block_y),
                    "block_x": block_x, "block_y": block_y,
                    "center_x_m": origin_x + (block_x + 0.5) * block_size,
                    "center_y_m": origin_y + (block_y + 0.5) * block_size,
                    "cube_id": record.cube_id,
                })

    overlap_rows: list[dict[str, object]] = []
    for first, second in combinations(sorted(cube_groups), 2):
        shared = cube_groups[first] & cube_groups[second]
        if not shared:
            continue
        overlap_rows.append({
            "cube_a": first, "cube_b": second,
            "shared_spatial_groups": len(shared),
            "fraction_of_smaller_cube_groups": len(shared) / min(
                len(cube_groups[first]), len(cube_groups[second])
            ),
        })

    footprint_frame = pd.DataFrame(footprints).sort_values("cube_id")
    membership_frame = pd.DataFrame(membership).sort_values(
        ["spatial_group_id", "cube_id"]
    )
    overlap_frame = pd.DataFrame(overlap_rows, columns=[
        "cube_a", "cube_b", "shared_spatial_groups", "fraction_of_smaller_cube_groups"
    ]).sort_values(["shared_spatial_groups", "cube_a", "cube_b"], ascending=[False, True, True])
    footprint_frame.to_csv(local / "spatial_cube_footprints.csv", index=False)
    membership_frame.to_csv(local / "spatial_group_membership.csv", index=False)
    overlap_frame.to_csv(local / "spatial_cube_overlap.csv", index=False)

    repeated = membership_frame.groupby("spatial_group_id")["cube_id"].nunique()
    print(f"Cubes indexed: {len(footprint_frame)}")
    print(f"Unique {block_size:g} m spatial groups: {membership_frame['spatial_group_id'].nunique()}")
    print(f"Groups observed by multiple cubes: {int((repeated > 1).sum())}")
    print(f"Overlapping cube pairs: {len(overlap_frame)}")
    print(f"Reports written to: {local}")
    print("No fold assignment was made; Field 2 was not opened.")


if __name__ == "__main__":
    main()

