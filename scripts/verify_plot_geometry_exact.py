#!/usr/bin/env python
"""Verify exact plot containment within the observed Field 1 boundary."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chickpea_ssl.shapefile import read_polygon_geometries


def square_buffer(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), radius)
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    size = 2 * radius + 1
    windows = (
        integral[size:, size:] - integral[:-size, size:]
        - integral[size:, :-size] + integral[:-size, :-size]
    )
    return windows > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--boundary-buffer-m", type=float, default=1.5)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text())
    project = Path(config["project_root"])
    archive = project / "data" / "OneDrive_2026-07-31_raw"
    local = project / "metadata" / "local"
    gis = pd.read_csv(local / "supporting_gis_qc.csv")
    boundary_row = gis[
        gis["relative_path"].str.lower().str.endswith("field_boundary_polygon.shp")
    ].iloc[0]
    boundary_path = archive / boundary_row["relative_path"]
    boundary_shapes = read_polygon_geometries(boundary_path)

    margin = args.boundary_buffer_m + args.resolution_m
    left = boundary_row["bounds_left"] - margin
    right = boundary_row["bounds_right"] + margin
    bottom = boundary_row["bounds_bottom"] - margin
    top = boundary_row["bounds_top"] + margin
    width = math.ceil((right - left) / args.resolution_m)
    height = math.ceil((top - bottom) / args.resolution_m)
    transform = from_origin(left, top, args.resolution_m, args.resolution_m)
    boundary = rasterize(
        ((geometry, 1) for geometry in boundary_shapes),
        out_shape=(height, width), transform=transform,
        fill=0, dtype="uint8", all_touched=True,
    ).astype(bool)
    radius = math.ceil(args.boundary_buffer_m / args.resolution_m)
    buffered_boundary = square_buffer(boundary, radius)

    plots = gis[(gis["declared_geometry"] == "Polygon") & gis["plot_id"].notna()].copy()
    plots["preferred"] = plots["relative_path"].str.contains("bounds_polygon")
    plots = plots.sort_values(["plot_id", "preferred"], ascending=[True, False]).drop_duplicates("plot_id")
    rows: list[dict[str, object]] = []
    for _, row in plots.iterrows():
        path = archive / row["relative_path"]
        geometries = read_polygon_geometries(path)
        plot = rasterize(
            ((geometry, 1) for geometry in geometries),
            out_shape=(height, width), transform=transform,
            fill=0, dtype="uint8", all_touched=True,
        ).astype(bool)
        pixels = int(plot.sum())
        exact = int((plot & boundary).sum()) / pixels if pixels else 0.0
        buffered = int((plot & buffered_boundary).sum()) / pixels if pixels else 0.0
        rows.append({
            "plot_id": int(row["plot_id"]),
            "relative_path": row["relative_path"],
            "rasterized_plot_pixels": pixels,
            "fraction_inside_exact_boundary": exact,
            "fraction_inside_buffered_boundary": buffered,
            "boundary_buffer_m": args.boundary_buffer_m,
            "verification_resolution_m": args.resolution_m,
            "field1_geometry_confirmed": buffered >= 0.995,
            "eligible_as_chickpea_spatial_prior": buffered >= 0.995,
        })
    output = pd.DataFrame(rows).sort_values("plot_id")
    output.to_csv(local / "plot_exact_geometry_qc.csv", index=False)
    print(output["field1_geometry_confirmed"].value_counts().to_string())
    print(f"Minimum exact-boundary containment: {output['fraction_inside_exact_boundary'].min():.6f}")
    print(f"Minimum buffered containment: {output['fraction_inside_buffered_boundary'].min():.6f}")
    print("Field 2 was not opened.")
    print(f"Report: {local / 'plot_exact_geometry_qc.csv'}")


if __name__ == "__main__":
    main()
