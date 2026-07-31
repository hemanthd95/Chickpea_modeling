#!/usr/bin/env python
"""Separate plot field identity from current Field 1 cube coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.crs import CRS
import yaml


def intersection_fraction(plot: tuple[float, float, float, float],
                          cube: tuple[float, float, float, float]) -> float:
    left = max(plot[0], cube[0])
    bottom = max(plot[1], cube[1])
    right = min(plot[2], cube[2])
    top = min(plot[3], cube[3])
    intersection = max(0.0, right - left) * max(0.0, top - bottom)
    area = max(0.0, plot[2] - plot[0]) * max(0.0, plot[3] - plot[1])
    return intersection / area if area else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text())
    project = Path(config["project_root"])
    local = project / "metadata" / "local"
    catalog = pd.read_csv(local / "project_catalog.csv", dtype=str).fillna("")
    gis = pd.read_csv(local / "supporting_gis_qc.csv")
    batches = config["field1"].get(
        "reflectance_processing_batches", config["field1"].get("reflectance_dates", {})
    )

    cube_rows: list[dict[str, object]] = []
    headers = catalog[catalog["file_role"] == "reflectance_header"]
    for _, row in headers.iterrows():
        batch = row.get("processing_batch", "") or row["acquisition_date"]
        data_name = row["relative_path"][:-4]
        path = Path(batches[str(batch)]) / data_name
        with rasterio.open(path) as dataset:
            cube_rows.append({
                "cube_id": row["cube_id_inferred"],
                "left": dataset.bounds.left,
                "bottom": dataset.bounds.bottom,
                "right": dataset.bounds.right,
                "top": dataset.bounds.top,
                "crs": str(dataset.crs or ""),
            })
    cubes = pd.DataFrame(cube_rows)
    if cubes.empty:
        raise SystemExit("No Field 1 reflectance cubes found in the local catalog.")

    boundary_names = ("field_boundary_polygon.shp", "field1_polygon_plot.shp")
    boundaries = gis[
        gis["relative_path"].str.lower().map(
            lambda value: any(value.endswith(name) for name in boundary_names)
        )
    ]
    if boundaries.empty:
        raise SystemExit("No named Field 1 boundary candidates were found.")
    results: list[dict[str, object]] = []
    polygons = gis[
        (gis["declared_geometry"] == "Polygon") & gis["plot_id"].notna()
    ].copy()
    polygons["preferred"] = polygons["relative_path"].str.contains("bounds_polygon")
    polygons = (
        polygons.sort_values(["plot_id", "preferred"], ascending=[True, False])
        .drop_duplicates("plot_id", keep="first")
    )
    for _, row in polygons.iterrows():
        bounds = (row["bounds_left"], row["bounds_bottom"],
                  row["bounds_right"], row["bounds_top"])
        plot_crs = CRS.from_wkt(row["crs"]) if row.get("crs", "") else None
        comparable = [
            cube for _, cube in cubes.iterrows()
            if plot_crs is not None and cube["crs"] and CRS.from_string(cube["crs"]) == plot_crs
        ]
        cube_fractions = [
            (cube["cube_id"], intersection_fraction(
                bounds, (cube["left"], cube["bottom"], cube["right"], cube["top"])
            ))
            for cube in comparable
        ]
        boundary_fractions = []
        for _, boundary in boundaries.iterrows():
            if plot_crs is None or not boundary.get("crs", ""):
                continue
            if CRS.from_wkt(boundary["crs"]) != plot_crs:
                continue
            # A 1.5 m metadata-stage tolerance accommodates hand-drawn/RTK edge
            # differences; exact polygon containment remains a later gate.
            buffered = (
                boundary["bounds_left"] - 1.5, boundary["bounds_bottom"] - 1.5,
                boundary["bounds_right"] + 1.5, boundary["bounds_top"] + 1.5,
            )
            boundary_fractions.append((boundary["relative_path"], intersection_fraction(bounds, buffered)))
        if not cube_fractions:
            results.append({
                "relative_path": row["relative_path"],
                "plot_id": row.get("plot_id", ""),
                "field_identity_status": "crs_mismatch_or_missing",
                "cube_coverage_status": "crs_mismatch_or_missing",
                "best_field1_cube": "",
                "maximum_single_cube_bbox_coverage": 0.0,
                "intersecting_field1_cube_count": 0,
                "intersecting_field1_cubes": "",
                "best_boundary": "",
                "maximum_buffered_boundary_bbox_coverage": 0.0,
                "method": "boundary_identity_plus_independent_cube_coverage",
                "eligible_as_chickpea_spatial_prior": False,
            })
            continue
        overlapping = [(cube, fraction) for cube, fraction in cube_fractions if fraction > 0]
        best_cube, maximum = max(cube_fractions, key=lambda item: item[1])
        if maximum >= 0.95:
            coverage = "fully_covered_by_single_current_cube"
        elif maximum > 0:
            coverage = "partly_covered_by_current_cubes"
        else:
            coverage = "not_covered_by_current_cube_collection"
        best_boundary, boundary_maximum = max(
            boundary_fractions, key=lambda item: item[1], default=("", 0.0)
        )
        identity = (
            "field1_supported_by_named_boundary_bbox"
            if boundary_maximum >= 0.95 else "field_identity_pending_exact_geometry"
        )
        results.append({
            "relative_path": row["relative_path"],
            "plot_id": row.get("plot_id", ""),
            "field_identity_status": identity,
            "cube_coverage_status": coverage,
            "best_field1_cube": best_cube,
            "maximum_single_cube_bbox_coverage": maximum,
            "intersecting_field1_cube_count": len(overlapping),
            "intersecting_field1_cubes": "|".join(cube for cube, _ in overlapping),
            "best_boundary": best_boundary,
            "maximum_buffered_boundary_bbox_coverage": boundary_maximum,
            "method": "boundary_identity_plus_independent_cube_coverage",
            "eligible_as_chickpea_spatial_prior": False,
        })

    output = pd.DataFrame(results)
    output.to_csv(local / "plot_field_assignment.csv", index=False)
    print("Field identity (provisional):")
    print(output["field_identity_status"].value_counts().to_string())
    print("Current cube coverage (not field identity):")
    print(output["cube_coverage_status"].value_counts().to_string())
    print(f"Unique numbered plots assessed: {len(output)}")
    print("Field 2 imagery and coordinates were not opened.")
    print(f"Report: {local / 'plot_field_assignment.csv'}")


if __name__ == "__main__":
    main()
