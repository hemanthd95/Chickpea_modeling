#!/usr/bin/env python
"""Assign candidate plot layers using Field 1 cube coverage only."""

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

    results: list[dict[str, object]] = []
    polygons = gis[gis["declared_geometry"] == "Polygon"]
    for _, row in polygons.iterrows():
        bounds = (row["bounds_left"], row["bounds_bottom"],
                  row["bounds_right"], row["bounds_top"])
        plot_crs = CRS.from_wkt(row["crs"]) if row.get("crs", "") else None
        comparable = [
            cube for _, cube in cubes.iterrows()
            if plot_crs is not None and cube["crs"] and CRS.from_string(cube["crs"]) == plot_crs
        ]
        fractions = [
            (cube["cube_id"], intersection_fraction(
                bounds, (cube["left"], cube["bottom"], cube["right"], cube["top"])
            ))
            for cube in comparable
        ]
        if not fractions:
            results.append({
                "relative_path": row["relative_path"],
                "plot_id": row.get("plot_id", ""),
                "assignment": "crs_mismatch_or_missing",
                "best_field1_cube": "",
                "maximum_single_cube_bbox_coverage": 0.0,
                "intersecting_field1_cube_count": 0,
                "intersecting_field1_cubes": "",
                "method": "candidate_bbox_vs_observed_field1_cube_bounds",
                "eligible_as_chickpea_spatial_prior": False,
            })
            continue
        overlapping = [(cube, fraction) for cube, fraction in fractions if fraction > 0]
        best_cube, maximum = max(fractions, key=lambda item: item[1])
        if maximum >= 0.95:
            assignment = "field1_confirmed_by_cube_coverage"
        elif maximum > 0:
            assignment = "partial_field1_overlap_review"
        else:
            assignment = "outside_field1_coverage_pending"
        results.append({
            "relative_path": row["relative_path"],
            "plot_id": row.get("plot_id", ""),
            "assignment": assignment,
            "best_field1_cube": best_cube,
            "maximum_single_cube_bbox_coverage": maximum,
            "intersecting_field1_cube_count": len(overlapping),
            "intersecting_field1_cubes": "|".join(cube for cube, _ in overlapping),
            "method": "candidate_bbox_vs_observed_field1_cube_bounds",
            "eligible_as_chickpea_spatial_prior": assignment == "field1_confirmed_by_cube_coverage",
        })

    output = pd.DataFrame(results)
    output.to_csv(local / "plot_field_assignment.csv", index=False)
    print(output["assignment"].value_counts().to_string())
    print(f"Candidate polygons assessed: {len(output)}")
    print("Field 2 imagery and coordinates were not opened.")
    print(f"Report: {local / 'plot_field_assignment.csv'}")


if __name__ == "__main__":
    main()
