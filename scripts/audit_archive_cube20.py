#!/usr/bin/env python
"""Audit the newly observed Field 1 Cube 20 and mask variants."""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml


def role(path: Path) -> str:
    name = path.name.lower()
    for candidate in ("chickpea", "weed", "soil", "combined"):
        if candidate in name:
            return candidate
    return "unclassified"


def raster_row(path: Path, source: Path) -> tuple[dict[str, object], np.ndarray | None]:
    with rasterio.open(path) as dataset:
        array = dataset.read() if dataset.count <= 4 else None
        unique = ""
        nonzero = ""
        if array is not None:
            values = np.unique(array)
            unique = "|".join(map(str, values[:256]))
            nonzero = int(np.count_nonzero(array))
        transform = dataset.transform
        row = {
            "relative_path": str(path.relative_to(source)),
            "role": role(path),
            "driver": dataset.driver,
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "dtypes": "|".join(dataset.dtypes),
            "crs": str(dataset.crs or ""),
            "transform": "|".join(map(str, transform[:6])),
            "bounds_left": dataset.bounds.left,
            "bounds_bottom": dataset.bounds.bottom,
            "bounds_right": dataset.bounds.right,
            "bounds_top": dataset.bounds.top,
            "nodata": dataset.nodata,
            "unique_values_up_to_256": unique,
            "nonzero_values": nonzero,
            "size_bytes": path.stat().st_size,
        }
    return row, array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text())
    project = Path(config["project_root"])
    source = project / "data" / "OneDrive_2026-07-31_raw"
    trial = source / "Kusi chickpea project" / "Kusi cubes analysis trials"
    local = project / "metadata" / "local"

    reflectance = trial / "kusi_field1_target_Pika_L-GigE_20.bil-Georectify.bip"
    if not reflectance.is_file():
        raise SystemExit(f"Cube 20 reflectance not found: {reflectance}")

    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    issues: list[dict[str, str]] = []
    reflectance_row, _ = raster_row(reflectance, source)
    reflectance_row["role"] = "reflectance"
    reflectance_row["analytical_status"] = "valid_reflectance_grid"
    rows.append(reflectance_row)
    expected = (
        int(reflectance_row["width"])
        * int(reflectance_row["height"])
        * int(reflectance_row["bands"])
        * np.dtype(str(reflectance_row["dtypes"]).split("|")[0]).itemsize
    )
    if reflectance.stat().st_size != expected:
        issues.append({"relative_path": str(reflectance.relative_to(source)),
                       "issue": "reflectance_byte_size_mismatch"})
    if int(reflectance_row["bands"]) != 150:
        issues.append({"relative_path": str(reflectance.relative_to(source)),
                       "issue": "reflectance_band_count_not_150"})

    cube20_mask_name = re.compile(
        r"(?:field[_ ]?1.*(?:cube[_ ]?20|[_ ]20|__20).*mask|"
        r"mask.*field[_ ]?1.*(?:cube[_ ]?20|[_ ]20|__20))",
        flags=re.IGNORECASE,
    )
    candidates = sorted(
        path for path in trial.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".bip", ".tif", ".tiff"}
        and cube20_mask_name.search(path.name)
    )
    for path in candidates:
        try:
            row, array = raster_row(path, source)
            values = {
                int(float(value)) for value in str(row["unique_values_up_to_256"]).split("|")
                if value != ""
            }
            allowed = {0, 1, 2, 3} if row["role"] == "combined" else {0, 1, 255}
            same_grid = (
                row["width"] == reflectance_row["width"]
                and row["height"] == reflectance_row["height"]
                and row["crs"] == reflectance_row["crs"]
                and row["transform"] == reflectance_row["transform"]
            )
            if row["bands"] != 1 or not row["crs"] or not values.issubset(allowed):
                row["analytical_status"] = "visualization_only"
            elif same_grid:
                row["analytical_status"] = "exact_reflectance_grid"
            else:
                row["analytical_status"] = "alternate_georeferenced_grid_needs_nearest_reprojection"
            rows.append(row)
            if array is not None:
                arrays[str(path.relative_to(source))] = array
            if row["analytical_status"] == "visualization_only":
                issues.append({"relative_path": row["relative_path"],
                               "issue": "visualization_only_not_analytical_mask"})
            if row["nodata"] not in ("", None) and float(row["nodata"]) in values - {0}:
                issues.append({"relative_path": row["relative_path"],
                               "issue": "nodata_value_conflicts_with_foreground_value"})
        except Exception as error:
            issues.append({"relative_path": str(path.relative_to(source)),
                           "issue": f"read_error: {error}"})

    comparisons: list[dict[str, object]] = []
    for first, second in itertools.combinations(rows[1:], 2):
        if first["role"] != second["role"] or first["role"] == "unclassified":
            continue
        if "visualization_only" in (first["analytical_status"], second["analytical_status"]):
            continue
        a = arrays.get(str(first["relative_path"]))
        b = arrays.get(str(second["relative_path"]))
        if a is None or b is None or a.shape != b.shape:
            continue
        a_mask = np.any(a != 0, axis=0)
        b_mask = np.any(b != 0, axis=0)
        union = int(np.count_nonzero(a_mask | b_mask))
        intersection = int(np.count_nonzero(a_mask & b_mask))
        comparisons.append({
            "role": first["role"],
            "first": first["relative_path"],
            "second": second["relative_path"],
            "intersection_pixels": intersection,
            "union_pixels": union,
            "iou": intersection / union if union else 1.0,
            "arrays_identical": bool(np.array_equal(a, b)),
        })

    pd.DataFrame(rows).to_csv(local / "archive_cube20_raster_qc.csv", index=False)
    pd.DataFrame(comparisons, columns=[
        "role", "first", "second", "intersection_pixels", "union_pixels",
        "iou", "arrays_identical",
    ]).to_csv(local / "archive_cube20_mask_comparison.csv", index=False)
    pd.DataFrame(issues, columns=["relative_path", "issue"]).to_csv(
        local / "archive_cube20_issues.csv", index=False
    )
    print(f"Cube 20 rasters inspected: {len(rows)}")
    print(f"Same-role comparisons: {len(comparisons)}")
    print(f"Issues: {len(issues)}")
    print(f"Reports written to: {local}")


if __name__ == "__main__":
    main()
