#!/usr/bin/env python
"""Audit supporting Field 1 GIS/CSV metadata while keeping Field 2 locked."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml


LOCKED = "locked_field2_external_validation"


def plot_id(path: str) -> int | None:
    match = re.search(r"plot\s*([0-9]+)", Path(path).name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def access_role(relative_path: str, category: str) -> str:
    name = relative_path.lower()
    if (
        category == LOCKED
        or re.search(r"field[ _-]*2|tall[ _-]*grass", name)
        or re.search(r"kusi cubes analysis trials/cube[_ -]*34_georectify", name)
    ):
        return "locked_external_validation"
    if category == "experimental_plot_support" or re.search(
        r"field[ _-]*1", name
    ):
        return "field1_development_candidate"
    return "unassigned_support"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--inventory",
        type=Path,
        help="Defaults to metadata/local/onedrive_archive_inventory.csv",
    )
    args = parser.parse_args()

    try:
        import fiona
    except ImportError as error:
        raise SystemExit(
            "Fiona is required for vector metadata inspection. Install it in the "
            "chickpea_modeling environment with: conda install -c conda-forge fiona"
        ) from error

    config = yaml.safe_load(args.paths.read_text())
    project = Path(config["project_root"])
    source = project / "data" / "OneDrive_2026-07-31_raw"
    local = project / "metadata" / "local"
    inventory_path = args.inventory or local / "onedrive_archive_inventory.csv"
    inventory = pd.read_csv(inventory_path).fillna("")

    if not source.is_dir():
        raise SystemExit(f"Extracted archive directory not found: {source}")

    vector_rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    issues: list[dict[str, str]] = []

    inventory["access_role"] = inventory.apply(
        lambda row: access_role(str(row["relative_path"]), str(row["category"])),
        axis=1,
    )
    # Only explicit Field 1 candidates are opened. Locked and unassigned records
    # cannot influence development decisions.
    field1 = inventory[inventory["access_role"] == "field1_development_candidate"]
    for _, row in field1[field1["extension"] == ".shp"].iterrows():
        relative = str(row["relative_path"])
        path = source / relative
        try:
            with fiona.open(path) as dataset:
                geometry_types = sorted({
                    feature["geometry"]["type"]
                    for feature in dataset
                    if feature["geometry"] is not None
                })
                crs = dataset.crs_wkt or str(dataset.crs)
                bounds = dataset.bounds
                vector_rows.append({
                    "relative_path": relative,
                    "category": row["category"],
                    "plot_id": plot_id(relative),
                    "driver": dataset.driver,
                    "feature_count": len(dataset),
                    "declared_geometry": dataset.schema.get("geometry", ""),
                    "observed_geometry_types": "|".join(geometry_types),
                    "attribute_fields": "|".join(dataset.schema.get("properties", {}).keys()),
                    "crs": crs,
                    "bounds_left": bounds[0],
                    "bounds_bottom": bounds[1],
                    "bounds_right": bounds[2],
                    "bounds_top": bounds[3],
                    "content_role": row["access_role"],
                })
                if not crs:
                    issues.append({"relative_path": relative, "issue": "missing_crs"})
                if len(dataset) == 0:
                    issues.append({"relative_path": relative, "issue": "zero_features"})
        except Exception as error:  # report corrupt/unreadable GIS without stopping batch
            issues.append({"relative_path": relative, "issue": f"read_error: {error}"})

    for _, row in field1[field1["extension"] == ".csv"].iterrows():
        relative = str(row["relative_path"])
        path = source / relative
        try:
            table = pd.read_csv(path)
            csv_rows.append({
                "relative_path": relative,
                "category": row["category"],
                "plot_id": plot_id(relative),
                "rows": len(table),
                "columns": len(table.columns),
                "column_names": "|".join(map(str, table.columns)),
                "content_role": row["access_role"],
            })
        except Exception as error:
            issues.append({"relative_path": relative, "issue": f"csv_read_error: {error}"})

    vector = pd.DataFrame(vector_rows)
    csv = pd.DataFrame(csv_rows)
    issue_table = pd.DataFrame(issues, columns=["relative_path", "issue"])
    vector.to_csv(local / "supporting_gis_qc.csv", index=False)
    csv.to_csv(local / "supporting_csv_qc.csv", index=False)
    issue_table.to_csv(local / "supporting_gis_issues.csv", index=False)

    observed_plots = sorted({int(value) for value in vector["plot_id"].dropna()})
    locked_count = int((inventory["access_role"] == "locked_external_validation").sum())
    unassigned_count = int((inventory["access_role"] == "unassigned_support").sum())
    print(f"Field 1 shapefiles inspected: {len(vector)}")
    print(f"Field 1 CSV files inspected: {len(csv)}")
    print(f"Numbered plot IDs observed: {len(observed_plots)}")
    print(f"Plot IDs: {', '.join(map(str, observed_plots))}")
    print(f"GIS/CSV issues: {len(issue_table)}")
    print(f"Locked records skipped without opening: {locked_count}")
    print(f"Unassigned records skipped without opening: {unassigned_count}")
    print(f"Reports written to: {local}")


if __name__ == "__main__":
    main()
