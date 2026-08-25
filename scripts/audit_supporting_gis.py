#!/usr/bin/env python
"""Audit supporting Field 1 GIS/CSV metadata while keeping Field 2 locked."""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

import pandas as pd
import yaml


LOCKED = "locked_field2_external_validation"
SHAPE_TYPES = {
    0: "Null",
    1: "Point",
    3: "PolyLine",
    5: "Polygon",
    8: "MultiPoint",
    11: "PointZ",
    13: "PolyLineZ",
    15: "PolygonZ",
    18: "MultiPointZ",
    21: "PointM",
    23: "PolyLineM",
    25: "PolygonM",
    28: "MultiPointM",
    31: "MultiPatch",
}


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
    if re.search(r"field[ _-]*1", name):
        return "field1_development_candidate"
    if category == "experimental_plot_support":
        return "spatial_assignment_pending"
    return "unassigned_support"


def read_shapefile_metadata(path: Path) -> dict[str, object]:
    """Read SHP/DBF/PRJ metadata without optional GIS dependencies."""
    with path.open("rb") as stream:
        header = stream.read(100)
        if len(header) != 100 or struct.unpack(">i", header[:4])[0] != 9994:
            raise ValueError("invalid SHP header")
        declared_code = struct.unpack("<i", header[32:36])[0]
        bounds = struct.unpack("<4d", header[36:68])
        observed_codes: set[int] = set()
        feature_count = 0
        while record_header := stream.read(8):
            if len(record_header) != 8:
                raise ValueError("truncated SHP record header")
            _, content_words = struct.unpack(">2i", record_header)
            content = stream.read(content_words * 2)
            if len(content) != content_words * 2:
                raise ValueError("truncated SHP record")
            if len(content) >= 4:
                observed_codes.add(struct.unpack("<i", content[:4])[0])
            feature_count += 1

    dbf_path = path.with_suffix(".dbf")
    with dbf_path.open("rb") as stream:
        dbf_header = stream.read(32)
        if len(dbf_header) != 32:
            raise ValueError("invalid DBF header")
        dbf_rows = struct.unpack("<I", dbf_header[4:8])[0]
        header_length = struct.unpack("<H", dbf_header[8:10])[0]
        field_bytes = stream.read(max(header_length - 33, 0))
        fields = []
        for offset in range(0, len(field_bytes), 32):
            descriptor = field_bytes[offset:offset + 32]
            if len(descriptor) < 32 or descriptor[0] == 0x0D:
                break
            name = descriptor[:11].split(b"\x00", 1)[0].decode("latin-1")
            field_type = chr(descriptor[11])
            fields.append(f"{name}:{field_type}")

    prj_path = path.with_suffix(".prj")
    crs = prj_path.read_text(errors="replace").strip() if prj_path.exists() else ""
    return {
        "feature_count": feature_count,
        "dbf_record_count": dbf_rows,
        "declared_geometry": SHAPE_TYPES.get(declared_code, f"Unknown({declared_code})"),
        "observed_geometry_types": "|".join(
            SHAPE_TYPES.get(code, f"Unknown({code})") for code in sorted(observed_codes)
        ),
        "attribute_fields": "|".join(fields),
        "crs": crs,
        "bounds": bounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--inventory",
        type=Path,
        help="Defaults to metadata/local/onedrive_archive_inventory.csv",
    )
    args = parser.parse_args()

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
    inspectable = inventory[inventory["access_role"].isin(
        ["field1_development_candidate", "spatial_assignment_pending"]
    )]
    for _, row in inspectable[inspectable["extension"] == ".shp"].iterrows():
        relative = str(row["relative_path"])
        path = source / relative
        try:
            metadata = read_shapefile_metadata(path)
            bounds = metadata.pop("bounds")
            vector_rows.append({
                "relative_path": relative,
                "category": row["category"],
                "plot_id": plot_id(relative),
                "driver": "ESRI Shapefile",
                **metadata,
                "bounds_left": bounds[0],
                "bounds_bottom": bounds[1],
                "bounds_right": bounds[2],
                "bounds_top": bounds[3],
                "content_role": row["access_role"],
            })
            if not metadata["crs"]:
                issues.append({"relative_path": relative, "issue": "missing_crs"})
            if metadata["feature_count"] == 0:
                issues.append({"relative_path": relative, "issue": "zero_features"})
            if metadata["feature_count"] != metadata["dbf_record_count"]:
                issues.append({
                    "relative_path": relative,
                    "issue": "shp_dbf_record_count_mismatch",
                })
        except Exception as error:  # report corrupt/unreadable GIS without stopping batch
            issues.append({"relative_path": relative, "issue": f"read_error: {error}"})

    for _, row in inspectable[inspectable["extension"] == ".csv"].iterrows():
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
    print(f"Candidate shapefiles inspected: {len(vector)}")
    print(f"Candidate CSV files inspected: {len(csv)}")
    print(f"Numbered plot IDs observed: {len(observed_plots)}")
    print(f"Plot IDs: {', '.join(map(str, observed_plots))}")
    print(f"GIS/CSV issues: {len(issue_table)}")
    print(f"Locked records skipped without opening: {locked_count}")
    print(f"Unassigned records skipped without opening: {unassigned_count}")
    print(f"Reports written to: {local}")


if __name__ == "__main__":
    main()
