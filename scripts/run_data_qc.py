#!/usr/bin/env python
"""Metadata-only QC for Field 1 reflectance, masks, and label tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ENVI_BYTES = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 6: 8, 9: 16,
              12: 2, 13: 4, 14: 8, 15: 8}


def source_root(config: dict, source: str, processing_batch: str) -> Path:
    field1 = config["field1"]
    if source == "field1.reflectance":
        batches = field1.get("reflectance_processing_batches", field1.get("reflectance_dates", {}))
        return Path(batches[str(processing_batch)])
    mapping = {
        "field1.derived_ssl": field1["derived_ssl"],
        "field1.label_csvs": field1["label_csvs"],
        "field1.masks_emmanuel": field1["masks_emmanuel"],
    }
    return Path(mapping[source])


def scalar(metadata: dict, key: str, default: int = 0) -> int:
    return int(str(metadata.get(key, default)).strip())


def inspect_envi(row: pd.Series, config: dict) -> dict[str, object]:
    from spectral.io import envi

    header = source_root(config, row["source"], row.get("processing_batch", row["acquisition_date"])) / row["relative_path"]
    metadata = envi.read_envi_header(str(header))
    lines, samples, bands = (scalar(metadata, key) for key in ("lines", "samples", "bands"))
    dtype_code = scalar(metadata, "data type")
    offset = scalar(metadata, "header offset")
    data_path = Path(str(header)[:-4]) if str(header).lower().endswith(".hdr") else header.with_suffix("")
    expected = lines * samples * bands * ENVI_BYTES.get(dtype_code, 0) + offset
    actual = data_path.stat().st_size if data_path.is_file() else -1
    wavelengths = metadata.get("wavelength", [])
    wavelength_values = [float(value) for value in wavelengths] if wavelengths else []
    return {
        "cube_id": row["cube_id_inferred"],
        "acquisition_date": row["acquisition_date"],
        "processing_batch": row.get("processing_batch", ""),
        "file_role": row["file_role"],
        "header": row["relative_path"],
        "lines": lines,
        "samples": samples,
        "bands": bands,
        "interleave": str(metadata.get("interleave", "")).lower(),
        "data_type": dtype_code,
        "byte_order": scalar(metadata, "byte order"),
        "header_offset": offset,
        "wavelength_count": len(wavelength_values),
        "wavelength_min": min(wavelength_values) if wavelength_values else np.nan,
        "wavelength_max": max(wavelength_values) if wavelength_values else np.nan,
        "has_map_info": bool(metadata.get("map info")),
        "has_coordinate_system": bool(metadata.get("coordinate system string")),
        "expected_bytes": expected,
        "actual_bytes": actual,
        "byte_size_match": expected == actual,
        "data_file_exists": data_path.is_file(),
    }


def inspect_mask(row: pd.Series, config: dict) -> dict[str, object]:
    import rasterio

    path = source_root(config, row["source"], row.get("processing_batch", row["acquisition_date"])) / row["relative_path"]
    with rasterio.open(path) as dataset:
        scale = max(dataset.height / 512, dataset.width / 512, 1)
        out_height = max(1, round(dataset.height / scale))
        out_width = max(1, round(dataset.width / scale))
        sample = dataset.read(
            1, out_shape=(out_height, out_width), masked=True,
            resampling=rasterio.enums.Resampling.nearest,
        )
        values = np.unique(sample.compressed())
        return {
            "cube_id": row["cube_id_inferred"],
            "file_role": row["file_role"],
            "relative_path": row["relative_path"],
            "height": dataset.height,
            "width": dataset.width,
            "bands": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs or ""),
            "nodata": dataset.nodata,
            "sample_unique_count": len(values),
            "sample_unique_values": json.dumps(values[:50].tolist()),
        }


def inspect_label(row: pd.Series, config: dict) -> dict[str, object]:
    path = source_root(config, row["source"], row.get("processing_batch", row["acquisition_date"])) / row["relative_path"]
    sample = pd.read_csv(path, nrows=1000)
    candidate_columns = [
        column for column in sample.columns
        if any(token in column.lower() for token in ("label", "class", "mask", "weed", "soil", "chick"))
    ]
    candidate_values = {
        column: sample[column].dropna().astype(str).value_counts().head(20).to_dict()
        for column in candidate_columns
    }
    return {
        "cube_id": row["cube_id_inferred"],
        "relative_path": row["relative_path"],
        "size_bytes": row["size_bytes"],
        "column_count": len(sample.columns),
        "columns": json.dumps(sample.columns.tolist()),
        "candidate_label_columns": json.dumps(candidate_columns),
        "candidate_sample_values": json.dumps(candidate_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text())
    output = Path(config["project_root"]) / "metadata" / "local"
    catalog_path = args.catalog or output / "project_catalog.csv"
    catalog = pd.read_csv(catalog_path, dtype=str).fillna("")
    catalog["size_bytes"] = pd.to_numeric(catalog["size_bytes"], errors="coerce").fillna(0).astype(int)

    issues: list[dict[str, str]] = []
    envi_rows: list[dict[str, object]] = []
    headers = catalog[
        catalog["file_role"].isin(["reflectance_header", "source_mask_header"])
    ]
    for _, row in headers.iterrows():
        try:
            result = inspect_envi(row, config)
            envi_rows.append(result)
            if not result["byte_size_match"]:
                issues.append({"severity": "error", "cube_id": result["cube_id"],
                               "check": "envi_byte_size", "detail": result["header"]})
            if result["file_role"] == "reflectance_header" and result["wavelength_count"] != result["bands"]:
                issues.append({"severity": "error", "cube_id": result["cube_id"],
                               "check": "wavelength_count", "detail": result["header"]})
        except Exception as exc:
            issues.append({"severity": "error", "cube_id": row["cube_id_inferred"],
                           "check": "envi_read", "detail": f'{row["relative_path"]}: {exc}'})

    mask_rows: list[dict[str, object]] = []
    masks = catalog[(catalog["source"] == "field1.masks_emmanuel") &
                    catalog["extension"].isin([".tif", ".tiff"])]
    for _, row in masks.iterrows():
        try:
            mask_rows.append(inspect_mask(row, config))
        except Exception as exc:
            issues.append({"severity": "error", "cube_id": row["cube_id_inferred"],
                           "check": "mask_read", "detail": f'{row["relative_path"]}: {exc}'})

    label_rows: list[dict[str, object]] = []
    labels = catalog[catalog["file_role"] == "label_table"]
    for _, row in labels.iterrows():
        try:
            label_rows.append(inspect_label(row, config))
        except Exception as exc:
            issues.append({"severity": "error", "cube_id": row["cube_id_inferred"],
                           "check": "label_read", "detail": f'{row["relative_path"]}: {exc}'})

    duplicates = catalog[catalog.duplicated("fingerprint", keep=False)]
    for fingerprint, group in duplicates.groupby("fingerprint"):
        if group["cube_id_inferred"].nunique() > 1:
            issues.append({
                "severity": "critical", "cube_id": "|".join(sorted(group["cube_id_inferred"].unique())),
                "check": "cross_cube_duplicate", "detail": " | ".join(group["relative_path"]),
            })

    pd.DataFrame(envi_rows).to_csv(output / "envi_qc.csv", index=False)
    pd.DataFrame(mask_rows).to_csv(output / "mask_qc.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output / "label_csv_qc.csv", index=False)
    pd.DataFrame(issues, columns=["severity", "cube_id", "check", "detail"]).to_csv(
        output / "qc_issues.csv", index=False
    )
    print(f"ENVI headers inspected: {len(envi_rows)}/{len(headers)}")
    print(f"Mask rasters inspected: {len(mask_rows)}/{len(masks)}")
    print(f"Label tables sampled: {len(label_rows)}/{len(labels)}")
    print(f"QC issues: {len(issues)}")
    print(f"Reports written to: {output}")


if __name__ == "__main__":
    main()
