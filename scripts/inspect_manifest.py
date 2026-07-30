#!/usr/bin/env python
"""Report image dimensions, mask values, and likely pairing problems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from spectral import envi


def envi_shape(path: Path) -> tuple[int, int, int]:
    candidates = (Path(f"{path}.hdr"), path.with_suffix(".hdr"))
    header = next((item for item in candidates if item.is_file()), None)
    if header is None:
        raise FileNotFoundError(f"No ENVI header found for {path}")
    image = envi.open(str(header), str(path))
    return tuple(int(value) for value in image.shape)


def raster_info(path: Path) -> tuple[tuple[int, int, int], list[float] | str]:
    with rasterio.open(path) as src:
        array = src.read()
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            values: list[float] | str = []
        elif finite.size <= 5_000_000:
            unique = np.unique(finite)
            values = unique[:30].astype(float).tolist()
            if unique.size > 30:
                values = f"{unique.size} unique values; first 30: {values}"
        else:
            values = "not enumerated (>5 million finite pixels)"
        return (src.height, src.width, src.count), values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    table = pd.read_csv(args.manifest, dtype=str).fillna("")
    reports: list[dict[str, object]] = []
    roles = (
        "reflectance", "pca", "deriv1", "deriv2", "ndvi",
        "mask_combined", "mask_chickpea", "mask_weed", "mask_soil",
    )
    for _, row in table.iterrows():
        report: dict[str, object] = {
            "cube_id": row["cube_id"], "field_id": row["field_id"]
        }
        spatial_shapes: set[tuple[int, int]] = set()
        for role in roles:
            value = row.get(role, "")
            if not value:
                continue
            path = Path(value)
            try:
                if path.suffix.lower() in {".tif", ".tiff"}:
                    shape, unique = raster_info(path)
                    report[f"{role}_values"] = json.dumps(unique)
                else:
                    shape = envi_shape(path)
                report[f"{role}_shape"] = "x".join(map(str, shape))
                spatial_shapes.add(shape[:2])
            except Exception as exc:  # report all data issues in one pass
                report[f"{role}_error"] = str(exc)
        report["spatial_match"] = len(spatial_shapes) <= 1
        reports.append(report)

    output = pd.DataFrame(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote inspection for {len(output)} cubes to {args.output}")
    print(f"Spatially consistent rows: {int(output['spatial_match'].sum())}/{len(output)}")
    error_columns = [column for column in output if column.endswith("_error")]
    error_count = int(output[error_columns].notna().sum().sum()) if error_columns else 0
    print(f"Read errors: {error_count}")


if __name__ == "__main__":
    main()
