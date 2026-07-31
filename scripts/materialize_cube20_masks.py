#!/usr/bin/env python
"""Materialize mutually exclusive Cube 20 masks on its reflectance grid."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import yaml


LABELS = {"soil": 1, "chickpea": 2, "weed": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--decisions", default=Path("configs/data_decisions.yaml"), type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text())
    decisions = yaml.safe_load(args.decisions.read_text())["archive_cube20"]
    project = Path(config["project_root"])
    archive = project / "data" / "OneDrive_2026-07-31_raw"
    source_mask = archive / decisions["categorical_mask"]
    reflectance = archive / decisions["reflectance"]
    destination = project / "data" / "processed" / "field1_cube20"
    local = project / "metadata" / "local"

    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(
            f"Destination is not empty: {destination}\n"
            "No file was overwritten. Review or move the existing processed output."
        )
    destination.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_mask) as source, rasterio.open(reflectance) as target:
        source_classes = source.read(1)
        observed = set(np.unique(source_classes).tolist())
        if not observed.issubset({0, 1, 2, 3}):
            raise SystemExit(f"Unexpected categorical values: {sorted(observed)}")
        class_map = np.zeros((target.height, target.width), dtype=np.uint8)
        reproject(
            source=source_classes,
            destination=class_map,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=target.transform,
            dst_crs=target.crs,
            resampling=Resampling.nearest,
        )
        profile = {
            "driver": "GTiff", "height": target.height, "width": target.width,
            "count": 1, "dtype": "uint8", "crs": target.crs,
            "transform": target.transform, "compress": "deflate", "predictor": 2,
        }

    rows: list[dict[str, object]] = []
    products = {"class_map": class_map}
    products.update({mask_role: (class_map == value).astype(np.uint8)
                     for mask_role, value in LABELS.items()})
    for product_role, array in products.items():
        path = destination / f"{product_role}.tif"
        with rasterio.open(path, "w", **profile) as dataset:
            dataset.write(array, 1)
            dataset.update_tags(
                source_categorical_mask=str(source_mask),
                source_reflectance=str(reflectance),
                transformation="nearest-neighbor reprojection; no synthetic observations",
                label_mapping="0=unlabeled,1=soil,2=chickpea,3=weed",
            )
        rows.append({
            "product_role": product_role,
            "relative_path": str(path.relative_to(project)),
            "pixels_nonzero": int(np.count_nonzero(array)),
            "height": array.shape[0],
            "width": array.shape[1],
            "sha256": sha256(path),
        })

    binary_sum = sum((class_map == value).astype(np.uint8) for value in LABELS.values())
    if int(binary_sum.max()) > 1:
        raise RuntimeError("Materialized class masks are not mutually exclusive.")
    pd.DataFrame(rows).to_csv(local / "archive_cube20_processed_mask_qc.csv", index=False)
    print(f"Processed directory: {destination}")
    print(f"Target grid: {class_map.shape[1]} × {class_map.shape[0]}")
    for mask_role, value in LABELS.items():
        print(f"{mask_role}: {int((class_map == value).sum()):,} pixels")
    print(f"Unlabeled: {int((class_map == 0).sum()):,} pixels")
    print("Class masks are mutually exclusive; raw files were not modified.")
    print(f"QC: {local / 'archive_cube20_processed_mask_qc.csv'}")


if __name__ == "__main__":
    main()
