#!/usr/bin/env python
"""Compare Cube 20 binary masks against its categorical mask after alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import yaml


LABELS = {"soil": 1, "chickpea": 2, "weed": 3}


def align_binary(path: Path, reference: rasterio.io.DatasetReader) -> np.ndarray:
    with rasterio.open(path) as source:
        raw = source.read(1)
        binary = (raw != 0).astype(np.uint8)
        destination = np.zeros((reference.height, reference.width), dtype=np.uint8)
        reproject(
            source=binary,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            resampling=Resampling.nearest,
        )
        return destination.astype(bool)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    intersection = int(np.count_nonzero(reference & candidate))
    union = int(np.count_nonzero(reference | candidate))
    candidate_count = int(candidate.sum())
    reference_count = int(reference.sum())
    return {
        "reference_pixels": reference_count,
        "candidate_pixels": candidate_count,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": intersection / union if union else 1.0,
        "precision_vs_categorical": intersection / candidate_count if candidate_count else 0.0,
        "recall_vs_categorical": intersection / reference_count if reference_count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text())
    project = Path(config["project_root"])
    trial = project / "data" / "OneDrive_2026-07-31_raw" / "Kusi chickpea project" / "Kusi cubes analysis trials"
    local = project / "metadata" / "local"
    categorical_path = trial / "Field_1_Cube20_combined_mask.tiff"
    reflectance_path = trial / "kusi_field1_target_Pika_L-GigE_20.bil-Georectify.bip"
    candidates = {
        "weed_original": ("weed", trial / "Field1_Cube_20_weed_mask.tiff"),
        "weed_new": ("weed", trial / "Field1_Cube_20_weed_mask_new.tiff"),
        "chickpea_alternate_grid": ("chickpea", trial / "kusi_field1__20-Georectify_qgis_chickpea_Mask.bip"),
        "chickpea_reflectance_grid": ("chickpea", trial / "Pre_processed_Field1_20_chickpea_mask.bip"),
        "soil_original": ("soil", trial / "soil_mask_Field1_Cube20.tiff"),
        "soil_gps": ("soil", trial / "soil_mask_Field1_Cube20_gps.tiff"),
    }

    rows: list[dict[str, object]] = []
    with rasterio.open(categorical_path) as categorical:
        labels = categorical.read(1)
        if not set(np.unique(labels)).issubset({0, 1, 2, 3}):
            raise SystemExit("Selected categorical mask contains values outside 0,1,2,3.")
        for name, (mask_role, path) in candidates.items():
            candidate = align_binary(path, categorical)
            row = {
                "candidate_name": name,
                "role": mask_role,
                "relative_path": str(path.relative_to(project)),
                **metrics(labels == LABELS[mask_role], candidate),
            }
            rows.append(row)

    target_rows: list[dict[str, object]] = []
    with rasterio.open(categorical_path) as categorical, rasterio.open(reflectance_path) as target:
        for mask_role, value in LABELS.items():
            source = (categorical.read(1) == value).astype(np.uint8)
            destination = np.zeros((target.height, target.width), dtype=np.uint8)
            reproject(
                source=source, destination=destination,
                src_transform=categorical.transform, src_crs=categorical.crs,
                dst_transform=target.transform, dst_crs=target.crs,
                resampling=Resampling.nearest,
            )
            target_rows.append({
                "role": mask_role,
                "categorical_label": value,
                "source_pixels": int(source.sum()),
                "reflectance_grid_pixels": int(destination.sum()),
                "target_height": target.height,
                "target_width": target.width,
            })

    agreement = pd.DataFrame(rows).sort_values(["role", "iou"], ascending=[True, False])
    agreement.to_csv(local / "archive_cube20_categorical_agreement.csv", index=False)
    pd.DataFrame(target_rows).to_csv(
        local / "archive_cube20_reprojected_class_counts.csv", index=False
    )
    print(agreement[["candidate_name", "role", "iou", "precision_vs_categorical", "recall_vs_categorical"]].to_string(index=False))
    print("No processed mask was written; this step is comparison only.")
    print(f"Reports written to: {local}")


if __name__ == "__main__":
    main()
