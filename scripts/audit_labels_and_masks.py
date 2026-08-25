#!/usr/bin/env python
"""Deep label/mask audit using observed Field 1 data only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_label_column(path: Path) -> tuple[np.ndarray, Counter]:
    chunks: list[np.ndarray] = []
    counts: Counter = Counter()
    for chunk in pd.read_csv(path, usecols=["Label"], chunksize=250_000):
        values = pd.to_numeric(chunk["Label"], errors="coerce").to_numpy()
        finite = values[np.isfinite(values)].astype(np.int16)
        chunks.append(finite)
        counts.update(finite.tolist())
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.int16), counts


def full_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_binary_mask(path: Path, height: int, width: int) -> tuple[np.ndarray | None, list[int]]:
    import rasterio

    with rasterio.open(path) as dataset:
        if dataset.count != 1 or dataset.height != height or dataset.width != width:
            return None, []
        array = dataset.read(1)
    unique = np.unique(array).astype(int).tolist()
    if set(unique).issubset({0, 1, 255}) and len(unique) <= 3:
        return array > 0, unique
    return None, unique


def iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text())
    root = Path(config["project_root"])
    local = root / "metadata" / "local"
    catalog = pd.read_csv(local / "project_catalog.csv", dtype=str).fillna("")
    envi_qc = pd.read_csv(local / "envi_qc.csv", dtype=str).fillna("")
    reflectance = envi_qc[envi_qc["file_role"] == "reflectance_header"].copy()
    reflectance["lines"] = reflectance["lines"].astype(int)
    reflectance["samples"] = reflectance["samples"].astype(int)
    dimensions = reflectance.set_index("cube_id")[["lines", "samples"]].to_dict("index")

    label_root = Path(config["field1"]["label_csvs"])
    mask_root = Path(config["field1"]["masks_emmanuel"])
    label_audit: list[dict[str, object]] = []
    count_match_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []

    label_rows = catalog[catalog["file_role"] == "label_table"]
    for _, label_row in label_rows.iterrows():
        cube = label_row["cube_id_inferred"]
        height = dimensions[cube]["lines"]
        width = dimensions[cube]["samples"]
        label_path = label_root / label_row["relative_path"]
        values, counts = read_label_column(label_path)
        expected = height * width
        matches = len(values) == expected
        label_audit.append({
            "cube_id": cube,
            "label_path": label_row["relative_path"],
            "rows": len(values),
            "expected_pixels": expected,
            "row_count_match": matches,
            "table_type": "full_raster" if matches else "labeled_pixel_table",
            "label_values": json.dumps(sorted(counts)),
            "label_counts": json.dumps(dict(sorted(counts.items()))),
            "sha256": full_sha256(label_path),
        })
        cube_masks = catalog[
            (catalog["cube_id_inferred"] == cube)
            & catalog["file_role"].isin(["chickpea_mask", "weed_mask", "soil_mask"])
            & catalog["extension"].isin([".tif", ".tiff"])
        ]
        valid_masks: list[tuple[str, str, np.ndarray]] = []
        for _, mask_row in cube_masks.iterrows():
            mask, unique = read_binary_mask(
                mask_root / mask_row["relative_path"], height, width
            )
            if mask is None:
                count_match_rows.append({
                    "cube_id": cube, "label_value": "", "mask_role": mask_row["file_role"],
                    "mask_path": mask_row["relative_path"], "label_count": "",
                    "mask_positive_pixels": "", "absolute_difference": "",
                    "relative_difference": "",
                    "status": f"invalid_binary_mask:{unique[:20]}",
                })
                continue
            valid_masks.append((mask_row["file_role"], mask_row["relative_path"], mask))
            for label_value in sorted(counts):
                label_count = counts[label_value]
                mask_count = int(mask.sum())
                count_match_rows.append({
                    "cube_id": cube,
                    "label_value": label_value,
                    "mask_role": mask_row["file_role"],
                    "mask_path": mask_row["relative_path"],
                    "label_count": label_count,
                    "mask_positive_pixels": mask_count,
                    "absolute_difference": abs(label_count - mask_count),
                    "relative_difference": abs(label_count - mask_count) / max(mask_count, 1),
                    "status": "ok",
                })
        for index, (role_a, path_a, mask_a) in enumerate(valid_masks):
            for role_b, path_b, mask_b in valid_masks[index + 1:]:
                if role_a == role_b:
                    variant_rows.append({
                        "cube_id": cube, "mask_role": role_a,
                        "mask_a": path_a, "mask_b": path_b,
                        "iou": iou(mask_a, mask_b),
                        "identical": bool(np.array_equal(mask_a, mask_b)),
                    })

    label_frame = pd.DataFrame(label_audit)
    count_match_frame = pd.DataFrame(count_match_rows)
    variant_frame = pd.DataFrame(
        variant_rows,
        columns=["cube_id", "mask_role", "mask_a", "mask_b", "iou", "identical"],
    )
    label_frame.to_csv(local / "label_deep_audit.csv", index=False)
    count_match_frame.to_csv(local / "label_mask_count_match.csv", index=False)
    variant_frame.to_csv(local / "mask_variant_comparison.csv", index=False)
    hash_duplicates = label_frame[
        label_frame.duplicated("sha256", keep=False)
    ].sort_values(["sha256", "cube_id"])
    hash_duplicates.to_csv(local / "label_sha256_duplicates.csv", index=False)

    print(f"Label tables audited: {len(label_frame)}")
    print("Table structure: labeled-pixel tables (not full raster tables)")
    print(f"Label-to-mask count comparisons: {len(count_match_frame)}")
    print(f"Same-role mask variant comparisons: {len(variant_frame)}")
    print(f"Rows in full-SHA duplicate groups: {len(hash_duplicates)}")
    print(f"Reports written to: {local}")


if __name__ == "__main__":
    main()
