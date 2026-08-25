#!/usr/bin/env python
"""Visualize every observed Field 1 chickpea mask without other classes."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records


BLACK_GREEN = ListedColormap(["#000000", "#21D65B"])
BLACK_RED = ListedColormap(["#000000", "#EF476F"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def downsample_nearest(mask: np.ndarray, maximum: int) -> np.ndarray:
    step = max(1, math.ceil(max(mask.shape) / maximum))
    return mask[::step, ::step]


def mask_bbox(mask: np.ndarray) -> tuple[int | None, int | None, int | None, int | None]:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None, None, None, None
    return int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--max-overview-dimension", default=600, type=int)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    manifest = local / "authoritative_manifest.csv"
    reports = local / "reports" / "mask_visualization"
    single_output = reports / "chickpea_only"
    comparison_output = reports / "source_vs_authoritative"
    single_output.mkdir(parents=True, exist_ok=True)
    comparison_output.mkdir(parents=True, exist_ok=True)

    records = sorted(
        [record for record in load_records(args.paths, manifest) if record.chickpea_mask],
        key=lambda record: record.cube_id,
    )
    if not records:
        raise ValueError("No Field 1 chickpea masks were found in the authoritative manifest")

    summary_rows = []
    overview_tiles = []
    output_paths = []
    for number, record in enumerate(records, start=1):
        with rasterio.open(record.chickpea_mask) as dataset:
            source = dataset.read(1) > 0
        labels = authoritative_class_map(record)
        authoritative = labels == 1
        if source.shape != authoritative.shape:
            raise ValueError(f"Chickpea mask shape mismatch: {record.cube_id}")
        removed = source & ~authoritative
        added = authoritative & ~source
        if added.any():
            raise ValueError(f"Authoritative processing unexpectedly added chickpea pixels: {record.cube_id}")

        pixels = int(authoritative.sum())
        source_pixels = int(source.sum())
        removed_pixels = int(removed.sum())
        height, width = authoritative.shape
        coverage = pixels / authoritative.size
        row_occupancy = float(np.any(authoritative, axis=1).mean())
        column_occupancy = float(np.any(authoritative, axis=0).mean())
        row_min, row_max, column_min, column_max = mask_bbox(authoritative)

        single_path = single_output / f"{record.cube_id}_chickpea_only.png"
        figure, axis = plt.subplots(figsize=(8, 10), constrained_layout=True)
        axis.imshow(authoritative.astype(np.uint8), cmap=BLACK_GREEN, vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(
            f"{record.cube_id} — authoritative chickpea only\n"
            f"{pixels:,} pixels ({coverage:.2%} of raster)",
            fontsize=13,
        )
        axis.axis("off")
        figure.savefig(single_path, dpi=220, facecolor="white")
        plt.close(figure)
        output_paths.append(single_path)

        comparison_path = comparison_output / f"{record.cube_id}_source_vs_authoritative.png"
        figure, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        axes[0].imshow(source.astype(np.uint8), cmap=BLACK_GREEN, vmin=0, vmax=1, interpolation="nearest")
        axes[0].set_title(f"Source chickpea\n{source_pixels:,} pixels")
        axes[1].imshow(authoritative.astype(np.uint8), cmap=BLACK_GREEN, vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title(f"Authoritative chickpea\n{pixels:,} pixels")
        axes[2].imshow(removed.astype(np.uint8), cmap=BLACK_RED, vmin=0, vmax=1, interpolation="nearest")
        axes[2].set_title(f"Removed by class precedence\n{removed_pixels:,} pixels")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"{record.cube_id} chickpea-mask provenance", fontsize=15)
        figure.savefig(comparison_path, dpi=180, facecolor="white")
        plt.close(figure)
        output_paths.append(comparison_path)

        tile = downsample_nearest(authoritative, args.max_overview_dimension)
        overview_tiles.append((record.cube_id, tile, pixels, coverage))
        summary_rows.append({
            "cube_id": record.cube_id,
            "source_chickpea_pixels": source_pixels,
            "authoritative_chickpea_pixels": pixels,
            "removed_by_class_precedence": removed_pixels,
            "removed_fraction_of_source": removed_pixels / source_pixels if source_pixels else 0.0,
            "raster_height": height,
            "raster_width": width,
            "authoritative_raster_fraction": coverage,
            "rows_containing_chickpea_fraction": row_occupancy,
            "columns_containing_chickpea_fraction": column_occupancy,
            "mask_row_min": row_min,
            "mask_row_max": row_max,
            "mask_column_min": column_min,
            "mask_column_max": column_max,
            "source_mask": str(record.chickpea_mask),
        })
        print(
            f"Rendered {number}/{len(records)} {record.cube_id}: "
            f"{pixels:,} authoritative chickpea pixels",
            flush=True,
        )

    columns = 4
    rows = math.ceil(len(overview_tiles) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.8 * rows), constrained_layout=True)
    flat_axes = np.asarray(axes).reshape(-1)
    for axis in flat_axes:
        axis.set_facecolor("black")
        axis.axis("off")
    for axis, (cube_id, tile, pixels, coverage) in zip(flat_axes, overview_tiles):
        axis.imshow(tile.astype(np.uint8), cmap=BLACK_GREEN, vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"{cube_id}\n{pixels:,} pixels | {coverage:.2%}", color="black", fontsize=11)
    figure.suptitle(
        "Field 1 authoritative chickpea masks only\n"
        "Green = chickpea used by the supervised model; black = every other pixel",
        fontsize=17,
    )
    overview_path = reports / "field1_chickpea_masks_overview.png"
    figure.savefig(overview_path, dpi=220, facecolor="white")
    plt.close(figure)
    output_paths.append(overview_path)

    summary = pd.DataFrame(summary_rows)
    summary_path = reports / "field1_chickpea_mask_summary.csv"
    summary.to_csv(summary_path, index=False)
    output_paths.append(summary_path)

    contract = {
        "status": "field1_chickpea_only_masks_visualized",
        "field": "Field 1",
        "field2_accessed": False,
        "source_masks_modified": False,
        "synthetic_scientific_observations": False,
        "displayed_class": "authoritative chickpea (class_id 1)",
        "class_precedence": "weed > chickpea > soil",
        "cubes_rendered": len(records),
        "output_hashes": {str(path.relative_to(reports)): sha256(path) for path in output_paths},
        "source_hashes": {"authoritative_manifest": sha256(manifest)},
        "notes": [
            "The chickpea-only overview contains no soil, weed, RGB, or synthetic pixels.",
            "Individual provenance panels compare source and authoritative masks without editing either.",
            "PNG downsampling uses nearest-neighbour display only; summary counts use full-resolution masks.",
        ],
    }
    contract_path = local / "contracts" / "field1_chickpea_mask_visualization_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"\nRendered chickpea masks: {len(records)}")
    print(f"Individual masks: {single_output}")
    print(f"Source comparisons: {comparison_output}")
    print(f"Summary: {summary_path}")
    print(f"Overview: {overview_path}")
    print(f"Contract: {contract_path}")
    print("Only observed Field 1 masks were read; no source mask was modified.")


if __name__ == "__main__":
    main()
