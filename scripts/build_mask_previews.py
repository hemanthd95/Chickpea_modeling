#!/usr/bin/env python
"""Create observed-data RGB/mask QA previews; never edits source files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


COLORS = {
    "chickpea_mask": (0.1, 1.0, 0.1),
    "weed_mask": (1.0, 0.1, 0.8),
    "soil_mask": (1.0, 0.65, 0.1),
}


def root_for(config: dict, source: str, processing_batch: str) -> Path:
    if source == "field1.reflectance":
        field1 = config["field1"]
        batches = field1.get("reflectance_processing_batches", field1.get("reflectance_dates", {}))
        return Path(batches[str(processing_batch)])
    return Path(config["field1"]["masks_emmanuel"])


def stretch(rgb: np.ndarray) -> np.ndarray:
    output = np.empty_like(rgb, dtype=np.float32)
    for index in range(3):
        channel = rgb[..., index].astype(np.float32)
        valid = channel[np.isfinite(channel) & (channel > 0)]
        low, high = np.percentile(valid, (2, 98)) if len(valid) else (0, 1)
        output[..., index] = np.clip((channel - low) / max(high - low, 1), 0, 1)
    return output


def load_rgb(header: Path, max_size: int = 900) -> tuple[np.ndarray, int]:
    from spectral.io import envi

    image = envi.open(str(header), str(Path(str(header)[:-4])))
    wavelengths = np.asarray(image.metadata["wavelength"], dtype=float)
    indices = [int(np.argmin(abs(wavelengths - target))) for target in (650, 550, 470)]
    rows, columns, _ = image.shape
    step = max(1, math.ceil(max(rows, columns) / max_size))
    memory = image.open_memmap()
    rgb = np.stack([memory[::step, ::step, index] for index in indices], axis=-1)
    return stretch(rgb), step


def load_mask(path: Path, out_height: int, out_width: int) -> np.ndarray | None:
    import rasterio

    with rasterio.open(path) as dataset:
        if dataset.count != 1:
            return None
        array = dataset.read(
            1, out_shape=(out_height, out_width),
            resampling=rasterio.enums.Resampling.nearest,
        )
        if not set(np.unique(array).tolist()).issubset({0, 1, 255}):
            return None
        return array > 0


def overlay(rgb: np.ndarray, masks: list[tuple[str, np.ndarray]]) -> np.ndarray:
    result = rgb.copy()
    for role, mask in masks:
        color = np.asarray(COLORS[role], dtype=np.float32)
        interior = (
            np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0)
            & np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1)
        )
        boundary = mask & ~interior
        result[boundary] = 0.15 * result[boundary] + 0.85 * color
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    config = yaml.safe_load(args.paths.read_text())
    project = Path(config["project_root"])
    local = project / "metadata" / "local"
    output = local / "mask_previews"
    output.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(local / "project_catalog.csv", dtype=str).fillna("")

    completed: list[dict[str, object]] = []
    overview_tiles: list[tuple[str, np.ndarray]] = []
    label_cubes = sorted(catalog.loc[catalog["file_role"] == "label_table", "cube_id_inferred"].unique())
    for cube in label_cubes:
        reflectance = catalog[
            (catalog["cube_id_inferred"] == cube)
            & (catalog["file_role"] == "reflectance_header")
        ].iloc[0]
        header = root_for(config, reflectance["source"], reflectance.get("processing_batch", reflectance["acquisition_date"])) / reflectance["relative_path"]
        rgb, _ = load_rgb(header)
        cube_masks = catalog[
            (catalog["cube_id_inferred"] == cube)
            & catalog["file_role"].isin(COLORS)
            & catalog["extension"].isin([".tif", ".tiff"])
        ]
        valid: list[tuple[str, np.ndarray]] = []
        skipped: list[str] = []
        for _, row in cube_masks.iterrows():
            mask_path = root_for(config, row["source"], row.get("processing_batch", row["acquisition_date"])) / row["relative_path"]
            mask = load_mask(mask_path, rgb.shape[0], rgb.shape[1])
            if mask is None:
                skipped.append(row["relative_path"])
            else:
                valid.append((row["file_role"], mask))
        preview = overlay(rgb, valid)
        figure, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        axes[0].imshow(rgb); axes[0].set_title(f"{cube}: RGB 650/550/470 nm")
        axes[1].imshow(preview); axes[1].set_title("Contours: chickpea green, weed magenta, soil orange")
        for axis in axes: axis.axis("off")
        figure.savefig(output / f"{cube}.png", dpi=160)
        plt.close(figure)
        overview_step = max(1, math.ceil(max(preview.shape[:2]) / 320))
        overview_tiles.append((cube, preview[::overview_step, ::overview_step]))
        completed.append({"cube_id": cube, "valid_mask_files": len(valid),
                          "skipped_mask_files": len(skipped), "skipped": " | ".join(skipped)})
    pd.DataFrame(completed).to_csv(local / "mask_preview_summary.csv", index=False)
    rows = math.ceil(len(overview_tiles) / 3)
    overview, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows), constrained_layout=True)
    for axis in np.asarray(axes).reshape(-1):
        axis.axis("off")
    for axis, (cube, tile) in zip(np.asarray(axes).reshape(-1), overview_tiles):
        axis.imshow(tile)
        axis.set_title(cube)
    overview.savefig(local / "mask_preview_overview.png", dpi=160)
    plt.close(overview)
    print(f"Created {len(completed)} cube previews in {output}")
    print(f"Summary: {local / 'mask_preview_summary.csv'}")
    print(f"Overview: {local / 'mask_preview_overview.png'}")


if __name__ == "__main__":
    main()
