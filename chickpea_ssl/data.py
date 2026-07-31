"""Authoritative observed-data access for Pika-L cubes and masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from spectral.io import envi
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CubeRecord:
    cube_id: str
    acquisition_date: str
    header: Path
    data: Path
    chickpea_mask: Path | None
    weed_mask: Path | None
    soil_mask: Path | None


def load_records(paths_file: Path, manifest_file: Path) -> list[CubeRecord]:
    config = yaml.safe_load(paths_file.read_text())
    manifest = pd.read_csv(manifest_file, dtype=str).fillna("")
    mask_root = Path(config["field1"]["masks_emmanuel"])
    records: list[CubeRecord] = []
    for _, row in manifest.iterrows():
        date_root = Path(config["field1"]["reflectance_dates"][row["acquisition_date"]])
        def optional_mask(role: str) -> Path | None:
            return mask_root / row[role] if row[role] else None
        records.append(CubeRecord(
            cube_id=row["cube_id"], acquisition_date=row["acquisition_date"],
            header=date_root / row["reflectance_header"],
            data=date_root / row["reflectance_bip"],
            chickpea_mask=optional_mask("chickpea_mask"),
            weed_mask=optional_mask("weed_mask"),
            soil_mask=optional_mask("soil_mask"),
        ))
    return records


def load_band_indices(bands_file: Path, section: str = "primary") -> np.ndarray:
    configuration = yaml.safe_load(bands_file.read_text())[section]
    first = int(configuration["first_band_index"])
    last = int(configuration["last_band_index_inclusive"])
    return np.arange(first, last + 1, dtype=int)


class EnviCube:
    def __init__(self, record: CubeRecord, band_indices: np.ndarray | None = None):
        self.record = record
        self.image = envi.open(str(record.header), str(record.data))
        self.array = self.image.open_memmap()
        self.wavelengths = np.asarray(self.image.metadata["wavelength"], dtype=np.float32)
        self.band_indices = (
            np.arange(self.array.shape[-1]) if band_indices is None
            else np.asarray(band_indices, dtype=int)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.array.shape[0], self.array.shape[1], len(self.band_indices)

    def patch(self, row: int, column: int, size: int) -> np.ndarray:
        radius = size // 2
        if size % 2 != 1:
            raise ValueError("Patch size must be odd")
        if row < radius or column < radius or row + radius >= self.array.shape[0] or column + radius >= self.array.shape[1]:
            raise IndexError("Patch extends beyond cube boundary")
        patch = self.array[
            row - radius:row + radius + 1,
            column - radius:column + radius + 1,
            :,
        ][..., self.band_indices]
        return np.asarray(patch, dtype=np.float32)


def authoritative_class_map(record: CubeRecord) -> np.ndarray:
    """Return -1 unlabeled, 0 soil, 1 chickpea, 2 weed.

    Assignment order implements the evidence-backed precedence
    weed > chickpea > soil without editing source rasters.
    """
    import rasterio
    image = envi.open(str(record.header), str(record.data))
    height, width = image.shape[:2]
    labels = np.full((height, width), -1, dtype=np.int8)
    for value, path in ((0, record.soil_mask), (1, record.chickpea_mask), (2, record.weed_mask)):
        if path is None:
            continue
        with rasterio.open(path) as dataset:
            if (dataset.height, dataset.width) != (height, width):
                raise ValueError(f"Mask shape mismatch: {path}")
            labels[dataset.read(1) > 0] = value
    return labels


def spatial_block_ids(height: int, width: int, block_size: int = 256) -> np.ndarray:
    rows = np.arange(height)[:, None] // block_size
    columns = np.arange(width)[None, :] // block_size
    blocks_per_row = int(np.ceil(width / block_size))
    return rows * blocks_per_row + columns


class SpectralPatchDataset(Dataset):
    """Patch dataset backed by observed ENVI memmaps; creates no synthetic samples."""
    def __init__(self, cube: EnviCube, centers: np.ndarray, labels: np.ndarray,
                 patch_size: int = 15, mean: np.ndarray | None = None,
                 std: np.ndarray | None = None):
        self.cube = cube
        self.centers = np.asarray(centers, dtype=int)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        if len(self.centers) != len(self.labels):
            raise ValueError("centers and labels must have equal length")

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row, column = self.centers[index]
        patch = self.cube.patch(int(row), int(column), self.patch_size)
        if self.mean is not None and self.std is not None:
            patch = (patch - self.mean[None, None, :]) / np.maximum(self.std[None, None, :], 1e-6)
        patch = np.moveaxis(patch, -1, 0).copy()
        return torch.from_numpy(patch), torch.tensor(self.labels[index], dtype=torch.long)
