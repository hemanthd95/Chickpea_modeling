#!/usr/bin/env python
"""Compute leakage-safe, fold-specific spectral normalization statistics."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.data import EnviCube, load_band_indices, load_records
from chickpea_ssl.spatial import map_block_indices


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rng(seed: int, cube_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{cube_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def boundary_safe_mask(
    x: np.ndarray, y: np.ndarray, block_x: np.ndarray, block_y: np.ndarray,
    fold_lookup: dict[tuple[int, int], int], heldout_fold: int,
    block_size: float, origin_x: float, origin_y: float, buffer_m: float,
) -> np.ndarray:
    """Exclude training centres close enough for a patch to cross into heldout groups."""
    safe = np.ones(len(x), dtype=bool)
    local_x = x - (origin_x + block_x * block_size)
    local_y = y - (origin_y + block_y * block_size)
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            if offset_x == 0 and offset_y == 0:
                continue
            neighbour_fold = np.fromiter((
                fold_lookup.get((int(one_x + offset_x), int(one_y + offset_y)), 0)
                for one_x, one_y in zip(block_x, block_y)
            ), dtype=np.int8, count=len(block_x))
            touches_heldout = neighbour_fold == heldout_fold
            if not touches_heldout.any():
                continue
            distance_x = (
                local_x if offset_x < 0 else
                block_size - local_x if offset_x > 0 else np.zeros_like(local_x)
            )
            distance_y = (
                local_y if offset_y < 0 else
                block_size - local_y if offset_y > 0 else np.zeros_like(local_y)
            )
            safe &= ~(touches_heldout & (np.hypot(distance_x, distance_y) <= buffer_m))
    return safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--normalization", default=Path("configs/normalization.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    parser.add_argument("--spatial", default=Path("configs/spatial_splits.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    normal = yaml.safe_load(args.normalization.read_text())
    spatial = yaml.safe_load(args.spatial.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reports = local / "reports" / "normalization"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = local / "authoritative_manifest.csv"
    fold_file = contracts / "field1_spatial_folds.csv"
    fold_contract_file = contracts / "field1_spatial_fold_contract.yaml"
    for required in (manifest, fold_file, fold_contract_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    fold_contract = yaml.safe_load(fold_contract_file.read_text())
    if sha256(fold_file) != fold_contract["assignment_sha256"]:
        raise ValueError("Spatial-fold assignment hash does not match its frozen contract")

    assignments = pd.read_csv(fold_file)
    fold_lookup = {
        (int(row.block_x), int(row.block_y)): int(row.fold)
        for row in assignments.itertuples()
    }
    folds = sorted(assignments["fold"].unique())
    grouping = spatial["grouping"]
    block_size = float(grouping["block_size_m"])
    origin_x = float(grouping.get("origin_x_m", 0.0))
    origin_y = float(grouping.get("origin_y_m", 0.0))
    buffer_m = float(normal["boundary_exclusion_m"])
    if buffer_m != float(fold_contract["boundary_exclusion_m"]):
        raise ValueError("Normalization boundary exclusion differs from frozen fold contract")
    band_indices = load_band_indices(args.bands, normal["band_section"])
    band_count = len(band_indices)
    sums = {fold: np.zeros(band_count, dtype=np.float64) for fold in folds}
    sums_squared = {fold: np.zeros(band_count, dtype=np.float64) for fold in folds}
    minima = {fold: np.full(band_count, np.inf) for fold in folds}
    maxima = {fold: np.full(band_count, -np.inf) for fold in folds}
    counts = {fold: 0 for fold in folds}
    heldout_counts = {fold: 0 for fold in folds}
    boundary_counts = {fold: 0 for fold in folds}
    cube_counts = {fold: 0 for fold in folds}
    wavelengths: np.ndarray | None = None
    total_sampled = 0

    for record in load_records(args.paths, manifest):
        cube = EnviCube(record, band_indices)
        if wavelengths is None:
            wavelengths = cube.wavelengths[band_indices]
        elif not np.allclose(wavelengths, cube.wavelengths[band_indices]):
            raise ValueError(f"Wavelength mismatch: {record.cube_id}")
        valid = np.any(cube.array > 0, axis=2)
        coordinates = np.argwhere(valid)
        sample_count = min(int(normal["samples_per_cube"]), len(coordinates))
        rng = stable_rng(int(normal["seed"]), record.cube_id)
        selected = coordinates[rng.choice(len(coordinates), size=sample_count, replace=False)]
        spectra = np.asarray(
            cube.array[selected[:, 0], selected[:, 1]][:, band_indices], dtype=np.float64
        )
        with rasterio.open(record.data) as dataset:
            transform = dataset.transform
        rows, columns = selected[:, 0], selected[:, 1]
        x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
        y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f
        block_x, block_y = map_block_indices(
            x, y, block_size, origin_x, origin_y
        )
        sample_folds = np.fromiter((
            fold_lookup.get((int(one_x), int(one_y)), 0)
            for one_x, one_y in zip(block_x, block_y)
        ), dtype=np.int8, count=sample_count)
        if (sample_folds == 0).any():
            raise ValueError(f"Sampled pixel outside frozen groups: {record.cube_id}")
        total_sampled += sample_count
        for heldout in folds:
            heldout_mask = sample_folds == heldout
            safe = boundary_safe_mask(
                x, y, block_x, block_y, fold_lookup, int(heldout), block_size,
                origin_x, origin_y, buffer_m,
            )
            boundary_removed = (~heldout_mask) & (~safe)
            training = (~heldout_mask) & safe
            heldout_counts[heldout] += int(heldout_mask.sum())
            boundary_counts[heldout] += int(boundary_removed.sum())
            if not training.any():
                continue
            values = spectra[training]
            counts[heldout] += len(values)
            cube_counts[heldout] += 1
            sums[heldout] += values.sum(axis=0)
            sums_squared[heldout] += np.square(values).sum(axis=0)
            minima[heldout] = np.minimum(minima[heldout], values.min(axis=0))
            maxima[heldout] = np.maximum(maxima[heldout], values.max(axis=0))

    rows = []
    summary_rows = []
    for heldout in folds:
        mean = sums[heldout] / counts[heldout]
        variance = np.maximum(sums_squared[heldout] / counts[heldout] - np.square(mean), 0)
        standard_deviation = np.sqrt(variance)
        for position, band_index in enumerate(band_indices):
            rows.append({
                "heldout_fold": int(heldout), "band_index": int(band_index),
                "wavelength_nm": float(wavelengths[position]),
                "training_sample_count": counts[heldout],
                "mean": mean[position], "standard_deviation": standard_deviation[position],
                "minimum": minima[heldout][position], "maximum": maxima[heldout][position],
            })
        summary_rows.append({
            "heldout_fold": int(heldout), "total_cube_balanced_samples": total_sampled,
            "heldout_samples_excluded": heldout_counts[heldout],
            "boundary_samples_excluded": boundary_counts[heldout],
            "training_samples": counts[heldout], "contributing_cubes": cube_counts[heldout],
        })
    statistics = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    statistics.to_csv(contracts / "field1_fold_normalization.csv", index=False)
    summary.to_csv(reports / "fold_normalization_summary.csv", index=False)
    statistics.to_csv(reports / "fold_normalization_diagnostics.csv", index=False)

    contract = {
        "status": "frozen_training_only_normalization",
        "field": "Field 1", "field2_accessed": False,
        "mode": normal["mode"], "seed": int(normal["seed"]),
        "samples_per_cube": int(normal["samples_per_cube"]),
        "band_section": normal["band_section"], "band_count": band_count,
        "boundary_exclusion_m": buffer_m,
        "standard_deviation": normal["standard_deviation"],
        "fold_assignment_sha256": sha256(fold_file),
        "normalization_csv_sha256": sha256(contracts / "field1_fold_normalization.csv"),
        "configuration_sha256": {
            "normalization": sha256(args.normalization), "spectral_bands": sha256(args.bands),
            "spatial_splits": sha256(args.spatial), "manifest": sha256(manifest),
        },
    }
    contract_path = contracts / "field1_fold_normalization_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for heldout, frame in statistics.groupby("heldout_fold"):
        axes[0].plot(frame["wavelength_nm"], frame["mean"], label=f"Hold out fold {heldout}")
        axes[1].plot(frame["wavelength_nm"], frame["standard_deviation"], label=f"Hold out fold {heldout}")
    axes[0].set_title("Training-only spectral means")
    axes[1].set_title("Training-only spectral standard deviations")
    for axis in axes:
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Stored reflectance value (uint16 scale)")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.suptitle("Field 1 fold-specific normalization diagnostics", fontsize=14)
    preview = reports / "fold_normalization_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)
    print(f"Cube-balanced spectra sampled: {total_sampled:,}")
    print(summary.to_string(index=False))
    print(f"Contract: {contract_path}")
    print(f"Visual QC: {preview}")
    print("Statistics use training groups only; Field 2 was not accessed.")


if __name__ == "__main__":
    main()
