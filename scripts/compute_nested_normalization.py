#!/usr/bin/env python
"""Compute nested training-only normalization excluding outer and inner spatial folds."""

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
    x: np.ndarray,
    y: np.ndarray,
    block_x: np.ndarray,
    block_y: np.ndarray,
    fold_lookup: dict[tuple[int, int], int],
    excluded_fold: int,
    block_size: float,
    origin_x: float,
    origin_y: float,
    buffer_m: float,
) -> np.ndarray:
    safe = np.ones(len(x), dtype=bool)
    local_x = x - (origin_x + block_x * block_size)
    local_y = y - (origin_y + block_y * block_size)
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            if offset_x == 0 and offset_y == 0:
                continue
            neighbour_fold = np.fromiter(
                (
                    fold_lookup.get(
                        (int(one_x + offset_x), int(one_y + offset_y)), 0
                    )
                    for one_x, one_y in zip(block_x, block_y)
                ),
                dtype=np.int8,
                count=len(block_x),
            )
            touches = neighbour_fold == excluded_fold
            if not touches.any():
                continue
            distance_x = (
                local_x
                if offset_x < 0
                else block_size - local_x
                if offset_x > 0
                else np.zeros_like(local_x)
            )
            distance_y = (
                local_y
                if offset_y < 0
                else block_size - local_y
                if offset_y > 0
                else np.zeros_like(local_y)
            )
            safe &= ~(touches & (np.hypot(distance_x, distance_y) <= buffer_m))
    return safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--normalization", default=Path("configs/normalization.yaml"), type=Path
    )
    parser.add_argument(
        "--bands", default=Path("configs/spectral_bands.yaml"), type=Path
    )
    parser.add_argument(
        "--spatial", default=Path("configs/spatial_splits.yaml"), type=Path
    )
    parser.add_argument(
        "--protocol",
        default=Path("configs/supervised_primary_evaluation.yaml"),
        type=Path,
    )
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    normal = yaml.safe_load(args.normalization.read_text())
    spatial = yaml.safe_load(args.spatial.read_text())
    protocol_config = yaml.safe_load(args.protocol.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reports = local / "reports" / "nested_normalization"
    reports.mkdir(parents=True, exist_ok=True)

    manifest = local / "authoritative_manifest.csv"
    fold_file = contracts / "field1_spatial_folds.csv"
    fold_contract_file = contracts / "field1_spatial_fold_contract.yaml"
    protocol_contract_file = (
        contracts / protocol_config["contract_filename"]
    )
    role_file = contracts / protocol_config["role_table_filename"]
    for required in (
        manifest, fold_file, fold_contract_file, protocol_contract_file, role_file
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    fold_contract = yaml.safe_load(fold_contract_file.read_text())
    if sha256(fold_file) != fold_contract["assignment_sha256"]:
        raise ValueError("Spatial-fold assignment hash mismatch")
    protocol = yaml.safe_load(protocol_contract_file.read_text())
    if protocol.get("status") != "supervised_primary_evaluation_protocol_frozen":
        raise ValueError("Primary evaluation protocol is not frozen")
    if protocol.get("field2_accessed") is not False:
        raise ValueError("Primary evaluation protocol violated Field 2 lock")
    if sha256(role_file) != protocol["nested_role_table_sha256"]:
        raise ValueError("Nested role-table hash mismatch")

    roles = pd.read_csv(role_file)
    outer_folds = sorted(roles["outer_evaluation_fold"].unique().astype(int))
    inner_by_outer = {
        int(outer): int(
            roles[
                (roles.outer_evaluation_fold == outer)
                & (roles.role == "inner_validation")
            ]["spatial_fold"].iloc[0]
        )
        for outer in outer_folds
    }
    assignments = pd.read_csv(fold_file)
    fold_lookup = {
        (int(row.block_x), int(row.block_y)): int(row.fold)
        for row in assignments.itertuples()
    }

    grouping = spatial["grouping"]
    block_size = float(grouping["block_size_m"])
    origin_x = float(grouping.get("origin_x_m", 0.0))
    origin_y = float(grouping.get("origin_y_m", 0.0))
    buffer_m = float(normal["boundary_exclusion_m"])
    if buffer_m != float(fold_contract["boundary_exclusion_m"]):
        raise ValueError("Boundary exclusion differs from frozen spatial contract")

    band_indices = load_band_indices(args.bands, normal["band_section"])
    band_count = len(band_indices)
    sums = {outer: np.zeros(band_count, dtype=np.float64) for outer in outer_folds}
    sums_squared = {
        outer: np.zeros(band_count, dtype=np.float64) for outer in outer_folds
    }
    minima = {
        outer: np.full(band_count, np.inf, dtype=np.float64)
        for outer in outer_folds
    }
    maxima = {
        outer: np.full(band_count, -np.inf, dtype=np.float64)
        for outer in outer_folds
    }
    counts = {outer: 0 for outer in outer_folds}
    outer_counts = {outer: 0 for outer in outer_folds}
    inner_counts = {outer: 0 for outer in outer_folds}
    boundary_counts = {outer: 0 for outer in outer_folds}
    cube_counts = {outer: 0 for outer in outer_folds}
    wavelengths: np.ndarray | None = None
    total_sampled = 0

    for record in load_records(args.paths, manifest):
        cube = EnviCube(record, band_indices)
        current_wavelengths = cube.wavelengths[band_indices]
        if wavelengths is None:
            wavelengths = current_wavelengths
        elif not np.allclose(wavelengths, current_wavelengths):
            raise ValueError(f"Wavelength mismatch: {record.cube_id}")

        valid = np.any(cube.array > 0, axis=2)
        coordinates = np.argwhere(valid)
        sample_count = min(int(normal["samples_per_cube"]), len(coordinates))
        rng = stable_rng(int(normal["seed"]), record.cube_id)
        selected = coordinates[
            rng.choice(len(coordinates), size=sample_count, replace=False)
        ]
        spectra = np.asarray(
            cube.array[selected[:, 0], selected[:, 1]][:, band_indices],
            dtype=np.float64,
        )
        with rasterio.open(record.data) as dataset:
            transform = dataset.transform
        rows, columns = selected[:, 0], selected[:, 1]
        x = (
            transform.a * (columns + 0.5)
            + transform.b * (rows + 0.5)
            + transform.c
        )
        y = (
            transform.d * (columns + 0.5)
            + transform.e * (rows + 0.5)
            + transform.f
        )
        block_x, block_y = map_block_indices(
            x, y, block_size, origin_x, origin_y
        )
        sample_folds = np.fromiter(
            (
                fold_lookup.get((int(one_x), int(one_y)), 0)
                for one_x, one_y in zip(block_x, block_y)
            ),
            dtype=np.int8,
            count=sample_count,
        )
        if (sample_folds == 0).any():
            raise ValueError(
                f"Sampled pixel outside frozen groups: {record.cube_id}"
            )
        total_sampled += sample_count

        for outer in outer_folds:
            inner = inner_by_outer[outer]
            outer_mask = sample_folds == outer
            inner_mask = sample_folds == inner
            safe_outer = boundary_safe_mask(
                x, y, block_x, block_y, fold_lookup, outer, block_size,
                origin_x, origin_y, buffer_m,
            )
            safe_inner = boundary_safe_mask(
                x, y, block_x, block_y, fold_lookup, inner, block_size,
                origin_x, origin_y, buffer_m,
            )
            role_excluded = outer_mask | inner_mask
            boundary_removed = (~role_excluded) & (~safe_outer | ~safe_inner)
            training = (~role_excluded) & safe_outer & safe_inner
            outer_counts[outer] += int(outer_mask.sum())
            inner_counts[outer] += int(inner_mask.sum())
            boundary_counts[outer] += int(boundary_removed.sum())
            if not training.any():
                continue
            values = spectra[training]
            counts[outer] += len(values)
            cube_counts[outer] += 1
            sums[outer] += values.sum(axis=0)
            sums_squared[outer] += np.square(values).sum(axis=0)
            minima[outer] = np.minimum(minima[outer], values.min(axis=0))
            maxima[outer] = np.maximum(maxima[outer], values.max(axis=0))

    if wavelengths is None:
        raise RuntimeError("No spectra were sampled")

    statistic_rows = []
    summary_rows = []
    for outer in outer_folds:
        if counts[outer] == 0:
            raise ValueError(f"No training spectra for outer fold {outer}")
        inner = inner_by_outer[outer]
        mean = sums[outer] / counts[outer]
        variance = np.maximum(
            sums_squared[outer] / counts[outer] - np.square(mean), 0
        )
        standard_deviation = np.sqrt(variance)
        if (standard_deviation <= 0).any():
            raise ValueError(f"Zero-variance band for outer fold {outer}")
        for position, band_index in enumerate(band_indices):
            statistic_rows.append(
                {
                    "outer_fold": int(outer),
                    "inner_validation_fold": int(inner),
                    "band_index": int(band_index),
                    "wavelength_nm": float(wavelengths[position]),
                    "training_sample_count": counts[outer],
                    "mean": mean[position],
                    "standard_deviation": standard_deviation[position],
                    "minimum": minima[outer][position],
                    "maximum": maxima[outer][position],
                }
            )
        summary_rows.append(
            {
                "outer_fold": int(outer),
                "inner_validation_fold": int(inner),
                "total_cube_balanced_samples": total_sampled,
                "outer_samples_excluded": outer_counts[outer],
                "inner_samples_excluded": inner_counts[outer],
                "boundary_samples_excluded": boundary_counts[outer],
                "training_samples": counts[outer],
                "contributing_cubes": cube_counts[outer],
            }
        )

    statistics = pd.DataFrame(statistic_rows)
    summary = pd.DataFrame(summary_rows)
    statistics_path = contracts / "field1_nested_normalization.csv"
    summary_path = reports / "nested_normalization_summary.csv"
    diagnostics_path = reports / "nested_normalization_diagnostics.csv"
    statistics.to_csv(statistics_path, index=False)
    summary.to_csv(summary_path, index=False)
    statistics.to_csv(diagnostics_path, index=False)

    figure, axes = plt.subplots(
        1, 2, figsize=(15, 5.5), constrained_layout=True
    )
    for outer, frame in statistics.groupby("outer_fold"):
        inner = int(frame["inner_validation_fold"].iloc[0])
        label = f"Outer {outer}; inner {inner}"
        axes[0].plot(frame["wavelength_nm"], frame["mean"], label=label)
        axes[1].plot(
            frame["wavelength_nm"], frame["standard_deviation"], label=label
        )
    axes[0].set_title("Nested-training spectral means")
    axes[1].set_title("Nested-training spectral standard deviations")
    for axis in axes:
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Stored reflectance value (uint16 scale)")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.suptitle(
        "Field 1 nested training-only normalization\n"
        "Both outer test and inner validation folds excluded",
        fontsize=14,
    )
    preview = reports / "nested_normalization_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)

    contract = {
        "status": "frozen_nested_training_only_normalization",
        "field": "Field 1",
        "field2_accessed": False,
        "synthetic_scientific_observations": False,
        "outer_to_inner_fold": inner_by_outer,
        "mode": normal["mode"],
        "seed": int(normal["seed"]),
        "samples_per_cube": int(normal["samples_per_cube"]),
        "band_section": normal["band_section"],
        "band_count": band_count,
        "boundary_exclusion_m": buffer_m,
        "standard_deviation": normal["standard_deviation"],
        "normalization_csv_sha256": sha256(statistics_path),
        "summary_sha256": sha256(summary_path),
        "visual_sha256": sha256(preview),
        "source_hashes": {
            "protocol_contract": sha256(protocol_contract_file),
            "nested_roles": sha256(role_file),
            "fold_assignment": sha256(fold_file),
            "manifest": sha256(manifest),
            "normalization_config": sha256(args.normalization),
            "spectral_bands": sha256(args.bands),
            "spatial_splits": sha256(args.spatial),
        },
    }
    contract_path = (
        contracts / "field1_nested_normalization_contract.yaml"
    )
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Cube-balanced spectra sampled: {total_sampled:,}")
    print(summary.to_string(index=False))
    print(f"Contract: {contract_path}")
    print(f"Visual QC: {preview}")
    print(
        "Statistics exclude outer and inner folds; Field 2 was not accessed."
    )


if __name__ == "__main__":
    main()
