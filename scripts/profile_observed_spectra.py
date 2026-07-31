#!/usr/bin/env python
"""Profile actual Field 1 spectra for evidence-based band QC."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--samples-per-cube", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from spectral.io import envi

    config = yaml.safe_load(args.paths.read_text())
    project = Path(config["project_root"])
    local = project / "metadata" / "local"
    manifest = pd.read_csv(local / "authoritative_manifest.csv", dtype=str).fillna("")
    rng = np.random.default_rng(args.seed)
    samples: list[np.ndarray] = []
    cube_rows: list[dict[str, object]] = []
    wavelengths: np.ndarray | None = None

    for _, row in manifest.iterrows():
        root = Path(config["field1"]["reflectance_dates"][row["acquisition_date"]])
        header = root / row["reflectance_header"]
        data = root / row["reflectance_bip"]
        image = envi.open(str(header), str(data))
        array = image.open_memmap()
        current_wavelengths = np.asarray(image.metadata["wavelength"], dtype=float)
        if wavelengths is None:
            wavelengths = current_wavelengths
        elif not np.allclose(wavelengths, current_wavelengths):
            raise ValueError(f"Wavelength mismatch in {row['cube_id']}")
        valid = np.any(array > 0, axis=2)
        coordinates = np.argwhere(valid)
        count = min(args.samples_per_cube, len(coordinates))
        selected = coordinates[rng.choice(len(coordinates), size=count, replace=False)]
        spectra = np.asarray(array[selected[:, 0], selected[:, 1], :], dtype=np.uint16)
        samples.append(spectra)
        cube_rows.append({
            "cube_id": row["cube_id"], "acquisition_date": row["acquisition_date"],
            "valid_pixels": len(coordinates), "sampled_pixels": count,
            "zero_fraction_sample": float((spectra == 0).mean()),
            "saturation_fraction_sample": float((spectra == 65535).mean()),
        })

    observed = np.concatenate(samples, axis=0)
    profile = pd.DataFrame({
        "band_index": np.arange(observed.shape[1]),
        "wavelength_nm": wavelengths,
        "minimum": observed.min(axis=0),
        "p01": np.percentile(observed, 1, axis=0),
        "median": np.median(observed, axis=0),
        "p99": np.percentile(observed, 99, axis=0),
        "maximum": observed.max(axis=0),
        "zero_fraction": (observed == 0).mean(axis=0),
        "saturation_fraction": (observed == 65535).mean(axis=0),
        "robust_range": np.percentile(observed, 99, axis=0) - np.percentile(observed, 1, axis=0),
    })
    profile.to_csv(local / "spectral_band_profile.csv", index=False)
    pd.DataFrame(cube_rows).to_csv(local / "spectral_cube_profile.csv", index=False)
    print(f"Observed spectra sampled: {len(observed):,}")
    print(f"Cubes profiled: {len(cube_rows)}")
    print(f"Band profile: {local / 'spectral_band_profile.csv'}")
    print(f"Cube profile: {local / 'spectral_cube_profile.csv'}")


if __name__ == "__main__":
    main()
