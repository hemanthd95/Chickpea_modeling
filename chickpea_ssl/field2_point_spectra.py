"""Prediction-free raw Field 2 spectra for frozen MAIN annotation points."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from spectral.io import envi

from chickpea_ssl.field2_readiness import parse_wavelengths, sha256


PYTHON_BAND_START = 3
PYTHON_BAND_STOP_INCLUSIVE = 113
INDEX_TARGETS_NM = {
    "NDVI": {"a": 800.0, "b": 670.0},
    "GNDVI": {"a": 800.0, "b": 550.0},
    "NDRE": {"a": 800.0, "b": 720.0},
}


def normalized_difference(a: float, b: float) -> float | None:
    denominator = float(a) + float(b)
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return None
    value = (float(a) - float(b)) / denominator
    return float(value) if np.isfinite(value) else None


class PredictionFreeSpectrumStore:
    """Read only source reflectance and frozen support; never load a model artifact."""

    def __init__(self, project: Path, inventory_path: Path, support_manifest_path: Path, main_frame: pd.DataFrame):
        self.project = project.resolve()
        inventory = pd.read_csv(inventory_path, keep_default_na=False)
        reflectance = inventory.loc[inventory.product_type == "reflectance"].copy()
        support = pd.read_csv(support_manifest_path, keep_default_na=False)
        if len(reflectance) != 40 or reflectance.cube_id.nunique() != 40 or len(support) != 40:
            raise ValueError("Raw spectrum service requires the exact 40-cube source/support inventory")
        if len(main_frame) != 800 or set(main_frame.sampling_frame) != {"main"}:
            raise ValueError("Raw spectrum service accepts only the frozen 800-point MAIN frame")
        self.sources = {str(row.cube_id): row for row in reflectance.itertuples(index=False)}
        self.support = {str(row.cube_id): row for row in support.itertuples(index=False)}
        self.samples = {str(row.sample_id): row for row in main_frame.itertuples(index=False)}
        self._images = {}; self._wavelengths = {}; self._support_arrays = {}

    def _cube(self, cube_id: str):
        if cube_id not in self.sources:
            raise KeyError(cube_id)
        if cube_id not in self._images:
            source = self.sources[cube_id]
            header = self.project / str(source.header_path); binary = self.project / str(source.binary_path)
            if sha256(binary) != str(source.sha256):
                raise ValueError(f"Frozen reflectance hash changed: {cube_id}")
            image = envi.open(str(header), str(binary))
            wavelengths = parse_wavelengths(image.metadata)
            if len(wavelengths) != int(source.bands):
                raise ValueError(f"Reflectance wavelength count changed: {cube_id}")
            support_row = self.support[cube_id]; support_path = self.project / str(support_row.mask_path)
            if sha256(support_path) != str(support_row.mask_sha256):
                raise ValueError(f"Frozen support hash changed: {cube_id}")
            with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                with rasterio.open(support_path) as dataset:
                    support_array = dataset.read(1) == 1
            if support_array.shape != (image.nrows, image.ncols):
                raise ValueError(f"Reflectance/support grid mismatch: {cube_id}")
            self._images[cube_id] = image; self._wavelengths[cube_id] = wavelengths
            self._support_arrays[cube_id] = support_array
        return self._images[cube_id], self._wavelengths[cube_id], self._support_arrays[cube_id]

    @staticmethod
    def _indices(values: np.ndarray, wavelengths: np.ndarray) -> dict:
        result = {}
        for name, targets in INDEX_TARGETS_NM.items():
            index_a = int(np.argmin(np.abs(wavelengths - targets["a"])))
            index_b = int(np.argmin(np.abs(wavelengths - targets["b"])))
            wavelength_a, wavelength_b = float(wavelengths[index_a]), float(wavelengths[index_b])
            result[name] = {
                "value": normalized_difference(values[index_a], values[index_b]),
                "formula": f"(R{wavelength_a:.2f} - R{wavelength_b:.2f}) / (R{wavelength_a:.2f} + R{wavelength_b:.2f})",
                "selected_wavelengths_nm": [wavelength_a, wavelength_b],
                "threshold_applied": False,
            }
        return result

    def sample_spectrum(self, sample_id: str) -> dict:
        if sample_id not in self.samples:
            raise KeyError("Unknown or non-MAIN sample")
        sample = self.samples[sample_id]; cube_id = str(sample.cube_id)
        image, all_wavelengths, support = self._cube(cube_id)
        row, column = int(sample.row), int(sample.column)
        if not support[row, column]:
            raise ValueError("Frozen MAIN sample is outside valid support")
        band_slice = slice(PYTHON_BAND_START, PYTHON_BAND_STOP_INCLUSIVE + 1)
        wavelengths = np.asarray(all_wavelengths[band_slice], dtype=float)
        center = np.asarray(image[row, column, band_slice], dtype=float).reshape(-1)
        row_start, row_stop = max(0, row - 1), min(image.nrows, row + 2)
        column_start, column_stop = max(0, column - 1), min(image.ncols, column + 2)
        neighborhood = np.asarray(image[row_start:row_stop, column_start:column_stop, band_slice], dtype=float)
        valid = support[row_start:row_stop, column_start:column_stop]
        spectra = neighborhood[valid]
        finite_rows = np.all(np.isfinite(spectra), axis=1)
        median = np.median(spectra[finite_rows], axis=0) if finite_rows.any() else np.full_like(center, np.nan)
        if len(center) != 111 or len(wavelengths) != 111:
            raise ValueError("Expected exact Python bands 3-113 (111 bands)")
        return {
            "sample_id": sample_id, "cube_id": cube_id, "row": row, "column": column,
            "source": "raw_stored_reflectance", "python_band_range_inclusive": [3, 113],
            "envi_band_range_one_based_inclusive": [4, 114], "wavelength_units": "nm",
            "wavelengths_nm": wavelengths.tolist(), "center_reflectance": center.tolist(),
            "valid_3x3_median_reflectance": median.tolist(),
            "valid_3x3_spectrum_count": int(finite_rows.sum()),
            "raw_band_indices": self._indices(center, wavelengths),
            "stored_scalar_product_is_authoritative_ndvi": False,
            "model_prediction_probability_or_threshold_used": False,
        }
