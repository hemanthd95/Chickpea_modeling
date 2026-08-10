#!/usr/bin/env python
"""Prepare aligned observed-data layers for investigator chickpea-region polygons."""

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
import numpy as np
import pandas as pd
import rasterio
from spectral.io import envi
import yaml

from chickpea_ssl.data import load_records
from scripts.audit_standardized_planter_turn_bands import (
    pca_rgb,
    spatial_derivatives,
    standardized_preview_pca,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stretch(channel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = channel[valid & np.isfinite(channel)]
    if not len(values):
        return np.zeros(channel.shape, dtype=np.float32)
    lower, upper = np.percentile(values, [2, 98])
    if upper <= lower:
        return np.zeros(channel.shape, dtype=np.float32)
    result = np.clip((channel.astype(np.float32) - lower) / (upper - lower), 0, 1)
    result[~valid] = 0
    return result


def nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.nanargmin(np.abs(wavelengths - target_nm)))


def load_current_chickpea(path: Path | None, shape: tuple[int, int], step: int) -> np.ndarray:
    if path is None:
        return np.zeros((math.ceil(shape[0] / step), math.ceil(shape[1] / step)), dtype=bool)
    with rasterio.open(path) as dataset:
        if (dataset.height, dataset.width) != shape:
            raise ValueError(f"Chickpea mask shape mismatch: {path}")
        return dataset.read(1)[::step, ::step] > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    report_root = local / "reports" / "chickpea_region_annotation"
    layers_root = report_root / "layers"
    contracts = local / "contracts"
    layers_root.mkdir(parents=True, exist_ok=True)
    contracts.mkdir(parents=True, exist_ok=True)

    settings = config["planter_pass_reconstruction"]["standardized_turn_band_audit"]
    optional = set(config["investigator_guidance"]["optional_mask_expansion"]["cubes"])
    excluded = set(config["investigator_guidance"]["phenology_exclusions_from_refined_primary_supervised_analysis"])
    all_records = load_records(args.paths, local / "authoritative_manifest.csv")
    records = sorted(
        [r for r in all_records if r.chickpea_mask is not None or r.cube_id in optional],
        key=lambda r: tuple(int(x) if x.isdigit() else x for x in r.cube_id.replace("field1_cube", "").split("_")),
    )
    found = {r.cube_id for r in records}
    missing = sorted(optional - found)
    if missing:
        raise ValueError(
            "Optional cubes are absent from authoritative_manifest.csv: " + ", ".join(missing)
        )

    manifest_rows: list[dict] = []
    overview_tiles = []
    input_hashes = []
    max_preview = int(settings["maximum_preview_dimension_pixels"])
    seed = int(settings["seed"])

    for number, record in enumerate(records, start=1):
        image = envi.open(str(record.header), str(record.data))
        memory = image.open_memmap()
        height, width = memory.shape[:2]
        wavelengths = np.asarray(image.metadata["wavelength"], dtype=np.float32)
        with rasterio.open(record.data) as reference:
            transform = reference.transform
            crs = str(reference.crs or "")
        step = max(1, math.ceil(max(height, width) / max_preview))
        scores, valid, explained, _, diagnostics = standardized_preview_pca(
            memory, step, int(settings["maximum_pca_fit_samples"]), seed
        )
        pca = pca_rgb(scores, valid)
        first, second, _ = spatial_derivatives(scores, valid)

        nir_index = nearest_band(wavelengths, 800.0)
        red_index = nearest_band(wavelengths, 670.0)
        green_index = nearest_band(wavelengths, 550.0)
        observed = np.asarray(memory[::step, ::step, :])
        false_colour = np.dstack([
            stretch(observed[..., nir_index], valid),
            stretch(observed[..., red_index], valid),
            stretch(observed[..., green_index], valid),
        ])
        nir = observed[..., nir_index].astype(np.float32)
        red = observed[..., red_index].astype(np.float32)
        denominator = nir + red
        ndvi = np.full(nir.shape, np.nan, dtype=np.float32)
        usable = valid & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
        ndvi[usable] = (nir[usable] - red[usable]) / denominator[usable]

        current = load_current_chickpea(record.chickpea_mask, (height, width), step)
        current_overlay = false_colour.copy()
        current_overlay[current] = (
            0.25 * current_overlay[current] + 0.75 * np.array([0.0, 1.0, 0.2])
        )

        cube_root = layers_root / record.cube_id
        cube_root.mkdir(parents=True, exist_ok=True)
        layer_paths = {
            "false_colour": cube_root / "nir_red_green.png",
            "pca": cube_root / "pca_rgb.png",
            "first_difference": cube_root / "first_difference_magnitude.png",
            "second_difference": cube_root / "second_difference_magnitude.png",
            "ndvi": cube_root / "ndvi.png",
            "current_chickpea": cube_root / "current_chickpea_overlay.png",
        }
        plt.imsave(layer_paths["false_colour"], false_colour)
        plt.imsave(layer_paths["pca"], pca)
        plt.imsave(layer_paths["first_difference"], first, cmap="gray", vmin=0, vmax=1)
        plt.imsave(layer_paths["second_difference"], second, cmap="gray", vmin=0, vmax=1)
        plt.imsave(layer_paths["ndvi"], ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        plt.imsave(layer_paths["current_chickpea"], current_overlay)

        role = "label_expansion_candidate" if record.cube_id in optional else (
            "sensitivity_only" if record.cube_id in excluded else "primary_candidate"
        )
        manifest_rows.append({
            "cube_id": record.cube_id,
            "analysis_role": role,
            "has_current_chickpea_mask": record.chickpea_mask is not None,
            "preview_height": scores.shape[0],
            "preview_width": scores.shape[1],
            "preview_step": step,
            "original_height": height,
            "original_width": width,
            "transform_a": transform.a,
            "transform_b": transform.b,
            "transform_c": transform.c,
            "transform_d": transform.d,
            "transform_e": transform.e,
            "transform_f": transform.f,
            "crs": crs,
            "nir_band_index": nir_index,
            "nir_wavelength_nm": float(wavelengths[nir_index]),
            "red_band_index": red_index,
            "red_wavelength_nm": float(wavelengths[red_index]),
            "green_band_index": green_index,
            "green_wavelength_nm": float(wavelengths[green_index]),
            "current_chickpea_preview_pixels": int(current.sum()),
            "pca_explained_variance_1": float(explained[0]),
            "pca_explained_variance_2": float(explained[1]),
            "pca_explained_variance_3": float(explained[2]),
            **diagnostics,
            **{f"{name}_path": str(path) for name, path in layer_paths.items()},
        })
        input_hashes.append({
            "cube_id": record.cube_id,
            "header": str(record.header),
            "header_sha256": sha256(record.header),
        })
        overview_tiles.append((record.cube_id, role, current_overlay))
        print(
            f"Prepared {number}/{len(records)} {record.cube_id}: {role}; "
            f"current chickpea mask={'yes' if record.chickpea_mask else 'no'}"
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = report_root / "chickpea_region_annotation_layer_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    columns = 4
    rows = math.ceil(len(overview_tiles) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4.2 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (cube_id, role, tile) in zip(axes, overview_tiles):
        axis.imshow(tile)
        axis.set_title(f"{cube_id}\n{role}")
    figure.suptitle(
        "Field 1 chickpea-region polygon annotation layers\n"
        "NIR-red-green; green overlay = current chickpea mask; no label changed",
        fontsize=16,
    )
    overview_path = report_root / "chickpea_region_annotation_layers_overview.png"
    figure.savefig(overview_path, dpi=180, facecolor="white")
    plt.close(figure)

    contract = {
        "status": "investigator_polygon_annotation_ready",
        "field": "Field 1",
        "cubes": len(manifest),
        "primary_candidates": int((manifest.analysis_role == "primary_candidate").sum()),
        "phenology_sensitivity_only": sorted(excluded),
        "label_expansion_candidates": sorted(optional),
        "polygon_is_spatial_prior_not_pixel_class": True,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "input_headers": input_hashes,
    }
    contract_path = contracts / "field1_chickpea_region_annotation_layers_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"Annotation-ready cubes: {len(manifest)}")
    print(f"Layer manifest: {manifest_path}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print("No authoritative mask was changed; Field 2 remained locked.")


if __name__ == "__main__":
    main()
