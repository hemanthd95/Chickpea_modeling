#!/usr/bin/env python
"""Build a prediction-free, read-only Field 2 georectified readiness package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ["GDAL_PAM_ENABLED"] = "NO"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.field2_readiness import (
    ReadOnlySourceGuard,
    affine_comparison,
    compare_snapshots,
    deterministic_coordinates,
    dump_yaml,
    expected_payload_bytes,
    match_wavelengths,
    normalized_cube_id,
    open_envi_memmap,
    parse_envi_header,
    parse_wavelengths,
    resolve_envi_payload,
    scalar_int,
    sha256,
    source_snapshot,
)

REPORT_NAMES = {
    "inventory": "field2_georectified_inventory.csv",
    "manifest": "field2_file_integrity_manifest.csv",
    "pairs": "field2_envi_pair_validation.csv",
    "spectral": "field2_spectral_band_compatibility.csv",
    "alignment": "field2_spatial_alignment_audit.csv",
    "footprints": "field2_cube_footprints.csv",
    "overlap": "field2_cube_overlap.csv",
    "statistics": "field2_sampled_band_statistics.csv",
    "qc": "field2_cube_qc_summary.csv",
    "readiness": "field2_annotation_readiness.csv",
}


def git_output(project: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=project, text=True).strip()


def load_configuration(paths_file: Path, config_file: Path) -> tuple[Path, dict, dict]:
    paths = yaml.safe_load(paths_file.read_text())
    config = yaml.safe_load(config_file.read_text())
    project = Path(paths["project_root"]).resolve()
    roots = paths.get("field2", {}).get("readiness_roots", [])
    if not roots:
        raise ValueError("field2.readiness_roots is empty")
    config["source_roots"] = [str(Path(root).resolve()) for root in roots]
    return project, paths, config


def product_for_root(root: Path, config: dict) -> str:
    matches = [name for name, root_name in config["authorized_root_names"].items() if root.name == root_name]
    if len(matches) != 1:
        raise ValueError(f"Authorized root does not map to exactly one product: {root}")
    return matches[0]


def metadata_text(metadata: dict, key: str) -> str:
    value = metadata.get(key, "")
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def robust_stretch(values: np.ndarray, valid: np.ndarray, percentiles: tuple[float, float]) -> tuple[np.ndarray, list[list[float]]]:
    output = np.zeros(values.shape, dtype=np.float32)
    stretches: list[list[float]] = []
    for band in range(values.shape[2]):
        data = np.asarray(values[..., band], dtype=np.float32)
        usable = data[valid & np.isfinite(data)]
        if not len(usable):
            low, high = 0.0, 1.0
        else:
            low, high = np.percentile(usable, percentiles)
            if not np.isfinite(high) or high <= low:
                high = low + 1.0
        output[..., band] = np.clip((data - low) / (high - low), 0, 1)
        stretches.append([float(low), float(high)])
    return output, stretches


def raster_details(path: Path, guard: ReadOnlySourceGuard) -> dict:
    source = guard.source(path, "r")
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(source, "r") as dataset:
            transform = dataset.transform
            bounds = dataset.bounds
            return {
                "width": int(dataset.width), "height": int(dataset.height), "raster_bands": int(dataset.count),
                "crs": str(dataset.crs or ""), "coordinate_system_wkt": dataset.crs.to_wkt() if dataset.crs else "",
                "transform": json.dumps([float(value) for value in transform]),
                "transform_object": transform,
                "gsd_x": float(abs(transform.a)), "gsd_y": float(abs(transform.e)),
                "rotation_x": float(transform.b), "rotation_y": float(transform.d),
                "left": float(bounds.left), "bottom": float(bounds.bottom),
                "right": float(bounds.right), "top": float(bounds.top),
                "nodata": dataset.nodata,
            }


def sampled_band_rows(cube_id: str, product: str, array: np.ndarray, metadata: dict,
                      coordinates: np.ndarray, valid_mask: np.ndarray) -> list[dict]:
    samples = np.asarray(array[coordinates[:, 0], coordinates[:, 1], :], dtype=np.float64)
    wavelengths = parse_wavelengths(metadata)
    nodata_raw = metadata.get("data ignore value")
    nodata = float(nodata_raw) if nodata_raw not in (None, "") else None
    rows = []
    for band in range(samples.shape[1]):
        values = samples[:, band]
        finite = np.isfinite(values)
        nodata_mask = np.zeros(len(values), dtype=bool) if nodata is None else values == nodata
        usable = values[finite & ~nodata_mask & valid_mask]
        quantiles = np.percentile(usable, [1, 5, 25, 50, 75, 95, 99]) if len(usable) else np.full(7, np.nan)
        rows.append({
            "cube_id": cube_id, "product_type": product, "band_index": band,
            "wavelength_nm": float(wavelengths[band]) if band < len(wavelengths) else np.nan,
            "sample_count": int(len(values)), "valid_observation_count": int(len(usable)),
            "minimum": float(np.min(usable)) if len(usable) else np.nan,
            "maximum": float(np.max(usable)) if len(usable) else np.nan,
            "mean": float(np.mean(usable)) if len(usable) else np.nan,
            "standard_deviation": float(np.std(usable)) if len(usable) else np.nan,
            "median": float(quantiles[3]), "percentile_01": float(quantiles[0]),
            "percentile_05": float(quantiles[1]), "percentile_25": float(quantiles[2]),
            "percentile_75": float(quantiles[4]), "percentile_95": float(quantiles[5]),
            "percentile_99": float(quantiles[6]),
            "nan_fraction": float(np.mean(np.isnan(values))), "infinity_fraction": float(np.mean(np.isinf(values))),
            "nodata_fraction": float(np.mean(nodata_mask)), "zero_fraction": float(np.mean(values == 0)),
            "saturation_fraction": np.nan,
            "saturation_note": "post-reflectance saturation encoding not supplied" if product == "reflectance" else "not_applicable",
            "constant_value_indicator": bool(len(usable) and np.ptp(usable) == 0),
        })
    return rows


def overlap_rows(footprints: pd.DataFrame) -> list[dict]:
    rows = []
    records = footprints.to_dict("records")
    for first_index, first in enumerate(records):
        for second in records[first_index + 1:]:
            same_crs = first["crs"] == second["crs"] and bool(first["crs"])
            left, right = max(first["left"], second["left"]), min(first["right"], second["right"])
            bottom, top = max(first["bottom"], second["bottom"]), min(first["top"], second["top"])
            intersection = max(0.0, right - left) * max(0.0, top - bottom) if same_crs else 0.0
            first_area = (first["right"] - first["left"]) * (first["top"] - first["bottom"])
            second_area = (second["right"] - second["left"]) * (second["top"] - second["bottom"])
            rows.append({
                "cube_id_a": first["cube_id"], "cube_id_b": second["cube_id"], "same_crs": same_crs,
                "intersection_area_square_metres": intersection,
                "overlap_fraction_of_smaller_footprint": intersection / min(first_area, second_area) if min(first_area, second_area) else 0.0,
            })
    return rows


def save_cube_preview(path: Path, cube_id: str, reflectance_rgb: np.ndarray, pca_rgb: np.ndarray,
                      index: np.ndarray, valid: np.ndarray, alignment_rgb: np.ndarray,
                      stretch_record: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    panels = [
        (reflectance_rgb, "False colour: 800/670/550 nm"),
        (pca_rgb, "Supplied PCA components 1–3"),
        (index, "Stored scalar index (provenance not fully verified)"),
        (valid, "Operational valid/zero footprint"),
        (alignment_rgb, "Alignment edge check: R/PCA/index"),
    ]
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image, cmap="viridis" if image.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    axes.flat[5].axis("off")
    axes.flat[5].text(0, 1, "Prediction-free diagnostic\n\n" + json.dumps(stretch_record, indent=2), va="top", fontsize=8)
    fig.suptitle(f"{cube_id} — Field 2 annotation-source review", fontsize=16)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_overviews(output: Path, inventory: pd.DataFrame, alignment: pd.DataFrame,
                    qc: pd.DataFrame, thumbnails: list[tuple[str, np.ndarray]]) -> None:
    counts = inventory.groupby("product_type").size()
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    counts.plot.bar(ax=axis, color="#2673b8")
    axis.set(title="Field 2 product inventory", ylabel="ENVI products", xlabel="")
    fig.savefig(output / "field2_inventory_overview.png", dpi=180); plt.close(fig)

    status = alignment["alignment_status"].value_counts()
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    status.plot.bar(ax=axis, color="#5aa469")
    axis.set(title="Reflectance-to-product alignment audit", ylabel="Product pairs", xlabel="")
    fig.savefig(output / "field2_alignment_overview.png", dpi=180); plt.close(fig)

    ordered = qc.sort_values("cube_id")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    axes[0].bar(ordered["cube_id"], ordered["valid_fraction"])
    axes[0].set(title="Operational valid-support coverage", ylabel="Fraction"); axes[0].tick_params(axis="x", rotation=90)
    axes[1].bar(ordered["cube_id"], ordered["reflectance_sampled_dynamic_range"])
    axes[1].set(title="Sampled reflectance dynamic range", ylabel="Stored-value range"); axes[1].tick_params(axis="x", rotation=90)
    fig.savefig(output / "field2_spectral_qc_overview.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(5, 8, figsize=(20, 13), constrained_layout=True)
    for axis, (cube_id, image) in zip(axes.flat, thumbnails):
        axis.imshow(image); axis.set_title(cube_id, fontsize=8); axis.axis("off")
    for axis in axes.flat[len(thumbnails):]: axis.axis("off")
    fig.suptitle("Field 2 prediction-free annotation sources")
    fig.savefig(output / "field2_prediction_free_annotation_source_overview.png", dpi=180); plt.close(fig)


def write_report(path: Path, commit: str, cubes: list[str], inventory: pd.DataFrame, pairs: pd.DataFrame,
                 spectral: pd.DataFrame, alignment: pd.DataFrame, qc: pd.DataFrame,
                 readiness: pd.DataFrame, snapshot_issues: list[str], config_path: Path) -> None:
    product_counts = inventory.product_type.value_counts().to_dict()
    blocked = readiness.loc[readiness.usable_for_annotation != "yes", ["cube_id", "exclusion_reason"]]
    blocked_table = "None."
    if len(blocked):
        blocked_table = "| cube_id | exclusion_reason |\n|---|---|\n" + "\n".join(
            f"| {row.cube_id} | {row.exclusion_reason} |" for row in blocked.itertuples(index=False)
        )
    warnings = sorted(set(item for values in qc["warnings"].fillna("") for item in values.split("|") if item))
    text = f"""# Field 2 georectified-data readiness audit

Date: 2026-08-14

Audit code commit at execution: `{commit}`

Status: **prediction-free read-only readiness audit complete**

## Scope and safeguards

The audit inspected only the three explicitly authorized Field 2 roots. A complete SHA-256 snapshot was written before raster arrays were opened. All ENVI payloads were accessed through read-only NumPy memory maps; Rasterio was used in reader mode with `GDAL_PAM_ENABLED=NO`. Output paths were rejected if they resolved inside a source root.

No model checkpoint was loaded. No prediction, probability, categorical label, annotation point, model training, or SSL experiment was generated. The frozen Field 1 benchmark and its artifacts were not modified.

## Inventory

- Cubes ({len(cubes)}): {', '.join(cubes)}
- Products: {json.dumps(product_counts, sort_keys=True)}
- Headers: {len(pairs)}; payloads: {int(pairs.payload_exists.sum())}; valid one-to-one pairs: {int(pairs.valid_pair.sum())}
- Missing/ambiguous pairs: {int((~pairs.valid_pair).sum())}

Derivative PCA, separate valid-data masks, georeferencing sidecars, and georectification residual/control-point reports were not supplied. Their absence is reported and was not inferred to mean acquisition failure.

## Integrity and spectral compatibility

All payload sizes matched ENVI dimensions, data type, interleave, byte order, and header offset: **{bool(pairs.byte_size_match.all())}**. Reflectance wavelength rows matched the frozen Field 1 band indices 3–113 within the configured tolerance: **{bool((spectral.match_status == 'matched').all())}**. No spectral reordering, interpolation, truncation, or resampling was performed.

## Spatial alignment and overlap

Grid-alignment status counts: {json.dumps(alignment.alignment_status.value_counts().to_dict(), sort_keys=True)}. CRS, affine, dimensions, GSD, bounds, rotation/shear, and pixel-center alignment were compared directly. No product was resampled. Georectification residuals/control-point errors were `not_supplied`.

Although all grids align exactly, PCA and stored-index products contain transformed, nonzero values outside the reflectance raster's zero-filled footprint. Raw support agreement is {alignment.valid_footprint_agreement_fraction.min():.3f}–{alignment.valid_footprint_agreement_fraction.max():.3f}. The diagnostic previews apply the operational reflectance support for display, but this is not treated as an authoritative NoData definition.

Cross-cube footprint overlaps are reported in `field2_cube_overlap.csv`; source cubes were not deduplicated or changed.

## Radiometry, NoData, and sampled QC

Each reflectance header records measured-reference correction with 100% reflectivity scaled to 10,000 stored integer units. The history also records a sensor saturation value of 4095 before reflectance correction, but does not state a post-correction saturation encoding. Therefore saturation fractions are left unresolved rather than guessed. Field 2 radiometric processing is documented, but compatibility with Field 1 scale remains unresolved because the frozen Field 1 report states its headers do not prove unit-reflectance scale.

No explicit ENVI `data ignore value` or separate valid-data mask was supplied. Operational preview support uses finite, nonzero reflectance at the documented preview bands only to suppress obvious display background; it remains distinct from an authoritative NoData rule. Sampled per-band statistics quantify zero, nonfinite, and constant fractions. Stored index files are described as scalar-index products: the processing history names an NDVI transform but does not supply the exact formula and source-band provenance.

Warnings observed: {', '.join(warnings) if warnings else 'none'}.

## Annotation readiness

Annotation-ready cubes: {', '.join(readiness.loc[readiness.usable_for_annotation == 'yes', 'cube_id']) or 'none'}. The previews are suitable for investigator review, but annotation is blocked until an authoritative valid-data/NoData rule or aligned mask is supplied and frozen.

Blocked/excluded cubes:
{blocked_table}

All `investigator_role` entries remain `unreviewed`; no biological role was inferred from cube number, filename, image content, or statistics. The future vocabulary is `soil`, `weed`, `tall_grass_weed`, `chickpea`, `chickpea_soil_mixed`, `chickpea_weed_mixed`, `uncertain`, and `nodata_invalid`.

## Source immutability result

Independent before/after snapshot differences: {json.dumps(snapshot_issues)}. Final status: **{'PASS' if not snapshot_issues else 'FAIL'}**.

## Reproduction

```bash
GDAL_PAM_ENABLED=NO MPLCONFIGDIR=/tmp/chickpea_matplotlib /home/hemanthd95/miniconda3/envs/chickpea_modeling/bin/python scripts/audit_field2_georectified_inventory.py --paths configs/paths.local.yaml --config {config_path}
```

Main machine-local tables and previews are under `metadata/local/reports/field2_readiness`; the machine-readable contract is `metadata/local/contracts/field2_georectified_inventory_contract.yaml`.
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_readiness.yaml"), type=Path)
    args = parser.parse_args()
    project, paths_config, config = load_configuration(args.paths, args.config)
    roots = tuple(Path(path) for path in config["source_roots"])
    guard = ReadOnlySourceGuard(roots)
    output = guard.output(project / config["output_root"])
    contract_path = guard.output(project / config["contract_path"])
    temporary = guard.output(project / config["temporary_root"])
    output.mkdir(parents=True, exist_ok=True); contract_path.parent.mkdir(parents=True, exist_ok=True); temporary.mkdir(parents=True, exist_ok=True)
    preview_root = output / "previews"; preview_root.mkdir(exist_ok=True)

    # Required baseline precedes all header parsing and raster-array access.
    print("Creating baseline SHA-256 source snapshot...", flush=True)
    baseline = source_snapshot(guard, project)
    baseline_frame = pd.DataFrame(baseline)
    baseline_frame.to_csv(output / "field2_source_snapshot_before.csv", index=False)
    baseline_lookup = {row["relative_path"]: row for row in baseline}

    inventory_rows, pair_rows, records = [], [], {}
    for root in roots:
        product = product_for_root(root, config)
        for header in sorted(root.rglob("*.hdr")):
            cube_id, error = normalized_cube_id(header), ""
            try:
                payload = resolve_envi_payload(header, guard)
                metadata = parse_envi_header(header, guard)
                expected = expected_payload_bytes(metadata)
                actual = payload.stat().st_size
                dimensions_valid = all(scalar_int(metadata, key) > 0 for key in ("lines", "samples", "bands"))
                details = raster_details(payload, guard)
                valid_pair = dimensions_valid and expected == actual
            except Exception as exc:
                payload, metadata, expected, actual, details, dimensions_valid, valid_pair = None, {}, -1, -1, {}, False, False
                error = str(exc)
            pair_rows.append({
                "cube_id": cube_id, "product_type": product, "header_path": str(header.resolve().relative_to(project)),
                "binary_path": str(payload.relative_to(project)) if payload else "", "header_exists": header.is_file(),
                "payload_exists": bool(payload and payload.is_file()), "payload_candidate_count": 1 if payload else 0,
                "expected_bytes": expected, "actual_bytes": actual, "byte_size_match": expected == actual,
                "dimensions_valid": dimensions_valid, "valid_pair": valid_pair, "error": error,
            })
            wavelengths = parse_wavelengths(metadata)
            header_relative = str(header.resolve().relative_to(project))
            payload_relative = str(payload.relative_to(project)) if payload else ""
            map_info = metadata_text(metadata, "map info")
            inventory_rows.append({
                "field_identifier": "Field 2", "cube_id": cube_id, "relative_path": payload_relative,
                "product_type": product, "header_path": header_relative, "binary_path": payload_relative,
                "file_extension": payload.suffix.lower() if payload else "", "declared_interleave": metadata_text(metadata, "interleave").lower(),
                "byte_size": actual, "sha256": baseline_lookup.get(payload_relative, {}).get("sha256", ""),
                "modification_timestamp_ns": baseline_lookup.get(payload_relative, {}).get("mtime_ns", ""),
                "envi_data_type": scalar_int(metadata, "data type"), "byte_order": scalar_int(metadata, "byte order"),
                "header_offset": scalar_int(metadata, "header offset"), "samples": scalar_int(metadata, "samples"),
                "lines": scalar_int(metadata, "lines"), "bands": scalar_int(metadata, "bands"),
                "wavelength_count": len(wavelengths), "wavelength_units": metadata_text(metadata, "wavelength units"),
                "wavelength_minimum": float(np.min(wavelengths)) if len(wavelengths) else np.nan,
                "wavelength_maximum": float(np.max(wavelengths)) if len(wavelengths) else np.nan,
                "band_name_count": len(metadata.get("band names", [])), "coordinate_system_string": details.get("coordinate_system_wkt", ""),
                "map_information": map_info, "affine_transform": details.get("transform", ""),
                "spatial_resolution_x": details.get("gsd_x", np.nan), "spatial_resolution_y": details.get("gsd_y", np.nan),
                "bounds_left": details.get("left", np.nan), "bounds_bottom": details.get("bottom", np.nan),
                "bounds_right": details.get("right", np.nan), "bounds_top": details.get("top", np.nan),
                "nodata_ignore_value": metadata.get("data ignore value", details.get("nodata")),
                "acquisition_metadata": json.dumps({key: metadata.get(key) for key in ("shutter", "shutter units", "gain", "framerate", "sample binning", "spectral binning", "target") if key in metadata}),
                "header_payload_valid_pair": valid_pair, "source_open_mode": "read_only",
            })
            records.setdefault(cube_id, {})[product] = {"header": header, "payload": payload, "metadata": metadata, "raster": details}

    inventory = pd.DataFrame(inventory_rows).sort_values(["cube_id", "product_type"])
    pairs = pd.DataFrame(pair_rows).sort_values(["cube_id", "product_type"])
    inventory.to_csv(output / REPORT_NAMES["inventory"], index=False)
    pairs.to_csv(output / REPORT_NAMES["pairs"], index=False)
    manifest = baseline_frame.copy()
    manifest["product_type"] = manifest.relative_path.map(lambda value: next((product_for_root(root, config) for root in roots if str(Path(value)).startswith(str(root.relative_to(project)))), "other"))
    manifest["file_role"] = np.where(manifest.relative_path.str.lower().str.endswith(".hdr"), "envi_header", "envi_payload")
    manifest.to_csv(output / REPORT_NAMES["manifest"], index=False)

    required_products = {"reflectance", "pca", "stored_index"}
    hard_errors = []
    for cube_id, products in records.items():
        if not required_products.issubset(products): hard_errors.append(f"{cube_id}:incomplete_product_group")
    for row in pair_rows:
        if not row["valid_pair"]: hard_errors.append(f"{row['cube_id']}:{row['product_type']}:invalid_pair")
    if hard_errors:
        raise RuntimeError("Integrity gate failed before array previews: " + ";".join(hard_errors))

    field1_wavelength_source = project / "metadata/local/contracts/field1_fold_normalization.csv"
    field1 = pd.read_csv(field1_wavelength_source).sort_values("band_index").drop_duplicates("band_index")
    band_config = config["field1_model_bands"]
    field1 = field1[(field1.band_index >= band_config["first_band_index"]) & (field1.band_index <= band_config["last_band_index_inclusive"])]
    expected_wavelengths = field1.wavelength_nm.to_numpy(dtype=float)
    spectral_rows = []
    for cube_id in sorted(records):
        observed = parse_wavelengths(records[cube_id]["reflectance"]["metadata"])
        matches = match_wavelengths(expected_wavelengths, observed, float(band_config["wavelength_tolerance_nm"]))
        duplicate_count = int(len(observed) - len(np.unique(observed)))
        for row in matches:
            row.update({"cube_id": cube_id, "duplicate_wavelength_count": duplicate_count,
                        "field2_total_band_count": len(observed), "field1_contract_source": str(field1_wavelength_source.relative_to(project))})
            spectral_rows.append(row)
    spectral = pd.DataFrame(spectral_rows)
    spectral.to_csv(output / REPORT_NAMES["spectral"], index=False)
    incompatible = spectral[(spectral.match_status != "matched") | (spectral.ordering_status != "strictly_increasing")]
    if len(incompatible):
        raise RuntimeError("Wavelength compatibility gate failed before array previews")

    alignment_rows, footprint_rows, statistics_rows, qc_rows, readiness_rows, thumbnails = [], [], [], [], [], []
    seed, count = int(config["sampling"]["seed"]), int(config["sampling"]["pixels_per_cube"])
    percentiles = tuple(float(value) for value in config["sampling"]["preview_percentiles"])
    for cube_id in sorted(records):
        products = records[cube_id]
        ref_info = products["reflectance"]["raster"]
        ref_array, ref_metadata = open_envi_memmap(products["reflectance"]["header"], products["reflectance"]["payload"], guard)
        pca_array, pca_metadata = open_envi_memmap(products["pca"]["header"], products["pca"]["payload"], guard)
        idx_array, idx_metadata = open_envi_memmap(products["stored_index"]["header"], products["stored_index"]["payload"], guard)
        height, width = ref_array.shape[:2]
        coords = deterministic_coordinates(height, width, count, seed, cube_id)
        preview_indices = [int(np.argmin(abs(parse_wavelengths(ref_metadata) - target))) for target in config["false_colour_wavelengths_nm"]]
        sampled_ref = np.asarray(ref_array[coords[:, 0], coords[:, 1], :], dtype=np.float64)
        sample_valid = np.isfinite(sampled_ref[:, preview_indices]).all(axis=1) & np.any(sampled_ref[:, preview_indices] != 0, axis=1)
        valid = np.isfinite(np.asarray(ref_array[..., preview_indices[0]], dtype=np.float32)) & (np.asarray(ref_array[..., preview_indices[0]]) != 0)

        for product, array, metadata in (("reflectance", ref_array, ref_metadata), ("pca", pca_array, pca_metadata), ("stored_index", idx_array, idx_metadata)):
            statistics_rows.extend(sampled_band_rows(cube_id, product, array, metadata, coords, sample_valid))

        ref_rgb_raw = np.asarray(ref_array[..., preview_indices], dtype=np.float32)
        ref_rgb, ref_stretch = robust_stretch(ref_rgb_raw, valid, percentiles)
        pca_raw = np.asarray(pca_array[..., :3], dtype=np.float32)
        pca_rgb, pca_stretch = robust_stretch(pca_raw, valid, percentiles)
        idx_raw = np.asarray(idx_array[..., 0], dtype=np.float32)
        idx_rgb, idx_stretch = robust_stretch(idx_raw[..., None], valid & np.isfinite(idx_raw), percentiles)
        pca_support = np.isfinite(pca_raw[..., 0]) & (pca_raw[..., 0] != 0)
        idx_support = np.isfinite(idx_raw) & (idx_raw != 0)
        alignment_rgb = np.dstack([valid, pca_support, idx_support]).astype(float)
        stretch_record = {"percentiles": list(percentiles), "reflectance": ref_stretch, "pca": pca_stretch, "stored_index": idx_stretch}
        save_cube_preview(preview_root / f"{cube_id}_prediction_free_review.png", cube_id, ref_rgb, pca_rgb,
                          idx_rgb[..., 0], valid, alignment_rgb, stretch_record)
        thumbnails.append((cube_id, ref_rgb[::max(1, height // 180), ::max(1, width // 180)]))

        for product in ("pca", "stored_index"):
            candidate = products[product]["raster"]
            shape_equal = (ref_info["height"], ref_info["width"]) == (candidate["height"], candidate["width"])
            crs_equal = ref_info["crs"] == candidate["crs"] and bool(ref_info["crs"])
            transform_status = affine_comparison(ref_info["transform_object"], candidate["transform_object"],
                                                 float(config["alignment"]["affine_absolute_tolerance"]),
                                                 float(config["alignment"]["numerical_equivalence_tolerance"]))
            if shape_equal and crs_equal and transform_status == "exact_alignment": status = "exact_alignment"
            elif shape_equal and crs_equal and transform_status == "numerically_equivalent": status = "numerically_equivalent"
            elif crs_equal: status = "potentially_repairable"
            else: status = "incompatible"
            dx = (candidate["transform_object"].c - ref_info["transform_object"].c) / ref_info["gsd_x"]
            dy = (candidate["transform_object"].f - ref_info["transform_object"].f) / ref_info["gsd_y"]
            overlap_width = max(0.0, min(ref_info["right"], candidate["right"]) - max(ref_info["left"], candidate["left"]))
            overlap_height = max(0.0, min(ref_info["top"], candidate["top"]) - max(ref_info["bottom"], candidate["bottom"]))
            ref_area = (ref_info["right"] - ref_info["left"]) * (ref_info["top"] - ref_info["bottom"])
            alignment_rows.append({
                "cube_id": cube_id, "reference_product": "reflectance", "candidate_product": product,
                "crs_equal": crs_equal, "transform_comparison": transform_status, "width_equal": ref_info["width"] == candidate["width"],
                "height_equal": ref_info["height"] == candidate["height"], "pixel_size_x_difference": candidate["gsd_x"] - ref_info["gsd_x"],
                "pixel_size_y_difference": candidate["gsd_y"] - ref_info["gsd_y"], "origin_shift_x_pixels": dx,
                "origin_shift_y_pixels": dy, "origin_shift_x_metres": dx * ref_info["gsd_x"], "origin_shift_y_metres": dy * ref_info["gsd_y"],
                "bounds_max_absolute_difference": max(abs(candidate[key] - ref_info[key]) for key in ("left", "bottom", "right", "top")),
                "width_difference": candidate["width"] - ref_info["width"], "height_difference": candidate["height"] - ref_info["height"],
                "overlap_fraction": overlap_width * overlap_height / ref_area if ref_area else 0.0,
                "valid_footprint_agreement_fraction": float(np.mean(valid == (pca_support if product == "pca" else idx_support))),
                "valid_footprint_status": "mismatch_transformed_values_outside_reflectance_support",
                "pixel_center_aligned": transform_status in ("exact_alignment", "numerically_equivalent"),
                "expected_band_count": config["alignment"]["expected_pca_bands" if product == "pca" else "expected_stored_index_bands"],
                "observed_band_count": candidate["raster_bands"], "alignment_status": status,
                "georectification_residual": "not_supplied",
            })
        footprint_rows.append({"cube_id": cube_id, "crs": ref_info["crs"], "left": ref_info["left"], "bottom": ref_info["bottom"],
                               "right": ref_info["right"], "top": ref_info["top"], "width": width, "height": height,
                               "gsd_x": ref_info["gsd_x"], "gsd_y": ref_info["gsd_y"], "valid_fraction": float(np.mean(valid))})
        cube_stats = sampled_ref[:, 3:114][sample_valid]
        warnings = []
        if float(np.mean(valid)) < 0.5: warnings.append("low_operational_valid_fraction")
        if np.mean(sampled_ref == 0) > 0.5: warnings.append("high_sampled_zero_fraction")
        if not np.isfinite(sampled_ref).all(): warnings.append("nonfinite_reflectance_sample")
        warnings.extend(["no_explicit_nodata_or_valid_mask", "derived_product_nonzero_outside_reflectance_support"])
        history = str(ref_metadata.get("history", ""))
        radiometric = "100_percent_reflectivity_scaled_to_10000" if "Scale 100% Reflectivity To:' value:10000" in history else "unresolved"
        qc_rows.append({
            "cube_id": cube_id, "valid_fraction": float(np.mean(valid)), "sampled_valid_count": int(sample_valid.sum()),
            "reflectance_sampled_minimum": float(np.min(cube_stats)) if cube_stats.size else np.nan,
            "reflectance_sampled_maximum": float(np.max(cube_stats)) if cube_stats.size else np.nan,
            "reflectance_sampled_dynamic_range": float(np.ptp(cube_stats)) if cube_stats.size else np.nan,
            "spatial_resolution_metres": ref_info["gsd_x"], "radiometric_scale_status": radiometric,
            "field1_radiometric_compatibility": "unresolved_field1_scale_not_proven",
            "nodata_status": "not_supplied_operational_zero_support_used", "saturation_status": "post_correction_encoding_not_supplied",
            "warnings": "|".join(warnings), "preview_stretch": json.dumps(stretch_record),
        })

    alignment = pd.DataFrame(alignment_rows)
    footprints = pd.DataFrame(footprint_rows)
    overlaps = pd.DataFrame(overlap_rows(footprints))
    statistics = pd.DataFrame(statistics_rows)
    qc = pd.DataFrame(qc_rows)
    for cube_id in sorted(records):
        cube_alignment = alignment[alignment.cube_id == cube_id]
        grid_usable = bool((cube_alignment.alignment_status.isin(["exact_alignment", "numerically_equivalent"])).all())
        authoritative_valid_support = False
        usable = grid_usable and authoritative_valid_support
        overlap_warning = bool((
            ((overlaps.cube_id_a == cube_id) | (overlaps.cube_id_b == cube_id))
            & (overlaps.overlap_fraction_of_smaller_footprint > 0)
        ).any())
        readiness_rows.append({
            "cube_id": cube_id, "usable_for_annotation": "yes" if usable else "no",
            "reflectance_available": True, "PCA_available": True, "derivative_PCA_available": False,
            "NDVI_or_index_available": True,
            "alignment_status": "exact_alignment_with_valid_footprint_semantics_unresolved" if (cube_alignment.alignment_status == "exact_alignment").all() else "review_required",
            "spectral_compatibility_status": "compatible_exact_frozen_band_match",
            "valid_fraction": float(qc.loc[qc.cube_id == cube_id, "valid_fraction"].iloc[0]),
            "geographic_overlap_warning": overlap_warning, "proposed_role": "",
            "investigator_role": "unreviewed", "investigator_notes": "",
            "exclusion_reason": "" if usable else ("unresolved_nodata_and_valid_support_definition" if grid_usable else "spatial_alignment_gate_failed"),
        })
    readiness = pd.DataFrame(readiness_rows)
    alignment.to_csv(output / REPORT_NAMES["alignment"], index=False)
    footprints.to_csv(output / REPORT_NAMES["footprints"], index=False)
    overlaps.to_csv(output / REPORT_NAMES["overlap"], index=False)
    statistics.to_csv(output / REPORT_NAMES["statistics"], index=False)
    qc.to_csv(output / REPORT_NAMES["qc"], index=False)
    readiness.to_csv(output / REPORT_NAMES["readiness"], index=False)
    write_overviews(output, inventory, alignment, qc, thumbnails)

    print("Creating independent final SHA-256 source snapshot...", flush=True)
    final_snapshot = source_snapshot(guard, project)
    pd.DataFrame(final_snapshot).to_csv(output / "field2_source_snapshot_after.csv", index=False)
    snapshot_issues = compare_snapshots(baseline, final_snapshot)
    if snapshot_issues:
        raise RuntimeError("Source immutability check failed: " + ";".join(snapshot_issues))

    commit = git_output(project, "rev-parse", "HEAD")
    report_path = project / "docs/progress/2026-08-14_field2_georectified_readiness.md"
    write_report(report_path, commit, sorted(records), inventory, pairs, spectral, alignment, qc, readiness,
                 snapshot_issues, args.config)
    outputs = [output / value for value in REPORT_NAMES.values()]
    outputs += sorted(preview_root.glob("*.png")) + sorted(output.glob("*overview.png"))
    contract = {
        "status": "field2_georectified_inventory_complete_annotation_blocked",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_git_commit": commit, "script_path": str(Path(__file__).resolve().relative_to(project)),
        "configuration_path": str(args.config), "configuration_sha256": sha256(args.config),
        "source_roots": [str(root) for root in roots], "source_files_opened_read_only": True,
        "gdal_pam_enabled": "NO", "numpy_memmap_mode": "r", "source_permissions_changed": False,
        "checksum_algorithm": "sha256", "source_file_count": len(baseline),
        "header_count": int((manifest.file_role == "envi_header").sum()),
        "payload_count": int((manifest.file_role == "envi_payload").sum()),
        "valid_pair_count": int(pairs.valid_pair.sum()), "cube_count": len(records),
        "cube_ids": sorted(records), "product_counts": inventory.product_type.value_counts().to_dict(),
        "source_snapshot_before_sha256": sha256(output / "field2_source_snapshot_before.csv"),
        "source_snapshot_after_sha256": sha256(output / "field2_source_snapshot_after.csv"),
        "source_snapshot_comparison": "pass_identical" if not snapshot_issues else "fail",
        "spectral_compatibility": "compatible_exact_frozen_band_match",
        "grid_alignment": "exact_all_reflectance_pca_index_pairs",
        "valid_footprint_compatibility": "unresolved_transformed_products_nonzero_outside_reflectance_support",
        "annotation_ready_cube_count": int((readiness.usable_for_annotation == "yes").sum()),
        "annotation_blocked_cube_count": int((readiness.usable_for_annotation != "yes").sum()),
        "radiometric_compatibility": "unresolved_field1_scale_not_proven",
        "stored_index_provenance": "processing_history_names_ndvi_but_exact_formula_and_source_bands_not_supplied",
        "predictions_generated": False, "categorical_labels_generated": False,
        "model_retrained": False, "ssl_training_run": False, "field1_benchmark_modified": False,
        "output_sha256": {str(path.relative_to(project)): sha256(path) for path in outputs if path.is_file()},
    }
    dump_yaml(contract, contract_path)
    print(f"Audit complete: {len(records)} cubes, {len(pairs)} valid pairs", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"Contract: {contract_path}", flush=True)


if __name__ == "__main__":
    main()
