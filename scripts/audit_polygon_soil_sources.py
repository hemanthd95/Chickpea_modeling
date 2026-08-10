#!/usr/bin/env python
"""Audit soil evidence inside investigator chickpea-region polygons.

This read-only diagnostic was added after direct NDVI calculated from stored
Pika-L band values classified zero soil pixels in every annotated polygon.
It compares that diagnostic calculation with the existing, provenance-tracked
soil masks and inventories stored NDVI products.  It never changes labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
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
from rasterio.features import rasterize
from spectral.io import envi
import yaml

from chickpea_ssl.data import load_records
from scripts.audit_chickpea_region_annotations import (
    metres_per_pixel,
    normalized_ring,
    observed_support,
    planter_alley_features,
    reconcile_exports,
)


CMAP = ListedColormap([
    "#000000",  # outside polygon / NoData
    "#20C76F",  # investigator polygon core
    "#8B5A2B",  # existing NDVI-derived soil mask
    "#3B82F6",  # plausible stored NDVI product only
    "#FDE047",  # existing mask and stored product agree
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    usable = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    result[usable] = numerator[usable] / denominator[usable]
    return result


def nearest_band(wavelengths: np.ndarray, target_nm: float) -> int:
    return int(np.nanargmin(np.abs(wavelengths - target_nm)))


def source_root(paths: dict, row: pd.Series) -> Path:
    field1 = paths["field1"]
    source = str(row["source"])
    if source == "field1.derived_ssl":
        return Path(field1["derived_ssl"])
    if source == "field1.masks_emmanuel":
        return Path(field1["masks_emmanuel"])
    if source == "field1.reflectance":
        batches = field1.get("reflectance_processing_batches", field1.get("reflectance_dates", {}))
        return Path(batches[str(row["processing_batch"])])
    raise ValueError(f"Unsupported catalog source for NDVI audit: {source}")


def product_array(path: Path, expected_shape: tuple[int, int]) -> tuple[np.ndarray | None, dict]:
    result = {"read_status": "", "bands": 0, "dtype": "", "q01": np.nan,
              "median": np.nan, "q99": np.nan, "plausible_ndvi_scale": False}
    try:
        with rasterio.open(path) as dataset:
            result["bands"] = dataset.count
            result["dtype"] = dataset.dtypes[0]
            if (dataset.height, dataset.width) != expected_shape:
                result["read_status"] = (
                    f"shape_mismatch_{dataset.height}x{dataset.width}_vs_"
                    f"{expected_shape[0]}x{expected_shape[1]}"
                )
                return None, result
            array = dataset.read(1, masked=True).astype(np.float32).filled(np.nan)
    except Exception as exc:
        result["read_status"] = f"read_error:{type(exc).__name__}:{exc}"
        return None, result
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        result["read_status"] = "no_finite_values"
        return None, result
    q01, median, q99 = np.quantile(finite, [0.01, 0.50, 0.99])
    result.update({"read_status": "readable", "q01": float(q01),
                   "median": float(median), "q99": float(q99)})
    # This is deliberately a plausibility gate, not an inferred rescaling rule.
    plausible = bool(q01 >= -1.2 and q99 <= 1.2 and q99 > q01)
    result["plausible_ndvi_scale"] = plausible
    return array, result


def agreement(reference: np.ndarray, candidate: np.ndarray, support: np.ndarray) -> dict:
    ref = reference & support
    cand = candidate & support
    intersection = int((ref & cand).sum())
    union = int((ref | cand).sum())
    return {
        "reference_soil_pixels": int(ref.sum()),
        "candidate_soil_pixels": int(cand.sum()),
        "intersection_pixels": intersection,
        "iou": float(intersection / union) if union else np.nan,
        "precision_vs_reference": float(intersection / cand.sum()) if cand.any() else np.nan,
        "recall_vs_reference": float(intersection / ref.sum()) if ref.any() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--annotations-root", type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    policy = config.get("polygon_soil_source_audit", {})
    threshold = float(policy.get("soil_ndvi_threshold", 0.30))
    seam_snap = float(config.get("chickpea_region_polygon_audit", {}).get(
        "seam_snap_maximum_m", 0.30
    ))
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    annotation_root = args.annotations_root or local / "annotations" / "chickpea_regions"
    json_path = annotation_root / "field1_chickpea_region_annotations.json"
    csv_path = annotation_root / "field1_chickpea_region_vertices.csv"
    geojson_path = annotation_root / "field1_chickpea_region_annotations.geojson"
    _, geojson, _ = reconcile_exports(json_path, csv_path, geojson_path)

    polygons: dict[str, list[dict]] = defaultdict(list)
    for feature in geojson["features"]:
        cube_id = str(feature["properties"]["cube_id"])
        coordinates = feature["geometry"]["coordinates"][0]
        if coordinates and coordinates[0] == coordinates[-1]:
            coordinates = coordinates[:-1]
        ring, _, _ = normalized_ring(coordinates, seam_snap)
        polygons[cube_id].append({"type": "Polygon", "coordinates": [ring]})

    alley_path = local / "annotations" / "planter_tracks" / "field1_planter_track_annotations.geojson"
    alleys = planter_alley_features(alley_path, seam_snap)
    records = {
        record.cube_id: record for record in load_records(
            args.paths, local / "authoritative_manifest.csv"
        ) if record.cube_id in polygons
    }
    catalog = pd.read_csv(local / "project_catalog.csv", dtype=str).fillna("")
    ndvi_catalog = catalog[
        catalog["file_role"].eq("ndvi")
        & catalog["extension"].isin([".bip", ".bil", ".bsq", ".tif", ".tiff"])
    ].copy()

    reports = local / "reports" / "mask_refinement" / "polygon_soil_source_audit"
    reports.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict] = []
    inventory_rows: list[dict] = []
    panels: list[tuple[str, np.ndarray, dict]] = []
    output_paths: list[Path] = []

    for number, cube_id in enumerate(sorted(records, key=lambda item: int(item.split("cube")[-1])), 1):
        record = records[cube_id]
        image = envi.open(str(record.header), str(record.data))
        wavelengths = np.asarray(image.metadata["wavelength"], dtype=np.float32)
        red_index = nearest_band(wavelengths, 670.0)
        nir_index = nearest_band(wavelengths, 800.0)
        with rasterio.open(record.data) as reference:
            shape = (reference.height, reference.width)
            transform = reference.transform
            observed = observed_support(reference)
            red, nir = reference.read([red_index + 1, nir_index + 1]).astype(np.float32)
        red[~observed] = np.nan
        nir[~observed] = np.nan
        raw_ndvi = safe_ratio(nir - red, nir + red)
        raw_soil = observed & np.isfinite(raw_ndvi) & (raw_ndvi < threshold)

        core = rasterize(
            [(geometry, 1) for geometry in polygons[cube_id]], out_shape=shape,
            transform=transform, fill=0, dtype="uint8", all_touched=False,
        ).astype(bool) & observed
        alley_shapes = [(geometry, 1) for geometry in alleys.get(cube_id, [])]
        alley = (
            rasterize(alley_shapes, out_shape=shape, transform=transform, fill=0,
                      dtype="uint8", all_touched=True).astype(bool)
            if alley_shapes else np.zeros(shape, dtype=bool)
        )
        eligible = core & ~alley

        authoritative_soil = None
        if record.soil_mask is not None:
            with rasterio.open(record.soil_mask) as dataset:
                if (dataset.height, dataset.width) != shape:
                    raise ValueError(f"Soil-mask shape mismatch for {cube_id}: {record.soil_mask}")
                authoritative_soil = dataset.read(1) > 0

        raw_metrics = (
            agreement(authoritative_soil, raw_soil, observed)
            if authoritative_soil is not None else {}
        )
        candidates = ndvi_catalog[ndvi_catalog["cube_id_inferred"].eq(cube_id)]
        product_options: list[tuple[float, Path, np.ndarray, dict]] = []
        for _, row in candidates.iterrows():
            path = source_root(paths, row) / row["relative_path"]
            array, metadata = product_array(path, shape)
            inventory = {
                "cube_id": cube_id, "source": row["source"],
                "relative_path": row["relative_path"], "absolute_path": str(path),
                **metadata,
            }
            if array is not None and metadata["plausible_ndvi_scale"]:
                product_soil = observed & np.isfinite(array) & (array < threshold)
                if authoritative_soil is not None:
                    metrics = agreement(authoritative_soil, product_soil, observed)
                    inventory.update({f"agreement_{key}": value for key, value in metrics.items()})
                    score = float(metrics["iou"]) if np.isfinite(metrics["iou"]) else -1.0
                else:
                    score = -1.0
                product_options.append((score, path, product_soil, inventory))
            inventory_rows.append(inventory)

        best_product = None
        best_path = ""
        best_metrics: dict = {}
        product_status = "none"
        if product_options:
            product_options.sort(key=lambda item: (-item[0], str(item[1])))
            _, selected_path, best_product, selected_inventory = product_options[0]
            best_path = str(selected_path)
            product_status = (
                "compared_with_authoritative_soil_mask"
                if authoritative_soil is not None else "plausible_scale_unvalidated_without_soil_mask"
            )
            if authoritative_soil is not None:
                best_metrics = agreement(authoritative_soil, best_product, observed)

        comparison_rows.append({
            "cube_id": cube_id,
            "has_authoritative_soil_mask": authoritative_soil is not None,
            "eligible_polygon_core_pixels": int(eligible.sum()),
            "authoritative_soil_inside_eligible_core": (
                int((authoritative_soil & eligible).sum()) if authoritative_soil is not None else np.nan
            ),
            "authoritative_soil_fraction_inside_eligible_core": (
                float((authoritative_soil & eligible).sum() / eligible.sum())
                if authoritative_soil is not None and eligible.any() else np.nan
            ),
            "raw_band_ndvi_soil_inside_eligible_core": int((raw_soil & eligible).sum()),
            "raw_band_ndvi_soil_fraction_inside_eligible_core": (
                float((raw_soil & eligible).sum() / eligible.sum()) if eligible.any() else np.nan
            ),
            "raw_band_ndvi_iou_vs_authoritative_soil": raw_metrics.get("iou", np.nan),
            "stored_ndvi_candidates": len(candidates),
            "best_stored_ndvi_product": best_path,
            "stored_ndvi_product_status": product_status,
            "best_product_soil_inside_eligible_core": (
                int((best_product & eligible).sum()) if best_product is not None else np.nan
            ),
            "best_product_iou_vs_authoritative_soil": best_metrics.get("iou", np.nan),
        })

        qc = np.zeros(shape, dtype=np.uint8)
        qc[eligible] = 1
        if authoritative_soil is not None:
            qc[eligible & authoritative_soil] = 2
        if best_product is not None:
            qc[eligible & best_product] = 3
        if authoritative_soil is not None and best_product is not None:
            qc[eligible & authoritative_soil & best_product] = 4
        panels.append((cube_id, qc, comparison_rows[-1]))
        print(
            f"Audited {number}/{len(records)} {cube_id}: "
            f"authoritative soil in core={comparison_rows[-1]['authoritative_soil_inside_eligible_core']}; "
            f"raw-band soil={comparison_rows[-1]['raw_band_ndvi_soil_inside_eligible_core']}; "
            f"stored NDVI candidates={len(candidates)}"
        )

    comparison = pd.DataFrame(comparison_rows)
    inventory = pd.DataFrame(inventory_rows)
    comparison_path = reports / "polygon_soil_source_comparison.csv"
    inventory_path = reports / "stored_ndvi_product_inventory.csv"
    comparison.to_csv(comparison_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    output_paths.extend([comparison_path, inventory_path])

    columns = 4
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, rows * 4.1), squeeze=False)
    for axis, panel in zip(axes.flat, panels):
        cube_id, qc, summary = panel
        step = max(1, math.ceil(max(qc.shape) / 650))
        axis.imshow(qc[::step, ::step], cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
        auth = summary["authoritative_soil_inside_eligible_core"]
        product_iou = summary["best_product_iou_vs_authoritative_soil"]
        auth_text = "none" if pd.isna(auth) else f"{int(auth):,}"
        iou_text = "n/a" if pd.isna(product_iou) else f"{product_iou:.3f}"
        axis.set_title(f"{cube_id}\nsoil in polygons={auth_text}; product IoU={iou_text}", fontsize=8)
        axis.set_axis_off()
    for axis in axes.flat[len(panels):]:
        axis.set_axis_off()
    figure.suptitle(
        "Field 1 polygon soil-source audit\n"
        "green=eligible polygon; brown=existing soil mask; blue=stored NDVI product; yellow=agreement",
        fontsize=14,
    )
    overview_path = reports / "polygon_soil_source_audit_overview.png"
    figure.savefig(overview_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    output_paths.append(overview_path)

    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    contract = {
        "status": "soil_source_audit_complete_no_label_change",
        "field": "Field 1",
        "soil_ndvi_threshold": threshold,
        "raw_reflectance_dn_ndvi_role": "diagnostic_only_not_accepted_for_soil_labels",
        "existing_soil_mask_role": "preferred_observed_soil_source_pending_audit_result",
        "stored_ndvi_product_role": "candidate_requires_scale_and_alignment_validation",
        "label_expansion_cubes_without_soil_mask": "remain_unresolved_until_ndvi_product_is_validated",
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {
            "annotations_json": sha256(json_path),
            "vertices_csv": sha256(csv_path),
            "annotations_geojson": sha256(geojson_path),
            "planter_annotations_geojson": sha256(alley_path),
        },
        "output_hashes": {str(path.relative_to(reports)): sha256(path) for path in output_paths},
    }
    contract_path = contracts / "field1_polygon_soil_source_audit_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Cubes audited: {len(comparison)}")
    print(f"Cubes with existing soil masks: {int(comparison.has_authoritative_soil_mask.sum())}")
    print(f"Stored NDVI files inventoried: {len(inventory)}")
    print(f"Comparison: {comparison_path}")
    print(f"NDVI inventory: {inventory_path}")
    print(f"Visual QC: {overview_path}")
    print(f"Contract: {contract_path}")
    print("Audit only: no mask changed, no model retrained, and Field 2 remained locked.")


if __name__ == "__main__":
    main()
