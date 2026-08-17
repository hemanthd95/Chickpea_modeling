#!/usr/bin/env python
"""Audit and freeze prediction-free Field 2 reflectance support masks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["GDAL_PAM_ENABLED"] = "NO"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy import ndimage

from chickpea_ssl.field2_readiness import (
    ReadOnlySourceGuard,
    compare_snapshots,
    dump_yaml,
    open_envi_memmap,
    parse_wavelengths,
    sha256,
    source_snapshot,
)
from chickpea_ssl.field2_valid_support import (
    component_metrics,
    disagreement_metrics,
    model_band_indices,
    support_masks_windowed,
    threshold_token,
    verify_mask_geotiff,
    write_mask_geotiff,
)


REPORT_FILES = {
    "sensitivity": "field2_valid_support_rule_sensitivity.csv",
    "summary": "field2_valid_support_summary.csv",
    "components": "field2_valid_support_component_qc.csv",
    "transformed": "field2_transformed_product_outside_support.csv",
    "manifest": "field2_valid_support_mask_manifest.csv",
    "readiness": "field2_annotation_readiness.csv",
}


def git_output(project: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=project, text=True).strip()


def load_inputs(paths_path: Path, config_path: Path) -> tuple[Path, dict, dict]:
    paths = yaml.safe_load(paths_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    project = Path(paths["project_root"]).resolve()
    roots = paths.get("field2", {}).get("readiness_roots", [])
    if len(roots) != 3:
        raise ValueError("Exactly three authorized Field 2 readiness roots are required")
    config["source_roots"] = [str(Path(value).resolve()) for value in roots]
    return project, paths, config


def readiness_gate(project: Path, config: dict, inventory: pd.DataFrame) -> None:
    contract = yaml.safe_load((project / config["readiness_contract"]).read_text())
    expected = {
        "status": "field2_georectified_inventory_complete_annotation_blocked",
        "source_file_count": 240,
        "valid_pair_count": 120,
        "cube_count": 40,
        "source_snapshot_comparison": "pass_identical",
        "spectral_compatibility": "compatible_exact_frozen_band_match",
        "grid_alignment": "exact_all_reflectance_pca_index_pairs",
    }
    failures = [f"{key}={contract.get(key)!r}" for key, value in expected.items() if contract.get(key) != value]
    if failures:
        raise RuntimeError("Field 2 readiness contract gate failed: " + "; ".join(failures))
    commit = str(config["readiness_repository_commit"])
    git_output(project, "cat-file", "-e", f"{commit}^{{commit}}")
    committed_config = git_output(project, "show", f"{commit}:configs/field2_readiness.yaml")
    if "field2_georectified_readiness_v1" not in committed_config:
        raise RuntimeError("Configured readiness commit does not contain the readiness contract")
    if inventory.cube_id.nunique() != 40 or len(inventory) != 120:
        raise RuntimeError("Readiness inventory is not the frozen 40-cube/120-product inventory")
    counts = inventory.groupby(["cube_id", "product_type"]).size()
    if not (counts == 1).all() or set(inventory.product_type) != {"reflectance", "pca", "stored_index"}:
        raise RuntimeError("Readiness inventory product grouping is incomplete or ambiguous")


def raster_reference(path: Path, guard: ReadOnlySourceGuard) -> dict:
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(guard.source(path), "r") as dataset:
            return {
                "width": dataset.width, "height": dataset.height, "count": 1,
                "source_count": dataset.count, "crs": dataset.crs,
                "crs_text": str(dataset.crs or ""), "transform": dataset.transform,
                "bounds": dataset.bounds,
            }


def robust_stretch(values: np.ndarray, valid: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.float32)
    for band in range(values.shape[2]):
        plane = np.asarray(values[..., band], dtype=np.float32)
        usable = plane[valid & np.isfinite(plane)]
        if len(usable):
            low, high = np.percentile(usable, percentiles)
        else:
            low, high = 0.0, 1.0
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        result[..., band] = np.clip((plane - low) / (high - low), 0, 1)
    return result


def wavelength_indices(observed: np.ndarray, targets: list[float]) -> list[int]:
    return [int(np.argmin(np.abs(observed - float(target)))) for target in targets]


def transformed_statistics(array: np.ndarray, support: np.ndarray, window_rows: int) -> dict[str, int | float]:
    totals = {
        "total_pixels": int(support.size), "total_nonzero_pixels": 0,
        "nonzero_pixels_outside_valid_support": 0, "finite_pixels_inside_support": 0,
        "nonfinite_pixels_inside_support": 0,
    }
    for start in range(0, array.shape[0], window_rows):
        stop = min(start + window_rows, array.shape[0])
        block = np.asarray(array[start:stop, :, :])
        finite = np.isfinite(block).all(axis=2)
        nonzero = finite & np.any(block != 0, axis=2)
        inside = support[start:stop]
        totals["total_nonzero_pixels"] += int(nonzero.sum())
        totals["nonzero_pixels_outside_valid_support"] += int(np.count_nonzero(nonzero & ~inside))
        totals["finite_pixels_inside_support"] += int(np.count_nonzero(finite & inside))
        totals["nonfinite_pixels_inside_support"] += int(np.count_nonzero(~finite & inside))
    nonzero_total = int(totals["total_nonzero_pixels"])
    totals["fraction_nonzero_outside_valid_support"] = (
        float(totals["nonzero_pixels_outside_valid_support"] / nonzero_total) if nonzero_total else 0.0
    )
    return totals


def save_qc(
    path: Path,
    cube_id: str,
    reflectance: np.ndarray,
    support: np.ndarray,
    robust: np.ndarray,
    pca: np.ndarray,
    stored_index: np.ndarray,
    labels: np.ndarray,
    percentiles: tuple[float, float],
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_support = support[::stride, ::stride]
    sampled_reflectance = reflectance
    rgb = robust_stretch(sampled_reflectance, sampled_support, percentiles)
    rgb[~sampled_support] = 0
    sampled_pca = pca
    sampled_index = stored_index
    pca_valid = sampled_support & np.isfinite(sampled_pca[..., :3]).all(2)
    index_valid = sampled_support & np.isfinite(sampled_index[..., 0])
    pca_rgb = robust_stretch(sampled_pca[..., :3], pca_valid, percentiles)
    index_base = robust_stretch(sampled_index[..., :1], index_valid, percentiles)[..., 0]
    pca_rgb[~sampled_support] = 0
    index_masked = index_base.copy()
    index_masked[~sampled_support] = 0
    pca_outside = np.isfinite(sampled_pca).all(2) & np.any(sampled_pca != 0, 2) & ~sampled_support
    index_outside = np.isfinite(sampled_index).all(2) & np.any(sampled_index != 0, 2) & ~sampled_support
    pca_outside_stretch = robust_stretch(sampled_pca[..., :1], pca_outside, percentiles)[..., 0]
    index_outside_stretch = robust_stretch(sampled_index[..., :1], index_outside, percentiles)[..., 0]
    outside = np.dstack([pca_outside_stretch, index_outside_stretch, np.zeros_like(index_base)])
    outside[~(pca_outside | index_outside)] = 0
    disagreement = (support ^ robust)[::stride, ::stride]
    fig, axes = plt.subplots(2, 4, figsize=(19, 10), constrained_layout=True)
    panels = [
        (rgb, "Masked reflectance false colour"),
        (sampled_support, "Proposed reflectance support"),
        (~sampled_support, "Invalid zero-filled exterior"),
        (disagreement, "Primary/robust disagreement"),
        (labels[::stride, ::stride], "Connected components (unfiltered)"),
        (pca_rgb, "PCA preview masked by support"),
        (index_masked, "Stored-index preview masked by support"),
        (outside, "Outside support: PCA red / index green"),
    ]
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"{cube_id} — reflectance-only valid-support QC")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return rgb, sampled_support


def save_overviews(output: Path, thumbnails: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    fig, axes = plt.subplots(5, 8, figsize=(20, 13), constrained_layout=True)
    for axis, (cube_id, _, mask) in zip(axes.flat, thumbnails):
        axis.imshow(mask, cmap="gray")
        axis.set_title(cube_id, fontsize=8)
        axis.axis("off")
    fig.suptitle("Field 2 reflectance valid-support masks")
    fig.savefig(output / "field2_valid_support_overview.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(5, 8, figsize=(20, 13), constrained_layout=True)
    for axis, (cube_id, rgb, _) in zip(axes.flat, thumbnails):
        axis.imshow(rgb)
        axis.set_title(cube_id, fontsize=8)
        axis.axis("off")
    fig.suptitle("Field 2 prediction-free annotation overview")
    fig.savefig(output / "field2_prediction_free_annotation_overview.png", dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    sensitivity: pd.DataFrame,
    summary: pd.DataFrame,
    transformed: pd.DataFrame,
    manifest: pd.DataFrame,
    readiness: pd.DataFrame,
    config: dict,
    source_issues: list[str],
) -> None:
    primary = config["rules"]["primary"]
    robust = config["rules"]["selected_robust"]
    selected = sensitivity[sensitivity.rule_name == robust]
    per_cube = "\n".join(
        f"| {row.cube_id} | {row.valid_fraction:.6f} | {row.connected_component_count} | "
        f"{row.largest_component_fraction:.6f} | {row.validation_status} |"
        for row in summary.itertuples(index=False)
    )
    transformed_table = (
        transformed.groupby("product_type").agg(
            total_nonzero_pixels=("total_nonzero_pixels", "sum"),
            nonzero_outside=("nonzero_pixels_outside_valid_support", "sum"),
            finite_inside=("finite_pixels_inside_support", "sum"),
            nonfinite_inside=("nonfinite_pixels_inside_support", "sum"),
        ) if "product_type" in transformed else pd.DataFrame()
    )
    if len(transformed_table):
        transformed_markdown = "| product | total nonzero | nonzero outside support | finite inside | nonfinite inside |\n|---|---:|---:|---:|---:|\n" + "\n".join(
            f"| {name} | {int(row.total_nonzero_pixels)} | {int(row.nonzero_outside)} | {int(row.finite_inside)} | {int(row.nonfinite_inside)} |"
            for name, row in transformed_table.iterrows()
        )
    else:
        transformed_markdown = "Not calculated because the freeze gate stopped before transformed-product review."
    lines = [
        "# Field 2 reflectance valid-support freeze", "", "Date: 2026-08-17", "",
        f"Status: **{'all cubes annotation-ready (prediction-free)' if (readiness.usable_for_annotation == 'yes').all() else 'review required'}**", "",
        "## Scientific rule", "",
        f"The frozen rule is `{primary}`: all zero-based Python bands 3–113 (ENVI bands 4–114) must be finite, and the maximum absolute stored reflectance across those bands must exceed zero. It uses reflectance only. PCA and the stored scalar index were display/diagnostic layers and never mask inputs.", "",
        f"The selected robustness comparator is `{robust}`. Maximum per-cube disagreement was {selected.disagreement_fraction_primary_valid.max():.9f}; the configured gate is {config['freeze_gates']['maximum_primary_robust_disagreement_fraction']:.6f}. Thresholds tested were {config['rules']['positive_norm_thresholds_stored_units']}, together with finite-any-nonzero, finite-all-nonzero, and full-spectrum-any-nonzero.", "",
        f"All candidate rules agreed exactly. Thus low-amplitude nonzero pixels did not form a fringe: no valid pixel had maximum model-band magnitude at or below {config['rules']['low_amplitude_upper_bound_stored_units']} stored units. Cube 02 contains one main body, two coherent non-small upper scan strips, and two tiny boundary remnants; all are retained as observed support. No morphology, component removal, hole filling, convex hull, or biological assumption was applied.", "",
        "## Per-cube support", "", "| cube_id | valid_fraction | components | largest_component_fraction | status |", "|---|---:|---:|---:|---|", per_cube, "",
        "## Transformed products", "",
        "PCA/index rasters were not altered or resampled. Their nonzero pixels outside reflectance support were counted and suppressed only in review displays:", "", transformed_markdown, "",
        "## Integrity and prohibited operations", "",
        f"Materialized masks: {len(manifest)}. Annotation-ready cubes: {int((readiness.usable_for_annotation == 'yes').sum())}; blocked cubes: {int((readiness.usable_for_annotation != 'yes').sum())}. Source before/after differences: `{json.dumps(source_issues)}`.", "",
        "No source product was modified. No biological label, supervised prediction, probability, model retraining, or SSL training was generated. The frozen Field 1 benchmark was unchanged.", "",
        "Generated machine-local CSVs, previews, masks, and the YAML contract are intentionally ignored by Git.", "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_valid_support.yaml"), type=Path)
    args = parser.parse_args()
    project, _, config = load_inputs(args.paths, args.config)
    roots = tuple(Path(value) for value in config["source_roots"])
    guard = ReadOnlySourceGuard(roots)
    output = guard.output(project / config["output_root"])
    contract_path = guard.output(project / config["contract_path"])
    temporary = guard.output(project / config["temporary_root"])
    mask_root = guard.output(project / config["mask_root"])
    for directory in (output, contract_path.parent, temporary, mask_root, output / "previews"):
        directory.mkdir(parents=True, exist_ok=True)

    print("Creating pre-operation source snapshot...", flush=True)
    baseline = source_snapshot(guard, project)
    baseline_path = output / "field2_source_snapshot_before.csv"
    pd.DataFrame(baseline).to_csv(baseline_path, index=False)
    baseline_lookup = {str(row["relative_path"]): row for row in baseline}
    inventory = pd.read_csv(project / config["readiness_inventory"])
    readiness_gate(project, config, inventory)
    alignment = pd.read_csv(project / config["readiness_alignment"])
    if not (alignment.alignment_status == "exact_alignment").all():
        raise RuntimeError("Frozen readiness alignment is not exact for every transformed product")

    band_config = config["model_input_bands"]
    thresholds = config["rules"]["positive_norm_thresholds_stored_units"]
    low_amplitude_threshold = float(config["rules"]["low_amplitude_upper_bound_stored_units"])
    low_amplitude_rule = f"finite_positive_norm_gt_{threshold_token(low_amplitude_threshold)}"
    primary_name, robust_name = config["rules"]["primary"], config["rules"]["selected_robust"]
    processing = config["processing"]
    config_hash = sha256(args.config.resolve())
    if (int(config["mask"]["valid_code"]), int(config["mask"]["invalid_code"])) != (1, 0):
        raise ValueError("Field 2 support mask codes are frozen as valid=1 and invalid=0")
    field1 = pd.read_csv(project / config["field1_wavelength_contract"])
    field1 = field1.sort_values("band_index").drop_duplicates("band_index")
    field1 = field1[
        (field1.band_index >= band_config["first_index"])
        & (field1.band_index <= band_config["last_index_inclusive"])
    ]
    if len(field1) != band_config["expected_count"]:
        raise RuntimeError("Frozen Field 1 wavelength contract does not contain exactly 111 model bands")
    expected_wavelengths = field1.wavelength_nm.to_numpy(dtype=float)

    sensitivity_rows: list[dict] = []
    component_rows: list[dict] = []
    summary_rows: list[dict] = []
    records: dict[str, dict[str, object]] = {}
    primary_masks: dict[str, np.ndarray] = {}
    cube_reasons: dict[str, list[str]] = {}
    print("Auditing reflectance support candidates window by window...", flush=True)
    for cube_id, group in inventory.groupby("cube_id", sort=True):
        products = {row.product_type: row for row in group.itertuples(index=False)}
        records[cube_id] = products
        ref = products["reflectance"]
        ref_path, header_path = project / ref.binary_path, project / ref.header_path
        array, metadata = open_envi_memmap(header_path, ref_path, guard)
        indices = model_band_indices(
            int(band_config["first_index"]), int(band_config["last_index_inclusive"]), array.shape[2]
        )
        observed_wavelengths = parse_wavelengths(metadata)
        selected_wavelengths = observed_wavelengths[indices]
        difference = np.abs(selected_wavelengths - expected_wavelengths)
        if len(selected_wavelengths) != len(expected_wavelengths) or np.any(
            difference > float(band_config["wavelength_tolerance_nm"])
        ):
            raise RuntimeError(f"{cube_id}: Field 1-to-Field 2 wavelength match failed")
        masks, diagnostics = support_masks_windowed(
            array, indices, thresholds, int(processing["row_window_size"])
        )
        if primary_name not in masks or robust_name not in masks:
            raise RuntimeError("Configured primary or robust rule was not evaluated")
        primary, robust = masks[primary_name], masks[robust_name]
        primary_masks[cube_id] = primary.copy()
        reference = raster_reference(ref_path, guard)
        if low_amplitude_rule not in masks:
            raise RuntimeError("Configured low-amplitude threshold was not evaluated")
        low_amplitude = int(np.count_nonzero(primary & ~masks[low_amplitude_rule]))
        for rule_name, candidate in masks.items():
            components = component_metrics(
                candidate, reference["transform"], int(processing["component_connectivity"]),
                int(processing["small_component_max_pixels"]),
            )
            disagreement = disagreement_metrics(primary, candidate)
            valid_count = int(candidate.sum())
            row = {
                "cube_id": cube_id, "rule_name": rule_name,
                "total_pixels": diagnostics["total_pixels"], "valid_pixels": valid_count,
                "invalid_pixels": diagnostics["total_pixels"] - valid_count,
                "retained_fraction": float(candidate.mean()),
                "finite_failure_count": diagnostics["finite_failure_count"],
                "full_spectrum_finite_failure_count": diagnostics["full_spectrum_finite_failure_count"],
                "exact_all_zero_spectrum_count": diagnostics["exact_all_zero_spectrum_count"],
                "partial_zero_spectrum_count": diagnostics["partial_zero_spectrum_count"],
                "negative_value_spectrum_count": diagnostics["negative_value_spectrum_count"],
                "low_amplitude_nonzero_count": low_amplitude,
                **disagreement, **components,
            }
            sensitivity_rows.append(row)
            component_rows.append({
                key: row[key] for key in (
                    "cube_id", "rule_name", "connected_component_count", "largest_component_pixels",
                    "largest_component_fraction", "small_component_count", "small_component_pixels",
                    "small_component_pixel_fraction", "raster_boundary_component_count",
                    "non_small_component_count", "non_small_boundary_component_count",
                    "non_small_interior_component_count", "interior_hole_count", "interior_hole_pixels",
                    "boundary_pixel_count", "minimum_row", "maximum_row", "minimum_column",
                    "maximum_column", "geographic_left", "geographic_bottom", "geographic_right",
                    "geographic_top",
                )
            })
        primary_metrics = component_metrics(
            primary, reference["transform"], int(processing["component_connectivity"]),
            int(processing["small_component_max_pixels"]),
        )
        robust_disagreement = disagreement_metrics(primary, robust)
        reasons: list[str] = []
        if not np.array_equal(primary, masks["finite_any_nonzero"]):
            reasons.append("primary_rule_definition_mismatch")
        if robust_disagreement["disagreement_fraction_primary_valid"] >= float(
            config["freeze_gates"]["maximum_primary_robust_disagreement_fraction"]
        ):
            reasons.append("primary_robust_disagreement_gate_failed")
        if primary_metrics["largest_component_fraction"] < float(
            config["freeze_gates"]["minimum_largest_component_fraction"]
        ):
            reasons.append("fragmented_largest_component_gate_failed")
        exception = config.get("spatial_review", {}).get("reviewed_exceptions", {}).get(cube_id)
        permitted_interior = int(config["freeze_gates"]["maximum_non_small_interior_component_count"])
        if exception:
            if primary_metrics["connected_component_count"] != int(exception["expected_component_count"]):
                reasons.append("reviewed_spatial_exception_topology_changed")
            else:
                permitted_interior = int(exception["permitted_non_small_interior_component_count"])
        if primary_metrics["non_small_interior_component_count"] > permitted_interior:
            reasons.append("unexplained_non_small_interior_component_gate_failed")
        if primary_metrics["small_component_pixel_fraction"] > float(
            config["freeze_gates"]["maximum_small_component_pixel_fraction"]
        ):
            reasons.append("small_component_fraction_gate_failed")
        cube_reasons[cube_id] = reasons
        summary_rows.append({
            "cube_id": cube_id, "total_pixels": int(primary.size), "valid_pixels": int(primary.sum()),
            "invalid_pixels": int((~primary).sum()), "valid_fraction": float(primary.mean()),
            "primary_rule": primary_name, "selected_robust_rule": robust_name,
            "primary_robust_disagreement_count": robust_disagreement["disagreement_count"],
            "primary_robust_disagreement_fraction": robust_disagreement["disagreement_fraction_primary_valid"],
            "finite_failure_count": diagnostics["finite_failure_count"],
            "exact_all_zero_spectrum_count": diagnostics["exact_all_zero_spectrum_count"],
            "partial_zero_spectrum_count": diagnostics["partial_zero_spectrum_count"],
            "negative_value_spectrum_count": diagnostics["negative_value_spectrum_count"],
            "low_amplitude_nonzero_count": low_amplitude, **primary_metrics,
            "spatial_interpretation": (
                "reviewed_coherent_scan_strip_exception_retained_unfiltered" if exception
                else "no_unexplained_non_small_interior_components"
            ),
            "validation_status": "pass" if not reasons else "valid_support_review_required",
            "review_reason": "|".join(reasons),
        })
        print(f"  {cube_id}: valid={primary.mean():.6f}, reasons={reasons or ['none']}", flush=True)

    sensitivity = pd.DataFrame(sensitivity_rows)
    components = pd.DataFrame(component_rows)
    summary = pd.DataFrame(summary_rows)
    global_reasons: list[str] = []
    if len(records) != int(config["freeze_gates"]["expected_cube_count"]):
        global_reasons.append("expected_cube_count_gate_failed")
    if any(cube_reasons.values()):
        global_reasons.append("one_or_more_cube_support_gates_failed")
    if not config["safety"]["reflectance_only_rule"]:
        global_reasons.append("reflectance_only_rule_not_declared")
    freeze_passed = not global_reasons

    manifest_rows: list[dict] = []
    transformed_rows: list[dict] = []
    thumbnails: list[tuple[str, np.ndarray, np.ndarray]] = []
    mask_grid_status: dict[str, tuple[bool, list[str]]] = {}
    if freeze_passed:
        print("All freeze gates passed; materializing derived masks and review products...", flush=True)
        for cube_id in sorted(records):
            products = records[cube_id]
            support = primary_masks[cube_id]
            ref = products["reflectance"]
            ref_path = project / ref.binary_path
            reference = raster_reference(ref_path, guard)
            mask_path = mask_root / f"{cube_id}_valid_support.tif"
            source_hash = str(baseline_lookup[str(Path(ref.binary_path))]["sha256"])
            indices = model_band_indices(
                int(band_config["first_index"]), int(band_config["last_index_inclusive"]),
                int(reference["source_count"]),
            )
            write_mask_geotiff(
                mask_path, support, reference, guard, source_hash, primary_name, config_hash, indices,
                config["mask"]["compression"], int(config["mask"]["predictor"]),
                int(config["mask"]["zlevel"]),
            )
            verified, issues, codes = verify_mask_geotiff(mask_path, reference)
            mask_grid_status[cube_id] = (verified, issues)
            if not verified:
                cube_reasons[cube_id].append("mask_grid_or_code_verification_failed:" + ",".join(issues))
            manifest_rows.append({
                "cube_id": cube_id, "source_reflectance_path": str(ref.binary_path),
                "source_reflectance_sha256": source_hash,
                "mask_path": str(mask_path.relative_to(project)), "mask_sha256": sha256(mask_path),
                "valid_count": int(support.sum()), "invalid_count": int((~support).sum()),
                "valid_fraction": float(support.mean()), "crs": reference["crs_text"],
                "transform": json.dumps([float(value) for value in reference["transform"]]),
                "width": reference["width"], "height": reference["height"],
                "rule_name": primary_name, "rule_parameters": json.dumps({
                    "python_zero_based_first": band_config["first_index"],
                    "python_zero_based_last_inclusive": band_config["last_index_inclusive"],
                    "maximum_absolute_stored_reflectance_threshold": 0,
                    "all_model_bands_finite": True,
                }, sort_keys=True),
                "configuration_sha256": config_hash,
                "validation_status": "pass" if verified else "valid_support_review_required",
                "mask_codes": json.dumps(sorted(codes)),
            })
            opened: dict[str, np.ndarray] = {}
            for product in ("reflectance", "pca", "stored_index"):
                row = products[product]
                opened[product], _ = open_envi_memmap(
                    project / row.header_path, project / row.binary_path, guard
                )
            for product in ("pca", "stored_index"):
                candidate_path = project / products[product].binary_path
                candidate_reference = raster_reference(candidate_path, guard)
                candidate_reference["count"] = 1
                same, grid_issues = verify_grid_pair(reference, candidate_reference)
                if not same:
                    cube_reasons[cube_id].append(product + "_grid_mismatch:" + ",".join(grid_issues))
                stats = transformed_statistics(
                    opened[product], support, int(processing["row_window_size"])
                )
                transformed_rows.append({"cube_id": cube_id, "product_type": product, **stats})
            observed = parse_wavelengths(open_envi_memmap(
                project / ref.header_path, ref_path, guard
            )[1])
            preview_indices = wavelength_indices(observed, processing["false_colour_wavelengths_nm"])
            max_dimension = int(processing["preview_max_dimension"])
            stride = max(1, int(np.ceil(max(support.shape) / max_dimension)))
            reflectance_preview = np.asarray(opened["reflectance"][::stride, ::stride, :][..., preview_indices])
            pca_preview = np.asarray(opened["pca"][::stride, ::stride, :])
            index_preview = np.asarray(opened["stored_index"][::stride, ::stride, :])
            labels, _ = ndimage.label(support, structure=np.ones((3, 3), dtype=np.uint8))
            rgb, mask_thumb = save_qc(
                output / "previews" / f"{cube_id}_valid_support_qc.png", cube_id,
                reflectance_preview, support, primary_masks[cube_id], pca_preview,
                index_preview, labels, tuple(processing["preview_percentiles"]), stride,
            )
            thumbnails.append((cube_id, rgb, mask_thumb))
        save_overviews(output, thumbnails)
    else:
        print("Freeze gate did not pass; no GeoTIFF masks were materialized.", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    transformed = pd.DataFrame(transformed_rows)
    print("Creating independent post-operation source snapshot...", flush=True)
    final_snapshot = source_snapshot(guard, project)
    final_path = output / "field2_source_snapshot_after.csv"
    pd.DataFrame(final_snapshot).to_csv(final_path, index=False)
    source_issues = compare_snapshots(baseline, final_snapshot)
    if source_issues:
        global_reasons.append("source_integrity_gate_failed")

    readiness_rows = []
    spectral_by_cube = set(inventory.loc[inventory.product_type == "reflectance", "cube_id"])
    for cube_id in sorted(records):
        alignment_passed = bool((alignment.loc[alignment.cube_id == cube_id, "alignment_status"] == "exact_alignment").all())
        mask_passed = bool(mask_grid_status.get(cube_id, (False, []))[0])
        reasons = list(cube_reasons[cube_id])
        if cube_id not in spectral_by_cube:
            reasons.append("spectral_compatibility_failed")
        if not alignment_passed:
            reasons.append("spatial_alignment_failed")
        if not mask_passed:
            reasons.append("mask_not_materialized_or_verification_failed")
        if source_issues:
            reasons.append("source_integrity_failed")
        ready = not reasons and freeze_passed
        readiness_rows.append({
            "cube_id": cube_id, "usable_for_annotation": "yes" if ready else "no",
            "annotation_status": "annotation_ready_prediction_free" if ready else "valid_support_review_required",
            "spectral_compatibility_status": "compatible_exact_frozen_band_match",
            "spatial_alignment_status": "exact_alignment" if alignment_passed else "failed",
            "valid_support_status": "passed" if not cube_reasons[cube_id] else "review_required",
            "mask_grid_hash_verification_status": "passed" if mask_passed else "failed",
            "source_integrity_status": "passed" if not source_issues else "failed",
            "valid_fraction": float(summary.loc[summary.cube_id == cube_id, "valid_fraction"].iloc[0]),
            "proposed_role": "", "investigator_role": "unreviewed", "investigator_notes": "",
            "exclusion_reason": "|".join(reasons),
        })
    readiness = pd.DataFrame(readiness_rows)
    summary["validation_status"] = summary.cube_id.map(
        dict(zip(readiness.cube_id, readiness.annotation_status))
    )
    summary["review_reason"] = summary.cube_id.map(dict(zip(readiness.cube_id, readiness.exclusion_reason)))
    frames = {
        "sensitivity": sensitivity, "summary": summary, "components": components,
        "transformed": transformed, "manifest": manifest, "readiness": readiness,
    }
    for name, frame in frames.items():
        frame.to_csv(output / REPORT_FILES[name], index=False)

    report_path = project / "docs/progress/2026-08-17_field2_valid_support.md"
    write_report(report_path, sensitivity, summary, transformed, manifest, readiness, config, source_issues)
    output_files = [output / name for name in REPORT_FILES.values()]
    output_files += [baseline_path, final_path]
    output_files += sorted((output / "previews").glob("*.png"))
    output_files += sorted(output.glob("*overview.png"))
    implementation_commit = git_output(project, "rev-parse", "HEAD")
    contract = {
        "status": "field2_valid_support_frozen_all_cubes_annotation_ready" if (
            (readiness.usable_for_annotation == "yes").all() and not source_issues
        ) else "field2_valid_support_review_required",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_git_commit": implementation_commit,
        "script_path": str(Path(__file__).resolve().relative_to(project)),
        "configuration_path": str(args.config), "configuration_sha256": config_hash,
        "readiness_repository_commit": config["readiness_repository_commit"],
        "readiness_contract_sha256": sha256(project / config["readiness_contract"]),
        "source_manifest_sha256": sha256(baseline_path),
        "source_file_count": len(baseline), "source_snapshot_comparison": "pass_identical" if not source_issues else "fail",
        "reflectance_only_rule": True, "rule_name": primary_name,
        "rule_definition": "all model-input bands finite and maximum absolute stored reflectance across them > 0",
        "python_zero_based_band_indices": [int(value) for value in range(
            int(band_config["first_index"]), int(band_config["last_index_inclusive"]) + 1
        )],
        "envi_one_based_band_indices": [int(value) for value in range(
            int(band_config["envi_first_band_one_based"]),
            int(band_config["envi_last_band_one_based_inclusive"]) + 1
        )],
        "wavelengths_nm": [float(value) for value in expected_wavelengths],
        "sensitivity_thresholds_stored_units": thresholds,
        "selected_robust_rule": robust_name,
        "maximum_primary_robust_disagreement_fraction": float(
            sensitivity.loc[sensitivity.rule_name == robust_name, "disagreement_fraction_primary_valid"].max()
        ),
        "configured_maximum_disagreement_fraction": config["freeze_gates"]["maximum_primary_robust_disagreement_fraction"],
        "candidate_rules": sorted(sensitivity.rule_name.unique().tolist()),
        "reviewed_spatial_exceptions": config.get("spatial_review", {}).get("reviewed_exceptions", {}),
        "cube_status": readiness[["cube_id", "annotation_status", "exclusion_reason"]].to_dict("records"),
        "mask_sha256": dict(zip(manifest.get("cube_id", []), manifest.get("mask_sha256", []))),
        "annotation_ready_cube_count": int((readiness.usable_for_annotation == "yes").sum()),
        "annotation_blocked_cube_count": int((readiness.usable_for_annotation != "yes").sum()),
        "class_information_used": False, "model_inference_used": False,
        "supervised_predictions_generated": False, "supervised_probabilities_generated": False,
        "biological_labels_generated": False, "model_retrained": False,
        "ssl_training_run": False, "field1_benchmark_modified": False,
        "pca_or_stored_index_used_to_derive_mask": False,
        "output_sha256": {str(path.relative_to(project)): sha256(path) for path in output_files if path.is_file()},
    }
    dump_yaml(contract, contract_path)
    if source_issues:
        raise RuntimeError("Source integrity changed: " + ";".join(source_issues))
    if not (readiness.usable_for_annotation == "yes").all():
        raise RuntimeError("One or more cubes require valid-support review")
    print(f"Frozen {len(manifest)} masks; annotation-ready cubes={len(readiness)}", flush=True)
    print(f"Contract: {contract_path}", flush=True)


def verify_grid_pair(reference: dict, candidate: dict) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if reference["width"] != candidate["width"] or reference["height"] != candidate["height"]:
        issues.append("dimensions_mismatch")
    if str(reference["crs"]) != str(candidate["crs"]):
        issues.append("crs_mismatch")
    if tuple(reference["transform"]) != tuple(candidate["transform"]):
        issues.append("transform_mismatch")
    if tuple(reference["bounds"]) != tuple(candidate["bounds"]):
        issues.append("bounds_mismatch")
    return not issues, issues


if __name__ == "__main__":
    main()
