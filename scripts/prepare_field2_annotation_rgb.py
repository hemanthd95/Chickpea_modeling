#!/usr/bin/env python
"""Prepare immutable full-resolution natural RGB annotation displays."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["GDAL_PAM_ENABLED"] = "NO"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.field2_blind_review import (
    atomic_write_bytes, atomic_write_yaml, natural_rgb_png_bytes,
    render_natural_rgb_uint8, select_natural_rgb_bands,
    validate_frozen_sampling_contract,
)
from chickpea_ssl.field2_readiness import (
    ReadOnlySourceGuard, compare_snapshots, open_envi_memmap, parse_wavelengths,
    sha256, source_snapshot,
)


def git_output(project: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=project, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve()
    display, expected = config["annotation_display"], list(config["expected_cube_ids"])
    preview_root = project / display["preview_root"]
    manifest_path = project / display["preview_manifest"]
    contract_path = project / display["display_contract"]
    existing = [path for path in (preview_root, manifest_path, contract_path) if path.exists()]
    if existing:
        raise SystemExit(f"REFUSED: immutable annotation-display output already exists: {existing}")

    sampling_contract_path = project / config["sampling"]["sampling_contract"]
    sampling_result = validate_frozen_sampling_contract(project, config, sampling_contract_path)
    sampling_hashes_before = {
        name: sha256(project / reference["path"])
        for name, reference in sampling_result["contract"]["frozen_outputs"].items()
        if name in {"main", "reserve", "combined"}
    }
    valid_contract_path = project / config["inputs"]["valid_support_contract"]
    valid_manifest_path = project / config["inputs"]["valid_support_manifest"]
    valid_contract = yaml.safe_load(valid_contract_path.read_text())
    if valid_contract.get("status") != "field2_valid_support_frozen_all_cubes_annotation_ready":
        raise RuntimeError("Frozen valid-support contract is not annotation-ready")
    if sha256(valid_manifest_path) != config["valid_support_manifest_sha256"]:
        raise RuntimeError("Frozen valid-support manifest checksum changed")

    roots = tuple(Path(value) for value in paths["field2"]["readiness_roots"])
    guard = ReadOnlySourceGuard(roots)
    source_before = source_snapshot(guard, project)
    inventory = pd.read_csv(project / config["inputs"]["readiness_inventory"]).fillna("")
    support_manifest = pd.read_csv(valid_manifest_path).fillna("")
    if support_manifest.cube_id.astype(str).tolist() != expected:
        raise RuntimeError("Valid-support manifest does not contain the exact 40-cube inventory")
    targets = {name: float(value) for name, value in display["rgb_target_wavelengths_nm"].items()}
    tolerance = float(display["maximum_wavelength_difference_nm"])
    percentiles = tuple(float(value) for value in display["stretch_percentiles"])
    records = []
    preview_root.mkdir(parents=True, exist_ok=False)
    print("Preparing 40 full-resolution reflectance-only natural RGB displays...", flush=True)
    for cube_id in expected:
        source_row = inventory[(inventory.cube_id == cube_id) & (inventory.product_type == "reflectance")].iloc[0]
        support_row = support_manifest[support_manifest.cube_id == cube_id].iloc[0]
        source_path, mask_path = project / str(source_row.binary_path), project / str(support_row.mask_path)
        if sha256(source_path) != str(source_row.sha256) or str(source_row.sha256) != str(support_row.source_reflectance_sha256):
            raise RuntimeError(f"{cube_id}: frozen reflectance checksum mismatch")
        if sha256(mask_path) != str(support_row.mask_sha256):
            raise RuntimeError(f"{cube_id}: frozen support-mask checksum mismatch")
        reflectance, metadata = open_envi_memmap(
            project / str(source_row.header_path), source_path, guard,
        )
        selected = select_natural_rgb_bands(parse_wavelengths(metadata), targets, tolerance)
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(mask_path, "r") as dataset:
                support = dataset.read(1) == 1
                if dataset.width != reflectance.shape[1] or dataset.height != reflectance.shape[0]:
                    raise RuntimeError(f"{cube_id}: reflectance/support dimensions differ")
        rgb, stretch = render_natural_rgb_uint8(reflectance, support, selected, percentiles)
        if np.any(rgb[~support] != 0):
            raise RuntimeError(f"{cube_id}: unsupported RGB pixels are not black")
        first_bytes, repeated_bytes = natural_rgb_png_bytes(rgb), natural_rgb_png_bytes(rgb)
        if first_bytes != repeated_bytes:
            raise RuntimeError(f"{cube_id}: RGB PNG rendering is not deterministic")
        preview_path = preview_root / f"{cube_id}_natural_rgb.png"
        atomic_write_bytes(preview_path, first_bytes)
        records.append({
            "cube_id": cube_id, "width": rgb.shape[1], "height": rgb.shape[0],
            "preview_step": 1, "natural_rgb_path": str(preview_path.relative_to(project)),
            "natural_rgb_sha256": sha256(preview_path),
            "source_reflectance_sha256": str(source_row.sha256),
            "support_mask_sha256": str(support_row.mask_sha256),
            "selected_bands_json": json.dumps(selected, sort_keys=True),
            "stretch_json": json.dumps(stretch, sort_keys=True),
            "outside_support_black": True, "reflectance_only": True,
        })
        wavelengths = ", ".join(
            f"{name}={selected[name]['selected_wavelength_nm']:.3f} nm"
            for name in ("red", "green", "blue")
        )
        print(f"  {cube_id}: {wavelengths}", flush=True)

    frame = pd.DataFrame(records)
    buffer = io.StringIO(); frame.to_csv(buffer, index=False, lineterminator="\n")
    atomic_write_bytes(manifest_path, buffer.getvalue().encode())
    if compare_snapshots(source_before, source_snapshot(guard, project)):
        raise RuntimeError("Field 2 source inventory changed during RGB preparation")
    if any(sha256(project / row.mask_path) != row.mask_sha256 for row in support_manifest.itertuples(index=False)):
        raise RuntimeError("A frozen valid-support mask changed during RGB preparation")
    sampling_hashes_after = {
        name: sha256(project / reference["path"])
        for name, reference in sampling_result["contract"]["frozen_outputs"].items()
        if name in {"main", "reserve", "combined"}
    }
    if sampling_hashes_after != sampling_hashes_before:
        raise RuntimeError("A frozen sampling table changed during RGB preparation")
    contract = {
        "version": "field2_annotation_display_addendum_v1",
        "status": "field2_annotation_display_natural_rgb_frozen",
        "display_recipe_version": display["version"],
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "materialization_git_commit": git_output(project, "rev-parse", "HEAD"),
        "cube_count": len(frame), "cube_ids": frame.cube_id.tolist(),
        "recipe": {
            "target_wavelengths_nm": targets,
            "nearest_available_band": True,
            "maximum_wavelength_difference_nm": tolerance,
            "stretch_percentiles_inside_valid_support": list(percentiles),
            "stretch_scope": "one cube-level stretch per channel",
            "output_type": "uint8 RGB 0-255",
            "outside_valid_support": "black RGB 0,0,0",
            "adaptive_or_point_specific_enhancement": False,
        },
        "input_contracts": {
            "valid_support_contract": {"path": str(valid_contract_path.relative_to(project)), "sha256": sha256(valid_contract_path)},
            "valid_support_manifest": {"path": str(valid_manifest_path.relative_to(project)), "sha256": sha256(valid_manifest_path)},
            "sampling_contract": {"path": str(sampling_contract_path.relative_to(project)), "sha256": sha256(sampling_contract_path)},
            "field2_source_manifest_sha256": config["source_manifest_sha256"],
        },
        "preview_manifest": {"path": str(manifest_path.relative_to(project)), "sha256": sha256(manifest_path)},
        "cube_displays": [
            {
                "cube_id": row.cube_id, "natural_rgb_path": row.natural_rgb_path,
                "natural_rgb_sha256": row.natural_rgb_sha256,
                "source_reflectance_sha256": row.source_reflectance_sha256,
                "support_mask_sha256": row.support_mask_sha256,
                "selected_bands": json.loads(row.selected_bands_json),
                "stretch": json.loads(row.stretch_json),
            }
            for row in frame.itertuples(index=False)
        ],
        "sampling_table_sha256": sampling_hashes_before,
        "provenance": {
            "display_only": True, "prediction_free": True,
            "reflectance_and_frozen_valid_support_only": True,
            "pca_used_to_construct_rgb": False, "stored_scalar_index_used_to_construct_rgb": False,
            "field1_labels_or_model_outputs_used": False, "supervised_checkpoint_loaded": False,
            "predictions_probabilities_or_pseudo_labels_generated": False,
            "biological_class_suggested": False,
        },
    }
    atomic_write_yaml(contract_path, contract)
    print(f"FROZEN: {len(frame)} deterministic natural RGB displays", flush=True)
    print(f"Contract: {contract_path}", flush=True)


if __name__ == "__main__":
    main()
