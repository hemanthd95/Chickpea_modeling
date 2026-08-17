#!/usr/bin/env python
"""Prepare prediction-free layers for Field 2 investigator cube-role review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["GDAL_PAM_ENABLED"] = "NO"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
import yaml

from chickpea_ssl.field2_blind_review import PREVIEW_LAYERS, atomic_write_bytes, atomic_write_yaml
from chickpea_ssl.field2_readiness import (
    ReadOnlySourceGuard,
    compare_snapshots,
    open_envi_memmap,
    parse_wavelengths,
    sha256,
    source_snapshot,
)


def csv_boolean(value: object) -> bool:
    """Parse a CSV boolean without treating the string ``False`` as true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Unrecognized CSV boolean value: {value!r}")


def robust_stretch(values: np.ndarray, valid: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.float32)
    for band in range(values.shape[2]):
        plane = np.asarray(values[..., band], dtype=np.float32)
        usable = plane[valid & np.isfinite(plane)]
        low, high = np.percentile(usable, percentiles) if len(usable) else (0.0, 1.0)
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        result[..., band] = np.clip((plane - low) / (high - low), 0, 1)
    return result


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        plt.imsave(temporary, image)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_sheet(path: Path, cube_id: str, layers: dict[str, np.ndarray], metadata: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [
        (layers["false_colour"], "False-colour reflectance"),
        (layers["pca"], "Supplied PCA components 1–3"),
        (layers["stored_index"], "Stored scalar index (display only)"),
        (layers["valid_support"], "Frozen reflectance support"),
        (layers["support_outline"], "Support outline on reflectance"),
    ]
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0, 1,
        "AUTOMATIC NON-BIOLOGICAL METADATA\n\n" + "\n".join(f"{key}: {value}" for key, value in metadata.items()),
        va="top", family="monospace", fontsize=9,
    )
    fig.suptitle(f"{cube_id} — prediction-free cube-role review")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        fig.savefig(temporary, dpi=150)
        plt.close(fig)
        temporary.replace(path)
    except Exception:
        plt.close(fig); temporary.unlink(missing_ok=True); raise


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
    roots = tuple(Path(value) for value in paths["field2"]["readiness_roots"])
    guard = ReadOnlySourceGuard(roots)
    output = guard.output(project / config["review"]["package_root"])
    manifest_path = guard.output(project / config["review"]["package_manifest"])
    contract_path = guard.output(project / config["review"]["package_contract"])
    assets = output / "layers"
    assets.mkdir(parents=True, exist_ok=True)

    valid_contract_path = project / config["inputs"]["valid_support_contract"]
    valid_contract = yaml.safe_load(valid_contract_path.read_text())
    if valid_contract.get("status") != "field2_valid_support_frozen_all_cubes_annotation_ready":
        raise RuntimeError("Frozen valid-support contract is not annotation-ready")
    if valid_contract.get("script_git_commit") != config["valid_support_commit"]:
        raise RuntimeError("Valid-support implementation commit mismatch")
    if valid_contract.get("source_manifest_sha256") != config["source_manifest_sha256"]:
        raise RuntimeError("Frozen source-manifest hash mismatch")

    baseline = source_snapshot(guard, project)
    buffer = io.StringIO(); pd.DataFrame(baseline).to_csv(buffer, index=False)
    current_source_hash = hashlib.sha256(buffer.getvalue().encode()).hexdigest()
    if current_source_hash != config["source_manifest_sha256"]:
        raise RuntimeError("Current Field 2 source snapshot differs from the frozen manifest")

    inventory = pd.read_csv(project / config["inputs"]["readiness_inventory"]).fillna("")
    old_readiness = pd.read_csv(project / config["inputs"]["readiness_annotation"]).fillna("")
    support_readiness = pd.read_csv(project / config["inputs"]["valid_support_readiness"]).fillna("")
    support_manifest_path = project / config["inputs"]["valid_support_manifest"]
    support_manifest = pd.read_csv(support_manifest_path).fillna("")
    footprints = pd.read_csv(project / config["inputs"]["cube_footprints"]).fillna("")
    expected = list(config["expected_cube_ids"])
    if support_manifest.cube_id.astype(str).tolist() != expected:
        raise RuntimeError("The frozen support manifest does not contain the exact configured 40 cubes")
    if sha256(support_manifest_path) != config["valid_support_manifest_sha256"]:
        raise RuntimeError("Frozen valid-support manifest checksum mismatch")
    if not (support_readiness.annotation_status == "annotation_ready_prediction_free").all():
        raise RuntimeError("One or more cubes is not prediction-free annotation-ready")

    baseline_lookup = {row["relative_path"]: row["sha256"] for row in baseline}
    records = []
    percentiles = tuple(float(value) for value in config["review"]["preview_percentiles"])
    targets = [float(value) for value in config["review"]["false_colour_wavelengths_nm"]]
    print("Preparing 40 prediction-free cube-role review packages...", flush=True)
    for cube_id in expected:
        group = inventory[inventory.cube_id == cube_id]
        if set(group.product_type) != {"reflectance", "pca", "stored_index"} or len(group) != 3:
            raise RuntimeError(f"{cube_id}: incomplete prediction-free product inventory")
        products = {row.product_type: row for row in group.itertuples(index=False)}
        support_row = support_manifest[support_manifest.cube_id == cube_id].iloc[0]
        mask_path = project / str(support_row.mask_path)
        if sha256(mask_path) != support_row.mask_sha256:
            raise RuntimeError(f"{cube_id}: support-mask checksum mismatch")
        arrays = {}
        metadata = {}
        for product, row in products.items():
            if baseline_lookup.get(str(row.binary_path)) != str(row.sha256):
                raise RuntimeError(f"{cube_id}:{product}: source checksum mismatch")
            arrays[product], metadata[product] = open_envi_memmap(
                project / str(row.header_path), project / str(row.binary_path), guard
            )
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(mask_path, "r") as dataset:
                support = dataset.read(1) == 1
                transform, bounds, crs = dataset.transform, dataset.bounds, dataset.crs
                width, height = dataset.width, dataset.height
        stride = max(1, int(np.ceil(max(height, width) / int(config["review"]["preview_max_dimension"]))))
        sampled_support = support[::stride, ::stride]
        wavelengths = parse_wavelengths(metadata["reflectance"])
        false_indices = [int(np.argmin(np.abs(wavelengths - target))) for target in targets]
        ref_raw = np.asarray(arrays["reflectance"][::stride, ::stride, :][..., false_indices])
        pca_raw = np.asarray(arrays["pca"][::stride, ::stride, :][..., :3])
        index_raw = np.asarray(arrays["stored_index"][::stride, ::stride, 0], dtype=np.float32)
        false_colour = robust_stretch(ref_raw, sampled_support, percentiles)
        pca = robust_stretch(pca_raw, sampled_support & np.isfinite(pca_raw).all(2), percentiles)
        index_valid = sampled_support & np.isfinite(index_raw)
        usable_index = index_raw[index_valid]
        low, high = np.percentile(usable_index, percentiles) if len(usable_index) else (0.0, 1.0)
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        index_scaled = np.clip((index_raw - low) / (high - low), 0, 1)
        stored_index = plt.get_cmap("viridis")(index_scaled)[..., :3]
        false_colour[~sampled_support] = 0
        pca[~sampled_support] = 0
        stored_index[~sampled_support] = 0
        valid_support = np.repeat(sampled_support[..., None], 3, axis=2).astype(np.float32)
        boundary = sampled_support & ~ndimage.binary_erosion(sampled_support, structure=np.ones((3, 3)), border_value=0)
        support_outline = false_colour.copy()
        support_outline[boundary] = np.asarray([1.0, 0.85, 0.0])
        layers = {
            "false_colour": false_colour, "pca": pca, "stored_index": stored_index,
            "valid_support": valid_support, "support_outline": support_outline,
        }
        cube_root = assets / cube_id
        cube_root.mkdir(parents=True, exist_ok=True)
        layer_paths = {}
        for layer, image in layers.items():
            path = cube_root / f"{layer}.png"
            save_image(path, image)
            layer_paths[layer] = path
        footprint = footprints[footprints.cube_id == cube_id].iloc[0]
        prior = old_readiness[old_readiness.cube_id == cube_id].iloc[0]
        ready = support_readiness[support_readiness.cube_id == cube_id].iloc[0]
        metadata_panel = {
            "dimensions": f"{width} × {height}", "GSD": f"{abs(transform.a):.4f} m",
            "valid fraction": f"{float(support_row.valid_fraction):.6f}",
            "bounds": f"{bounds.left:.3f}, {bounds.bottom:.3f}, {bounds.right:.3f}, {bounds.top:.3f}",
            "overlap warning": str(prior.geographic_overlap_warning),
            "spectral QC": str(ready.spectral_compatibility_status),
            "alignment": str(ready.spatial_alignment_status),
            "role source": "investigator only; currently unreviewed",
        }
        sheet = cube_root / "review_sheet.png"
        save_sheet(sheet, cube_id, layers, metadata_panel)
        layer_paths["review_sheet"] = sheet
        hashes = {layer: sha256(layer_paths[layer]) for layer in PREVIEW_LAYERS}
        records.append({
            "cube_id": cube_id, "width": width, "height": height, "gsd_m": abs(transform.a),
            "valid_fraction": float(support_row.valid_fraction), "crs": str(crs),
            "bounds": json.dumps([bounds.left, bounds.bottom, bounds.right, bounds.top]),
            "transform": json.dumps([float(value) for value in transform]),
            "footprint_overlap_warning": csv_boolean(prior.geographic_overlap_warning),
            "spectral_qc_status": str(ready.spectral_compatibility_status),
            "alignment_status": str(ready.spatial_alignment_status),
            "preview_width": int(false_colour.shape[1]), "preview_height": int(false_colour.shape[0]),
            "preview_step": stride, "source_reflectance_sha256": str(support_row.source_reflectance_sha256),
            "support_mask_sha256": str(support_row.mask_sha256),
            **{f"{layer}_path": str(layer_paths[layer].relative_to(project)) for layer in PREVIEW_LAYERS},
            "preview_sha256_json": json.dumps(hashes, sort_keys=True),
            "automatic_metadata_only": True, "investigator_role": "unreviewed",
        })
        print(f"  {cube_id}: {false_colour.shape[1]}×{false_colour.shape[0]} preview", flush=True)

    frame = pd.DataFrame(records)
    buffer = io.StringIO(); frame.to_csv(buffer, index=False)
    atomic_write_bytes(manifest_path, buffer.getvalue().encode())
    final_snapshot = source_snapshot(guard, project)
    source_issues = compare_snapshots(baseline, final_snapshot)
    if source_issues:
        raise RuntimeError("Field 2 source integrity changed: " + ";".join(source_issues))
    if any(sha256(project / row.mask_path) != row.mask_sha256 for row in support_manifest.itertuples(index=False)):
        raise RuntimeError("A frozen support mask changed during review preparation")
    contract = {
        "status": "field2_cube_role_review_package_ready_unreviewed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_git_commit": git_output(project, "rev-parse", "HEAD"),
        "configuration_path": str(args.config), "configuration_sha256": sha256(args.config.resolve()),
        "cube_count": len(frame), "cube_ids": frame.cube_id.tolist(),
        "package_manifest": str(manifest_path.relative_to(project)),
        "package_manifest_sha256": sha256(manifest_path),
        "valid_support_manifest_sha256": sha256(support_manifest_path),
        "source_manifest_sha256": current_source_hash,
        "preview_layer_names": list(PREVIEW_LAYERS),
        "all_biological_roles": "unreviewed",
        "automatic_biological_role_assignment": False,
        "supervised_checkpoints_opened": False,
        "predictions_or_probabilities_used": False,
        "field1_labels_or_masks_used": False,
        "source_snapshot_comparison": "pass_identical",
    }
    atomic_write_yaml(contract_path, contract)
    print(f"READY: {len(frame)} prediction-free cube-role review packages", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
