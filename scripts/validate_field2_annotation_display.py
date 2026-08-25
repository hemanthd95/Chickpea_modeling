#!/usr/bin/env python
"""Validate the immutable Field 2 natural-RGB annotation-display addendum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.field2_blind_review import (
    natural_rgb_png_bytes, render_natural_rgb_uint8, select_natural_rgb_bands,
    validate_annotation_display_contract,
)
from chickpea_ssl.field2_readiness import (
    ReadOnlySourceGuard, compare_snapshots, open_envi_memmap, parse_wavelengths,
    sha256, source_snapshot,
)


def verify_reproducible_rerender(project: Path, paths: dict, config: dict, manifest: pd.DataFrame) -> None:
    """Re-render every cube in memory and require its frozen PNG hash exactly."""
    guard = ReadOnlySourceGuard(tuple(Path(value) for value in paths["field2"]["readiness_roots"]))
    before = source_snapshot(guard, project)
    inventory = pd.read_csv(project / config["inputs"]["readiness_inventory"]).fillna("")
    support_manifest = pd.read_csv(project / config["inputs"]["valid_support_manifest"]).fillna("").set_index("cube_id")
    display = config["annotation_display"]
    targets = {name: float(value) for name, value in display["rgb_target_wavelengths_nm"].items()}
    tolerance = float(display["maximum_wavelength_difference_nm"])
    percentiles = tuple(float(value) for value in display["stretch_percentiles"])
    for frozen in manifest.itertuples(index=False):
        cube_id = str(frozen.cube_id)
        source = inventory[(inventory.cube_id == cube_id) & (inventory.product_type == "reflectance")].iloc[0]
        support_row = support_manifest.loc[cube_id]
        reflectance, metadata = open_envi_memmap(
            project / str(source.header_path), project / str(source.binary_path), guard,
        )
        selected = select_natural_rgb_bands(parse_wavelengths(metadata), targets, tolerance)
        if selected != json.loads(str(frozen.selected_bands_json)):
            raise ValueError(f"Re-render selected-band recipe changed: {cube_id}")
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(project / str(support_row.mask_path), "r") as dataset:
                support = dataset.read(1) == 1
        rgb, stretch = render_natural_rgb_uint8(reflectance, support, selected, percentiles)
        if stretch != json.loads(str(frozen.stretch_json)):
            raise ValueError(f"Re-render cube-level stretch changed: {cube_id}")
        rerendered_hash = hashlib.sha256(natural_rgb_png_bytes(rgb)).hexdigest()
        if rerendered_hash != str(frozen.natural_rgb_sha256):
            raise ValueError(f"Re-rendered natural RGB hash changed: {cube_id}")
        if np.any(rgb[~support] != 0):
            raise ValueError(f"Re-render produced nonblack unsupported pixels: {cube_id}")
    changes = compare_snapshots(before, source_snapshot(guard, project))
    if changes:
        raise ValueError(f"Field 2 sources changed during display re-render: {changes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve()
    contract_path = project / config["annotation_display"]["display_contract"]
    result = validate_annotation_display_contract(project, config, contract_path)
    contract, manifest = result["contract"], result["manifest"]
    verify_reproducible_rerender(project, paths, config, manifest)
    subprocess.check_call(
        ["git", "cat-file", "-e", f"{contract['materialization_git_commit']}^{{commit}}"], cwd=project,
    )
    selected_sets = sorted(set(manifest.selected_bands_json.astype(str)))
    print("Field 2 natural-RGB annotation-display validation passed")
    print(f"Contract: {contract_path}")
    print(f"Contract SHA-256: {sha256(contract_path)}")
    print(f"Materialization commit: {contract['materialization_git_commit']}")
    print(f"Natural RGB previews: {len(manifest)}")
    print(f"Distinct selected-band recipes: {len(selected_sets)}")
    print("Deterministic in-memory re-render: all 40 preview hashes identical")
    print("Outside-support pixels: black for all cubes")
    print("Frozen main/reserve/combined sampling hashes: unchanged")


if __name__ == "__main__":
    main()
