#!/usr/bin/env python
"""Validate the immutable Field 2 natural-RGB annotation-display addendum."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_blind_review import validate_annotation_display_contract
from chickpea_ssl.field2_readiness import sha256


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
    print("Outside-support pixels: black for all cubes")
    print("Frozen main/reserve/combined sampling hashes: unchanged")


if __name__ == "__main__":
    main()
