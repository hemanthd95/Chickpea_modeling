#!/usr/bin/env python
"""Validate frozen prediction-free Field 2 main and reserve sampling frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_blind_review import validate_frozen_sampling_contract
from chickpea_ssl.field2_readiness import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve()
    contract_path = project / config["sampling"]["sampling_contract"]
    result = validate_frozen_sampling_contract(project, config, contract_path)
    contract, audit = result["contract"], result["audit"]
    subprocess.check_call(
        ["git", "cat-file", "-e", f"{contract['materialization_git_commit']}^{{commit}}"], cwd=project,
    )
    print("Field 2 frozen blind-sampling contract validation passed")
    print(f"Contract: {contract_path}")
    print(f"Contract SHA-256: {sha256(contract_path)}")
    print(f"Materialization commit: {contract['materialization_git_commit']}")
    print(f"Main: {len(result['main'])}; reserve: {len(result['reserve'])}; combined: {len(result['combined'])}")
    print(f"Separation: {audit['combined_nearest_neighbor_separation_m']}")
    print(f"Spatial blocks: {audit['spatial_block_coverage']}")
    print("Reserve release: locked; separate immutable authorization required")


if __name__ == "__main__":
    main()
