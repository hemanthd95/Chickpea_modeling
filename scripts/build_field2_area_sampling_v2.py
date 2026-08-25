#!/usr/bin/env python
"""Materialize the immutable capacity-aware Field 2 area sampling v2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ["GDAL_PAM_ENABLED"] = "NO"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_area_sampling_v2 import DOMAINS, materialize
from chickpea_ssl.field2_readiness import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_area_sampling_v2.yaml"), type=Path)
    args = parser.parse_args()
    paths_config = yaml.safe_load(args.paths.read_text())
    project = Path(paths_config["project_root"]).resolve()
    config = yaml.safe_load(args.config.read_text())
    result = materialize(project, config, paths_config)
    print("FROZEN: Field 2 prediction-free area-stratified sampling v2")
    print(f"Main: {len(result['main'])}; reserve locked: {len(result['reserve'])}")
    for frame_name in ("main", "reserve"):
        frame = result[frame_name]
        print(frame_name + ": " + ", ".join(
            f"{domain}={int((frame.sampling_domain == domain).sum())}" for domain in DOMAINS
        ))
    contract_path = project / config["outputs"]["contract"]
    print(f"Contract: {contract_path}")
    print(f"Contract SHA-256: {sha256(contract_path)}")


if __name__ == "__main__":
    main()
