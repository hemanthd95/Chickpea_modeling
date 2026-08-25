#!/usr/bin/env python
"""Independently validate frozen Field 2 area-stratified sampling v2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_area_sampling_v2 import DOMAINS, load_verified_v2
from chickpea_ssl.field2_readiness import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_area_sampling_v2.yaml"), type=Path)
    args = parser.parse_args()
    project = Path(yaml.safe_load(args.paths.read_text())["project_root"]).resolve()
    config = yaml.safe_load(args.config.read_text())
    result = load_verified_v2(project, config)
    print("Field 2 area-stratified sampling v2 validation passed")
    print(f"Contract SHA-256: {sha256(project / config['outputs']['contract'])}")
    print(f"Main: {len(result['main'])}; reserve locked: {len(result['reserve'])}")
    for name in ("main", "reserve"):
        frame = result[name]
        print(name + ": " + ", ".join(
            f"{domain}={int((frame.sampling_domain == domain).sum())}" for domain in DOMAINS
        ))
    print("Reserve API exposure: zero")


if __name__ == "__main__":
    main()
