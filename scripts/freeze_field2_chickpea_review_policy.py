#!/usr/bin/env python
"""Freeze the predeclared raw-reflectance NDVI and chickpea-review policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ["GDAL_PAM_ENABLED"] = "NO"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_chickpea_review import materialize_policy
from chickpea_ssl.field2_readiness import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_chickpea_review_policy.yaml"), type=Path)
    args = parser.parse_args()
    project = Path(yaml.safe_load(args.paths.read_text())["project_root"]).resolve()
    policy_path = args.config.resolve()
    policy = yaml.safe_load(policy_path.read_text())
    result = materialize_policy(project, args.paths, policy, policy_path)
    print("FROZEN: prediction-free Field 2 chickpea-review policy")
    for key, value in result["contract"]["counts"].items():
        print(f"{key}: {value}")
    path = project / policy["outputs"]["contract"]
    print(f"Contract: {path}")
    print(f"Contract SHA-256: {sha256(path)}")


if __name__ == "__main__":
    main()
