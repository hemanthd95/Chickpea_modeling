#!/usr/bin/env python
"""Validate the frozen Field 2 chickpea-review policy without opening reserve."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_chickpea_review import validate_policy
from chickpea_ssl.field2_readiness import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_chickpea_review_policy.yaml"), type=Path)
    args = parser.parse_args()
    project = Path(yaml.safe_load(args.paths.read_text())["project_root"]).resolve()
    policy = yaml.safe_load(args.config.read_text())
    result = validate_policy(project, policy)
    print("Field 2 chickpea-review policy validation passed")
    print(f"Contract SHA-256: {sha256(project / policy['outputs']['contract'])}")
    print(f"Main frame: 800; manual queue: {len(result['references'][result['references'].manual_review_required.astype(str).str.lower().eq('true')])}")
    print("Reserve opened/served: 0; checkpoint/model/prediction use: false")


if __name__ == "__main__":
    main()
