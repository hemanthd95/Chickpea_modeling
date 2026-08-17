#!/usr/bin/env python
"""Validate the immutable Field 2 investigator role contract and dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from chickpea_ssl.field2_blind_review import validate_frozen_role_contract
from chickpea_ssl.field2_readiness import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve()
    contract_path = project / config["review"]["role_contract"]
    result = validate_frozen_role_contract(project, config, contract_path)
    contract = result["contract"]
    subprocess.check_call(
        ["git", "cat-file", "-e", f"{contract['freeze_git_commit']}^{{commit}}"],
        cwd=project,
    )
    sampling_frame = project / config["sampling"]["sampling_frame"]
    sampling_contract = project / config["sampling"]["sampling_contract"]
    if contract.get("blind_point_sampling_frame_generated") is not False:
        raise ValueError("Role contract does not declare the blind sampling frame absent")
    if sampling_frame.exists() or sampling_contract.exists():
        raise ValueError("Blind point-sampling output exists during role-contract validation")
    print("Field 2 frozen cube-role contract validation passed")
    print(f"Contract: {contract_path}")
    print(f"Contract SHA-256: {sha256(contract_path)}")
    print(f"Freeze commit: {contract['freeze_git_commit']}")
    print(f"Role totals: {result['totals']['primary_roles']}")
    print(f"Biological-flag totals: {result['totals']['biological_flags']}")
    print("Blind point-sampling frame: absent")


if __name__ == "__main__":
    main()
