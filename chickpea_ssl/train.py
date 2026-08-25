"""Starter entry point.

This command validates configuration and data boundaries before expensive GPU
training. Dataset-specific ENVI header interpretation is added after the generated
manifest has been reviewed on the workstation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from .model import SimCLR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    manifest_path = Path(config["data"]["manifest"])
    if not manifest_path.is_file():
        raise SystemExit(
            f"Manifest not found: {manifest_path}\n"
            "Run scripts/inventory_data.py, review the CSV, then validate it."
        )
    manifest = pd.read_csv(manifest_path)
    if manifest["field_id"].nunique() < 2:
        raise SystemExit("At least two field_id groups are required for held-out evaluation.")

    channels = 5 + 5 + 5 + 1
    model = SimCLR(
        in_channels=channels,
        embedding_dim=config["model"]["embedding_dim"],
        projection_dim=config["model"]["projection_dim"],
    )
    parameters = sum(p.numel() for p in model.parameters())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Preflight passed: {len(manifest)} cubes, "
          f"{manifest['field_id'].nunique()} fields")
    print(f"Device: {device}; trainable parameters: {parameters:,}")
    print("Next gate: confirm ENVI metadata, mask encodings, and cube-to-mask shapes.")


if __name__ == "__main__":
    main()

