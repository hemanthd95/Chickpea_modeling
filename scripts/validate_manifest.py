#!/usr/bin/env python
"""Fail early on missing, duplicate, or incompatible manifest entries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "cube_id", "field_id", "pca", "deriv1", "deriv2", "ndvi",
    "mask_combined", "mask_chickpea", "mask_weed", "mask_soil",
)
REQUIRED_FEATURES = ("pca", "deriv1", "deriv2", "ndvi")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    table = pd.read_csv(args.manifest, dtype=str).fillna("")
    errors: list[str] = []
    for column in REQUIRED_COLUMNS:
        if column not in table:
            errors.append(f"Missing column: {column}")
    if errors:
        raise SystemExit("\n".join(errors))

    if table["cube_id"].duplicated().any():
        errors.append("cube_id values must be unique")
    for row_index, row in table.iterrows():
        label = row["cube_id"] or f"row {row_index + 2}"
        if not row["field_id"]:
            errors.append(f"{label}: field_id is empty (needed for leakage-safe splits)")
        for column in REQUIRED_FEATURES:
            value = row[column]
            if not value:
                errors.append(f"{label}: {column} is empty")
            elif not Path(value).is_file():
                errors.append(f"{label}: {column} not found: {value}")
        combined = row["mask_combined"]
        separate = [row["mask_chickpea"], row["mask_weed"], row["mask_soil"]]
        if combined:
            if not Path(combined).is_file():
                errors.append(f"{label}: mask_combined not found: {combined}")
        elif all(separate):
            for column, value in zip(
                ("mask_chickpea", "mask_weed", "mask_soil"), separate
            ):
                if not Path(value).is_file():
                    errors.append(f"{label}: {column} not found: {value}")
        else:
            errors.append(
                f"{label}: provide mask_combined or all three class masks"
            )
        duplicate_columns = [c for c in table.columns if c.startswith("duplicate_")]
        for column in duplicate_columns:
            if row[column]:
                errors.append(f"{label}: resolve {column}: {row[column]}")

    if errors:
        print(f"Manifest validation failed with {len(errors)} issue(s):")
        raise SystemExit("\n".join(f"- {item}" for item in errors))
    print(f"Manifest is structurally valid: {len(table)} cubes across "
          f"{table['field_id'].nunique()} fields")


if __name__ == "__main__":
    main()
