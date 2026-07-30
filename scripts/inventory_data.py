#!/usr/bin/env python
"""Build a reviewable manifest from a directory of hyperspectral derivatives."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROLES = {
    "deriv2": (r"second.?deriv", r"deriv.?2", r"\bsd\b"),
    "deriv1": (r"first.?deriv", r"deriv.?1", r"\bfd\b"),
    "ndvi": (r"ndvi",),
    "mask_chickpea": (r"chickpea.*mask", r"mask.*chickpea"),
    "mask_weed": (r"weed.*mask", r"mask.*weed"),
    "mask_soil": (r"soil.*mask", r"mask.*soil"),
    "alley_mask": (r"alley.*mask", r"mask.*alley"),
    "pca": (r"pca",),
}
SUPPORTED = {".bip", ".bil", ".bsq", ".hdr", ".tif", ".tiff", ".npy", ".npz"}


def role_for(path: Path) -> str | None:
    name = path.name.lower()
    for role, patterns in ROLES.items():
        if any(re.search(pattern, name) for pattern in patterns):
            return role
    return None


def cube_key(path: Path) -> str:
    name = path.stem.lower()
    tokens = (
        "pca_transform", "secondderivative", "second_derivative",
        "firstderivative", "first_derivative", "deriv1", "deriv2",
        "chickpea_mask", "weed_mask", "soil_mask", "ndvi", "pca",
    )
    for token in tokens:
        name = name.replace(token, "")
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or path.parent.name.lower().replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Data directory does not exist: {args.root}")

    rows: dict[str, dict[str, str]] = {}
    unresolved: list[str] = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        role = role_for(path)
        if role is None:
            unresolved.append(str(path.resolve()))
            continue
        key = cube_key(path)
        row = rows.setdefault(key, {"cube_id": key, "field_id": ""})
        if role in row:
            row[f"duplicate_{role}"] = f"{row[role]} | {path.resolve()}"
        else:
            row[role] = str(path.resolve())

    columns = [
        "cube_id", "field_id", "pca", "deriv1", "deriv2", "ndvi",
        "mask_chickpea", "mask_weed", "mask_soil", "tall_grass_points",
        "alley_mask",
    ]
    frame = pd.DataFrame(rows.values())
    for column in columns:
        if column not in frame:
            frame[column] = ""
    extras = sorted(set(frame.columns) - set(columns))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[columns + extras].sort_values("cube_id").to_csv(args.output, index=False)

    print(f"Wrote {len(frame)} candidate cube rows to {args.output}")
    print(f"Unresolved supported files: {len(unresolved)}")
    for item in unresolved[:25]:
        print(f"  {item}")
    if len(unresolved) > 25:
        print(f"  ... and {len(unresolved) - 25} more")


if __name__ == "__main__":
    main()

