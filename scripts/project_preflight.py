#!/usr/bin/env python
"""Inventory authorized Field 1 data without reading locked Field 2 labels."""

from __future__ import annotations

import argparse
import hashlib
import platform
import re
from pathlib import Path

import pandas as pd
import yaml

ALLOWED_SUFFIXES = {
    ".bip", ".bil", ".bsq", ".hdr", ".tif", ".tiff", ".csv", ".gpkg",
    ".shp", ".dbf", ".shx", ".prj", ".npy", ".npz",
}


def infer_cube_id(text: str) -> str:
    normalized = text.lower()
    field = re.search(r"field\s*[_-]?(\d+)", normalized)
    unit = re.search(r"(?:cube|plot)\s*[_-]?(\d+)", normalized)
    if field and unit:
        return f"field{field.group(1)}_cube{unit.group(1)}"
    return f"cube{unit.group(1)}" if unit else ""


def lightweight_fingerprint(path: Path) -> str:
    """Hash metadata and small edge samples, not entire multi-GB cubes."""
    digest = hashlib.sha256()
    stat = path.stat()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    with path.open("rb") as stream:
        digest.update(stream.read(65_536))
        if stat.st_size > 65_536:
            stream.seek(max(0, stat.st_size - 65_536))
            digest.update(stream.read(65_536))
    return digest.hexdigest()


def inventory(root: Path, source: str, date: str = "") -> list[dict[str, object]]:
    if not root.is_dir():
        raise FileNotFoundError(f"{source} directory not found: {root}")
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        rows.append({
            "source": source,
            "acquisition_date": date,
            "cube_id_inferred": infer_cube_id(str(path.relative_to(root))),
            "relative_path": str(path.relative_to(root)),
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "fingerprint": lightweight_fingerprint(path),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed in the active environment. Run the "
            "environment.yml update and activate chickpea_modeling."
        ) from exc
    config = yaml.safe_load(args.paths.read_text())
    project_root = Path(config["project_root"])
    output_dir = project_root / "metadata" / "local"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    field1 = config["field1"]
    rows.extend(inventory(Path(field1["derived_ssl"]), "field1.derived_ssl"))
    rows.extend(inventory(Path(field1["label_csvs"]), "field1.label_csvs"))
    rows.extend(inventory(Path(field1["masks_emmanuel"]), "field1.masks_emmanuel"))
    for date, root in field1["reflectance_dates"].items():
        rows.extend(inventory(Path(root), "field1.reflectance", str(date)))

    if config["field2"].get("locked") is not True:
        raise SystemExit("Field 2 must remain locked during development.")

    catalog = pd.DataFrame(rows)
    catalog.to_csv(output_dir / "project_catalog.csv", index=False)
    summary = (
        catalog.groupby(["source", "acquisition_date", "extension"], dropna=False)
        .agg(files=("relative_path", "count"), bytes=("size_bytes", "sum"))
        .reset_index()
    )
    summary.to_csv(output_dir / "catalog_summary.csv", index=False)

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")
    print(f"Cataloged {len(catalog)} observed Field 1 files.")
    print(f"Catalog: {output_dir / 'project_catalog.csv'}")
    print(f"Summary: {output_dir / 'catalog_summary.csv'}")
    print("Field 2 remains locked and was not inventoried.")


if __name__ == "__main__":
    main()
