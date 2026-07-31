#!/usr/bin/env python
"""Safely extract and inventory the supporting OneDrive archive."""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def classify(path: PurePosixPath) -> str:
    name = str(path).lower()
    if (
        re.search(r"field[ _-]*2|tall[ _-]*grass", name)
        or re.search(r"kusi cubes analysis trials/cube[_ -]*34_georectify", name)
    ):
        return "locked_field2_external_validation"
    if re.search(r"plot|treatment|experiment|design|boundary", name):
        return "experimental_plot_support"
    if path.suffix.lower() in {".shp", ".dbf", ".shx", ".prj", ".gpkg", ".kml"}:
        return "gis_support"
    return "supporting_material"


def safe_member(info: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {info.filename}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"Symbolic links are not allowed: {info.filename}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text())
    project = Path(config["project_root"])
    destination = args.destination or project / "data" / "OneDrive_2026-07-31_raw"
    local = project / "metadata" / "local"
    local.mkdir(parents=True, exist_ok=True)

    if not args.archive.is_file():
        raise SystemExit(f"Archive not found: {args.archive}")
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(
            f"Destination is not empty: {destination}\n"
            "Nothing was overwritten. Choose a new destination or review the existing import."
        )
    destination.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.archive) as archive:
        members = [(info, safe_member(info)) for info in archive.infolist()]
        for info, relative in members:
            target = destination.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite: {target}")
            with archive.open(info) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            rows.append({
                "relative_path": str(relative),
                "category": classify(relative),
                "extension": target.suffix.lower(),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
                "content_inspected": False,
            })

    inventory = pd.DataFrame(rows)
    inventory.to_csv(local / "onedrive_archive_inventory.csv", index=False)
    summary = (
        inventory.groupby(["category", "extension"], dropna=False)
        .agg(files=("relative_path", "count"), bytes=("size_bytes", "sum"))
        .reset_index()
    )
    summary.to_csv(local / "onedrive_archive_summary.csv", index=False)
    (local / "onedrive_archive_sha256.txt").write_text(
        f"{sha256(args.archive)}  {args.archive.name}\n"
    )
    locked = int((inventory["category"] == "locked_field2_external_validation").sum())
    print(f"Archive preserved at: {args.archive}")
    print(f"Extracted files: {len(inventory)}")
    print(f"Destination: {destination}")
    print(f"Locked Field 2/tall-grass files: {locked}")
    print(f"Inventory: {local / 'onedrive_archive_inventory.csv'}")
    print(f"Summary: {local / 'onedrive_archive_summary.csv'}")
    print("No file contents or Field 2 coordinates were interpreted.")


if __name__ == "__main__":
    main()
