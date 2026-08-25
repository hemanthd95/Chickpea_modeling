#!/usr/bin/env python
"""Build a provenance-aware manifest and quantify selected mask conflicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MASK_ROLES = ("chickpea_mask", "weed_mask", "soil_mask")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def valid_binary_rows(mask_qc: pd.DataFrame, cube: str, role: str,
                      height: int, width: int) -> pd.DataFrame:
    rows = mask_qc[
        (mask_qc["cube_id"] == cube)
        & (mask_qc["file_role"] == role)
        & (mask_qc["bands"] == 1)
        & (mask_qc["height"] == height)
        & (mask_qc["width"] == width)
    ].copy()
    def is_binary(text: str) -> bool:
        try:
            return set(json.loads(text)).issubset({0, 1, 255})
        except (TypeError, ValueError):
            return False
    return rows[rows["sample_unique_values"].map(is_binary)]


def choose_mask(candidates: pd.DataFrame, override: str) -> tuple[str, str]:
    if override:
        selected = candidates[candidates["relative_path"] == override]
        return (override, "override") if len(selected) == 1 else ("", "invalid_override")
    if len(candidates) == 1:
        return candidates.iloc[0]["relative_path"], "unique_valid_candidate"
    if len(candidates) == 0:
        return "", "missing_valid_binary_mask"
    return "", "ambiguous_multiple_candidates"


def load_bool(path: Path) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as dataset:
        return dataset.read(1) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--decisions", default=Path("configs/data_decisions.yaml"), type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.paths.read_text())
    decisions = yaml.safe_load(args.decisions.read_text())
    project = Path(config["project_root"])
    local = project / "metadata" / "local"
    catalog = pd.read_csv(local / "project_catalog.csv", dtype=str).fillna("")
    mask_qc = pd.read_csv(local / "mask_qc.csv", dtype=str).fillna("")
    for column in ("height", "width", "bands"):
        mask_qc[column] = pd.to_numeric(mask_qc[column], errors="coerce").fillna(0).astype(int)
    envi_qc = pd.read_csv(local / "envi_qc.csv", dtype=str).fillna("")
    reflectance_qc = envi_qc[envi_qc["file_role"] == "reflectance_header"].copy()
    reflectance_qc[["lines", "samples"]] = reflectance_qc[["lines", "samples"]].astype(int)
    qc_by_cube = reflectance_qc.set_index("cube_id")
    mask_root = Path(config["field1"]["masks_emmanuel"])
    excluded_tables = decisions["label_table_exclusions"]
    overrides = decisions.get("mask_overrides", {})

    manifest_rows: list[dict[str, object]] = []
    spatial_rows: list[dict[str, object]] = []
    reflectance_headers = catalog[catalog["file_role"] == "reflectance_header"]
    for _, header_row in reflectance_headers.sort_values("cube_id_inferred").iterrows():
        cube = header_row["cube_id_inferred"]
        height = int(qc_by_cube.loc[cube, "lines"])
        width = int(qc_by_cube.loc[cube, "samples"])
        row: dict[str, object] = {
            "cube_id": cube,
            "acquisition_date": header_row["acquisition_date"],
            "processing_batch": header_row.get("processing_batch", ""),
            "reflectance_source": "processing_batch",
            "mask_source": "emmanuel",
            "reflectance_header": header_row["relative_path"],
            "reflectance_bip": header_row["relative_path"][:-4],
            "height": height,
            "width": width,
            "bands": 150,
        }
        selected_arrays: dict[str, np.ndarray] = {}
        for role in MASK_ROLES:
            candidates = valid_binary_rows(mask_qc, cube, role, height, width)
            override = overrides.get(cube, {}).get(role, "")
            selected, status = choose_mask(candidates, override)
            row[role] = selected
            row[f"{role}_status"] = status
            if selected:
                selected_arrays[role] = load_bool(mask_root / selected)

        label_candidates = catalog[
            (catalog["cube_id_inferred"] == cube)
            & (catalog["file_role"] == "label_table")
        ]
        label_path = label_candidates.iloc[0]["relative_path"] if len(label_candidates) else ""
        if cube in excluded_tables:
            row["label_table"] = ""
            row["label_table_status"] = "excluded_duplicate_association"
        else:
            row["label_table"] = label_path
            row["label_table_status"] = "available" if label_path else "not_available"
        row["weed_ssl_ready"] = bool(row["weed_mask"])
        row["raster_supervised_ready"] = all(bool(row[role]) for role in MASK_ROLES)
        row["legacy_table_ready"] = bool(row["label_table"])
        manifest_rows.append(row)

        counts = {role: int(array.sum()) for role, array in selected_arrays.items()}
        chickpea = selected_arrays.get("chickpea_mask")
        weed = selected_arrays.get("weed_mask")
        soil = selected_arrays.get("soil_mask")
        def pair_overlap(first: np.ndarray | None, second: np.ndarray | None) -> int:
            return int(np.logical_and(first, second).sum()) if first is not None and second is not None else 0
        conflict = np.zeros((height, width), dtype=np.uint8)
        for array in selected_arrays.values():
            conflict += array.astype(np.uint8)
        union = conflict > 0
        spatial_rows.append({
            "cube_id": cube,
            "available_masks": len(selected_arrays),
            "chickpea_pixels": counts.get("chickpea_mask", 0),
            "weed_pixels": counts.get("weed_mask", 0),
            "soil_pixels": counts.get("soil_mask", 0),
            "union_pixels": int(union.sum()),
            "unclassified_pixels": int((~union).sum()),
            "overlap_pixels": int((conflict > 1).sum()),
            "triple_overlap_pixels": int((conflict > 2).sum()),
            "chickpea_weed_overlap": pair_overlap(chickpea, weed),
            "chickpea_soil_overlap": pair_overlap(chickpea, soil),
            "weed_soil_overlap": pair_overlap(weed, soil),
            "overlap_fraction_of_union": float((conflict > 1).sum() / max(union.sum(), 1)),
        })

    archive_decision = decisions.get("archive_cube20", {})
    processed_qc_path = local / "archive_cube20_processed_mask_qc.csv"
    if archive_decision.get("status") == "approved_for_deterministic_mask_materialization" and processed_qc_path.is_file():
        archive_root = project / "data" / "OneDrive_2026-07-31_raw"
        reflectance_data = archive_root / archive_decision["reflectance"]
        reflectance_header = Path(f"{reflectance_data}.hdr")
        processed = project / "data" / "processed" / "field1_cube20"
        masks = {role: processed / f"{role}.tif" for role in ("chickpea", "weed", "soil")}
        required = [reflectance_data, reflectance_header, *masks.values()]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Approved Cube 20 products are missing: {missing}")
        processed_qc = pd.read_csv(processed_qc_path).set_index("relative_path")
        for path in masks.values():
            relative = str(path.relative_to(project))
            if relative not in processed_qc.index or sha256(path) != processed_qc.loc[relative, "sha256"]:
                raise ValueError(f"Cube 20 processed-mask hash mismatch: {relative}")
        selected_arrays = {f"{role}_mask": load_bool(path) for role, path in masks.items()}
        height, width = selected_arrays["chickpea_mask"].shape
        manifest_rows.append({
            "cube_id": "field1_cube20",
            "acquisition_date": str(config["field1"].get("acquisition_date", "2025-05-06")),
            "processing_batch": "archive_recovered_2026-07-31",
            "reflectance_source": "archive_cube20", "mask_source": "project_relative",
            "reflectance_header": str(reflectance_header.relative_to(archive_root)),
            "reflectance_bip": str(reflectance_data.relative_to(archive_root)),
            "height": height, "width": width, "bands": 150,
            "chickpea_mask": str(masks["chickpea"].relative_to(project)),
            "chickpea_mask_status": "derived_from_authoritative_categorical_mask",
            "weed_mask": str(masks["weed"].relative_to(project)),
            "weed_mask_status": "derived_from_authoritative_categorical_mask",
            "soil_mask": str(masks["soil"].relative_to(project)),
            "soil_mask_status": "derived_from_authoritative_categorical_mask",
            "label_table": "", "label_table_status": "not_available",
            "weed_ssl_ready": True, "raster_supervised_ready": True,
            "legacy_table_ready": False,
        })
        chickpea, weed, soil = (selected_arrays[f"{role}_mask"] for role in ("chickpea", "weed", "soil"))
        conflict = chickpea.astype(np.uint8) + weed.astype(np.uint8) + soil.astype(np.uint8)
        union = conflict > 0
        spatial_rows.append({
            "cube_id": "field1_cube20", "available_masks": 3,
            "chickpea_pixels": int(chickpea.sum()), "weed_pixels": int(weed.sum()),
            "soil_pixels": int(soil.sum()), "union_pixels": int(union.sum()),
            "unclassified_pixels": int((~union).sum()),
            "overlap_pixels": int((conflict > 1).sum()),
            "triple_overlap_pixels": int((conflict > 2).sum()),
            "chickpea_weed_overlap": int((chickpea & weed).sum()),
            "chickpea_soil_overlap": int((chickpea & soil).sum()),
            "weed_soil_overlap": int((weed & soil).sum()),
            "overlap_fraction_of_union": float((conflict > 1).sum() / max(union.sum(), 1)),
        })

    manifest = pd.DataFrame(manifest_rows)
    spatial = pd.DataFrame(spatial_rows)
    manifest.to_csv(local / "authoritative_manifest.csv", index=False)
    spatial.to_csv(local / "spatial_mask_qc.csv", index=False)
    print(f"Cubes in manifest: {len(manifest)}")
    print(f"Weed-SSL ready: {int(manifest['weed_ssl_ready'].sum())}")
    print(f"Raster-supervised ready: {int(manifest['raster_supervised_ready'].sum())}")
    print(f"Legacy-table ready: {int(manifest['legacy_table_ready'].sum())}")
    print(f"Cubes with any mask overlap: {int((spatial['overlap_pixels'] > 0).sum())}")
    print(f"Manifest: {local / 'authoritative_manifest.csv'}")
    print(f"Spatial QC: {local / 'spatial_mask_qc.csv'}")


if __name__ == "__main__":
    main()
