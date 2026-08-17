#!/usr/bin/env python
"""Build the deterministic blind Field 2 point frame after role freeze only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["GDAL_PAM_ENABLED"] = "NO"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates
import yaml

from chickpea_ssl.field2_blind_review import (
    atomic_write_bytes,
    atomic_write_yaml,
    deduplicate_geographic_views,
    geographic_ground_cell_ids,
    require_frozen_role_contract,
    scalar_index_rank_strata,
    spatial_group_ids,
    stable_selection_rank,
    stratified_deterministic_sample,
)
from chickpea_ssl.field2_readiness import ReadOnlySourceGuard, open_envi_memmap, sha256


def git_output(project: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=project, text=True).strip()


def collapsed_cube_candidates(
    cube_id: str,
    support: np.ndarray,
    scalar_index: np.ndarray,
    transform,
    crs: str,
    role_record: dict,
    source_sha256: str,
    mask_sha256: str,
    sampling: dict,
) -> pd.DataFrame:
    valid = support & np.isfinite(scalar_index)
    rows, columns = np.nonzero(valid)
    values = np.asarray(scalar_index[rows, columns], dtype=np.float64)
    if not len(values):
        raise RuntimeError(f"{cube_id}: no finite stored-index pixels inside frozen support")
    strata = scalar_index_rank_strata(values, sampling["scalar_index_rank_strata"])
    pixel_columns = columns.astype(np.float64) + 0.5
    pixel_rows = rows.astype(np.float64) + 0.5
    x = transform.a * pixel_columns + transform.b * pixel_rows + transform.c
    y = transform.d * pixel_columns + transform.e * pixel_rows + transform.f
    cell_size = float(sampling["geographic_ground_cell_size_m"])
    cell_x = np.floor(x / cell_size).astype(np.int64)
    cell_y = np.floor(y / cell_size).astype(np.int64)
    cube_seed = stable_selection_rank(int(sampling["seed"]), cube_id)
    priorities = (
        rows.astype(np.uint64) * np.uint64(73856093)
        ^ columns.astype(np.uint64) * np.uint64(19349663)
        ^ np.uint64(cube_seed)
    )
    order = np.lexsort((priorities, cell_y, cell_x))
    ordered_x, ordered_y = cell_x[order], cell_y[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (ordered_x[1:] != ordered_x[:-1]) | (ordered_y[1:] != ordered_y[:-1])
    keep = order[first]
    selected_x, selected_y = x[keep], y[keep]
    ground_ids = geographic_ground_cell_ids(crs, selected_x, selected_y, cell_size)
    spatial_ids = spatial_group_ids(crs, selected_x, selected_y, float(sampling["spatial_group_size_m"]))
    frame = pd.DataFrame({
        "cube_id": cube_id, "row": rows[keep], "column": columns[keep],
        "x": selected_x, "y": selected_y, "crs": crs,
        "valid_support_status": True, "spatial_group_id": spatial_ids,
        "geographic_ground_cell_id": ground_ids,
        "scalar_index_value": values[keep], "scalar_index_rank_stratum": strata[keep],
        "cube_evaluation_role": role_record["primary_role"],
        "cube_challenge_flags": json.dumps(role_record["flags"], sort_keys=True),
        "source_reflectance_sha256": source_sha256, "support_mask_sha256": mask_sha256,
    })
    frame["deterministic_selection_rank"] = [
        stable_selection_rank(int(sampling["seed"]), cell, cube_id, row, column)
        for cell, row, column in zip(ground_ids, rows[keep], columns[keep])
    ]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve()

    # Mandatory stop occurs before any Field 2 raster or transformed product is opened.
    role_contract_path = project / config["review"]["role_contract"]
    role_contract = require_frozen_role_contract(role_contract_path)
    expected = list(config["expected_cube_ids"])
    roles = {row["cube_id"]: row for row in role_contract["cube_roles"]}
    if list(roles) != expected:
        raise RuntimeError("Frozen role contract does not contain the exact 40-cube inventory")
    manifest_path = project / config["inputs"]["valid_support_manifest"]
    if sha256(manifest_path) != role_contract["field2_valid_support_manifest_sha256"]:
        raise RuntimeError("Valid-support manifest differs from the frozen role contract")
    if role_contract["field2_source_manifest_sha256"] != config["source_manifest_sha256"]:
        raise RuntimeError("Field 2 source manifest differs from the frozen role contract")

    roots = tuple(Path(value) for value in paths["field2"]["readiness_roots"])
    guard = ReadOnlySourceGuard(roots)
    inventory = pd.read_csv(project / config["inputs"]["readiness_inventory"]).fillna("")
    support_manifest = pd.read_csv(manifest_path).fillna("")
    sampling = config["sampling"]
    candidate_frames = []
    for cube_id in expected:
        role_record = roles[cube_id]
        if role_record["primary_role"] == "exclude_with_reason":
            continue
        source_rows = inventory[inventory.cube_id == cube_id]
        index_row = source_rows[source_rows.product_type == "stored_index"].iloc[0]
        support_row = support_manifest[support_manifest.cube_id == cube_id].iloc[0]
        mask_path = project / str(support_row.mask_path)
        if sha256(mask_path) != support_row.mask_sha256:
            raise RuntimeError(f"{cube_id}: support-mask checksum mismatch")
        index_array, _ = open_envi_memmap(
            project / str(index_row.header_path), project / str(index_row.binary_path), guard
        )
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(mask_path, "r") as dataset:
                support = dataset.read(1) == 1
                transform, crs = dataset.transform, str(dataset.crs)
        scalar_index = np.asarray(index_array[..., 0], dtype=np.float32)
        candidate_frames.append(collapsed_cube_candidates(
            cube_id, support, scalar_index, transform, crs, role_record,
            str(support_row.source_reflectance_sha256), str(support_row.mask_sha256), sampling,
        ))
    candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    reliability = sampling["repeated_view_reliability"]
    unique = deduplicate_geographic_views(
        candidates, int(sampling["seed"]),
        int(reliability["proposed_count"]) if reliability["enabled"] else 0,
    )
    selected = stratified_deterministic_sample(
        unique, sampling["proposed_points_per_cube_by_role"],
        sampling["scalar_index_rank_strata"], float(sampling["minimum_separation_m"]),
    )
    longitudes = np.full(len(selected), np.nan)
    latitudes = np.full(len(selected), np.nan)
    for crs, indices in selected.groupby("crs").groups.items():
        try:
            lon, lat = transform_coordinates(crs, "EPSG:4326", selected.loc[indices, "x"], selected.loc[indices, "y"])
            longitudes[indices] = lon; latitudes[indices] = lat
        except Exception:
            pass
    selected["longitude"] = longitudes
    selected["latitude"] = latitudes
    selected["sample_id"] = [
        "f2-" + hashlib.sha256(
            f"{sampling['seed']}:{row.cube_id}:{row.row}:{row.column}".encode()
        ).hexdigest()[:16]
        for row in selected.itertuples(index=False)
    ]
    selected["inclusion_probability_basis"] = "spatially_thinned_unique_ground_cells_within_cube_and_rank_stratum"
    columns = [
        "sample_id", "cube_id", "row", "column", "x", "y", "longitude", "latitude", "crs",
        "valid_support_status", "spatial_group_id", "geographic_ground_cell_id",
        "scalar_index_value", "scalar_index_rank_stratum", "cube_evaluation_role",
        "cube_challenge_flags", "inclusion_probability", "design_weight", "overlap_status",
        "deterministic_selection_rank", "sampling_stratum_population", "inclusion_probability_basis",
        "source_reflectance_sha256", "support_mask_sha256",
    ]
    selected = selected[columns].sort_values(["cube_id", "scalar_index_rank_stratum", "deterministic_selection_rank"])
    output = project / sampling["sampling_frame"]
    buffer = io.StringIO(); selected.to_csv(buffer, index=False)
    atomic_write_bytes(output, buffer.getvalue().encode())
    contract = {
        "status": "field2_blind_sampling_frame_frozen",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_git_commit": git_output(project, "rev-parse", "HEAD"),
        "configuration_path": str(args.config), "configuration_sha256": sha256(args.config.resolve()),
        "role_contract_path": str(role_contract_path.relative_to(project)),
        "role_contract_sha256": sha256(role_contract_path),
        "sampling_frame": str(output.relative_to(project)), "sampling_frame_sha256": sha256(output),
        "seed": sampling["seed"], "spatial_group_size_m": sampling["spatial_group_size_m"],
        "minimum_separation_m": sampling["minimum_separation_m"],
        "geographic_ground_cell_size_m": sampling["geographic_ground_cell_size_m"],
        "strata": sampling["scalar_index_rank_strata"],
        "strata_interpretation": "empirical stored-scalar-index ranks; not biological classes",
        "proposed_counts_used": sampling["proposed_points_per_cube_by_role"],
        "selected_count": len(selected),
        "geographic_duplicate_prevention": "one deterministic view per projected ground cell",
        "inclusion_probability_basis": "spatially thinned unique ground-cell frame within cube and rank stratum",
        "supervised_predictions_or_probabilities_used": False,
        "embeddings_used": False, "field1_labels_or_masks_used": False,
        "biological_point_labels_generated": False,
    }
    atomic_write_yaml(project / sampling["sampling_contract"], contract)
    print(f"FROZEN: {len(selected)} prediction-free blind annotation candidates", flush=True)
    print(f"File: {output}", flush=True)


if __name__ == "__main__":
    main()
