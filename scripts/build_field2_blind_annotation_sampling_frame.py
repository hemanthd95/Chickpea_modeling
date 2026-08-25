#!/usr/bin/env python
"""Materialize and freeze prediction-free main and reserve Field 2 point frames."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["GDAL_PAM_ENABLED"] = "NO"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates
from scipy.spatial import cKDTree
import yaml

from chickpea_ssl.field2_blind_review import (
    atomic_write_bytes, atomic_write_yaml, geographic_ground_cell_ids,
    minimum_separation_thin, require_frozen_role_contract, scalar_index_rank_strata,
    select_main_reserve_frames, spatial_group_ids, stable_selection_rank,
    validate_frozen_role_contract,
)
from chickpea_ssl.field2_readiness import (
    ReadOnlySourceGuard, compare_snapshots, open_envi_memmap, sha256, source_snapshot,
)


OUTPUT_COLUMNS = [
    "sample_id", "sampling_frame", "cube_id", "row", "column", "x", "y",
    "longitude", "latitude", "crs", "valid_support_status", "valid_context_status",
    "spatial_group_id", "geographic_ground_cell_id", "scalar_index_value",
    "scalar_index_rank_stratum", "cube_evaluation_role", "cube_challenge_flags",
    "cube_investigator_notes", "within_cube_ground_cell_pixel_count",
    "cross_cube_ground_cell_view_count", "first_stage_inclusion_probability",
    "sampling_stratum_block_population", "sampling_stratum_block_quota",
    "cube_rank_stratum_requested", "final_inclusion_probability",
    "overall_inclusion_probability", "design_weight", "overlap_status",
    "deterministic_selection_rank", "source_reflectance_sha256", "support_mask_sha256",
    "source_preview_checksums", "reserve_release_status",
]


def git_output(project: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=project, text=True).strip()


def frame_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame[OUTPUT_COLUMNS].to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode()


def collapsed_cube_candidates(
    cube_id: str, support: np.ndarray, scalar_index: np.ndarray, transform, crs: str,
    role_record: dict, source_sha256: str, mask_sha256: str, preview_hashes: str,
    preview_step: int, preview_width: int, preview_height: int, sampling: dict,
) -> pd.DataFrame:
    valid = support & np.isfinite(scalar_index)
    rows, columns = np.nonzero(valid)
    preview_x = (columns.astype(np.float64) + 0.5) / preview_step
    preview_y = (rows.astype(np.float64) + 0.5) / preview_step
    radius = int(sampling["annotation_context_radius_preview_pixels"])
    context = (
        (preview_x >= radius) & (preview_x < preview_width - radius)
        & (preview_y >= radius) & (preview_y < preview_height - radius)
    )
    rows, columns = rows[context], columns[context]
    values = np.asarray(scalar_index[rows, columns], dtype=np.float64)
    if not len(values):
        raise RuntimeError(f"{cube_id}: no finite valid-support pixels have full annotation context")
    strata = scalar_index_rank_strata(values, sampling["scalar_index_rank_strata"])
    pixel_columns, pixel_rows = columns.astype(float) + 0.5, rows.astype(float) + 0.5
    x = transform.a * pixel_columns + transform.b * pixel_rows + transform.c
    y = transform.d * pixel_columns + transform.e * pixel_rows + transform.f
    cell_size = float(sampling["geographic_ground_cell_size_m"])
    cell_x, cell_y = np.floor(x / cell_size).astype(np.int64), np.floor(y / cell_size).astype(np.int64)
    cube_seed = np.uint64(stable_selection_rank(int(sampling["seed"]), cube_id))
    priorities = rows.astype(np.uint64) * np.uint64(73856093) ^ columns.astype(np.uint64) * np.uint64(19349663) ^ cube_seed
    order = np.lexsort((priorities, cell_y, cell_x))
    ordered_x, ordered_y = cell_x[order], cell_y[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (ordered_x[1:] != ordered_x[:-1]) | (ordered_y[1:] != ordered_y[:-1])
    starts = np.flatnonzero(first)
    counts = np.diff(np.r_[starts, len(order)])
    keep = order[first]
    selected_x, selected_y = x[keep], y[keep]
    ground_ids = geographic_ground_cell_ids(crs, selected_x, selected_y, cell_size)
    blocks = spatial_group_ids(crs, selected_x, selected_y, float(sampling["spatial_group_size_m"]))
    frame = pd.DataFrame({
        "cube_id": cube_id, "row": rows[keep], "column": columns[keep],
        "x": selected_x, "y": selected_y, "crs": crs,
        "valid_support_status": True, "valid_context_status": True,
        "spatial_group_id": blocks, "geographic_ground_cell_id": ground_ids,
        "scalar_index_value": values[keep], "scalar_index_rank_stratum": strata[keep],
        "cube_evaluation_role": role_record["primary_role"],
        "cube_challenge_flags": json.dumps(role_record["flags"], sort_keys=True),
        "cube_investigator_notes": role_record["investigator_notes"],
        "within_cube_ground_cell_pixel_count": counts,
        "source_reflectance_sha256": source_sha256, "support_mask_sha256": mask_sha256,
        "source_preview_checksums": preview_hashes,
    })
    frame["deterministic_selection_rank"] = [
        stable_selection_rank(int(sampling["seed"]), cell, cube_id, row, column)
        for cell, row, column in zip(ground_ids, rows[keep], columns[keep])
    ]
    return frame


def unique_spatially_thinned_candidates(candidates: pd.DataFrame, sampling: dict) -> pd.DataFrame:
    work = candidates.copy()
    view_counts = work.groupby("geographic_ground_cell_id")["cube_id"].transform("size")
    work["cross_cube_ground_cell_view_count"] = view_counts.astype(int)
    work["overlap_status"] = np.where(view_counts > 1, "overlapping_view", "unique_view")
    work = work.sort_values(["geographic_ground_cell_id", "deterministic_selection_rank", "cube_id", "row", "column"])
    unique = work.groupby("geographic_ground_cell_id", sort=False).head(1).copy()
    unique.loc[unique.overlap_status == "overlapping_view", "overlap_status"] = "overlapping_view_selected"
    unique["first_stage_inclusion_probability"] = 1.0 / (
        unique["within_cube_ground_cell_pixel_count"] * unique["cross_cube_ground_cell_view_count"]
    )
    thinned = minimum_separation_thin(unique, float(sampling["minimum_separation_m"]))
    return thinned.sort_values(["cube_id", "deterministic_selection_rank"]).reset_index(drop=True)


def add_geographic_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reset_index(drop=True).copy()
    result["longitude"], result["latitude"] = np.nan, np.nan
    for crs, indices in result.groupby("crs").groups.items():
        try:
            lon, lat = transform_coordinates(crs, "EPSG:4326", result.loc[indices, "x"].tolist(), result.loc[indices, "y"].tolist())
            result.loc[indices, "longitude"], result.loc[indices, "latitude"] = lon, lat
        except Exception:
            pass
    return result


def finalize_frame(frame: pd.DataFrame, release_status: str) -> pd.DataFrame:
    result = add_geographic_coordinates(frame)
    result["reserve_release_status"] = release_status
    return result.sort_values(["cube_id", "scalar_index_rank_stratum", "spatial_group_id", "deterministic_selection_rank"]).reset_index(drop=True)


def nearest_neighbor_summary(frame: pd.DataFrame) -> dict[str, float]:
    distances = []
    for _, group in frame.groupby("crs"):
        if len(group) > 1:
            values, _ = cKDTree(group[["x", "y"]].to_numpy(float)).query(group[["x", "y"]].to_numpy(float), k=2)
            distances.extend(values[:, 1].tolist())
    return {"minimum_m": float(np.min(distances)), "median_m": float(np.median(distances)), "maximum_m": float(np.max(distances))}


def save_sampling_map(
    path: Path, manifest: pd.DataFrame, combined: pd.DataFrame, project: Path,
    include_reserve: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 8, figsize=(20, 13), constrained_layout=True)
    by_cube = {str(row.cube_id): row for row in manifest.itertuples(index=False)}
    for axis, cube_id in zip(axes.flat, manifest.cube_id.astype(str)):
        row = by_cube[cube_id]
        axis.imshow(plt.imread(project / str(row.false_colour_path)))
        points = combined[combined.cube_id == cube_id]
        frames = (("main", "o", "#ff2d55"), ("reserve", "x", "#00e5ff")) if include_reserve else (("main", "o", "#ff2d55"),)
        for name, marker, color in frames:
            selected = points[points.sampling_frame == name]
            axis.scatter((selected.column + .5) / int(row.preview_step), (selected.row + .5) / int(row.preview_step), s=18 if name == "main" else 23, marker=marker, c=color, linewidths=1.1, label=name)
        axis.set_title(cube_id, fontsize=8); axis.set_xticks([]); axis.set_yticks([])
    axes.flat[0].legend(loc="lower right", fontsize=6)
    fig.suptitle(
        "Field 2 prediction-free blind sampling — main circles / locked reserve crosses"
        if include_reserve else "Field 2 prediction-free area-stratified v2 — MAIN points only; reserve coordinates omitted"
    )
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor); temporary = Path(name)
    try:
        fig.savefig(temporary, dpi=170); plt.close(fig); temporary.replace(path)
    except Exception:
        plt.close(fig); temporary.unlink(missing_ok=True); raise


def verify_selected_support(combined: pd.DataFrame, support_manifest: pd.DataFrame, project: Path, radius: int, review_manifest: pd.DataFrame) -> tuple[int, int]:
    support_violations = context_violations = 0
    review_by_cube = review_manifest.set_index("cube_id")
    for row in support_manifest.itertuples(index=False):
        selected = combined[combined.cube_id == row.cube_id]
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(project / str(row.mask_path), "r") as dataset:
                support = dataset.read(1)
        support_violations += sum(support[int(point.row), int(point.column)] != 1 for point in selected.itertuples(index=False))
        review = review_by_cube.loc[row.cube_id]
        px, py = (selected.column.to_numpy(float) + .5) / int(review.preview_step), (selected.row.to_numpy(float) + .5) / int(review.preview_step)
        context_violations += int(np.count_nonzero((px < radius) | (px >= int(review.preview_width) - radius) | (py < radius) | (py >= int(review.preview_height) - radius)))
    return int(support_violations), int(context_violations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/field2_blind_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths, config = yaml.safe_load(args.paths.read_text()), yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"]).resolve(); sampling = config["sampling"]
    targets = {name: project / sampling[key] for name, key in {
        "main": "main_frame", "reserve": "reserve_frame", "combined": "combined_frame",
        "allocation_audit": "allocation_audit", "stratum_block_audit": "stratum_block_audit",
        "sampling_audit": "sampling_audit", "sampling_map": "sampling_map", "contract": "sampling_contract",
    }.items()}
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise SystemExit(f"REFUSED: immutable sampling output already exists: {existing}")
    role_contract_path = project / config["review"]["role_contract"]
    role_contract = require_frozen_role_contract(role_contract_path)
    validate_frozen_role_contract(project, config, role_contract_path)
    expected = list(config["expected_cube_ids"])
    role_records = {row["cube_id"]: row for row in role_contract["cube_roles"]}
    if list(role_records) != expected:
        raise RuntimeError("Frozen role contract does not contain the exact 40-cube inventory")
    role_by_cube = {cube: role_records[cube]["primary_role"] for cube in expected}
    manifest_path = project / config["inputs"]["valid_support_manifest"]
    if sha256(manifest_path) != role_contract["field2_valid_support_manifest_sha256"] or role_contract["field2_source_manifest_sha256"] != config["source_manifest_sha256"]:
        raise RuntimeError("Frozen source or valid-support manifest differs from the role contract")
    roots = tuple(Path(value) for value in paths["field2"]["readiness_roots"]); guard = ReadOnlySourceGuard(roots)
    source_before = source_snapshot(guard, project)
    inventory = pd.read_csv(project / config["inputs"]["readiness_inventory"]).fillna("")
    support_manifest = pd.read_csv(manifest_path).fillna("")
    review_manifest_path = project / config["review"]["package_manifest"]
    review_manifest = pd.read_csv(review_manifest_path).fillna("")
    if support_manifest.cube_id.tolist() != expected or review_manifest.cube_id.tolist() != expected:
        raise RuntimeError("Sampling inputs do not contain the exact ordered 40-cube inventory")
    preview_by_cube = review_manifest.set_index("cube_id")
    candidate_parts = []
    print("Building prediction-free eligible frames for 40 cubes...", flush=True)
    for cube_id in expected:
        index_row = inventory[(inventory.cube_id == cube_id) & (inventory.product_type == "stored_index")].iloc[0]
        support_row = support_manifest[support_manifest.cube_id == cube_id].iloc[0]
        mask_path = project / str(support_row.mask_path)
        if sha256(mask_path) != str(support_row.mask_sha256): raise RuntimeError(f"{cube_id}: support-mask checksum mismatch")
        index_array, _ = open_envi_memmap(project / str(index_row.header_path), project / str(index_row.binary_path), guard)
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(mask_path, "r") as dataset: support, transform, crs = dataset.read(1) == 1, dataset.transform, str(dataset.crs)
        review = preview_by_cube.loc[cube_id]
        candidate = collapsed_cube_candidates(cube_id, support, np.asarray(index_array[..., 0], dtype=np.float32), transform, crs, role_records[cube_id], str(support_row.source_reflectance_sha256), str(support_row.mask_sha256), str(review.preview_sha256_json), int(review.preview_step), int(review.preview_width), int(review.preview_height), sampling)
        candidate_parts.append(candidate); print(f"  {cube_id}: {len(candidate):,} within-cube ground cells", flush=True)
    candidates = pd.concat(candidate_parts, ignore_index=True)
    eligible = unique_spatially_thinned_candidates(candidates, sampling)
    print(f"Eligible after overlap de-duplication and separation: {len(eligible):,}", flush=True)
    selection_args = (eligible, role_by_cube, sampling["main_points_per_cube_by_role"], sampling["reserve_points_per_cube_by_role"], sampling["scalar_index_rank_strata"], int(sampling["seed"]))
    main_frame, reserve_frame = select_main_reserve_frames(*selection_args)
    repeated_main, repeated_reserve = select_main_reserve_frames(*selection_args)
    main_frame, reserve_frame = finalize_frame(main_frame, "released_for_main_annotation"), finalize_frame(reserve_frame, "locked_pending_separate_authorization")
    repeated_main, repeated_reserve = finalize_frame(repeated_main, "released_for_main_annotation"), finalize_frame(repeated_reserve, "locked_pending_separate_authorization")
    if frame_bytes(main_frame) != frame_bytes(repeated_main) or frame_bytes(reserve_frame) != frame_bytes(repeated_reserve): raise RuntimeError("Deterministic same-seed sampling reproduction failed")
    combined = pd.concat([main_frame, reserve_frame], ignore_index=True)
    if len(main_frame) != int(sampling["expected_main_total"]) or len(reserve_frame) != int(sampling["expected_reserve_total"]): raise RuntimeError("Main or reserve total differs from the predeclared allocation")
    if combined.sample_id.nunique() != len(combined) or combined.geographic_ground_cell_id.nunique() != len(combined): raise RuntimeError("Duplicate sample ID or geographic ground cell detected")
    if set(main_frame.sample_id) & set(reserve_frame.sample_id): raise RuntimeError("Main and reserve frames intersect")
    if not np.isfinite(combined.overall_inclusion_probability).all() or not (combined.overall_inclusion_probability > 0).all() or not np.isfinite(combined.design_weight).all() or not (combined.design_weight > 0).all(): raise RuntimeError("Invalid inclusion probability or design weight")
    allocation = pd.DataFrame([{"cube_id": cube, "cube_evaluation_role": role_by_cube[cube], "main_requested": int(sampling["main_points_per_cube_by_role"][role_by_cube[cube]]), "main_achieved": int((main_frame.cube_id == cube).sum()), "reserve_requested": int(sampling["reserve_points_per_cube_by_role"][role_by_cube[cube]]), "reserve_achieved": int((reserve_frame.cube_id == cube).sum())} for cube in expected])
    if not (allocation.main_requested == allocation.main_achieved).all() or not (allocation.reserve_requested == allocation.reserve_achieved).all(): raise RuntimeError("One or more cubes did not achieve its exact allocation")
    stratum_block = combined.groupby(["sampling_frame", "cube_id", "cube_evaluation_role", "scalar_index_rank_stratum"], sort=True).agg(points=("sample_id", "size"), spatial_blocks=("spatial_group_id", "nunique")).reset_index()
    support_violations, context_violations = verify_selected_support(combined, support_manifest, project, int(sampling["annotation_context_radius_preview_pixels"]), review_manifest)
    if support_violations or context_violations: raise RuntimeError(f"Selected-point validation failed: support={support_violations}, context={context_violations}")
    separation = nearest_neighbor_summary(combined)
    if separation["minimum_m"] + 1e-9 < float(sampling["minimum_separation_m"]): raise RuntimeError("Combined frame violates minimum separation")
    allocation_buffer, block_buffer = io.StringIO(), io.StringIO(); allocation.to_csv(allocation_buffer, index=False); stratum_block.to_csv(block_buffer, index=False)
    atomic_write_bytes(targets["main"], frame_bytes(main_frame)); atomic_write_bytes(targets["reserve"], frame_bytes(reserve_frame)); atomic_write_bytes(targets["combined"], frame_bytes(combined)); atomic_write_bytes(targets["allocation_audit"], allocation_buffer.getvalue().encode()); atomic_write_bytes(targets["stratum_block_audit"], block_buffer.getvalue().encode())
    save_sampling_map(targets["sampling_map"], review_manifest, combined, project)
    audit = {
        "status": "field2_blind_sampling_audit_passed", "main_count": len(main_frame), "reserve_count": len(reserve_frame), "combined_count": len(combined), "main_reserve_intersection": 0,
        "unique_sample_ids": int(combined.sample_id.nunique()), "duplicate_ground_cell_count": int(len(combined) - combined.geographic_ground_cell_id.nunique()), "valid_support_violations": support_violations, "invalid_context_or_border_violations": context_violations,
        "combined_nearest_neighbor_separation_m": separation,
        "spatial_block_coverage": {"main": int(main_frame.spatial_group_id.nunique()), "reserve": int(reserve_frame.spatial_group_id.nunique()), "combined": int(combined.spatial_group_id.nunique())},
        "rank_stratum_counts": {name: {stratum: int((frame.scalar_index_rank_stratum == stratum).sum()) for stratum in sampling["scalar_index_rank_strata"]} for name, frame in (("main", main_frame), ("reserve", reserve_frame))},
        "first_stage_inclusion_probability_range": [float(combined.first_stage_inclusion_probability.min()), float(combined.first_stage_inclusion_probability.max())],
        "final_inclusion_probability_range": [float(combined.final_inclusion_probability.min()), float(combined.final_inclusion_probability.max())],
        "overall_inclusion_probability_range": [float(combined.overall_inclusion_probability.min()), float(combined.overall_inclusion_probability.max())], "design_weight_range": [float(combined.design_weight.min()), float(combined.design_weight.max())],
        "deterministic_reproduction": "pass_identical_main_and_reserve_bytes", "scalar_index_interpretation": "within-cube empirical rank only; not an authoritative biological index", "prohibited_sources_used": False, "biological_filtering_used": False,
    }
    atomic_write_yaml(targets["sampling_audit"], audit)
    if compare_snapshots(source_before, source_snapshot(guard, project)): raise RuntimeError("Field 2 source inventory changed during sampling")
    if any(sha256(project / row.mask_path) != row.mask_sha256 for row in support_manifest.itertuples(index=False)): raise RuntimeError("A frozen valid-support mask changed during sampling")
    output_refs = {name: {"path": str(path.relative_to(project)), "sha256": sha256(path)} for name, path in targets.items() if name != "contract"}
    contract = {
        "version": "field2_blind_sampling_frame_contract_v1", "status": "field2_blind_sampling_frame_frozen", "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(), "materialization_git_commit": git_output(project, "rev-parse", "HEAD"), "seed": int(sampling["seed"]), "allocation_status": sampling["allocation_status"],
        "main_points_per_cube_by_role": sampling["main_points_per_cube_by_role"], "reserve_points_per_cube_by_role": sampling["reserve_points_per_cube_by_role"], "main_count": len(main_frame), "reserve_count": len(reserve_frame), "combined_count": len(combined),
        "sampling_rules": {"spatial_group_size_m": float(sampling["spatial_group_size_m"]), "minimum_separation_m": float(sampling["minimum_separation_m"]), "geographic_ground_cell_size_m": float(sampling["geographic_ground_cell_size_m"]), "annotation_context_radius_preview_pixels": int(sampling["annotation_context_radius_preview_pixels"]), "rank_strata": sampling["scalar_index_rank_strata"], "rank_interpretation": "within-cube stored-scalar-index empirical ranks; not biological classes", "tie_breaking": "SHA-256 ranks from seed, ground cell, cube, row, column; block round-robin", "morphological_or_biological_filtering": False},
        "inclusion_probability_definitions": {"first_stage": "1 / (valid-context pixels in selected within-cube ground cell × overlapping cube views)", "final": "selected quota / eligible population within cube × rank stratum × 5 m block", "overall": "first_stage × final", "design_weight": "1 / overall inclusion probability", "target_population": "prediction-free spatially thinned valid-context support"},
        "input_contracts": {"cube_role_contract": {"path": str(role_contract_path.relative_to(project)), "sha256": sha256(role_contract_path)}, "valid_support_manifest": {"path": str(manifest_path.relative_to(project)), "sha256": sha256(manifest_path)}, "review_package_manifest": {"path": str(review_manifest_path.relative_to(project)), "sha256": sha256(review_manifest_path)}, "field2_source_manifest_sha256": config["source_manifest_sha256"]},
        "frozen_outputs": output_refs,
        "reserve_release_policy": {"status": "locked_not_released_for_annotation", "authorization_required": "separate immutable reserve-release contract", "release_gates_after_800_main_reviews": sampling["reserve_release_gates_after_800_main_reviews"], "may_not_depend_on": sampling["reserve_release_prohibited_evidence"], "prespecified_support_gates_documented_in_configuration": True},
        "audit": audit,
        "provenance": {"prediction_free": True, "supervised_checkpoint_loaded": False, "predictions_or_probabilities_used": False, "pseudo_labels_used": False, "embeddings_ssl_features_or_clusters_used": False, "existing_biological_masks_used": False, "investigator_point_labels_used": False, "biological_classes_inferred_automatically": False},
    }
    atomic_write_yaml(targets["contract"], contract)
    print(f"FROZEN: main={len(main_frame)}, reserve={len(reserve_frame)}, combined={len(combined)}", flush=True); print(f"Contract: {targets['contract']}", flush=True)


if __name__ == "__main__": main()
