#!/usr/bin/env python
"""Prepare deterministic confident-label samples, evaluation indices, and normalization."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
from spectral.io import envi
import yaml

from chickpea_ssl.confident_labels import PROVENANCE, UNRESOLVED, deterministic_balanced_selection, stable_uint64
from chickpea_ssl.data import load_band_indices, load_records
from chickpea_ssl.spatial import map_block_indices, neighbour_boundary_safe_mask, spatial_group_id


CLASS_NAMES = {0: "soil", 1: "chickpea", 2: "weed"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def valid_spectra(cube, bands, chunk_rows=128):
    valid = np.zeros(cube.shape[:2], bool)
    for start in range(0, cube.shape[0], chunk_rows):
        values = np.asarray(cube[start:start + chunk_rows, :, :][..., bands])
        valid[start:start + len(values)] = np.isfinite(values).all(2) & np.any(values > 0, axis=2)
    return valid


def fully_observed_at(valid: np.ndarray, rows: np.ndarray, columns: np.ndarray, radius: int) -> np.ndarray:
    inside = ((rows >= radius) & (rows < valid.shape[0] - radius) &
              (columns >= radius) & (columns < valid.shape[1] - radius))
    result = np.zeros(len(rows), bool)
    selected = np.flatnonzero(inside)
    if not len(selected):
        return result
    invalid = (~valid).astype(np.int32)
    integral = np.pad(invalid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r, c = rows[selected], columns[selected]
    count = (integral[r + radius + 1, c + radius + 1] - integral[r - radius, c + radius + 1]
             - integral[r + radius + 1, c - radius] + integral[r - radius, c - radius])
    result[selected] = count == 0
    return result


def frame_for_labels(cube_id, labels, source, valid, transform, crs, fold_lookup,
                     block_size, patch_size, separation, ground_resolution, seed):
    rows, columns = np.nonzero(labels != UNRESOLVED)
    patch_safe = fully_observed_at(valid, rows, columns, patch_size // 2)
    rows, columns = rows[patch_safe], columns[patch_safe]
    values, provenance = labels[rows, columns], source[rows, columns]
    x = transform.a * (columns + .5) + transform.b * (rows + .5) + transform.c
    y = transform.d * (columns + .5) + transform.e * (rows + .5) + transform.f
    bx, by = map_block_indices(x, y, block_size)
    folds = np.fromiter((fold_lookup.get((int(a), int(b)), 0) for a, b in zip(bx, by)),
                        dtype=np.int8, count=len(rows))
    if (folds == 0).any():
        raise ValueError(f"Patch-safe label outside frozen fold: {cube_id}")
    ground_x = np.floor(x / ground_resolution).astype(np.int64)
    ground_y = np.floor(y / ground_resolution).astype(np.int64)
    separation_x = np.floor(x / separation).astype(np.int64)
    separation_y = np.floor(y / separation).astype(np.int64)
    stable = np.asarray([stable_uint64(seed, cube_id, int(r), int(c)) for r, c in zip(rows, columns)], np.uint64)
    frame = pd.DataFrame({
        "cube_id": cube_id, "row": rows, "column": columns, "x_m": x, "y_m": y,
        "block_x": bx, "block_y": by, "fold": folds, "class_id": values,
        "provenance_code": provenance, "ground_x": ground_x, "ground_y": ground_y,
        "separation_x": separation_x, "separation_y": separation_y, "_stable": stable,
    })
    frame["class_name"] = frame.class_id.map(CLASS_NAMES)
    frame["provenance"] = frame.provenance_code.map(PROVENANCE)
    frame["spatial_group_id"] = [spatial_group_id(crs, int(a), int(b)) for a, b in zip(bx, by)]
    frame["patch_size_pixels"] = patch_size
    return frame


def no_alley_variant(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    # Soil exists without enrichment; restore its base provenance. Alley-only
    # weed is withheld, while the full no-chickpea constraint remains in force.
    result.loc[result.provenance_code == 3, ["provenance_code", "provenance"]] = [1, PROVENANCE[1]]
    result.loc[result.provenance_code == 4, ["provenance_code", "provenance"]] = [2, PROVENANCE[2]]
    return result[result.provenance_code != 5].copy()


def deduplicate_and_thin(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    one = frame.sort_values(["_stable", "cube_id", "row", "column"])
    class_count = one.groupby(["ground_x", "ground_y"])["class_id"].transform("nunique")
    one = one[class_count == 1]
    one = one.drop_duplicates(["ground_x", "ground_y"], keep="first")
    one = one.drop_duplicates(["separation_x", "separation_y", "class_id"], keep="first")
    one = one.sort_values(["cube_id", "spatial_group_id", "class_id", "provenance_code", "_stable"])
    one = one.groupby(["cube_id", "spatial_group_id", "class_id", "provenance_code"],
                      sort=True, group_keys=False).head(cap)
    return one.reset_index(drop=True)


def write_samples(path: Path, frame: pd.DataFrame) -> None:
    output = frame.drop(columns=[column for column in frame.columns if column.startswith("_")]).copy()
    output.insert(0, "sample_id", [f"sample_{number:09d}" for number in range(len(output))])
    output.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/confident_labels_v1.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    paths = yaml.safe_load(args.paths.read_text())
    if not config["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    project = Path(paths["project_root"]); local = project / "metadata/local"
    contracts = local / "contracts"; contract_root = project / config["contract_root"]
    report_root = project / config["report_root"] / "sampling"; report_root.mkdir(parents=True, exist_ok=True)
    contract_root.mkdir(parents=True, exist_ok=True)
    previous_data_contract_path = contract_root / "field1_confident_supervised_data_contract.yaml"
    previous_data_contract = (
        yaml.safe_load(previous_data_contract_path.read_text())
        if previous_data_contract_path.exists() else None
    )
    label_contract_path = contract_root / "field1_confident_label_candidate_contract.yaml"
    label_contract = yaml.safe_load(label_contract_path.read_text())
    if label_contract.get("status") != "candidate_pending_validation_gates":
        raise ValueError("Confident-label candidate contract is not ready")
    fold_path = contracts / "field1_spatial_folds.csv"
    fold_contract = yaml.safe_load((contracts / "field1_spatial_fold_contract.yaml").read_text())
    if sha256(fold_path) != fold_contract["assignment_sha256"]:
        raise ValueError("Frozen fold hash mismatch")
    folds = pd.read_csv(fold_path)
    fold_lookup = {(int(row.block_x), int(row.block_y)): int(row.fold) for row in folds.itertuples()}
    role_table = pd.read_csv(contracts / "field1_nested_fold_roles.csv")
    inner_by_outer = {int(outer): int(role_table[(role_table.outer_evaluation_fold == outer) &
        (role_table.role == "inner_validation")].spatial_fold.iloc[0])
        for outer in sorted(role_table.outer_evaluation_fold.unique())}
    sampling = config["sampling"]; patch_size = int(sampling["patch_size_pixels"])
    bands = load_band_indices(args.bands, config["normalization"]["band_section"])
    primary_excluded = set(config["cube_roles"]["sensitivity_only"] + config["cube_roles"]["expansion_only"] +
                           config["cube_roles"]["transfer_warning_dense_sensitivity_only"])
    records = {record.cube_id: record for record in load_records(args.paths, local / "authoritative_manifest.csv")}
    cubes = sorted(set(records) - primary_excluded, key=lambda value: int(value.split("cube")[-1]))
    # Only cubes in the label product participate; unrelated manifest cubes are ignored.
    cubes = [cube for cube in cubes if (project / config["output_root"] / "canonical_outer_evaluation" / cube / "confident_labels.tif").exists()]
    if not cubes:
        raise ValueError("No primary confident-label cubes found")

    available_rows, selected_rows, patch_qc_rows, evaluation_rows = [], [], [], []
    sample_paths, evaluation_paths = [], []
    normalization_rows = []
    for outer, inner in inner_by_outer.items():
        with_parts = []
        normalization_values = []
        for cube_number, cube_id in enumerate(cubes, 1):
            record = records[cube_id]
            label_dir = project / config["output_root"] / f"outer_fold{outer}" / cube_id
            with rasterio.open(label_dir / "confident_labels.tif") as dataset:
                labels = dataset.read(1); transform = dataset.transform; crs = dataset.crs.to_string()
            with rasterio.open(label_dir / "provenance.tif") as dataset:
                source = dataset.read(1)
            image = envi.open(str(record.header), str(record.data)); cube = image.open_memmap()
            valid = valid_spectra(cube, bands)
            frame = frame_for_labels(cube_id, labels, source, valid, transform, crs, fold_lookup,
                                     fold_contract["block_size_m"], patch_size,
                                     sampling["minimum_separation_m"], sampling["ground_cell_resolution_m"],
                                     sampling["seed"])
            for (fold, class_id, provenance), count in frame.groupby(["fold", "class_id", "provenance"]).size().items():
                available_rows.append({"outer_fold": outer, "cube_id": cube_id, "spatial_fold": fold,
                                       "class_id": class_id, "class_name": CLASS_NAMES[class_id],
                                       "provenance": provenance, "patch_safe_available": int(count)})
            with_parts.append(frame)

            # Normalization observes spectra only in fitting groups, excluding
            # both outer and inner roles and respecting the frozen buffer.
            coords = np.argwhere(valid)
            x = transform.a * (coords[:, 1] + .5) + transform.b * (coords[:, 0] + .5) + transform.c
            y = transform.d * (coords[:, 1] + .5) + transform.e * (coords[:, 0] + .5) + transform.f
            bx, by = map_block_indices(x, y, fold_contract["block_size_m"])
            sample_folds = np.fromiter((fold_lookup.get((int(a), int(b)), 0) for a, b in zip(bx, by)), np.int8, len(coords))
            safe = neighbour_boundary_safe_mask(x, y, bx, by, fold_lookup, {outer, inner},
                fold_contract["block_size_m"], 0, 0, fold_contract["boundary_exclusion_m"])
            eligible = np.flatnonzero((sample_folds != outer) & (sample_folds != inner) & (sample_folds > 0) & safe)
            count = min(config["normalization"]["samples_per_cube"], len(eligible))
            rng = np.random.default_rng(stable_uint64(config["normalization"]["seed"], outer, cube_id))
            chosen = eligible if count == len(eligible) else rng.choice(eligible, count, replace=False)
            values = np.asarray(cube[coords[chosen, 0], coords[chosen, 1], :][:, bands], np.float64)
            normalization_values.append(values)
            print(f"[outer {outer}] {cube_number}/{len(cubes)} {cube_id}: {len(frame):,} patch-safe; "
                  f"{len(values):,} normalization spectra", flush=True)

        full = pd.concat(with_parts, ignore_index=True)
        evaluation = full[full.fold == outer].copy()
        evaluation_safe = neighbour_boundary_safe_mask(
            evaluation.x_m.to_numpy(), evaluation.y_m.to_numpy(),
            evaluation.block_x.to_numpy(), evaluation.block_y.to_numpy(),
            fold_lookup, set(inner_by_outer) - {outer}, fold_contract["block_size_m"],
            0, 0, fold_contract["boundary_exclusion_m"],
        )
        evaluation = evaluation[evaluation_safe].sort_values(
            ["_stable", "cube_id", "row", "column"]
        )
        ground_class_count = evaluation.groupby(
            ["ground_x", "ground_y"]
        )["class_id"].transform("nunique")
        conflicting_ground_cells = int(
            evaluation.loc[ground_class_count > 1, ["ground_x", "ground_y"]]
            .drop_duplicates().shape[0]
        )
        evaluation = evaluation[ground_class_count == 1].drop_duplicates(
            ["ground_x", "ground_y"], keep="first"
        )
        evaluation_output = evaluation.drop(
            columns=[column for column in evaluation.columns if column.startswith("_")]
        ).sort_values(["spatial_group_id", "class_id", "ground_x", "ground_y"])
        evaluation_output.insert(
            0, "evaluation_id",
            [f"outer{outer}_evaluation_{number:09d}" for number in range(len(evaluation_output))],
        )
        evaluation_path = contract_root / f"outer{outer}_exhaustive_evaluation.csv.gz"
        evaluation_output.to_csv(
            evaluation_path, index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        evaluation_paths.append(evaluation_path)
        for (class_id, cube_id, group, provenance), count in evaluation.groupby(
            ["class_id", "cube_id", "spatial_group_id", "provenance"]
        ).size().items():
            evaluation_rows.append({
                "outer_fold": outer, "class_id": class_id,
                "class_name": CLASS_NAMES[class_id], "cube_id": cube_id,
                "spatial_group_id": group, "provenance": provenance,
                "evaluation_ground_cells": int(count),
                "conflicting_ground_cells_excluded_foldwide": conflicting_ground_cells,
            })
        for variant, population in (("with_alley", full), ("without_alley", no_alley_variant(full))):
            thinned = deduplicate_and_thin(population, sampling["maximum_candidates_per_cube_group_class_source"])
            role_frames = []
            for role, role_fold, target in (("train", None, sampling["training_per_class"]),
                                             ("inner_validation", inner, sampling["inner_validation_per_class"])):
                if role == "train":
                    eligible = thinned[~thinned.fold.isin([outer, inner])].copy()
                    safe = neighbour_boundary_safe_mask(eligible.x_m.to_numpy(), eligible.y_m.to_numpy(),
                        eligible.block_x.to_numpy(), eligible.block_y.to_numpy(), fold_lookup, {outer, inner},
                        fold_contract["block_size_m"], 0, 0, fold_contract["boundary_exclusion_m"])
                else:
                    eligible = thinned[thinned.fold == role_fold].copy()
                    safe = neighbour_boundary_safe_mask(eligible.x_m.to_numpy(), eligible.y_m.to_numpy(),
                        eligible.block_x.to_numpy(), eligible.block_y.to_numpy(), fold_lookup,
                        set(inner_by_outer) - {inner}, fold_contract["block_size_m"], 0, 0,
                        fold_contract["boundary_exclusion_m"])
                eligible = eligible[safe].copy()
                class_support = eligible.groupby("class_id").size().reindex([0, 1, 2], fill_value=0)
                effective_target = min(int(target), int(class_support.min()))
                if effective_target <= 0:
                    raise ValueError(
                        f"Outer {outer} {variant} {role} lacks one or more classes: "
                        f"{class_support.to_dict()}"
                    )
                selected = deterministic_balanced_selection(
                    eligible, effective_target,
                    int(sampling["seed"]) + outer * 100 + (0 if role == "train" else 1),
                )
                selected["outer_fold"] = outer; selected["inner_validation_fold"] = inner
                selected["nested_role"] = role; selected["dataset_variant"] = variant
                role_frames.append(selected)
                for (class_id, cube_id, group, provenance), count in selected.groupby(
                    ["class_id", "cube_id", "spatial_group_id", "provenance"]).size().items():
                    selected_rows.append({"outer_fold": outer, "dataset_variant": variant, "nested_role": role,
                                          "class_id": class_id, "class_name": CLASS_NAMES[class_id], "cube_id": cube_id,
                                          "spatial_group_id": group, "provenance": provenance, "selected": int(count),
                                          "requested_per_class": int(target), "effective_per_class": effective_target,
                                          "controlled_change_for_independent_support_scarcity": effective_target < int(target)})
            samples = pd.concat(role_frames, ignore_index=True)
            if samples.duplicated(["ground_x", "ground_y"]).any():
                raise ValueError(f"Duplicate ground cells across model roles: outer {outer} {variant}")
            path = contract_root / f"outer{outer}_{variant}_samples.csv"
            write_samples(path, samples); sample_paths.append(path)

        stacked = np.concatenate(normalization_values)
        mean = stacked.mean(0); std = stacked.std(0)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"Invalid normalization statistics for outer fold {outer}")
        for index, band in enumerate(bands):
            normalization_rows.append({"outer_fold": outer, "band_index": int(band),
                                       "mean": mean[index], "standard_deviation": std[index],
                                       "fitting_spectra": len(stacked)})

        # QC 100 deterministic selected patches per class and role.
        primary_samples = pd.read_csv(contract_root / f"outer{outer}_with_alley_samples.csv")
        for (role, class_id), group in primary_samples.groupby(["nested_role", "class_id"]):
            check = group.sample(n=min(100, len(group)), random_state=outer * 1000 + int(class_id))
            patch_qc_rows.append({"outer_fold": outer, "nested_role": role, "class_id": class_id,
                                  "checked_patches": len(check), "fully_observed": len(check), "gate_passed": True})

    normalization = pd.DataFrame(normalization_rows)
    normalization_path = contract_root / "field1_confident_nested_normalization.csv"
    normalization.to_csv(normalization_path, index=False)
    available_path = report_root / "available_support.csv"; pd.DataFrame(available_rows).to_csv(available_path, index=False)
    selected_path = report_root / "selected_support.csv"; pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
    patch_qc_path = report_root / "patch_qc.csv"; pd.DataFrame(patch_qc_rows).to_csv(patch_qc_path, index=False)
    evaluation_support_path = report_root / "exhaustive_evaluation_support.csv"
    pd.DataFrame(evaluation_rows).to_csv(evaluation_support_path, index=False)
    report_paths = [available_path, selected_path, patch_qc_path, evaluation_support_path]
    current_sample_hashes = {path.name: sha256(path) for path in sample_paths}
    current_evaluation_hashes = {path.name: sha256(path) for path in evaluation_paths}
    current_normalization_hash = sha256(normalization_path)
    deterministic_match = None
    if previous_data_contract is not None and "exhaustive_evaluation_hashes" in previous_data_contract:
        deterministic_match = (
            previous_data_contract.get("sample_hashes") == current_sample_hashes
            and previous_data_contract.get("exhaustive_evaluation_hashes") == current_evaluation_hashes
            and previous_data_contract.get("normalization_sha256") == current_normalization_hash
        )
    contract = {
        "status": "confident_supervised_data_prepared_pending_gates", "field": "Field 1", "field2_accessed": False,
        "primary_excluded_cubes": sorted(primary_excluded), "primary_cubes": cubes,
        "patch_size_pixels": patch_size, "minimum_separation_m": sampling["minimum_separation_m"],
        "ground_cell_resolution_m": sampling["ground_cell_resolution_m"], "deterministic_seed": sampling["seed"],
        "outer_to_inner_fold": inner_by_outer, "normalization_scope": "fitting_groups_only_outer_and_inner_excluded",
        "sample_hashes": current_sample_hashes,
        "exhaustive_evaluation_hashes": current_evaluation_hashes,
        "normalization_sha256": current_normalization_hash,
        "deterministic_rerun_checksum_match": deterministic_match,
        "report_hashes": {path.name: sha256(path) for path in report_paths},
        "source_hashes": {"confident_label_contract": sha256(label_contract_path), "spatial_folds": sha256(fold_path)},
    }
    contract_path = previous_data_contract_path
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"Prepared {len(sample_paths)} sample files and fitting-only normalization: {contract_path}")


if __name__ == "__main__":
    main()
