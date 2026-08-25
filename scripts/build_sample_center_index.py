#!/usr/bin/env python
"""Build an observed-only, leakage-aware training-candidate centre index."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml

from chickpea_ssl.data import EnviCube, authoritative_class_map, load_records
from chickpea_ssl.spatial import (
    map_block_indices, neighbour_boundary_safe_mask, spatial_group_id,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    digest = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--sampling", default=Path("configs/sampling.yaml"), type=Path)
    parser.add_argument("--spatial", default=Path("configs/spatial_splits.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    sampling = yaml.safe_load(args.sampling.read_text())
    spatial = yaml.safe_load(args.spatial.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reports = local / "reports" / "sampling"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = local / "authoritative_manifest.csv"
    folds_path = contracts / "field1_spatial_folds.csv"
    fold_contract_path = contracts / "field1_spatial_fold_contract.yaml"
    for path in (manifest, folds_path, fold_contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fold_contract = yaml.safe_load(fold_contract_path.read_text())
    if sha256(folds_path) != fold_contract["assignment_sha256"]:
        raise ValueError("Frozen fold hash mismatch")
    assignments = pd.read_csv(folds_path)
    fold_lookup = {
        (int(row.block_x), int(row.block_y)): int(row.fold)
        for row in assignments.itertuples()
    }
    folds = sorted(assignments["fold"].unique())
    grouping = spatial["grouping"]
    block_size = float(grouping["block_size_m"])
    origin_x = float(grouping.get("origin_x_m", 0.0))
    origin_y = float(grouping.get("origin_y_m", 0.0))
    buffer_m = float(fold_contract["boundary_exclusion_m"])
    patch_size = int(sampling["patch_size_pixels"])
    if patch_size % 2 != 1:
        raise ValueError("Patch size must be odd")
    radius = patch_size // 2
    cap = int(sampling["max_centers_per_cube_group_class"])
    class_names = {int(key): value for key, value in sampling["class_names"].items()}
    frames: list[pd.DataFrame] = []
    validity_rows: list[dict[str, object]] = []

    for record in load_records(args.paths, manifest):
        if not any((record.soil_mask, record.chickpea_mask, record.weed_mask)):
            continue
        labels = authoritative_class_map(record)
        cube = EnviCube(record)
        observed = np.any(cube.array > 0, axis=2)
        eligible = labels >= 0
        eligible[:radius] = False
        eligible[-radius:] = False
        eligible[:, :radius] = False
        eligible[:, -radius:] = False
        rows, columns = np.nonzero(eligible)
        before_nodata_filter = len(rows)
        invalid = (~observed).astype(np.int32)
        integral = np.pad(invalid.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        row_min, row_max = rows - radius, rows + radius + 1
        column_min, column_max = columns - radius, columns + radius + 1
        invalid_count = (
            integral[row_max, column_max] - integral[row_min, column_max]
            - integral[row_max, column_min] + integral[row_min, column_min]
        )
        fully_observed = invalid_count == 0
        rows, columns = rows[fully_observed], columns[fully_observed]
        validity_rows.append({
            "cube_id": record.cube_id,
            "patch_safe_labeled_centers_before_nodata_filter": before_nodata_filter,
            "centers_excluded_for_patch_nodata": int((~fully_observed).sum()),
            "fully_observed_patch_centers": len(rows),
            "excluded_fraction": float((~fully_observed).sum() / max(before_nodata_filter, 1)),
        })
        values = labels[rows, columns]
        with rasterio.open(record.data) as dataset:
            transform = dataset.transform
            crs = dataset.crs.to_string()
            pixel_size = float(np.mean((
                np.hypot(transform.a, transform.d), np.hypot(transform.b, transform.e)
            )))
        x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
        y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f
        block_x, block_y = map_block_indices(x, y, block_size, origin_x, origin_y)
        sample_folds = np.fromiter((
            fold_lookup.get((int(one_x), int(one_y)), 0)
            for one_x, one_y in zip(block_x, block_y)
        ), dtype=np.int8, count=len(rows))
        if (sample_folds == 0).any():
            raise ValueError(f"Labeled centre outside frozen groups: {record.cube_id}")

        order = np.lexsort((values, block_y, block_x))
        ordered_keys = np.column_stack((block_x[order], block_y[order], values[order]))
        changes = np.r_[True, np.any(np.diff(ordered_keys, axis=0) != 0, axis=1), True]
        boundaries = np.flatnonzero(changes)
        selected_parts = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            candidates = order[start:end]
            one_x, one_y, one_class = ordered_keys[start]
            count = min(cap, len(candidates))
            rng = stable_rng(
                int(sampling["seed"]), record.cube_id, int(one_x), int(one_y), int(one_class)
            )
            selected_parts.append(
                candidates if count == len(candidates)
                else candidates[rng.choice(len(candidates), size=count, replace=False)]
            )
        selected = np.concatenate(selected_parts)
        selected_rows, selected_columns = rows[selected], columns[selected]
        selected_x, selected_y = x[selected], y[selected]
        selected_bx, selected_by = block_x[selected], block_y[selected]
        selected_folds, selected_values = sample_folds[selected], values[selected]
        frame = pd.DataFrame({
            "cube_id": record.cube_id, "row": selected_rows,
            "column": selected_columns, "x_m": selected_x, "y_m": selected_y,
            "block_x": selected_bx, "block_y": selected_by,
            "fold": selected_folds, "class_id": selected_values,
            "class_name": [class_names[int(value)] for value in selected_values],
            "patch_size_pixels": patch_size, "pixel_size_m": pixel_size,
        })
        frame["spatial_group_id"] = [
            spatial_group_id(crs, int(one_x), int(one_y))
            for one_x, one_y in zip(selected_bx, selected_by)
        ]
        frame["ssl_vegetation_center"] = frame["class_id"].isin(
            [int(value) for value in sampling["ssl_center_classes"]]
        )
        frame["weed_cluster_center"] = (
            frame["class_id"] == int(sampling["weed_cluster_center_class"])
        )
        for heldout in folds:
            training_safe = neighbour_boundary_safe_mask(
                selected_x, selected_y, selected_bx, selected_by, fold_lookup,
                {int(heldout)}, block_size, origin_x, origin_y, buffer_m,
            )
            validation_safe = neighbour_boundary_safe_mask(
                selected_x, selected_y, selected_bx, selected_by, fold_lookup,
                set(map(int, folds)) - {int(heldout)}, block_size, origin_x, origin_y,
                buffer_m,
            )
            frame[f"train_eligible_holdout_{heldout}"] = (
                (selected_folds != heldout) & training_safe
            )
            frame[f"validation_eligible_fold_{heldout}"] = (
                (selected_folds == heldout) & validation_safe
            )
        frames.append(frame)

    index = pd.concat(frames, ignore_index=True).sort_values(
        ["cube_id", "spatial_group_id", "class_id", "row", "column"]
    )
    index.insert(0, "sample_id", [f"sample_{number:09d}" for number in range(len(index))])
    index_path = contracts / "field1_training_candidate_centers.csv"
    index.to_csv(index_path, index=False)
    summary_rows = []
    for heldout in folds:
        for class_id, class_name in class_names.items():
            selected_class = index["class_id"] == class_id
            summary_rows.append({
                "heldout_fold": int(heldout), "class_id": class_id,
                "class_name": class_name,
                "indexed_centers": int(selected_class.sum()),
                "training_eligible_centers": int((
                    selected_class & index[f"train_eligible_holdout_{heldout}"]
                ).sum()),
                "validation_diagnostic_centers": int((
                    selected_class & index[f"validation_eligible_fold_{heldout}"]
                ).sum()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(reports / "sample_center_summary.csv", index=False)
    pd.DataFrame(validity_rows).to_csv(reports / "patch_nodata_exclusions_by_cube.csv", index=False)

    contract = {
        "status": "frozen_training_candidate_pool",
        "field": "Field 1", "field2_accessed": False,
        "use": sampling["index_use"],
        "primary_evaluation": sampling["primary_evaluation"],
        "seed": int(sampling["seed"]), "patch_size_pixels": patch_size,
        "max_centers_per_cube_group_class": cap,
        "require_fully_observed_patch": True,
        "center_index_sha256": sha256(index_path),
        "fold_assignment_sha256": sha256(folds_path),
        "configuration_sha256": sha256(args.sampling),
        "source_manifest_sha256": sha256(manifest),
        "notes": [
            "Every indexed centre is an observed labeled pixel; no samples were synthesized.",
            "Every indexed patch contains no all-band-zero spatial pixel.",
            "Class/group caps create a training candidate pool and must not define primary test prevalence.",
            "Primary held-out evaluation uses exhaustive tiled inference on authoritative masks.",
        ],
    }
    contract_path = contracts / "field1_training_candidate_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    fold_class = index.groupby(["fold", "class_name"]).size().unstack(fill_value=0)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fold_class.plot.bar(ax=axes[0], color={"soil": "#B07D42", "chickpea": "#2E8B57", "weed": "#7A4EAB"})
    axes[0].set_title("Observed training-candidate centres")
    axes[0].set_xlabel("Spatial fold")
    axes[0].set_ylabel("Indexed centres")
    axes[0].tick_params(axis="x", rotation=0)
    boundary_rows = []
    for heldout in folds:
        possible_train = index["fold"] != heldout
        possible_validation = index["fold"] == heldout
        boundary_rows.append({
            "fold": int(heldout),
            "training_removed_fraction": 1 - index.loc[possible_train, f"train_eligible_holdout_{heldout}"].mean(),
            "validation_removed_fraction": 1 - index.loc[possible_validation, f"validation_eligible_fold_{heldout}"].mean(),
        })
    boundary_frame = pd.DataFrame(boundary_rows).set_index("fold")
    boundary_frame.plot.bar(ax=axes[1], color=["#4477AA", "#EE6677"])
    axes[1].set_title("Candidate centres removed by 0.30 m boundary rule")
    axes[1].set_xlabel("Held-out fold")
    axes[1].set_ylabel("Removed fraction")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(["Training side", "Validation side"])
    figure.suptitle("Field 1 leakage-aware sample-centre QC", fontsize=14)
    preview = reports / "sample_center_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)
    print(f"Observed candidate centres indexed: {len(index):,}")
    print(f"Cubes contributing labels: {index['cube_id'].nunique()}")
    print(summary.to_string(index=False))
    print(f"Contract: {contract_path}")
    print(f"Visual QC: {preview}")
    print("Candidate pool only; primary evaluation remains exhaustive and Field 2 stays locked.")


if __name__ == "__main__":
    main()
