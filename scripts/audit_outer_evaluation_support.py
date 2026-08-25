#!/usr/bin/env python
"""Audit exhaustive Field 1 outer-test support after nested checkpoints are frozen."""

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
    map_block_indices,
    neighbour_boundary_safe_mask,
    spatial_group_id,
)


CLASS_NAMES = {0: "soil", 1: "chickpea", 2: "weed"}
CLASS_ORDER = ["soil", "chickpea", "weed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        default=Path("configs/supervised_primary_evaluation.yaml"),
        type=Path,
    )
    parser.add_argument(
        "--spatial", default=Path("configs/spatial_splits.yaml"), type=Path
    )
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    protocol_config = yaml.safe_load(args.protocol.read_text())
    spatial = yaml.safe_load(args.spatial.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reports = local / "reports" / "outer_evaluation_support"
    reports.mkdir(parents=True, exist_ok=True)

    manifest = local / "authoritative_manifest.csv"
    fold_path = contracts / "field1_spatial_folds.csv"
    fold_contract_path = contracts / "field1_spatial_fold_contract.yaml"
    protocol_path = contracts / protocol_config["contract_filename"]
    checkpoint_contract_path = (
        contracts / "field1_nested_supervised_checkpoints_contract.yaml"
    )
    for required in (
        manifest,
        fold_path,
        fold_contract_path,
        protocol_path,
        checkpoint_contract_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    fold_contract = yaml.safe_load(fold_contract_path.read_text())
    if sha256(fold_path) != fold_contract["assignment_sha256"]:
        raise ValueError("Frozen spatial-fold hash mismatch")
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("status") != "supervised_primary_evaluation_protocol_frozen":
        raise ValueError("Primary evaluation protocol is not frozen")
    if sha256(args.protocol) != protocol["configuration_sha256"]:
        raise ValueError("Primary evaluation configuration hash mismatch")
    checkpoints = yaml.safe_load(checkpoint_contract_path.read_text())
    if checkpoints.get("status") != "nested_supervised_checkpoints_frozen":
        raise ValueError("Nested checkpoints are not frozen")
    if checkpoints.get("outer_test_accessed") is not False:
        raise ValueError("Checkpoint contract indicates prior outer-test access")
    if checkpoints.get("total_checkpoints") != 45:
        raise ValueError("Expected 45 frozen nested checkpoints")
    if checkpoints.get("field2_accessed") is not False:
        raise ValueError("Checkpoint contract violated Field 2 lock")

    assignments = pd.read_csv(fold_path)
    fold_lookup = {
        (int(row.block_x), int(row.block_y)): int(row.fold)
        for row in assignments.itertuples()
    }
    folds = sorted(assignments["fold"].unique().astype(int))
    grouping = spatial["grouping"]
    block_size = float(grouping["block_size_m"])
    origin_x = float(grouping.get("origin_x_m", 0.0))
    origin_y = float(grouping.get("origin_y_m", 0.0))
    buffer_m = float(fold_contract["boundary_exclusion_m"])
    patch_size = int(protocol["training"]["patch_size_pixels"])
    if patch_size % 2 != 1:
        raise ValueError("Patch size must be odd")
    radius = patch_size // 2

    cube_rows = []
    group_rows = []
    validity_rows = []
    labeled_cubes = 0
    for record in load_records(args.paths, manifest):
        if not any((record.soil_mask, record.chickpea_mask, record.weed_mask)):
            continue
        labeled_cubes += 1
        labels = authoritative_class_map(record)
        cube = EnviCube(record)
        observed = np.any(cube.array > 0, axis=2)
        eligible = labels >= 0
        eligible[:radius] = False
        eligible[-radius:] = False
        eligible[:, :radius] = False
        eligible[:, -radius:] = False
        rows, columns = np.nonzero(eligible)
        before_nodata = len(rows)

        invalid = (~observed).astype(np.int32)
        integral = np.pad(
            invalid.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0))
        )
        row_min, row_max = rows - radius, rows + radius + 1
        column_min, column_max = columns - radius, columns + radius + 1
        invalid_count = (
            integral[row_max, column_max]
            - integral[row_min, column_max]
            - integral[row_max, column_min]
            + integral[row_min, column_min]
        )
        fully_observed = invalid_count == 0
        rows, columns = rows[fully_observed], columns[fully_observed]
        values = labels[rows, columns]

        with rasterio.open(record.data) as dataset:
            transform = dataset.transform
            crs = dataset.crs.to_string()
        x = (
            transform.a * (columns + 0.5)
            + transform.b * (rows + 0.5)
            + transform.c
        )
        y = (
            transform.d * (columns + 0.5)
            + transform.e * (rows + 0.5)
            + transform.f
        )
        block_x, block_y = map_block_indices(
            x, y, block_size, origin_x, origin_y
        )
        sample_folds = np.fromiter(
            (
                fold_lookup.get((int(one_x), int(one_y)), 0)
                for one_x, one_y in zip(block_x, block_y)
            ),
            dtype=np.int8,
            count=len(rows),
        )
        if (sample_folds == 0).any():
            raise ValueError(
                f"Labeled outer-test center outside frozen groups: {record.cube_id}"
            )

        retained_total = 0
        for outer in folds:
            safe = neighbour_boundary_safe_mask(
                x,
                y,
                block_x,
                block_y,
                fold_lookup,
                set(folds) - {outer},
                block_size,
                origin_x,
                origin_y,
                buffer_m,
            )
            selected = (sample_folds == outer) & safe
            retained_total += int(selected.sum())
            selected_values = values[selected]
            selected_bx = block_x[selected]
            selected_by = block_y[selected]
            for class_id, class_name in CLASS_NAMES.items():
                count = int((selected_values == class_id).sum())
                cube_rows.append(
                    {
                        "outer_fold": outer,
                        "cube_id": record.cube_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "eligible_outer_test_centers": count,
                    }
                )
            if selected.any():
                frame = pd.DataFrame(
                    {
                        "block_x": selected_bx,
                        "block_y": selected_by,
                        "class_id": selected_values,
                    }
                )
                grouped = frame.groupby(
                    ["block_x", "block_y", "class_id"]
                ).size()
                for (one_x, one_y, class_id), count in grouped.items():
                    group_rows.append(
                        {
                            "outer_fold": outer,
                            "spatial_group_id": spatial_group_id(
                                crs, int(one_x), int(one_y)
                            ),
                            "class_id": int(class_id),
                            "class_name": CLASS_NAMES[int(class_id)],
                            "cube_id": record.cube_id,
                            "eligible_outer_test_centers": int(count),
                        }
                    )
        validity_rows.append(
            {
                "cube_id": record.cube_id,
                "patch_safe_labeled_before_nodata": before_nodata,
                "excluded_for_patch_nodata": int((~fully_observed).sum()),
                "fully_observed_labeled_centers": len(rows),
                "retained_outer_test_centers_after_boundary": retained_total,
            }
        )
        print(
            f"Audited {record.cube_id}: {retained_total:,} exhaustive "
            "outer-test observations across folds",
            flush=True,
        )

    cube_support = pd.DataFrame(cube_rows)
    group_support = pd.DataFrame(group_rows)
    validity = pd.DataFrame(validity_rows)
    cube_support_path = reports / "outer_test_support_by_cube_class.csv"
    group_support_path = reports / "outer_test_support_by_group_cube_class.csv"
    validity_path = reports / "outer_test_patch_validity_by_cube.csv"
    cube_support.to_csv(cube_support_path, index=False)
    group_support.to_csv(group_support_path, index=False)
    validity.to_csv(validity_path, index=False)

    summary = (
        cube_support.groupby(["outer_fold", "class_id", "class_name"])
        ["eligible_outer_test_centers"]
        .agg(["sum", lambda values: int((values > 0).sum())])
        .reset_index()
        .rename(
            columns={
                "sum": "eligible_outer_test_centers",
                "<lambda_0>": "cubes_with_class",
            }
        )
    )
    group_counts = (
        group_support.groupby(["outer_fold", "class_id"])
        ["spatial_group_id"]
        .nunique()
        .rename("spatial_groups_with_class")
        .reset_index()
    )
    summary = summary.merge(group_counts, on=["outer_fold", "class_id"], how="left")
    summary["spatial_groups_with_class"] = (
        summary["spatial_groups_with_class"].fillna(0).astype(int)
    )
    summary_path = reports / "outer_test_support_summary.csv"
    summary.to_csv(summary_path, index=False)

    cube_presence = (
        cube_support.assign(
            present=cube_support["eligible_outer_test_centers"] > 0
        )
        .pivot_table(
            index=["outer_fold", "cube_id"],
            columns="class_name",
            values="present",
            aggfunc="max",
            fill_value=False,
        )
        .reset_index()
    )
    for class_name in CLASS_ORDER:
        if class_name not in cube_presence:
            cube_presence[class_name] = False
    cube_presence["classes_present"] = (
        cube_presence[CLASS_ORDER].sum(axis=1).astype(int)
    )
    cube_presence["contributes_any_outer_test_observation"] = (
        cube_presence["classes_present"] > 0
    )
    cube_presence["valid_three_class_cube_macro_f1"] = (
        cube_presence["classes_present"] == 3
    )
    presence_path = reports / "outer_test_cube_class_presence.csv"
    cube_presence.to_csv(presence_path, index=False)

    figure, axes = plt.subplots(
        1, 2, figsize=(15, 6), constrained_layout=True
    )
    count_pivot = summary.pivot(
        index="outer_fold",
        columns="class_name",
        values="eligible_outer_test_centers",
    )[CLASS_ORDER]
    colors = {"soil": "#B07D42", "chickpea": "#2E8B57", "weed": "#7A4EAB"}
    count_pivot.plot.bar(
        ax=axes[0], color=[colors[value] for value in CLASS_ORDER]
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Outer spatial fold")
    axes[0].set_ylabel("Eligible labeled observations (log scale)")
    axes[0].set_title("Exhaustive outer-test class support")
    axes[0].tick_params(axis="x", rotation=0)

    contributing_presence = cube_presence[
        cube_presence["contributes_any_outer_test_observation"]
    ]
    presence_counts = (
        contributing_presence.groupby(["outer_fold", "classes_present"])
        .size()
        .unstack(fill_value=0)
    )
    for value in (1, 2, 3):
        if value not in presence_counts:
            presence_counts[value] = 0
    presence_counts[[1, 2, 3]].plot.bar(
        stacked=True,
        ax=axes[1],
        color=["#D9D9D9", "#F2CF5B", "#4C78A8"],
    )
    axes[1].set_xlabel("Outer spatial fold")
    axes[1].set_ylabel("Contributing cube × fold records")
    axes[1].set_title("Cube-level class completeness")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(
        ["One class", "Two classes", "All three classes"],
        title="Classes present",
    )
    figure.suptitle(
        "Field 1 exhaustive outer-test support audit\n"
        "Labels opened only after all nested checkpoints were frozen",
        fontsize=14,
    )
    preview = reports / "outer_test_support_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)

    output_hashes = {
        "cube_support": sha256(cube_support_path),
        "group_support": sha256(group_support_path),
        "validity": sha256(validity_path),
        "summary": sha256(summary_path),
        "cube_presence": sha256(presence_path),
        "visual_qc": sha256(preview),
    }
    contract = {
        "status": "outer_test_support_audited",
        "field": "Field 1",
        "field2_accessed": False,
        "outer_test_labels_opened": True,
        "outer_test_predictions_generated": False,
        "synthetic_scientific_observations": False,
        "labeled_cubes_audited": labeled_cubes,
        "patch_size_pixels": patch_size,
        "boundary_exclusion_m": buffer_m,
        "total_eligible_outer_test_observations": int(
            summary["eligible_outer_test_centers"].sum()
        ),
        "total_cube_fold_combinations": int(len(cube_presence)),
        "noncontributing_cube_fold_combinations": int(
            (~cube_presence["contributes_any_outer_test_observation"]).sum()
        ),
        "contributing_cube_fold_records": int(
            cube_presence["contributes_any_outer_test_observation"].sum()
        ),
        "valid_three_class_cube_fold_records": int(
            cube_presence["valid_three_class_cube_macro_f1"].sum()
        ),
        "incomplete_contributing_cube_fold_records": int(
            (
                cube_presence["contributes_any_outer_test_observation"]
                & ~cube_presence["valid_three_class_cube_macro_f1"]
            ).sum()
        ),
        "output_hashes": output_hashes,
        "source_hashes": {
            "manifest": sha256(manifest),
            "spatial_folds": sha256(fold_path),
            "spatial_fold_contract": sha256(fold_contract_path),
            "primary_protocol": sha256(protocol_path),
            "nested_checkpoints": sha256(checkpoint_contract_path),
        },
        "notes": [
            "Outer labels were opened only after all 45 checkpoint hashes were frozen.",
            "No predictions were generated by this audit.",
            "Cube-level three-class macro-F1 is prohibited when any class is absent.",
            "Primary uncertainty remains clustered by frozen spatial group.",
        ],
    }
    contract_path = contracts / "field1_outer_test_support_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print("\nExhaustive outer-test support:")
    print(summary.to_string(index=False))
    print(f"Total eligible observations: {contract['total_eligible_outer_test_observations']:,}")
    print(
        "Contributing cube-fold records: "
        f"{contract['contributing_cube_fold_records']}; all three classes: "
        f"{contract['valid_three_class_cube_fold_records']}; incomplete: "
        f"{contract['incomplete_contributing_cube_fold_records']}; "
        "noncontributing combinations: "
        f"{contract['noncontributing_cube_fold_combinations']}"
    )
    print(f"Contract: {contract_path}")
    print(f"Visual QC: {preview}")
    print("Outer labels audited; no predictions or Field 2 access occurred.")


if __name__ == "__main__":
    main()
