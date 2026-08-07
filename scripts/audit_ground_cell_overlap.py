#!/usr/bin/env python
"""Audit repeated ground-cell support before overlap-sensitivity inference."""

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
from chickpea_ssl.spatial import map_block_indices, neighbour_boundary_safe_mask


CLASS_NAMES = {0: "soil", 1: "chickpea", 2: "weed"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_centres(record, radius, fold_lookup, folds, spatial, buffer_m):
    labels = authoritative_class_map(record)
    cube = EnviCube(record)
    observed = np.any(cube.array > 0, axis=2)
    eligible = labels >= 0
    eligible[:radius] = False
    eligible[-radius:] = False
    eligible[:, :radius] = False
    eligible[:, -radius:] = False
    rows, columns = np.nonzero(eligible)

    invalid = (~observed).astype(np.int32)
    integral = np.pad(invalid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r0, r1 = rows - radius, rows + radius + 1
    c0, c1 = columns - radius, columns + radius + 1
    invalid_count = (
        integral[r1, c1] - integral[r0, c1]
        - integral[r1, c0] + integral[r0, c0]
    )
    rows, columns = rows[invalid_count == 0], columns[invalid_count == 0]
    values = labels[rows, columns]

    with rasterio.open(record.data) as dataset:
        transform = dataset.transform
        crs = dataset.crs.to_string()
    x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
    y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f
    grouping = spatial["grouping"]
    block_size = float(grouping["block_size_m"])
    origin_x = float(grouping.get("origin_x_m", 0.0))
    origin_y = float(grouping.get("origin_y_m", 0.0))
    block_x, block_y = map_block_indices(x, y, block_size, origin_x, origin_y)
    sample_folds = np.fromiter(
        (fold_lookup.get((int(bx), int(by)), 0) for bx, by in zip(block_x, block_y)),
        dtype=np.int8,
        count=len(x),
    )
    if (sample_folds == 0).any():
        raise ValueError(f"Labeled center outside frozen groups: {record.cube_id}")
    selected_by_outer = {}
    for outer in folds:
        safe = neighbour_boundary_safe_mask(
            x, y, block_x, block_y, fold_lookup, set(folds) - {outer},
            block_size, origin_x, origin_y, buffer_m,
        )
        selected_by_outer[outer] = (sample_folds == outer) & safe
    return x, y, values, selected_by_outer, transform, crs


def deduplicate(x, y, labels, cube_order, cell_size, anchor_x, anchor_y):
    cell_x = np.floor((x - anchor_x) / cell_size).astype(np.int64)
    cell_y = np.floor((y - anchor_y) / cell_size).astype(np.int64)
    order = np.lexsort((cube_order, cell_y, cell_x))
    sx, sy, sl = cell_x[order], cell_y[order], labels[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1])
    first_positions = np.flatnonzero(first)
    first_labels = sl[first_positions]
    group_index = np.cumsum(first) - 1
    disagreements = sl != first_labels[group_index]
    return order[first], disagreements, first_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/outer_inference.yaml"), type=Path)
    parser.add_argument("--protocol", default=Path("configs/supervised_primary_evaluation.yaml"), type=Path)
    parser.add_argument("--spatial", default=Path("configs/spatial_splits.yaml"), type=Path)
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    protocol_config = yaml.safe_load(args.protocol.read_text())
    spatial = yaml.safe_load(args.spatial.read_text())
    sensitivity = config["overlap_sensitivity"]
    multipliers = [float(value) for value in sensitivity["native_gsd_multipliers"]]
    if multipliers != [1.0, 2.0]:
        raise ValueError("Expected predeclared native-GSD multipliers [1.0, 2.0]")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reports = local / "reports" / "overlap_sensitivity"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = local / "authoritative_manifest.csv"
    fold_path = contracts / "field1_spatial_folds.csv"
    fold_contract_path = contracts / "field1_spatial_fold_contract.yaml"
    protocol_path = contracts / protocol_config["contract_filename"]
    support_contract_path = contracts / "field1_outer_test_support_contract.yaml"
    evaluation_contract_path = contracts / "field1_supervised_outer_evaluation_contract.yaml"
    for required in (manifest, fold_path, fold_contract_path, protocol_path,
                     support_contract_path, evaluation_contract_path):
        if not required.exists():
            raise FileNotFoundError(required)

    fold_contract = yaml.safe_load(fold_contract_path.read_text())
    if sha256(fold_path) != fold_contract["assignment_sha256"]:
        raise ValueError("Frozen spatial-fold hash mismatch")
    support_contract = yaml.safe_load(support_contract_path.read_text())
    evaluation_contract = yaml.safe_load(evaluation_contract_path.read_text())
    expected_total = int(support_contract["total_eligible_outer_test_observations"])
    if expected_total != 12_782_603:
        raise ValueError("Unexpected exhaustive support total")
    if evaluation_contract.get("field2_accessed") is not False:
        raise ValueError("Field 2 lock was violated")

    assignments = pd.read_csv(fold_path)
    fold_lookup = {(int(r.block_x), int(r.block_y)): int(r.fold)
                   for r in assignments.itertuples()}
    folds = sorted(assignments.fold.unique().astype(int))
    buffer_m = float(fold_contract["boundary_exclusion_m"])
    protocol = yaml.safe_load(protocol_path.read_text())
    patch_size = int(protocol["training"]["patch_size_pixels"])
    radius = patch_size // 2

    records = sorted(
        [r for r in load_records(args.paths, manifest)
         if any((r.soil_mask, r.chickpea_mask, r.weed_mask))],
        key=lambda r: r.cube_id,
    )
    resolution_rows = []
    cached = []
    reference_crs = None
    for cube_order, record in enumerate(records):
        x, y, labels, selected, transform, crs = eligible_centres(
            record, radius, fold_lookup, folds, spatial, buffer_m
        )
        if reference_crs is None:
            reference_crs = crs
        elif crs != reference_crs:
            raise ValueError(f"CRS mismatch for {record.cube_id}: {crs}")
        x_step = float(np.hypot(transform.a, transform.d))
        y_step = float(np.hypot(transform.b, transform.e))
        area = float(abs(transform.a * transform.e - transform.b * transform.d))
        resolution_rows.append({
            "cube_id": record.cube_id,
            "crs": crs,
            "x_step_m": x_step,
            "y_step_m": y_step,
            "pixel_area_m2": area,
            "equivalent_gsd_m": np.sqrt(area),
        })
        cached.append((cube_order, record.cube_id, x, y, labels, selected))
        print(f"Indexed {record.cube_id}: {sum(int(v.sum()) for v in selected.values()):,} observations", flush=True)

    resolution = pd.DataFrame(resolution_rows)
    native_gsd = float(resolution.equivalent_gsd_m.median())
    grid_sizes = [native_gsd * value for value in multipliers]
    anchor_x = float(sensitivity.get("anchor_x_m", 0.0))
    anchor_y = float(sensitivity.get("anchor_y_m", 0.0))
    summary_rows, class_rows, cube_rows = [], [], []
    audited_total = 0

    for outer in folds:
        x_parts, y_parts, label_parts, order_parts, cube_parts = [], [], [], [], []
        for cube_order, cube_id, x, y, labels, selected_by_outer in cached:
            keep = selected_by_outer[outer]
            count = int(keep.sum())
            if count == 0:
                continue
            x_parts.append(x[keep]); y_parts.append(y[keep]); label_parts.append(labels[keep])
            order_parts.append(np.full(count, cube_order, dtype=np.int16))
            cube_parts.append(np.full(count, cube_order, dtype=np.int16))
        x_all, y_all = np.concatenate(x_parts), np.concatenate(y_parts)
        labels_all = np.concatenate(label_parts)
        cube_all = np.concatenate(cube_parts)
        cube_order_all = np.concatenate(order_parts)
        audited_total += len(x_all)

        for multiplier, cell_size in zip(multipliers, grid_sizes):
            retained, disagreements, retained_labels = deduplicate(
                x_all, y_all, labels_all, cube_order_all, cell_size, anchor_x, anchor_y
            )
            duplicate_count = len(x_all) - len(retained)
            disagreement_count = int(disagreements.sum())
            summary_rows.append({
                "outer_fold": outer,
                "native_gsd_multiplier": multiplier,
                "ground_cell_size_m": cell_size,
                "observed_outer_test_centers": len(x_all),
                "unique_ground_cells": len(retained),
                "duplicate_observations_removed": duplicate_count,
                "retained_fraction": len(retained) / len(x_all),
                "label_disagreement_observations": disagreement_count,
                "label_disagreement_fraction_all": disagreement_count / len(x_all),
                "label_disagreement_fraction_duplicates": (
                    disagreement_count / duplicate_count if duplicate_count else 0.0
                ),
            })
            for class_id, class_name in CLASS_NAMES.items():
                class_rows.append({
                    "outer_fold": outer,
                    "native_gsd_multiplier": multiplier,
                    "ground_cell_size_m": cell_size,
                    "class_id": class_id,
                    "class_name": class_name,
                    "unique_ground_cells": int((retained_labels == class_id).sum()),
                })
            retained_cubes = cube_all[retained]
            for cube_order, cube_id, *_ in cached:
                cube_rows.append({
                    "outer_fold": outer,
                    "native_gsd_multiplier": multiplier,
                    "ground_cell_size_m": cell_size,
                    "cube_id": cube_id,
                    "selected_ground_cells": int((retained_cubes == cube_order).sum()),
                })
        print(f"Audited outer fold {outer}: {len(x_all):,} observations", flush=True)

    if audited_total != expected_total:
        raise ValueError(f"Support mismatch: audited {audited_total:,}; expected {expected_total:,}")
    summary = pd.DataFrame(summary_rows)
    class_support = pd.DataFrame(class_rows)
    cube_support = pd.DataFrame(cube_rows)
    resolution_path = reports / "native_pixel_resolution.csv"
    summary_path = reports / "ground_cell_overlap_summary.csv"
    class_path = reports / "ground_cell_class_support.csv"
    cube_path = reports / "ground_cell_selection_by_cube.csv"
    resolution.to_csv(resolution_path, index=False)
    summary.to_csv(summary_path, index=False)
    class_support.to_csv(class_path, index=False)
    cube_support.to_csv(cube_path, index=False)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    axes[0].bar(resolution.cube_id, resolution.equivalent_gsd_m * 100)
    axes[0].axhline(native_gsd * 100, color="black", linestyle="--", label="Median")
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].set_ylabel("Equivalent native GSD (cm)")
    axes[0].set_title("Native pixel resolution")
    axes[0].legend()
    for multiplier, frame in summary.groupby("native_gsd_multiplier"):
        label = f"{multiplier:g}× native GSD"
        axes[1].plot(frame.outer_fold, frame.retained_fraction, marker="o", label=label)
        axes[2].plot(frame.outer_fold, frame.label_disagreement_fraction_duplicates,
                     marker="o", label=label)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Outer fold"); axes[1].set_ylabel("Unique cells / observations")
    axes[1].set_title("Overlap de-duplication retention"); axes[1].legend()
    axes[2].set_xlabel("Outer fold"); axes[2].set_ylabel("Disagreement / removed duplicates")
    axes[2].set_title("Cross-cube label disagreement"); axes[2].legend()
    figure.suptitle(
        "Field 1 observed overlap-sensitivity audit\n"
        "Deterministic lexicographic cube selection; no model inference",
        fontsize=15,
    )
    preview = reports / "ground_cell_overlap_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)

    contract = {
        "status": "ground_cell_overlap_sensitivity_audited",
        "field": "Field 1",
        "field2_accessed": False,
        "model_inference_performed": False,
        "synthetic_scientific_observations": False,
        "selection_rule": "first observation in lexicographic cube_id order",
        "grid_anchor_m": {"x": anchor_x, "y": anchor_y},
        "median_native_equivalent_gsd_m": native_gsd,
        "predeclared_native_gsd_multipliers": multipliers,
        "ground_cell_sizes_m": grid_sizes,
        "audited_outer_test_observations": audited_total,
        "output_hashes": {
            "native_resolution": sha256(resolution_path),
            "summary": sha256(summary_path),
            "class_support": sha256(class_path),
            "cube_support": sha256(cube_path),
            "visual_qc": sha256(preview),
        },
        "source_hashes": {
            "manifest": sha256(manifest),
            "spatial_folds": sha256(fold_path),
            "support_contract": sha256(support_contract_path),
            "outer_evaluation_contract": sha256(evaluation_contract_path),
            "configuration": sha256(args.config),
        },
        "notes": [
            "Both grid scales were declared before this audit and neither is selected by performance.",
            "The first cube rule is label-independent and deterministic.",
            "This audit quantifies support and label consistency before overlap-sensitivity inference.",
        ],
    }
    contract_path = contracts / "field1_ground_cell_overlap_audit_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print("\nGround-cell overlap audit:")
    print(summary.to_string(index=False))
    print(f"Native median equivalent GSD: {native_gsd:.6f} m")
    print(f"Contract: {contract_path}")
    print(f"Visual QC: {preview}")
    print("No model inference or Field 2 access occurred.")


if __name__ == "__main__":
    main()
