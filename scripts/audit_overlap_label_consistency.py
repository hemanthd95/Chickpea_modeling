#!/usr/bin/env python
"""Diagnose label disagreement among native-GSD repeated Field 1 observations."""

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

from audit_ground_cell_overlap import eligible_centres, sha256, CLASS_NAMES
from chickpea_ssl.data import authoritative_class_map, load_records
from chickpea_ssl.spatial import map_block_indices, spatial_group_id


CLASS_ORDER = ["soil", "chickpea", "weed"]


def class_interior_at_centres(labels, x, y, transform, radius):
    inverse = ~transform
    columns = np.floor(inverse.a * x + inverse.b * y + inverse.c).astype(np.int64)
    rows = np.floor(inverse.d * x + inverse.e * y + inverse.f).astype(np.int64)
    values = labels[rows, columns]
    interior = np.ones(len(rows), dtype=bool)
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            interior &= (
                labels[rows + row_offset, columns + column_offset] == values
            )
    return interior


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
    audit_config = config["overlap_label_consistency"]
    if float(audit_config["native_gsd_multiplier"]) != 1.0:
        raise ValueError("Label-consistency audit must use the predeclared 1× native GSD grid")
    radius = int(audit_config["interior_neighbourhood_radius_pixels"])
    if radius != 2:
        raise ValueError("Expected predeclared two-pixel interior radius")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reports = local / "reports" / "overlap_sensitivity"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = local / "authoritative_manifest.csv"
    fold_path = contracts / "field1_spatial_folds.csv"
    fold_contract_path = contracts / "field1_spatial_fold_contract.yaml"
    protocol_path = contracts / protocol_config["contract_filename"]
    overlap_contract_path = contracts / "field1_ground_cell_overlap_audit_contract.yaml"
    for required in (manifest, fold_path, fold_contract_path, protocol_path, overlap_contract_path):
        if not required.exists():
            raise FileNotFoundError(required)

    overlap_contract = yaml.safe_load(overlap_contract_path.read_text())
    if overlap_contract.get("status") != "ground_cell_overlap_sensitivity_audited":
        raise ValueError("Ground-cell overlap audit is not complete")
    if overlap_contract.get("field2_accessed") is not False:
        raise ValueError("Field 2 lock was violated")
    native_gsd = float(overlap_contract["median_native_equivalent_gsd_m"])
    anchor = overlap_contract["grid_anchor_m"]
    anchor_x, anchor_y = float(anchor["x"]), float(anchor["y"])

    assignments = pd.read_csv(fold_path)
    fold_lookup = {(int(r.block_x), int(r.block_y)): int(r.fold)
                   for r in assignments.itertuples()}
    folds = sorted(assignments.fold.unique().astype(int))
    fold_contract = yaml.safe_load(fold_contract_path.read_text())
    buffer_m = float(fold_contract["boundary_exclusion_m"])
    protocol = yaml.safe_load(protocol_path.read_text())
    patch_radius = int(protocol["training"]["patch_size_pixels"]) // 2
    grouping = spatial["grouping"]
    block_size = float(grouping["block_size_m"])
    origin_x = float(grouping.get("origin_x_m", 0.0))
    origin_y = float(grouping.get("origin_y_m", 0.0))

    records = sorted(
        [r for r in load_records(args.paths, manifest)
         if any((r.soil_mask, r.chickpea_mask, r.weed_mask))],
        key=lambda r: r.cube_id,
    )
    cached, reference_crs = [], None
    for cube_order, record in enumerate(records):
        x, y, values, selected, transform, crs = eligible_centres(
            record, patch_radius, fold_lookup, folds, spatial, buffer_m
        )
        if reference_crs is None:
            reference_crs = crs
        elif crs != reference_crs:
            raise ValueError(f"CRS mismatch for {record.cube_id}")
        labels = authoritative_class_map(record)
        interior = class_interior_at_centres(labels, x, y, transform, radius)
        cached.append((cube_order, record.cube_id, x, y, values, interior, selected))
        print(f"Prepared {record.cube_id}", flush=True)

    pair_rows, edge_rows, group_rows, fold_rows = [], [], [], []
    total_observations = 0
    for outer in folds:
        xs, ys, labels, interiors, cube_orders = [], [], [], [], []
        for cube_order, cube_id, x, y, values, interior, selected in cached:
            keep = selected[outer]
            count = int(keep.sum())
            if count == 0:
                continue
            xs.append(x[keep]); ys.append(y[keep]); labels.append(values[keep])
            interiors.append(interior[keep])
            cube_orders.append(np.full(count, cube_order, dtype=np.int16))
        x = np.concatenate(xs); y = np.concatenate(ys)
        label = np.concatenate(labels); interior = np.concatenate(interiors)
        cube_order = np.concatenate(cube_orders)
        total_observations += len(x)

        cell_x = np.floor((x - anchor_x) / native_gsd).astype(np.int64)
        cell_y = np.floor((y - anchor_y) / native_gsd).astype(np.int64)
        order = np.lexsort((cube_order, cell_y, cell_x))
        sx, sy = cell_x[order], cell_y[order]
        sl, si, sc = label[order], interior[order], cube_order[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1])
        first_positions = np.flatnonzero(first)
        group_index = np.cumsum(first) - 1
        duplicate = ~first
        reference_label = sl[first_positions][group_index[duplicate]]
        reference_interior = si[first_positions][group_index[duplicate]]
        reference_cube = sc[first_positions][group_index[duplicate]]
        repeated_label = sl[duplicate]
        repeated_interior = si[duplicate]
        repeated_cube = sc[duplicate]
        disagreement = reference_label != repeated_label
        cross_cube = reference_cube != repeated_cube

        for relation_name, relation_mask in (
            ("all_repeated_observations", np.ones(len(disagreement), dtype=bool)),
            ("cross_cube_only", cross_cube),
            ("within_cube_only", ~cross_cube),
        ):
            count = int(relation_mask.sum())
            disagree_count = int((disagreement & relation_mask).sum())
            fold_rows.append({
                "outer_fold": outer,
                "source_relation": relation_name,
                "repeated_observations": count,
                "label_disagreements": disagree_count,
                "disagreement_fraction": disagree_count / count if count else 0.0,
            })

        for reference_class in CLASS_NAMES:
            for repeated_class in CLASS_NAMES:
                pair_mask = cross_cube & (reference_label == reference_class) & (repeated_label == repeated_class)
                pair_rows.append({
                    "outer_fold": outer,
                    "reference_class_id": reference_class,
                    "reference_class_name": CLASS_NAMES[reference_class],
                    "repeated_class_id": repeated_class,
                    "repeated_class_name": CLASS_NAMES[repeated_class],
                    "cross_cube_observations": int(pair_mask.sum()),
                })

        edge_category = np.select(
            [reference_interior & repeated_interior,
             reference_interior ^ repeated_interior],
            ["both_5x5_interior", "one_5x5_interior"],
            default="neither_5x5_interior",
        )
        for category in ("both_5x5_interior", "one_5x5_interior", "neither_5x5_interior"):
            mask = cross_cube & (edge_category == category)
            count = int(mask.sum())
            disagree_count = int((disagreement & mask).sum())
            edge_rows.append({
                "outer_fold": outer,
                "edge_status": category,
                "cross_cube_observations": count,
                "label_disagreements": disagree_count,
                "disagreement_fraction": disagree_count / count if count else 0.0,
            })

        duplicate_order = order[duplicate]
        dx, dy = x[duplicate_order], y[duplicate_order]
        bx, by = map_block_indices(dx, dy, block_size, origin_x, origin_y)
        frame = pd.DataFrame({
            "block_x": bx[cross_cube], "block_y": by[cross_cube],
            "disagreement": disagreement[cross_cube].astype(np.int8),
        })
        grouped = frame.groupby(["block_x", "block_y"]).disagreement.agg(["size", "sum"]).reset_index()
        for row in grouped.itertuples():
            group_rows.append({
                "outer_fold": outer,
                "spatial_group_id": spatial_group_id(reference_crs, int(row.block_x), int(row.block_y)),
                "block_x": int(row.block_x), "block_y": int(row.block_y),
                "cross_cube_repeated_observations": int(row.size),
                "label_disagreements": int(row.sum),
                "disagreement_fraction": float(row.sum / row.size),
            })
        print(f"Audited outer fold {outer}: {int(duplicate.sum()):,} repeated observations", flush=True)

    if total_observations != int(overlap_contract["audited_outer_test_observations"]):
        raise ValueError("Outer-test observation total changed")

    pairs = pd.DataFrame(pair_rows)
    edges = pd.DataFrame(edge_rows)
    groups = pd.DataFrame(group_rows)
    fold_summary = pd.DataFrame(fold_rows)
    pair_path = reports / "overlap_label_pair_counts.csv"
    edge_path = reports / "overlap_disagreement_by_edge_status.csv"
    group_path = reports / "overlap_disagreement_by_spatial_group.csv"
    fold_path_out = reports / "overlap_disagreement_by_fold.csv"
    pairs.to_csv(pair_path, index=False); edges.to_csv(edge_path, index=False)
    groups.to_csv(group_path, index=False); fold_summary.to_csv(fold_path_out, index=False)

    aggregate_pairs = pairs.groupby(["reference_class_name", "repeated_class_name"]).cross_cube_observations.sum().unstack(fill_value=0)
    aggregate_pairs = aggregate_pairs.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
    normalized_pairs = aggregate_pairs.div(aggregate_pairs.sum(axis=1), axis=0).fillna(0)
    aggregate_edges = edges.groupby("edge_status")[["cross_cube_observations", "label_disagreements"]].sum()
    aggregate_edges["disagreement_fraction"] = aggregate_edges.label_disagreements / aggregate_edges.cross_cube_observations

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    image = axes[0].imshow(normalized_pairs.values, vmin=0, vmax=1, cmap="Blues")
    axes[0].set_xticks(range(3), [v.title() for v in CLASS_ORDER])
    axes[0].set_yticks(range(3), [v.title() for v in CLASS_ORDER])
    axes[0].set_xlabel("Repeated-cube label"); axes[0].set_ylabel("Reference-cube label")
    axes[0].set_title("Cross-cube label agreement")
    for row in range(3):
        for column in range(3):
            axes[0].text(column, row, f"{normalized_pairs.iloc[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    order_names = ["both_5x5_interior", "one_5x5_interior", "neither_5x5_interior"]
    edge_values = aggregate_edges.reindex(order_names).disagreement_fraction
    axes[1].bar(["Both interior", "One interior", "Neither interior"], edge_values, color=["#4C78A8", "#F2CF5B", "#E45756"])
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Cross-cube disagreement fraction")
    axes[1].set_title("Boundary-versus-interior diagnosis")
    axes[1].tick_params(axis="x", rotation=15)

    weighted = groups.groupby(["block_x", "block_y"])[["cross_cube_repeated_observations", "label_disagreements"]].sum().reset_index()
    weighted["rate"] = weighted.label_disagreements / weighted.cross_cube_repeated_observations
    scatter = axes[2].scatter(weighted.block_x, weighted.block_y, c=weighted.rate, s=np.sqrt(weighted.cross_cube_repeated_observations), cmap="magma", vmin=0, vmax=1)
    axes[2].set_aspect("equal"); axes[2].set_xlabel("5 m block x"); axes[2].set_ylabel("5 m block y")
    axes[2].set_title("Spatial concentration of disagreement")
    figure.colorbar(scatter, ax=axes[2], fraction=0.046)
    figure.suptitle("Field 1 native-GSD repeated-view label consistency\n5×5 neighborhoods distinguish class interiors from boundaries", fontsize=15)
    preview = reports / "overlap_label_consistency_overview.png"
    figure.savefig(preview, dpi=200); plt.close(figure)

    contract = {
        "status": "overlap_label_consistency_audited",
        "field": "Field 1",
        "field2_accessed": False,
        "model_inference_performed": False,
        "synthetic_scientific_observations": False,
        "ground_cell_size_m": native_gsd,
        "interior_neighbourhood": f"{2 * radius + 1}x{2 * radius + 1} authoritative labels",
        "reference_selection": "first observation in lexicographic cube_id order",
        "output_hashes": {
            "label_pairs": sha256(pair_path), "edge_status": sha256(edge_path),
            "spatial_groups": sha256(group_path), "fold_summary": sha256(fold_path_out),
            "visual_qc": sha256(preview),
        },
        "source_hashes": {
            "manifest": sha256(manifest), "spatial_folds": sha256(fold_path),
            "overlap_audit": sha256(overlap_contract_path), "configuration": sha256(args.config),
        },
        "notes": [
            "The audit separates cross-cube repeats from any within-cube grid collisions.",
            "Interior status is defined before results using a centered 5x5 authoritative-label neighborhood.",
            "No label is changed and no model performance is computed.",
        ],
    }
    contract_path = contracts / "field1_overlap_label_consistency_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print("\nCross-cube repeated-view summary:")
    print(fold_summary[fold_summary.source_relation == "cross_cube_only"].to_string(index=False))
    print("\nBoundary/interior summary:")
    print(aggregate_edges.to_string())
    print(f"Contract: {contract_path}")
    print(f"Visual QC: {preview}")
    print("No labels were changed; no model inference or Field 2 access occurred.")


if __name__ == "__main__":
    main()
