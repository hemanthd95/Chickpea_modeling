#!/usr/bin/env python
"""Evaluate frozen supervised models under native-GSD overlap sensitivity rules."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import yaml

from chickpea_ssl.data import EnviCube, authoritative_class_map, load_band_indices, load_records
from chickpea_ssl.spatial import map_block_indices, neighbour_boundary_safe_mask, spatial_group_id
from run_supervised_outer_evaluation import (
    CLASS_NAMES, DISPLAY_NAMES, confusion_update, fully_observed_centers,
    gather_patches, load_models, matrix_from_rows, metrics_from_confusion, sha256,
)


POPULATIONS = ["first_view", "unanimous_label"]
POPULATION_NAMES = {
    "first_view": "One deterministic view",
    "unanimous_label": "Unanimous repeated labels",
}


def prepare_fold(args, outer, fold_lookup, folds, block_size, origin_x, origin_y,
                 buffer_m, radius, native_gsd, anchor_x, anchor_y):
    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    manifest = project / "metadata" / "local" / "authoritative_manifest.csv"
    records = sorted(
        [r for r in load_records(args.paths, manifest)
         if any((r.soil_mask, r.chickpea_mask, r.weed_mask))],
        key=lambda r: r.cube_id,
    )
    metadata, cell_x_parts, cell_y_parts, label_parts, cube_parts = [], [], [], [], []
    offset = 0
    for cube_order, record in enumerate(records):
        labels = authoritative_class_map(record)
        cube = EnviCube(record)
        observed = np.any(cube.array > 0, axis=2)
        rows, columns = fully_observed_centers(observed, radius)
        keep = labels[rows, columns] >= 0
        rows, columns = rows[keep], columns[keep]
        truth = labels[rows, columns].astype(np.int8)
        with rasterio.open(record.data) as dataset:
            transform, crs = dataset.transform, dataset.crs.to_string()
        x = transform.a * (columns + .5) + transform.b * (rows + .5) + transform.c
        y = transform.d * (columns + .5) + transform.e * (rows + .5) + transform.f
        bx, by = map_block_indices(x, y, block_size, origin_x, origin_y)
        sample_folds = np.fromiter(
            (fold_lookup.get((int(one_x), int(one_y)), 0) for one_x, one_y in zip(bx, by)),
            dtype=np.int8, count=len(x),
        )
        safe = neighbour_boundary_safe_mask(
            x, y, bx, by, fold_lookup, folds - {outer}, block_size,
            origin_x, origin_y, buffer_m,
        )
        selected = (sample_folds == outer) & safe
        if not selected.any():
            continue
        rows, columns, truth = rows[selected], columns[selected], truth[selected]
        x, y, bx, by = x[selected], y[selected], bx[selected], by[selected]
        count = len(rows)
        metadata.append({
            "record": record, "cube_id": record.cube_id, "rows": rows,
            "columns": columns, "truth": truth, "block_x": bx, "block_y": by,
            "crs": crs, "start": offset, "stop": offset + count,
        })
        offset += count
        cell_x_parts.append(np.floor((x - anchor_x) / native_gsd).astype(np.int64))
        cell_y_parts.append(np.floor((y - anchor_y) / native_gsd).astype(np.int64))
        label_parts.append(truth)
        cube_parts.append(np.full(count, cube_order, dtype=np.int16))

    cell_x, cell_y = np.concatenate(cell_x_parts), np.concatenate(cell_y_parts)
    labels, cube_order = np.concatenate(label_parts), np.concatenate(cube_parts)
    order = np.lexsort((cube_order, cell_y, cell_x))
    sx, sy, sl = cell_x[order], cell_y[order], labels[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (sx[1:] != sx[:-1]) | (sy[1:] != sy[:-1])
    first_positions = np.flatnonzero(first)
    group_index = np.cumsum(first) - 1
    first_labels = sl[first_positions]
    conflicts = np.zeros(len(first_positions), dtype=bool)
    np.logical_or.at(conflicts, group_index, sl != first_labels[group_index])
    first_indices = order[first]
    first_view = np.zeros(len(labels), dtype=bool); first_view[first_indices] = True
    unanimous = np.zeros(len(labels), dtype=bool)
    unanimous[first_indices[~conflicts]] = True
    return metadata, {"first_view": first_view, "unanimous_label": unanimous}, int(conflicts.sum())


def worker(args):
    outer, gpu = int(args.worker_fold), int(args.gpu)
    paths = yaml.safe_load(args.paths.read_text()); config = yaml.safe_load(args.config.read_text())
    protocol = yaml.safe_load(args.protocol.read_text()); spatial = yaml.safe_load(args.spatial.read_text())
    project = Path(paths["project_root"]); local = project / "metadata" / "local"; contracts = local / "contracts"
    root = local / "reports" / "models" / "supervised_overlap_sensitivity" / "folds" / f"outer_fold{outer}"
    root.mkdir(parents=True, exist_ok=True)
    required_names = [
        "field1_nested_supervised_checkpoints_contract.yaml",
        "field1_nested_normalization_contract.yaml",
        "field1_spatial_fold_contract.yaml",
        "field1_ground_cell_overlap_audit_contract.yaml",
        "field1_overlap_label_consistency_contract.yaml",
        "field1_supervised_outer_evaluation_contract.yaml",
    ]
    required = [contracts / value for value in required_names]
    normalization_path = contracts / "field1_nested_normalization.csv"
    fold_path = contracts / "field1_spatial_folds.csv"
    for path in required + [normalization_path, fold_path]:
        if not path.exists(): raise FileNotFoundError(path)
    source_contracts = [yaml.safe_load(path.read_text()) for path in required]
    if any(value.get("field2_accessed") is not False for value in source_contracts):
        raise ValueError("A source contract violates the Field 2 lock")
    checkpoint_contract, normalization_contract, fold_contract, overlap_contract, label_contract, _ = source_contracts
    if label_contract.get("status") != "overlap_label_consistency_audited":
        raise ValueError("Overlap label consistency audit is incomplete")
    sensitivity = config["overlap_model_sensitivity"]
    if list(sensitivity["populations"]) != POPULATIONS:
        raise ValueError("Unexpected overlap sensitivity populations")
    native_gsd = float(overlap_contract["median_native_equivalent_gsd_m"])
    anchor_x = float(overlap_contract["grid_anchor_m"]["x"]); anchor_y = float(overlap_contract["grid_anchor_m"]["y"])

    if not torch.cuda.is_available() or gpu >= torch.cuda.device_count(): raise RuntimeError(f"CUDA GPU {gpu} unavailable")
    torch.cuda.set_device(gpu); device = torch.device(f"cuda:{gpu}")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    architectures = list(protocol["architectures"]); seeds = [int(v) for v in protocol["model_seeds"]]
    band_indices = load_band_indices(args.bands, normalization_contract["band_section"])
    models = load_models(project, checkpoint_contract, outer, architectures, seeds, len(band_indices), device)
    stats = pd.read_csv(normalization_path)
    stats = stats[stats.outer_fold == outer].set_index("band_index").loc[band_indices]
    mean = stats["mean"].to_numpy(np.float32); std = np.maximum(stats["standard_deviation"].to_numpy(np.float32), 1e-6)
    assignments = pd.read_csv(fold_path)
    fold_lookup = {(int(r.block_x), int(r.block_y)): int(r.fold) for r in assignments.itertuples()}
    folds = set(assignments.fold.astype(int).unique()); grouping = spatial["grouping"]
    block_size = float(grouping["block_size_m"]); origin_x = float(grouping.get("origin_x_m", 0)); origin_y = float(grouping.get("origin_y_m", 0))
    radius = int(protocol["training"]["patch_size_pixels"]) // 2
    metadata, populations, conflicting_cells = prepare_fold(
        args, outer, fold_lookup, folds, block_size, origin_x, origin_y,
        float(fold_contract["boundary_exclusion_m"]), radius, native_gsd, anchor_x, anchor_y,
    )
    expected_first = int(pd.read_csv(local / "reports" / "overlap_sensitivity" / "ground_cell_overlap_summary.csv")
                         .query("outer_fold == @outer and native_gsd_multiplier == 1.0").unique_ground_cells.iloc[0])
    if int(populations["first_view"].sum()) != expected_first: raise ValueError("First-view support mismatch")

    fold_conf = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    group_conf = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    cube_conf = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    batch_size = int(yaml.safe_load((contracts / "field1_outer_inference_pilot_contract.yaml").read_text())["selected_batch_size"])
    amp = bool(config["evaluation"]["mixed_precision"]); timing_rows = []
    for number, item in enumerate(metadata, 1):
        started = time.monotonic(); global_slice = slice(item["start"], item["stop"])
        union = populations["first_view"][global_slice]
        if not union.any(): continue
        rows, columns, truth = item["rows"][union], item["columns"][union], item["truth"][union]
        bx, by = item["block_x"][union], item["block_y"][union]
        consensus = populations["unanimous_label"][global_slice][union]
        group_ids = np.asarray([spatial_group_id(item["crs"], int(x), int(y)) for x, y in zip(bx, by)], dtype=object)
        cube = EnviCube(item["record"], band_indices)
        raw = np.asarray(cube.array[..., band_indices], dtype=np.float32)
        normalized = (raw - mean[None, None, :]) / std[None, None, :]
        cube_tensor = torch.from_numpy(np.moveaxis(normalized, -1, 0).copy()).to(device)
        row_tensor = torch.from_numpy(rows).to(device); col_tensor = torch.from_numpy(columns).to(device)
        predicted = {a: np.empty(len(rows), dtype=np.int8) for a in architectures}
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                stop = min(start + batch_size, len(rows)); patches = gather_patches(cube_tensor, row_tensor[start:stop], col_tensor[start:stop], radius)
                with torch.amp.autocast(device_type="cuda", enabled=amp):
                    for architecture in architectures:
                        probability = torch.stack([m(patches).softmax(1) for m in models[architecture]]).mean(0)
                        predicted[architecture][start:stop] = probability.argmax(1).cpu().numpy()
        for population, pop_keep in (("first_view", np.ones(len(rows), bool)), ("unanimous_label", consensus)):
            for architecture in architectures:
                pred = predicted[architecture][pop_keep]; one_truth = truth[pop_keep]; one_groups = group_ids[pop_keep]
                confusion_update(fold_conf[(population, architecture)], one_truth, pred)
                confusion_update(cube_conf[(population, architecture, item["cube_id"])], one_truth, pred)
                order = np.argsort(one_groups); sorted_groups = one_groups[order]
                starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1]; stops = np.r_[starts[1:], len(order)]
                for a, b in zip(starts, stops):
                    confusion_update(group_conf[(population, architecture, str(sorted_groups[a]))], one_truth[order[a:b]], pred[order[a:b]])
        elapsed = time.monotonic() - started
        timing_rows.append({"outer_fold": outer, "gpu": gpu, "cube_id": item["cube_id"], "first_view_predictions": len(rows), "elapsed_seconds": elapsed})
        print(f"[outer {outer} cuda:{gpu}] cube {number}/{len(metadata)} {item['cube_id']}: {len(rows):,} cells in {elapsed:.1f}s", flush=True)
        del cube_tensor, row_tensor, col_tensor, raw, normalized; torch.cuda.empty_cache()

    rows_out = []
    for (population, architecture), matrix in fold_conf.items():
        for i in range(3):
            for j in range(3): rows_out.append({"outer_fold": outer, "population": population, "architecture": architecture, "level": "fold", "unit_id": str(outer), "true_class_id": i, "predicted_class_id": j, "count": int(matrix[i,j])})
    for source, level in ((cube_conf, "cube"), (group_conf, "spatial_group")):
        for (population, architecture, unit), matrix in source.items():
            for i in range(3):
                for j in range(3): rows_out.append({"outer_fold": outer, "population": population, "architecture": architecture, "level": level, "unit_id": unit, "true_class_id": i, "predicted_class_id": j, "count": int(matrix[i,j])})
    counts_path = root / "confusion_counts.csv"; pd.DataFrame(rows_out).to_csv(counts_path, index=False)
    pd.DataFrame(timing_rows).to_csv(root / "inference_timing.csv", index=False)
    contract = {"status": "overlap_sensitivity_fold_evaluated", "outer_fold": outer, "field2_accessed": False,
                "first_view_cells": int(populations["first_view"].sum()), "unanimous_label_cells": int(populations["unanimous_label"].sum()),
                "conflicting_repeated_cells_excluded": conflicting_cells, "confusion_counts_sha256": sha256(counts_path)}
    (root / "fold_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False))


def aggregate(args):
    paths = yaml.safe_load(args.paths.read_text()); protocol = yaml.safe_load(args.protocol.read_text())
    project = Path(paths["project_root"]); local = project / "metadata" / "local"; contracts = local / "contracts"
    root = local / "reports" / "models" / "supervised_overlap_sensitivity"; folds = [int(v) for v in protocol["outer_folds"]]
    frames, support_rows = [], []
    for outer in folds:
        folder = root / "folds" / f"outer_fold{outer}"; contract = yaml.safe_load((folder / "fold_contract.yaml").read_text()); path = folder / "confusion_counts.csv"
        if contract.get("status") != "overlap_sensitivity_fold_evaluated" or sha256(path) != contract["confusion_counts_sha256"]: raise ValueError(f"Outer fold {outer} incomplete")
        frames.append(pd.read_csv(path)); support_rows.append(contract)
    counts = pd.concat(frames, ignore_index=True); counts_path = root / "overlap_sensitivity_confusion_counts.csv"; counts.to_csv(counts_path, index=False)
    metric_rows, fold_rows = [], []
    for population in POPULATIONS:
        for architecture in protocol["architectures"]:
            selected = counts[(counts.population == population) & (counts.architecture == architecture) & (counts.level == "spatial_group")]
            matrices = [matrix_from_rows(group) / matrix_from_rows(group).sum() for _, group in selected.groupby(["outer_fold", "unit_id"])]
            metric_rows.append({"population": population, "population_name": POPULATION_NAMES[population], "architecture": architecture, "display_name": DISPLAY_NAMES[architecture], "spatial_groups": len(matrices), **metrics_from_confusion(np.sum(matrices, axis=0))})
            selected_folds = counts[(counts.population == population) & (counts.architecture == architecture) & (counts.level == "fold")]
            for outer in folds: fold_rows.append({"population": population, "architecture": architecture, "outer_fold": outer, **metrics_from_confusion(matrix_from_rows(selected_folds[selected_folds.outer_fold == outer]))})
    metrics = pd.DataFrame(metric_rows); fold_metrics = pd.DataFrame(fold_rows); support = pd.DataFrame(support_rows)
    metrics_path = root / "overlap_sensitivity_metrics.csv"; fold_metrics_path = root / "overlap_sensitivity_fold_metrics.csv"; support_path = root / "overlap_sensitivity_support.csv"
    metrics.to_csv(metrics_path, index=False); fold_metrics.to_csv(fold_metrics_path, index=False); support.to_csv(support_path, index=False)
    primary = str(protocol["primary_architecture"]); architectures = list(protocol["architectures"])
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    x=np.arange(len(architectures)); width=.35
    for offset,population in enumerate(POPULATIONS):
        frame=metrics[metrics.population==population].set_index("architecture").loc[architectures]
        axes[0].bar(x+(offset-.5)*width,frame.macro_f1,width,label=POPULATION_NAMES[population])
    axes[0].set_xticks(x,[DISPLAY_NAMES[a] for a in architectures],rotation=15,ha="right"); axes[0].set_ylim(0,1); axes[0].set_ylabel("Equal-group macro-F1"); axes[0].set_title("Architecture robustness"); axes[0].legend()
    class_x=np.arange(3)
    for offset,population in enumerate(POPULATIONS):
        row=metrics[(metrics.population==population)&(metrics.architecture==primary)].iloc[0]
        axes[1].bar(class_x+(offset-.5)*width,[row.soil_f1,row.chickpea_f1,row.weed_f1],width,label=POPULATION_NAMES[population])
    axes[1].set_xticks(class_x,[v.title() for v in CLASS_NAMES]); axes[1].set_ylim(0,1); axes[1].set_ylabel("Equal-group F1"); axes[1].set_title("Center + context by class"); axes[1].legend()
    for population in POPULATIONS:
        frame=fold_metrics[(fold_metrics.population==population)&(fold_metrics.architecture==primary)]
        axes[2].plot(frame.outer_fold,frame.macro_f1,marker="o",label=POPULATION_NAMES[population])
    axes[2].set_xticks(folds); axes[2].set_ylim(0,1); axes[2].set_xlabel("Outer fold"); axes[2].set_ylabel("Macro-F1"); axes[2].set_title("Geographic consistency"); axes[2].legend()
    figure.suptitle("Field 1 supervised overlap sensitivity\nFrozen models; one native-GSD cell per ground location",fontsize=15)
    preview=root/"supervised_overlap_sensitivity_overview.png"; figure.savefig(preview,dpi=200); plt.close(figure)
    outputs=[counts_path,metrics_path,fold_metrics_path,support_path,preview]
    contract={"status":"field1_supervised_overlap_sensitivity_complete","field":"Field 1","field2_accessed":False,
              "models_retrained":False,"architectures":architectures,"populations":POPULATIONS,
              "ground_cell_size_m":float(yaml.safe_load((contracts/"field1_ground_cell_overlap_audit_contract.yaml").read_text())["median_native_equivalent_gsd_m"]),
              "conflicting_labels_used_for_model_selection":False,"output_hashes":{p.name:sha256(p) for p in outputs},
              "source_hashes":{"configuration":sha256(args.config),"protocol_configuration":sha256(args.protocol),
                               "overlap_audit":sha256(contracts/"field1_ground_cell_overlap_audit_contract.yaml"),
                               "label_consistency_audit":sha256(contracts/"field1_overlap_label_consistency_contract.yaml"),
                               "primary_outer_evaluation":sha256(contracts/"field1_supervised_outer_evaluation_contract.yaml")}}
    contract_path=contracts/"field1_supervised_overlap_sensitivity_contract.yaml"; contract_path.write_text(yaml.safe_dump(contract,sort_keys=False))
    print("\nEqual-spatial-group overlap sensitivity:")
    print(metrics[["population_name","display_name","balanced_accuracy","macro_f1","soil_f1","chickpea_f1","weed_f1"]].to_string(index=False))
    print(f"Reports: {root}\nVisual QC: {preview}\nContract: {contract_path}")
    print("Frozen-model sensitivity complete; no retraining or Field 2 access occurred.")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--paths",required=True,type=Path); parser.add_argument("--config",default=Path("configs/outer_inference.yaml"),type=Path); parser.add_argument("--protocol",default=Path("configs/supervised_primary_evaluation.yaml"),type=Path); parser.add_argument("--bands",default=Path("configs/spectral_bands.yaml"),type=Path); parser.add_argument("--spatial",default=Path("configs/spatial_splits.yaml"),type=Path); parser.add_argument("--worker-fold",type=int); parser.add_argument("--gpu",type=int); args=parser.parse_args()
    if args.worker_fold is not None:
        if args.gpu is None: raise ValueError("--gpu required in worker mode")
        worker(args); return
    paths=yaml.safe_load(args.paths.read_text()); config=yaml.safe_load(args.config.read_text()); protocol=yaml.safe_load(args.protocol.read_text()); project=Path(paths["project_root"])
    root=project/"metadata"/"local"/"reports"/"models"/"supervised_overlap_sensitivity"/"folds"; gpus=[int(v) for v in config["evaluation"]["gpu_indices"]]; folds=[int(v) for v in protocol["outer_folds"]]
    def run(fold,gpu):
        folder=root/f"outer_fold{fold}"; contract_path=folder/"fold_contract.yaml"; counts=folder/"confusion_counts.csv"
        if contract_path.exists() and counts.exists():
            contract=yaml.safe_load(contract_path.read_text())
            if contract.get("status")=="overlap_sensitivity_fold_evaluated" and sha256(counts)==contract["confusion_counts_sha256"]: return fold,gpu,0,"already complete"
        command=[sys.executable,str(project/"scripts"/"run_supervised_overlap_sensitivity.py"),"--paths",str(args.paths.resolve()),"--config",str(args.config.resolve()),"--protocol",str(args.protocol.resolve()),"--bands",str(args.bands.resolve()),"--spatial",str(args.spatial.resolve()),"--worker-fold",str(fold),"--gpu",str(gpu)]
        started=time.monotonic(); completed=subprocess.run(command,cwd=project,check=False)
        if completed.returncode: raise RuntimeError(f"Outer fold {fold} failed")
        return fold,gpu,time.monotonic()-started,"evaluated"
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures=[executor.submit(run,fold,gpus[i%len(gpus)]) for i,fold in enumerate(folds)]
        for future in as_completed(futures):
            fold,gpu,elapsed,status=future.result(); print(f"[scheduler] outer fold {fold} cuda:{gpu}: {status}; elapsed={elapsed/60:.1f} min",flush=True)
    aggregate(args)


if __name__=="__main__": main()
