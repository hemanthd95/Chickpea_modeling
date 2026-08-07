#!/usr/bin/env python
"""Run exhaustive, leakage-safe Field 1 supervised outer evaluation.

Predictions are accumulated as confusion counts by frozen 5 m group and cube;
individual pixel probabilities are intentionally not persisted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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
from chickpea_ssl.model import build_supervised_model
from chickpea_ssl.spatial import map_block_indices, neighbour_boundary_safe_mask, spatial_group_id


CLASS_NAMES = ["soil", "chickpea", "weed"]
DISPLAY_NAMES = {
    "center_spectrum_mlp": "Center spectrum",
    "spatial_average_cnn": "Spatial-average CNN",
    "center_context_fusion": "Center + context",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def confusion_update(matrix: np.ndarray, truth: np.ndarray, predicted: np.ndarray) -> None:
    matrix += np.bincount(truth * 3 + predicted, minlength=9).reshape(3, 3)


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    total = matrix.sum()
    result = {
        "accuracy": float(diagonal.sum() / total) if total else np.nan,
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
    for class_id, class_name in enumerate(CLASS_NAMES):
        result[f"{class_name}_precision"] = float(precision[class_id])
        result[f"{class_name}_recall"] = float(recall[class_id])
        result[f"{class_name}_f1"] = float(f1[class_id])
        result[f"{class_name}_support"] = float(support[class_id])
    return result


def gather_patches(cube: torch.Tensor, rows: torch.Tensor, columns: torch.Tensor, radius: int) -> torch.Tensor:
    offsets = torch.arange(-radius, radius + 1, device=cube.device)
    rr = rows[:, None, None] + offsets[None, :, None]
    cc = columns[:, None, None] + offsets[None, None, :]
    return cube[:, rr, cc].permute(1, 0, 2, 3).contiguous()


def fully_observed_centers(observed: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.ones(observed.shape, dtype=bool)
    valid[:radius] = False; valid[-radius:] = False
    valid[:, :radius] = False; valid[:, -radius:] = False
    rows, columns = np.nonzero(valid)
    invalid = (~observed).astype(np.int32)
    integral = np.pad(invalid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    r0, r1 = rows - radius, rows + radius + 1
    c0, c1 = columns - radius, columns + radius + 1
    counts = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    keep = counts == 0
    return rows[keep], columns[keep]


def load_models(
    project: Path, checkpoint_contract: dict, outer_fold: int,
    architectures: list[str], seeds: list[int], bands: int, device: torch.device,
) -> dict[str, list[torch.nn.Module]]:
    hashes = checkpoint_contract["checkpoint_hashes"]
    root = project / "results" / "supervised_nested_training" / f"outer_fold{outer_fold}"
    result = {}
    for architecture in architectures:
        models = []
        for seed in seeds:
            path = root / f"{architecture}_seed{seed}_best.pt"
            key = f"outer{outer_fold}:{architecture}:seed{seed}"
            if not path.exists():
                raise FileNotFoundError(path)
            if sha256(path) != hashes[key]:
                raise ValueError(f"Frozen checkpoint hash mismatch: {key}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("architecture") != architecture or int(checkpoint.get("seed")) != seed:
                raise ValueError(f"Checkpoint metadata mismatch: {key}")
            model = build_supervised_model(architecture, bands, 3).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            models.append(model)
        result[architecture] = models
    return result


def worker(args: argparse.Namespace) -> None:
    outer = int(args.worker_fold)
    gpu = int(args.gpu)
    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    protocol_config = yaml.safe_load(args.protocol.read_text())
    spatial = yaml.safe_load(args.spatial.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    fold_output = local / "reports" / "models" / "supervised_outer_evaluation" / "folds" / f"outer_fold{outer}"
    fold_output.mkdir(parents=True, exist_ok=True)

    checkpoint_contract_path = contracts / "field1_nested_supervised_checkpoints_contract.yaml"
    support_contract_path = contracts / "field1_outer_test_support_contract.yaml"
    pilot_contract_path = contracts / "field1_outer_inference_pilot_contract.yaml"
    normalization_contract_path = contracts / "field1_nested_normalization_contract.yaml"
    normalization_path = contracts / "field1_nested_normalization.csv"
    fold_contract_path = contracts / "field1_spatial_fold_contract.yaml"
    fold_path = contracts / "field1_spatial_folds.csv"
    for required in (checkpoint_contract_path, support_contract_path, pilot_contract_path,
                     normalization_contract_path, normalization_path, fold_contract_path,
                     fold_path, local / "authoritative_manifest.csv"):
        if not required.exists(): raise FileNotFoundError(required)
    checkpoint_contract = yaml.safe_load(checkpoint_contract_path.read_text())
    support_contract = yaml.safe_load(support_contract_path.read_text())
    pilot_contract = yaml.safe_load(pilot_contract_path.read_text())
    normalization_contract = yaml.safe_load(normalization_contract_path.read_text())
    fold_contract = yaml.safe_load(fold_contract_path.read_text())
    if checkpoint_contract.get("status") != "nested_supervised_checkpoints_frozen":
        raise ValueError("Nested checkpoints are not frozen")
    if support_contract.get("status") != "outer_test_support_audited":
        raise ValueError("Outer support is not audited")
    if pilot_contract.get("status") != "outer_inference_engineering_pilot_passed":
        raise ValueError("Inference pilot has not passed")
    if any(value.get("field2_accessed") is not False for value in
           (checkpoint_contract, support_contract, pilot_contract, normalization_contract, fold_contract)):
        raise ValueError("A source contract violates the Field 2 lock")
    if sha256(normalization_path) != normalization_contract["normalization_csv_sha256"]:
        raise ValueError("Nested normalization hash mismatch")
    if sha256(fold_path) != fold_contract["assignment_sha256"]:
        raise ValueError("Spatial fold hash mismatch")

    if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA GPU {gpu} is unavailable")
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    architectures = list(protocol_config["architectures"])
    seeds = [int(value) for value in protocol_config["model_seeds"]]
    band_indices = load_band_indices(args.bands, normalization_contract["band_section"])
    models = load_models(project, checkpoint_contract, outer, architectures, seeds, len(band_indices), device)
    batch_size = int(pilot_contract["selected_batch_size"])
    amp = bool(config["evaluation"]["mixed_precision"])
    radius = int(protocol_config["training"]["patch_size_pixels"]) // 2

    stats = pd.read_csv(normalization_path)
    stats = stats[stats["outer_fold"] == outer].set_index("band_index").loc[band_indices]
    mean = stats["mean"].to_numpy(np.float32)
    std = np.maximum(stats["standard_deviation"].to_numpy(np.float32), 1e-6)
    assignments = pd.read_csv(fold_path)
    fold_lookup = {(int(r.block_x), int(r.block_y)): int(r.fold) for r in assignments.itertuples()}
    folds = set(assignments["fold"].astype(int).unique())
    grouping = spatial["grouping"]
    block_size = float(grouping["block_size_m"])
    origin_x = float(grouping.get("origin_x_m", 0.0)); origin_y = float(grouping.get("origin_y_m", 0.0))
    buffer_m = float(fold_contract["boundary_exclusion_m"])

    fold_conf = {architecture: np.zeros((3, 3), dtype=np.int64) for architecture in architectures}
    cube_conf = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    group_conf = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    timing_rows = []
    records = load_records(args.paths, local / "authoritative_manifest.csv")
    labeled = [record for record in records if any((record.soil_mask, record.chickpea_mask, record.weed_mask))]
    for number, record in enumerate(labeled, start=1):
        started = time.monotonic()
        labels = authoritative_class_map(record)
        cube = EnviCube(record, band_indices)
        observed = np.any(cube.array > 0, axis=2)
        rows, columns = fully_observed_centers(observed, radius)
        keep_labeled = labels[rows, columns] >= 0
        rows, columns = rows[keep_labeled], columns[keep_labeled]
        truth = labels[rows, columns].astype(np.int64)
        with rasterio.open(record.data) as dataset:
            transform = dataset.transform; crs = dataset.crs.to_string()
        x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
        y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f
        block_x, block_y = map_block_indices(x, y, block_size, origin_x, origin_y)
        sample_folds = np.fromiter((fold_lookup.get((int(bx), int(by)), 0) for bx, by in zip(block_x, block_y)),
                                   dtype=np.int8, count=len(rows))
        safe = neighbour_boundary_safe_mask(x, y, block_x, block_y, fold_lookup,
                                            folds - {outer}, block_size, origin_x, origin_y, buffer_m)
        selected = (sample_folds == outer) & safe
        if not selected.any():
            continue
        rows = rows[selected]; columns = columns[selected]; truth = truth[selected]
        block_x = block_x[selected]; block_y = block_y[selected]
        group_ids = np.asarray([spatial_group_id(crs, int(bx), int(by)) for bx, by in zip(block_x, block_y)], dtype=object)

        raw = np.asarray(cube.array[..., band_indices], dtype=np.float32)
        normalized = (raw - mean[None, None, :]) / std[None, None, :]
        cube_tensor = torch.from_numpy(np.moveaxis(normalized, -1, 0).copy()).to(device)
        row_tensor = torch.from_numpy(rows).to(device); column_tensor = torch.from_numpy(columns).to(device)
        predicted_by_architecture = {architecture: np.empty(len(rows), dtype=np.int8) for architecture in architectures}
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                stop = min(start + batch_size, len(rows))
                patches = gather_patches(cube_tensor, row_tensor[start:stop], column_tensor[start:stop], radius)
                with torch.amp.autocast(device_type="cuda", enabled=amp):
                    for architecture in architectures:
                        probability = torch.stack([model(patches).softmax(1) for model in models[architecture]]).mean(0)
                        predicted_by_architecture[architecture][start:stop] = probability.argmax(1).cpu().numpy()
        torch.cuda.synchronize(device)
        for architecture, predicted in predicted_by_architecture.items():
            confusion_update(fold_conf[architecture], truth, predicted)
            confusion_update(cube_conf[(architecture, record.cube_id)], truth, predicted)
            order = np.argsort(group_ids)
            sorted_groups = group_ids[order]; sorted_truth = truth[order]; sorted_pred = predicted[order]
            starts = np.r_[0, np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1]
            stops = np.r_[starts[1:], len(order)]
            for one_start, one_stop in zip(starts, stops):
                confusion_update(group_conf[(architecture, str(sorted_groups[one_start]))],
                                 sorted_truth[one_start:one_stop], sorted_pred[one_start:one_stop])
        elapsed = time.monotonic() - started
        timing_rows.append({"outer_fold": outer, "gpu": gpu, "cube_id": record.cube_id,
                            "eligible_predictions": len(rows), "elapsed_seconds": elapsed,
                            "predictions_per_second_all_architectures": len(rows) / elapsed})
        print(f"[outer {outer} cuda:{gpu}] cube {number}/{len(labeled)} {record.cube_id}: "
              f"{len(rows):,} observations in {elapsed:.1f}s", flush=True)
        del cube_tensor, row_tensor, column_tensor, raw, normalized
        torch.cuda.empty_cache()

    rows_out = []
    for architecture, matrix in fold_conf.items():
        for true_id in range(3):
            for pred_id in range(3):
                rows_out.append({"outer_fold": outer, "architecture": architecture, "level": "fold",
                                 "unit_id": str(outer), "true_class_id": true_id,
                                 "predicted_class_id": pred_id, "count": int(matrix[true_id, pred_id])})
    for (architecture, unit), matrix in cube_conf.items():
        for true_id in range(3):
            for pred_id in range(3):
                rows_out.append({"outer_fold": outer, "architecture": architecture, "level": "cube",
                                 "unit_id": unit, "true_class_id": true_id,
                                 "predicted_class_id": pred_id, "count": int(matrix[true_id, pred_id])})
    for (architecture, unit), matrix in group_conf.items():
        for true_id in range(3):
            for pred_id in range(3):
                rows_out.append({"outer_fold": outer, "architecture": architecture, "level": "spatial_group",
                                 "unit_id": unit, "true_class_id": true_id,
                                 "predicted_class_id": pred_id, "count": int(matrix[true_id, pred_id])})
    counts_path = fold_output / "confusion_counts.csv"
    pd.DataFrame(rows_out).to_csv(counts_path, index=False)
    pd.DataFrame(timing_rows).to_csv(fold_output / "inference_timing.csv", index=False)
    predicted_total = int(next(iter(fold_conf.values())).sum())
    expected = int(pd.read_csv(local / "reports" / "outer_evaluation_support" / "outer_test_support_summary.csv")
                   .query("outer_fold == @outer")["eligible_outer_test_centers"].sum())
    if predicted_total != expected:
        raise ValueError(f"Outer {outer} predicted {predicted_total:,}; expected {expected:,}")
    contract = {"status": "outer_supervised_fold_evaluated", "outer_fold": outer,
                "field": "Field 1", "field2_accessed": False,
                "synthetic_scientific_observations": False,
                "individual_pixel_probabilities_saved": False,
                "architectures": architectures, "model_seeds": seeds,
                "ensemble_method": "arithmetic_mean_of_softmax_probabilities",
                "batch_size": batch_size, "eligible_observations": expected,
                "confusion_counts_sha256": sha256(counts_path)}
    (fold_output / "fold_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False))
    print(f"[outer {outer}] complete: {expected:,} observations × {len(architectures)} architectures", flush=True)


def matrix_from_rows(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.float64)
    for row in frame.itertuples(): matrix[int(row.true_class_id), int(row.predicted_class_id)] += float(row.count)
    return matrix


def aggregate(args: argparse.Namespace) -> None:
    paths = yaml.safe_load(args.paths.read_text()); config = yaml.safe_load(args.config.read_text())
    protocol = yaml.safe_load(args.protocol.read_text()); project = Path(paths["project_root"])
    local = project / "metadata" / "local"; contracts = local / "contracts"
    root = local / "reports" / "models" / "supervised_outer_evaluation"
    root.mkdir(parents=True, exist_ok=True)
    folds = [int(v) for v in protocol["outer_folds"]]; architectures = list(protocol["architectures"])
    frames = []
    for outer in folds:
        fold_dir = root / "folds" / f"outer_fold{outer}"
        contract = yaml.safe_load((fold_dir / "fold_contract.yaml").read_text())
        path = fold_dir / "confusion_counts.csv"
        if contract.get("status") != "outer_supervised_fold_evaluated" or sha256(path) != contract["confusion_counts_sha256"]:
            raise ValueError(f"Outer fold {outer} output is incomplete")
        frames.append(pd.read_csv(path))
    counts = pd.concat(frames, ignore_index=True)
    counts_path = root / "outer_confusion_counts.csv"; counts.to_csv(counts_path, index=False)

    pixel_rows, group_rows, cube_rows, fold_rows = [], [], [], []
    for architecture in architectures:
        arch_fold = counts[(counts.architecture == architecture) & (counts.level == "fold")]
        overall = matrix_from_rows(arch_fold)
        pixel_rows.append({"architecture": architecture, "display_name": DISPLAY_NAMES[architecture],
                           "outer_fold": "all", **metrics_from_confusion(overall)})
        for outer in folds:
            matrix = matrix_from_rows(arch_fold[arch_fold.outer_fold == outer])
            fold_rows.append({"architecture": architecture, "display_name": DISPLAY_NAMES[architecture],
                              "outer_fold": outer, **metrics_from_confusion(matrix)})
        selected_groups = counts[(counts.architecture == architecture) & (counts.level == "spatial_group")]
        normalized = []
        for (outer, unit), group in selected_groups.groupby(["outer_fold", "unit_id"]):
            matrix = matrix_from_rows(group); normalized.append(matrix / matrix.sum())
        equal_matrix = np.sum(normalized, axis=0)
        group_rows.append({"architecture": architecture, "display_name": DISPLAY_NAMES[architecture],
                           "spatial_groups": len(normalized), **metrics_from_confusion(equal_matrix)})
        selected_cubes = counts[(counts.architecture == architecture) & (counts.level == "cube")]
        for (outer, unit), group in selected_cubes.groupby(["outer_fold", "unit_id"]):
            matrix = matrix_from_rows(group); support = matrix.sum(axis=1); complete = bool((support > 0).all())
            metrics = metrics_from_confusion(matrix)
            if not complete:
                metrics["macro_f1"] = np.nan; metrics["balanced_accuracy"] = np.nan
            cube_rows.append({"architecture": architecture, "display_name": DISPLAY_NAMES[architecture],
                              "outer_fold": outer, "cube_id": unit, "classes_present": int((support > 0).sum()),
                              "valid_three_class_macro_f1": complete, **metrics})
    pixel = pd.DataFrame(pixel_rows); groups = pd.DataFrame(group_rows)
    cubes = pd.DataFrame(cube_rows); fold_metrics = pd.DataFrame(fold_rows)
    pixel.to_csv(root / "pixel_weighted_metrics.csv", index=False)
    groups.to_csv(root / "spatial_group_equal_weight_metrics.csv", index=False)
    cubes.to_csv(root / "cube_metrics.csv", index=False)
    fold_metrics.to_csv(root / "fold_metrics.csv", index=False)

    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    replicates = int(config["evaluation"]["bootstrap_replicates"]); bootstrap_rows = []
    for architecture in architectures:
        selected = counts[(counts.architecture == architecture) & (counts.level == "spatial_group")]
        matrices = []
        for _, group in selected.groupby(["outer_fold", "unit_id"]):
            matrix = matrix_from_rows(group); matrices.append(matrix / matrix.sum())
        matrices = np.stack(matrices)
        values = defaultdict(list)
        for _ in range(replicates):
            metric = metrics_from_confusion(matrices[rng.integers(0, len(matrices), len(matrices))].sum(0))
            for name in ("balanced_accuracy", "macro_f1", "soil_f1", "chickpea_f1", "weed_f1"):
                values[name].append(metric[name])
        point = metrics_from_confusion(matrices.sum(0))
        for name, samples in values.items():
            bootstrap_rows.append({"architecture": architecture, "metric": name, "point_estimate": point[name],
                                   "lower_95": float(np.quantile(samples, .025)),
                                   "upper_95": float(np.quantile(samples, .975)),
                                   "spatial_groups": len(matrices), "replicates": replicates})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(root / "spatial_group_cluster_bootstrap.csv", index=False)

    primary = str(protocol["primary_architecture"])
    primary_counts = counts[(counts.architecture == primary) & (counts.level == "fold")]
    primary_matrix = matrix_from_rows(primary_counts)
    row_norm = primary_matrix / np.maximum(primary_matrix.sum(1, keepdims=True), 1)
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    image = axes[0, 0].imshow(row_norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3): axes[0, 0].text(j, i, f"{row_norm[i,j]:.2f}", ha="center", va="center")
    axes[0, 0].set_xticks(range(3), [v.capitalize() for v in CLASS_NAMES]); axes[0, 0].set_yticks(range(3), [v.capitalize() for v in CLASS_NAMES])
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("True"); axes[0, 0].set_title("Primary row-normalized confusion")
    figure.colorbar(image, ax=axes[0, 0], fraction=.046)
    for architecture in architectures:
        selected = fold_metrics[fold_metrics.architecture == architecture]
        axes[0, 1].plot(selected.outer_fold, selected.macro_f1, marker="o", label=DISPLAY_NAMES[architecture])
    axes[0, 1].set_xticks(folds); axes[0, 1].set_ylim(0, 1); axes[0, 1].set_xlabel("Outer fold"); axes[0, 1].set_ylabel("Macro-F1")
    axes[0, 1].set_title("Spatial-fold generalization"); axes[0, 1].legend()
    x = np.arange(len(architectures)); width = .24
    for offset, class_name in enumerate(CLASS_NAMES):
        vals = [float(groups.loc[groups.architecture == a, f"{class_name}_f1"].iloc[0]) for a in architectures]
        axes[1, 0].bar(x + (offset-1)*width, vals, width, label=class_name.capitalize())
    axes[1, 0].set_xticks(x, [DISPLAY_NAMES[a] for a in architectures], rotation=15, ha="right")
    axes[1, 0].set_ylim(0, 1); axes[1, 0].set_ylabel("Equal-group F1"); axes[1, 0].set_title("Primary class performance"); axes[1, 0].legend()
    complete = cubes[cubes.valid_three_class_macro_f1]
    axes[1, 1].boxplot(
        [complete.loc[complete.architecture == a, "macro_f1"] for a in architectures],
        tick_labels=[DISPLAY_NAMES[a] for a in architectures],
    )
    axes[1, 1].set_ylim(0, 1); axes[1, 1].set_ylabel("Three-class macro-F1"); axes[1, 1].set_title("Complete cube × fold records only")
    axes[1, 1].tick_params(axis="x", rotation=15)
    figure.suptitle("Field 1 exhaustive nested supervised outer evaluation", fontsize=16)
    preview = root / "supervised_outer_evaluation_overview.png"; figure.savefig(preview, dpi=200); plt.close(figure)

    expected = int(yaml.safe_load((contracts / "field1_outer_test_support_contract.yaml").read_text())["total_eligible_outer_test_observations"])
    for architecture in architectures:
        observed = int(counts[(counts.architecture == architecture) & (counts.level == "fold")]["count"].sum())
        if observed != expected: raise ValueError(f"{architecture} evaluated {observed:,}; expected {expected:,}")
    outputs = [counts_path, root/"pixel_weighted_metrics.csv", root/"spatial_group_equal_weight_metrics.csv",
               root/"cube_metrics.csv", root/"fold_metrics.csv", root/"spatial_group_cluster_bootstrap.csv", preview]
    contract = {"status": "field1_supervised_outer_evaluation_complete", "field": "Field 1",
                "field2_accessed": False, "synthetic_scientific_observations": False,
                "architectures": architectures, "model_seeds": [int(v) for v in protocol["model_seeds"]],
                "eligible_observations_per_architecture": expected,
                "primary_summary": "equal_weight_frozen_5m_spatial_groups",
                "bootstrap_cluster_unit": "frozen_5m_spatial_group",
                "bootstrap_replicates": replicates,
                "individual_pixel_probabilities_saved": False,
                "output_hashes": {path.name: sha256(path) for path in outputs}}
    contract_path = contracts / "field1_supervised_outer_evaluation_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    print("\nPrimary equal-spatial-group metrics:")
    print(groups[["display_name", "balanced_accuracy", "macro_f1", "soil_f1", "chickpea_f1", "weed_f1"]].to_string(index=False))
    print(f"Reports: {root}"); print(f"Visual QC: {preview}"); print(f"Contract: {contract_path}")
    print("Exhaustive Field 1 supervised evaluation complete; Field 2 remained locked.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/outer_inference.yaml"), type=Path)
    parser.add_argument("--protocol", default=Path("configs/supervised_primary_evaluation.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    parser.add_argument("--spatial", default=Path("configs/spatial_splits.yaml"), type=Path)
    parser.add_argument("--worker-fold", type=int)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.worker_fold is not None:
        if args.gpu is None: raise ValueError("--gpu is required in worker mode")
        worker(args); return
    paths = yaml.safe_load(args.paths.read_text()); config = yaml.safe_load(args.config.read_text())
    protocol = yaml.safe_load(args.protocol.read_text()); project = Path(paths["project_root"])
    root = project / "metadata" / "local" / "reports" / "models" / "supervised_outer_evaluation" / "folds"
    gpu_indices = [int(v) for v in config["evaluation"]["gpu_indices"]]
    folds = [int(v) for v in protocol["outer_folds"]]
    def run(one_fold: int, gpu: int):
        fold_dir = root / f"outer_fold{one_fold}"; contract_path = fold_dir / "fold_contract.yaml"
        if contract_path.exists():
            contract = yaml.safe_load(contract_path.read_text()); counts = fold_dir / "confusion_counts.csv"
            if contract.get("status") == "outer_supervised_fold_evaluated" and counts.exists() and sha256(counts) == contract["confusion_counts_sha256"]:
                return one_fold, gpu, 0, "already complete"
        command = [sys.executable, str(project/"scripts"/"run_supervised_outer_evaluation.py"),
                   "--paths", str(args.paths.resolve()), "--config", str(args.config.resolve()),
                   "--protocol", str(args.protocol.resolve()), "--bands", str(args.bands.resolve()),
                   "--spatial", str(args.spatial.resolve()), "--worker-fold", str(one_fold), "--gpu", str(gpu)]
        started = time.monotonic(); completed = subprocess.run(command, cwd=project, check=False)
        if completed.returncode: raise RuntimeError(f"Outer fold {one_fold} failed with exit code {completed.returncode}")
        return one_fold, gpu, time.monotonic()-started, "evaluated"
    with ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
        futures = [executor.submit(run, fold, gpu_indices[i % len(gpu_indices)]) for i, fold in enumerate(folds)]
        for future in as_completed(futures):
            fold, gpu, elapsed, status = future.result()
            print(f"[scheduler] outer fold {fold} cuda:{gpu}: {status}; elapsed={elapsed/60:.1f} min", flush=True)
    aggregate(args)


if __name__ == "__main__": main()
