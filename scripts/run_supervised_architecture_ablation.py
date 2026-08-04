#!/usr/bin/env python
"""Run a controlled center-aware supervised architecture ablation on frozen Fold 1."""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from chickpea_ssl.data import IndexedPatchDataset, load_band_indices, load_records
from chickpea_ssl.model import build_supervised_model


CLASS_NAMES = np.asarray(["soil", "chickpea", "weed"], dtype=object)
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


def as_bool(series: pd.Series) -> pd.Series:
    return series if series.dtype == bool else series.astype(str).str.lower().eq("true")


def balanced_subset(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    parts = []
    for class_id, group in frame.groupby("class_id"):
        if len(group) < count:
            raise ValueError(f"Class {class_id} has only {len(group)} rows; requested {count}")
        parts.append(group.sample(n=count, random_state=seed + int(class_id)))
    return pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    targets, probabilities = [], []
    for patches, labels in loader:
        patches = patches.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(patches)
        loss_sum += float(criterion(logits, labels).item()) * len(labels)
        targets.append(labels.cpu().numpy())
        probabilities.append(logits.softmax(dim=1).cpu().numpy())
    truth = np.concatenate(targets)
    probability = np.concatenate(probabilities)
    return loss_sum / len(truth), truth, probability.argmax(axis=1), probability


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--config", default=Path("configs/supervised_architecture_ablation.yaml"),
        type=Path,
    )
    parser.add_argument(
        "--bands", default=Path("configs/spectral_bands.yaml"), type=Path
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not config.get("diagnostic_only", False):
        raise ValueError("Architecture ablation must remain diagnostic-only")
    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    report_stage = Path(config["report_stage"])
    if report_stage.is_absolute() or ".." in report_stage.parts:
        raise ValueError("Unsafe report_stage")
    reports = local / "reports" / report_stage
    reports.mkdir(parents=True, exist_ok=True)
    output = project / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)

    manifest = local / "authoritative_manifest.csv"
    centers_path = contracts / "field1_training_candidate_centers.csv"
    centers_contract_path = contracts / "field1_training_candidate_contract.yaml"
    normalization_path = contracts / "field1_fold_normalization.csv"
    normalization_contract_path = contracts / "field1_fold_normalization_contract.yaml"
    center_contract = yaml.safe_load(centers_contract_path.read_text())
    normalization_contract = yaml.safe_load(normalization_contract_path.read_text())
    if sha256(centers_path) != center_contract["center_index_sha256"]:
        raise ValueError("Candidate-center hash mismatch")
    if sha256(normalization_path) != normalization_contract["normalization_csv_sha256"]:
        raise ValueError("Normalization hash mismatch")

    heldout = int(config["heldout_fold"])
    sampling_seed = int(config["seed_for_observed_sample_selection"])
    centers = pd.read_csv(centers_path)
    train = balanced_subset(
        centers[as_bool(centers[f"train_eligible_holdout_{heldout}"])],
        int(config["samples_per_training_class"]),
        sampling_seed,
    )
    validation = balanced_subset(
        centers[as_bool(centers[f"validation_eligible_fold_{heldout}"])],
        int(config["samples_per_validation_class"]),
        sampling_seed + 1000,
    )
    band_indices = load_band_indices(args.bands, normalization_contract["band_section"])
    statistics = pd.read_csv(normalization_path)
    statistics = (
        statistics[statistics["heldout_fold"] == heldout]
        .set_index("band_index")
        .loc[band_indices]
    )
    records = load_records(args.paths, manifest)
    train_dataset = IndexedPatchDataset(
        records, train, band_indices, statistics["mean"].to_numpy(),
        statistics["standard_deviation"].to_numpy(),
    )
    validation_dataset = IndexedPatchDataset(
        records, validation, band_indices, statistics["mean"].to_numpy(),
        statistics["standard_deviation"].to_numpy(),
    )

    training = config["training"]
    gpu_index = int(training["gpu_index"])
    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA GPU {gpu_index} is unavailable")
    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    workers = int(training["num_workers"])
    common_loader = dict(
        batch_size=int(training["batch_size"]),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **common_loader
    )

    history_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    cube_rows: list[dict[str, object]] = []
    architectures = list(config["architectures"])
    seeds = [int(seed) for seed in config["model_seeds"]]
    total_runs = len(architectures) * len(seeds)
    run_number = 0

    for architecture in architectures:
        for seed in seeds:
            run_number += 1
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_dataset, shuffle=True, generator=generator, **common_loader
            )
            model = build_supervised_model(
                architecture, in_channels=len(band_indices), classes=3
            ).to(device)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            criterion = nn.CrossEntropyLoss()
            amp = bool(training["mixed_precision"])
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            best_f1 = -1.0
            epochs_without_improvement = 0
            best_path = output / f"{architecture}_seed{seed}_best.pt"
            print(
                f"Run {run_number}/{total_runs}: {architecture}, seed={seed}, "
                f"parameters={parameter_count:,}, device={device}",
                flush=True,
            )
            for epoch in range(1, int(training["epochs"]) + 1):
                model.train()
                loss_sum, seen = 0.0, 0
                started = time.monotonic()
                for batch_number, (patches, labels) in enumerate(train_loader, start=1):
                    patches = patches.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", enabled=amp):
                        logits = model(patches)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    loss_sum += float(loss.item()) * len(labels)
                    seen += len(labels)
                    if (
                        batch_number == 1
                        or batch_number % int(training["log_every_batches"]) == 0
                        or batch_number == len(train_loader)
                    ):
                        print(
                            f"  epoch {epoch} batch {batch_number}/{len(train_loader)} "
                            f"loss={loss_sum / seen:.4f} "
                            f"elapsed={(time.monotonic() - started) / 60:.2f} min",
                            flush=True,
                        )
                validation_loss, truth, predicted, _ = evaluate(
                    model, validation_loader, device
                )
                macro_f1 = f1_score(
                    truth, predicted, labels=[0, 1, 2], average="macro",
                    zero_division=0,
                )
                row = {
                    "architecture": architecture,
                    "seed": seed,
                    "epoch": epoch,
                    "training_loss": loss_sum / seen,
                    "validation_loss": validation_loss,
                    "validation_accuracy": accuracy_score(truth, predicted),
                    "validation_balanced_accuracy": balanced_accuracy_score(
                        truth, predicted
                    ),
                    "validation_macro_f1": macro_f1,
                }
                history_rows.append(row)
                print(f"  validation: {row}", flush=True)
                if macro_f1 > best_f1 + float(
                    training["early_stopping_minimum_delta"]
                ):
                    best_f1 = macro_f1
                    epochs_without_improvement = 0
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "architecture": architecture,
                            "seed": seed,
                            "epoch": epoch,
                            "config": config,
                        },
                        best_path,
                    )
                else:
                    epochs_without_improvement += 1
                if (
                    epoch >= int(training["minimum_epochs"])
                    and epochs_without_improvement
                    >= int(training["early_stopping_patience"])
                ):
                    print(f"  early stopping at epoch {epoch}", flush=True)
                    break

            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            validation_loss, truth, predicted, probability = evaluate(
                model, validation_loader, device
            )
            precision, recall, class_f1, support = precision_recall_fscore_support(
                truth, predicted, labels=[0, 1, 2], zero_division=0
            )
            for class_id, class_name in enumerate(CLASS_NAMES):
                class_rows.append({
                    "architecture": architecture,
                    "seed": seed,
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": precision[class_id],
                    "recall": recall[class_id],
                    "f1": class_f1[class_id],
                    "support": int(support[class_id]),
                })
            prediction_frame = validation[[
                "sample_id", "cube_id", "row", "column", "class_id", "class_name"
            ]].copy()
            prediction_frame["predicted_class_id"] = predicted
            prediction_frame["prediction_confidence"] = probability.max(axis=1)
            prediction_frame["correct"] = truth == predicted
            cube_macro_scores = []
            for cube_id, group in prediction_frame.groupby("cube_id"):
                indices = group.index.to_numpy()
                cube_truth = truth[indices]
                cube_predicted = predicted[indices]
                cube_macro = f1_score(
                    cube_truth, cube_predicted, labels=[0, 1, 2],
                    average="macro", zero_division=0,
                )
                cube_balanced = balanced_accuracy_score(cube_truth, cube_predicted)
                cube_macro_scores.append(cube_macro)
                cube_rows.append({
                    "architecture": architecture,
                    "seed": seed,
                    "cube_id": cube_id,
                    "patches": len(group),
                    "macro_f1": cube_macro,
                    "balanced_accuracy": cube_balanced,
                    "accuracy": accuracy_score(cube_truth, cube_predicted),
                })
            confusion = confusion_matrix(truth, predicted, labels=[0, 1, 2])
            pd.DataFrame(
                confusion,
                index=["true_soil", "true_chickpea", "true_weed"],
                columns=["pred_soil", "pred_chickpea", "pred_weed"],
            ).to_csv(reports / f"confusion_{architecture}_seed{seed}.csv")
            summary_rows.append({
                "architecture": architecture,
                "display_name": DISPLAY_NAMES[architecture],
                "seed": seed,
                "parameter_count": parameter_count,
                "best_epoch": int(checkpoint["epoch"]),
                "validation_loss": validation_loss,
                "validation_accuracy": accuracy_score(truth, predicted),
                "validation_balanced_accuracy": balanced_accuracy_score(
                    truth, predicted
                ),
                "validation_macro_f1": f1_score(
                    truth, predicted, labels=[0, 1, 2], average="macro",
                    zero_division=0,
                ),
                "soil_f1": class_f1[0],
                "chickpea_f1": class_f1[1],
                "weed_f1": class_f1[2],
                "worst_cube_macro_f1": min(cube_macro_scores),
                "mean_cube_macro_f1": float(np.mean(cube_macro_scores)),
            })
            del train_loader, model, optimizer, scaler
            gc.collect()
            torch.cuda.empty_cache()

    history = pd.DataFrame(history_rows)
    summary = pd.DataFrame(summary_rows)
    class_metrics = pd.DataFrame(class_rows)
    cube_metrics = pd.DataFrame(cube_rows)
    history.to_csv(reports / "architecture_ablation_history.csv", index=False)
    summary.to_csv(reports / "architecture_ablation_summary.csv", index=False)
    class_metrics.to_csv(
        reports / "architecture_ablation_class_metrics.csv", index=False
    )
    cube_metrics.to_csv(
        reports / "architecture_ablation_cube_metrics.csv", index=False
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    positions = np.arange(len(architectures))
    macro_groups = [
        summary.loc[summary.architecture == architecture, "validation_macro_f1"]
        for architecture in architectures
    ]
    axes[0].boxplot(macro_groups, positions=positions, widths=0.55)
    for position, values in zip(positions, macro_groups):
        axes[0].scatter(
            np.full(len(values), position), values, color="#3366AA", zorder=3
        )
    axes[0].set_xticks(positions, [DISPLAY_NAMES[value] for value in architectures],
                       rotation=15, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].set_title("Three-seed architecture stability")

    means = class_metrics.groupby(["architecture", "class_name"]).f1.mean().unstack()
    standard = class_metrics.groupby(["architecture", "class_name"]).f1.std().unstack()
    width = 0.24
    for offset, class_name in enumerate(CLASS_NAMES):
        axes[1].bar(
            positions + (offset - 1) * width,
            means.loc[architectures, class_name],
            width,
            yerr=standard.loc[architectures, class_name],
            capsize=3,
            label=class_name.capitalize(),
        )
    axes[1].set_xticks(positions, [DISPLAY_NAMES[value] for value in architectures],
                       rotation=15, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Class F1 (mean ± SD)")
    axes[1].set_title("Class-specific performance")
    axes[1].legend()

    for architecture in architectures:
        selected = summary[summary.architecture == architecture]
        axes[2].scatter(
            selected["worst_cube_macro_f1"],
            selected["validation_macro_f1"],
            s=70,
            label=DISPLAY_NAMES[architecture],
        )
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Worst-cube macro-F1")
    axes[2].set_ylabel("Overall validation macro-F1")
    axes[2].set_title("Accuracy versus spatial robustness")
    axes[2].legend()
    figure.suptitle(config["run_title"], fontsize=15)
    preview = reports / "supervised_architecture_ablation_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)

    aggregate = (
        summary.groupby(["architecture", "display_name"])
        .agg(
            runs=("seed", "size"),
            macro_f1_mean=("validation_macro_f1", "mean"),
            macro_f1_sd=("validation_macro_f1", "std"),
            chickpea_f1_mean=("chickpea_f1", "mean"),
            worst_cube_macro_f1_mean=("worst_cube_macro_f1", "mean"),
            parameter_count=("parameter_count", "first"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    aggregate.to_csv(reports / "architecture_ablation_aggregate.csv", index=False)
    contract = {
        "status": "diagnostic_architecture_ablation_complete",
        "field": "Field 1",
        "heldout_fold": heldout,
        "field2_accessed": False,
        "synthetic_training_samples": False,
        "training_samples": len(train),
        "validation_samples": len(validation),
        "model_seeds": seeds,
        "architectures": architectures,
        "candidate_center_sha256": sha256(centers_path),
        "normalization_sha256": sha256(normalization_path),
        "configuration_sha256": sha256(args.config),
        "selection_status": "no_architecture_selected_automatically",
    }
    (reports / "architecture_ablation_contract.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False)
    )
    print(aggregate.to_string(index=False), flush=True)
    print(f"Reports: {reports}", flush=True)
    print(f"Visual QC: {preview}", flush=True)
    print(
        "Diagnostic only; Field 2 was not accessed and no architecture was "
        "automatically selected.",
        flush=True,
    )


if __name__ == "__main__":
    main()
