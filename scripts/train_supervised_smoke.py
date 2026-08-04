#!/usr/bin/env python
"""Run a bounded Fold-1 supervised GPU diagnostic before full experiments."""

from __future__ import annotations

import argparse
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
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from chickpea_ssl.data import IndexedPatchDataset, load_band_indices, load_records
from chickpea_ssl.model import SupervisedClassifier


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
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); losses = []; targets = []; predictions = []; probabilities = []
    criterion = nn.CrossEntropyLoss()
    for patches, labels in loader:
        patches, labels = patches.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(patches)
        losses.append(float(criterion(logits, labels).item()) * len(labels))
        probability = logits.softmax(dim=1)
        targets.append(labels.cpu().numpy())
        predictions.append(probability.argmax(1).cpu().numpy())
        probabilities.append(probability.cpu().numpy())
    truth, predicted = np.concatenate(targets), np.concatenate(predictions)
    return sum(losses) / len(truth), truth, predicted, np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/supervised_smoke.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not config.get("diagnostic_only", False):
        raise ValueError("This entry point is restricted to diagnostic runs")
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    report_stage = Path(config.get("report_stage", "models/supervised_smoke_fold1"))
    if report_stage.is_absolute() or ".." in report_stage.parts:
        raise ValueError("report_stage must be a safe path relative to metadata/local/reports")
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
        raise ValueError("Candidate-centre hash mismatch")
    if sha256(normalization_path) != normalization_contract["normalization_csv_sha256"]:
        raise ValueError("Normalization hash mismatch")

    heldout = int(config["heldout_fold"])
    centers = pd.read_csv(centers_path)
    train_mask = as_bool(centers[f"train_eligible_holdout_{heldout}"])
    validation_mask = as_bool(centers[f"validation_eligible_fold_{heldout}"])
    train = balanced_subset(
        centers[train_mask], int(config["samples_per_training_class"]), seed
    )
    validation = balanced_subset(
        centers[validation_mask], int(config["samples_per_validation_class"]), seed + 1000
    )
    band_indices = load_band_indices(args.bands, normalization_contract["band_section"])
    statistics = pd.read_csv(normalization_path)
    statistics = statistics[statistics["heldout_fold"] == heldout].set_index("band_index").loc[band_indices]
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
    workers = int(training["num_workers"])
    loader_kwargs = dict(
        batch_size=int(training["batch_size"]), num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    gpu_count = torch.cuda.device_count()
    if torch.cuda.is_available():
        gpu_index = int(training.get("gpu_index", 0))
        if gpu_index < 0 or gpu_index >= gpu_count:
            raise ValueError(f"Requested GPU {gpu_index}, but {gpu_count} CUDA devices are visible")
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
    else:
        gpu_index = None
        device = torch.device("cpu")
    if training.get("use_all_visible_gpus", False):
        raise ValueError(
            "The diagnostic prohibits nn.DataParallel after the measured workstation stall; "
            "use one selected GPU. Full multi-GPU experiments will use DistributedDataParallel."
        )
    model = SupervisedClassifier(
        in_channels=len(band_indices), embedding_dim=int(config["model"]["embedding_dim"]), classes=3
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    amp = bool(training["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    best_f1 = -1.0
    best_path = output / "best_model.pt"
    total_epochs = int(training["epochs"])
    log_every = int(training.get("log_every_batches", 10))
    patience = int(training.get("early_stopping_patience", total_epochs))
    minimum_epochs = int(training.get("minimum_epochs", 1))
    minimum_delta = float(training.get("early_stopping_minimum_delta", 0.0))
    epochs_without_improvement = 0
    print(
        f"Starting diagnostic on {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}) "
        f"with {gpu_count} visible GPU(s): "
        f"train={len(train):,}, validation={len(validation):,}, "
        f"batches/epoch={len(train_loader):,}"
    )
    for epoch in range(1, total_epochs + 1):
        model.train(); loss_sum = 0.0; seen = 0; epoch_start = time.monotonic()
        for batch_number, (patches, labels) in enumerate(train_loader, start=1):
            patches, labels = patches.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(patches); loss = criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.item()) * len(labels); seen += len(labels)
            if batch_number == 1 or batch_number % log_every == 0 or batch_number == len(train_loader):
                elapsed = time.monotonic() - epoch_start
                print(
                    f"Epoch {epoch}/{total_epochs} batch {batch_number}/{len(train_loader)} "
                    f"loss={loss_sum / seen:.4f} elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
        validation_loss, truth, predicted, _ = evaluate(model, validation_loader, device)
        macro_f1 = f1_score(truth, predicted, average="macro")
        row = {
            "epoch": epoch, "training_loss": loss_sum / seen,
            "validation_loss": validation_loss,
            "validation_accuracy": accuracy_score(truth, predicted),
            "validation_balanced_accuracy": balanced_accuracy_score(truth, predicted),
            "validation_macro_f1": macro_f1,
        }
        history.append(row); print(row, flush=True)
        if macro_f1 > best_f1 + minimum_delta:
            best_f1 = macro_f1
            epochs_without_improvement = 0
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"model_state": state, "config": config, "epoch": epoch}, best_path)
        else:
            epochs_without_improvement += 1
        if epoch >= minimum_epochs and epochs_without_improvement >= patience:
            print(
                f"Early stopping after epoch {epoch}: no macro-F1 improvement "
                f"> {minimum_delta:.4f} for {patience} epochs.",
                flush=True,
            )
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    target_model.load_state_dict(checkpoint["model_state"])
    validation_loss, truth, predicted, probabilities = evaluate(model, validation_loader, device)
    prediction_frame = validation.copy()
    prediction_frame["true_class_id"] = truth
    prediction_frame["predicted_class_id"] = predicted
    prediction_frame["predicted_class_name"] = np.asarray(
        ["soil", "chickpea", "weed"], dtype=object
    )[predicted]
    prediction_frame["probability_soil"] = probabilities[:, 0]
    prediction_frame["probability_chickpea"] = probabilities[:, 1]
    prediction_frame["probability_weed"] = probabilities[:, 2]
    prediction_frame["prediction_confidence"] = probabilities.max(axis=1)
    prediction_frame["correct"] = truth == predicted
    prediction_frame.to_csv(reports / "validation_predictions.csv", index=False)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(reports / "training_history.csv", index=False)
    report = pd.DataFrame(classification_report(
        truth, predicted, target_names=["soil", "chickpea", "weed"], output_dict=True,
        zero_division=0,
    )).T.reset_index(names="class_or_average")
    report.to_csv(reports / "diagnostic_classification_report.csv", index=False)
    confusion = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    pd.DataFrame(confusion, index=["true_soil", "true_chickpea", "true_weed"],
                 columns=["pred_soil", "pred_chickpea", "pred_weed"]).to_csv(
        reports / "diagnostic_confusion_matrix.csv"
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(history_frame.epoch, history_frame.training_loss, marker="o", label="Training")
    axes[0].plot(history_frame.epoch, history_frame.validation_loss, marker="o", label="Validation")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Diagnostic learning curves"); axes[0].legend(); axes[0].grid(alpha=.2)
    sns.heatmap(confusion / confusion.sum(axis=1, keepdims=True), annot=True, fmt=".2f",
                cmap="Blues", xticklabels=["Soil", "Chickpea", "Weed"],
                yticklabels=["Soil", "Chickpea", "Weed"], ax=axes[1], vmin=0, vmax=1)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title("Row-normalized Fold-1 diagnostic confusion")
    figure.suptitle(
        config.get("run_title", "Supervised spectral-spatial pipeline diagnostic"),
        fontsize=14,
    )
    preview = reports / "supervised_smoke_overview.png"
    figure.savefig(preview, dpi=200); plt.close(figure)
    run_summary = {
        "status": "diagnostic_only",
        "heldout_fold": heldout,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(best_f1),
        "final_evaluated_validation_loss": float(validation_loss),
        "training_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "device": str(device),
        "field2_accessed": False,
        "synthetic_training_samples": False,
    }
    (reports / "run_summary.yaml").write_text(yaml.safe_dump(run_summary, sort_keys=False))
    print(f"Best diagnostic macro-F1: {best_f1:.4f} at epoch {checkpoint['epoch']}")
    print(f"Device: {device}; visible GPUs: {gpu_count}; train={len(train):,}; validation={len(validation):,}")
    print(f"Reports: {reports}"); print(f"Visual QC: {preview}")
    print("Diagnostic only—not a five-fold or exhaustive primary benchmark.")


if __name__ == "__main__":
    main()
