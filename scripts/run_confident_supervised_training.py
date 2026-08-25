#!/usr/bin/env python
"""Train frozen center-context models on leakage-safe confident labels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from chickpea_ssl.data import IndexedPatchDataset, load_band_indices, load_records
from chickpea_ssl.model import build_supervised_model


CLASS_NAMES = ["soil", "chickpea", "weed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def loader(dataset, batch_size, workers, shuffle, seed, prefetch):
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=True,
                   persistent_workers=workers > 0)
    if workers > 0:
        options["prefetch_factor"] = prefetch
    return DataLoader(dataset, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed) if shuffle else None, **options)


@torch.inference_mode()
def evaluate(model, data_loader, device, amp):
    model.eval(); criterion = nn.CrossEntropyLoss(); loss_sum = 0.; truth = []; probabilities = []
    started = time.monotonic()
    for patches, labels in data_loader:
        patches = patches.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(patches); loss = criterion(logits, labels)
        loss_sum += float(loss) * len(labels); truth.append(labels.cpu().numpy())
        probabilities.append(logits.softmax(1).float().cpu().numpy())
    truth = np.concatenate(truth); probability = np.concatenate(probabilities); elapsed = time.monotonic() - started
    return loss_sum / len(truth), truth, probability.argmax(1), probability, elapsed


def load_fold_data(args, outer, variant):
    config = yaml.safe_load(args.config.read_text()); paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"]); local = project / "metadata/local"; root = project / config["contract_root"]
    contract_path = root / "field1_confident_supervised_data_contract.yaml"
    contract = yaml.safe_load(contract_path.read_text())
    if contract.get("status") != "confident_supervised_data_prepared_pending_gates":
        raise ValueError("Confident supervised-data contract is not ready")
    sample_path = root / f"outer{outer}_{variant}_samples.csv"
    if sha256(sample_path) != contract["sample_hashes"][sample_path.name]:
        raise ValueError("Sample hash mismatch")
    frame = pd.read_csv(sample_path); train = frame[frame.nested_role == "train"].reset_index(drop=True)
    validation = frame[frame.nested_role == "inner_validation"].reset_index(drop=True)
    if set(train.fold) & set(validation.fold) or outer in set(train.fold) | set(validation.fold):
        raise ValueError("Nested role leakage detected")
    normalization_path = root / "field1_confident_nested_normalization.csv"
    if sha256(normalization_path) != contract["normalization_sha256"]:
        raise ValueError("Normalization hash mismatch")
    bands = load_band_indices(args.bands, config["normalization"]["band_section"])
    stats = pd.read_csv(normalization_path)
    stats = stats[stats.outer_fold == outer].set_index("band_index").loc[bands]
    records = load_records(args.paths, local / "authoritative_manifest.csv")
    train_dataset = IndexedPatchDataset(records, train, bands, stats["mean"].to_numpy(), stats["standard_deviation"].to_numpy())
    validation_dataset = IndexedPatchDataset(records, validation, bands, stats["mean"].to_numpy(), stats["standard_deviation"].to_numpy())
    return config, project, train, validation, train_dataset, validation_dataset, len(bands)


def require_device(allow_cpu: bool):
    if not torch.cuda.is_available():
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback (use --allow-cpu explicitly)")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Worker must own exactly one visible CUDA GPU; found {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def benchmark(args):
    device = require_device(args.allow_cpu)
    config, _, train, validation, train_dataset, _, bands = load_fold_data(args, 1, "with_alley")
    training = config["training"]; workers = int(training["num_workers"])
    print_startup(device, args.physical_gpu, 0, training, len(train), len(validation), 0)
    rows = []
    for batch_size in training["batch_size_candidates"]:
        data_loader = loader(train_dataset, int(batch_size), workers, True, 42, training["prefetch_factor"])
        model = build_supervised_model(training["architecture"], bands, 3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=training["mixed_precision"])
        torch.cuda.reset_peak_memory_stats(); seen = 0; started = time.monotonic(); end_previous = started
        data_seconds = compute_seconds = 0.
        for number, (patches, labels) in enumerate(data_loader, 1):
            data_seconds += time.monotonic() - end_previous
            patches = patches.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            compute_start = time.monotonic(); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=training["mixed_precision"]):
                loss = nn.functional.cross_entropy(model(patches), labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
            compute_seconds += time.monotonic() - compute_start; seen += len(labels); end_previous = time.monotonic()
            if number == 1: print(f"benchmark batch {batch_size}: batch 1 complete", flush=True)
            if number >= min(20, len(data_loader)): break
        elapsed = time.monotonic() - started; peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append({"batch_size": int(batch_size), "samples": seen, "elapsed_seconds": elapsed,
                     "samples_per_second": seen / elapsed, "data_loading_seconds": data_seconds,
                     "gpu_compute_seconds": compute_seconds, "peak_vram_gib": peak})
        print(rows[-1], flush=True); del data_loader, model, optimizer, scaler; gc.collect(); torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    safe = frame[frame.peak_vram_gib < 24]
    if safe.empty: raise RuntimeError("No benchmark batch size retained the required VRAM margin")
    selected = int(safe.sort_values("samples_per_second", ascending=False).iloc[0].batch_size)
    report = PROJECT_ROOT / config["report_root"] / "training"; report.mkdir(parents=True, exist_ok=True)
    frame["selected"] = frame.batch_size == selected; frame.to_csv(report / "real_patch_batch_benchmark.csv", index=False)
    print(f"Selected batch size: {selected}")


def print_startup(device, physical_gpu, outer, training, train_count, validation_count, batches):
    print(f"PyTorch: {torch.__version__}", flush=True); print(f"CUDA runtime: {torch.version.cuda}", flush=True)
    print(f"Visible GPU count: {torch.cuda.device_count()}", flush=True)
    if device.type == "cuda": print(f"Worker outer={outer} physical_gpu={physical_gpu} device={device} name={torch.cuda.get_device_name(0)}", flush=True)
    print(f"batch_size={training.get('selected_batch_size', training.get('batch_size'))}; train_samples={train_count}; "
          f"validation_samples={validation_count}; batches_per_epoch={batches}", flush=True)


def train_worker(args):
    outer = int(args.outer); device = require_device(args.allow_cpu)
    variants = args.variants.split(","); results = []
    for variant in variants:
        config, project, train, validation, train_dataset, validation_dataset, bands = load_fold_data(args, outer, variant)
        training = config["training"]
        benchmark_path = project / config["report_root"] / "training/real_patch_batch_benchmark.csv"
        if not benchmark_path.exists(): raise FileNotFoundError("Run --mode benchmark before training")
        benchmark_frame = pd.read_csv(benchmark_path); batch_size = int(benchmark_frame[benchmark_frame.selected.astype(str).str.lower().eq("true")].batch_size.iloc[0])
        training = dict(training); training["selected_batch_size"] = batch_size
        workers = int(training["num_workers"]); val_loader = loader(validation_dataset, batch_size, workers, False, 0, training["prefetch_factor"])
        train_loader_template = lambda seed: loader(train_dataset, batch_size, workers, True, seed, training["prefetch_factor"])
        print_startup(device, args.physical_gpu, outer, training, len(train), len(validation), math.ceil(len(train) / batch_size))
        seeds = [int(args.smoke_seed)] if args.mode == "smoke" else [int(value) for value in training["seeds"]]
        epochs = 1 if args.mode == "smoke" else int(training["epochs"])
        for seed in seeds:
            set_seed(seed); train_loader = train_loader_template(seed)
            model = build_supervised_model(training["architecture"], bands, 3).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
            criterion = nn.CrossEntropyLoss(); amp = bool(training["mixed_precision"])
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            run_root = "smoke" if args.mode == "smoke" else variant
            run_dir = project / "results/field1_confident_labels_v1" / run_root / f"outer{outer}"
            run_dir.mkdir(parents=True, exist_ok=True); checkpoint_path = run_dir / f"center_context_fusion_seed{seed}_best.pt"
            best_f1 = -1.; stale = 0; history = []; total_started = time.monotonic(); total_seen = 0
            peak = 0.; total_data = total_compute = 0.
            for epoch in range(1, epochs + 1):
                model.train(); loss_sum = seen = 0; epoch_start = time.monotonic(); end_previous = epoch_start
                torch.cuda.reset_peak_memory_stats()
                for batch_number, (patches, labels) in enumerate(train_loader, 1):
                    total_data += time.monotonic() - end_previous
                    patches = patches.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
                    compute_start = time.monotonic(); optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=amp): logits = model(patches); loss = criterion(logits, labels)
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
                    total_compute += time.monotonic() - compute_start; loss_sum += float(loss.detach()) * len(labels); seen += len(labels)
                    total_seen += len(labels); end_previous = time.monotonic()
                    if batch_number == 1 or batch_number % int(training["log_every_batches"]) == 0 or batch_number == len(train_loader):
                        print(f"outer {outer} {variant} seed {seed} epoch {epoch} batch {batch_number}/{len(train_loader)} "
                              f"loss={loss_sum/seen:.4f} throughput={seen/(time.monotonic()-epoch_start):,.0f}/s", flush=True)
                val_loss, truth, predicted, _, validation_seconds = evaluate(model, val_loader, device, amp)
                macro = f1_score(truth, predicted, labels=[0,1,2], average="macro", zero_division=0)
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
                row = {"outer_fold": outer, "variant": variant, "seed": seed, "epoch": epoch,
                       "training_loss": loss_sum/seen, "validation_loss": val_loss,
                       "validation_macro_f1": macro, "validation_balanced_accuracy": balanced_accuracy_score(truth, predicted),
                       "epoch_seconds": time.monotonic()-epoch_start, "validation_seconds": validation_seconds,
                       "samples_per_second": seen/(time.monotonic()-epoch_start), "peak_vram_gib": peak}
                history.append(row); print(f"inner validation: {row}", flush=True)
                if macro > best_f1 + float(training["early_stopping_minimum_delta"]):
                    best_f1 = macro; stale = 0
                    torch.save({"model_state": model.state_dict(), "architecture": training["architecture"],
                                "seed": seed, "outer_fold": outer, "variant": variant, "epoch": epoch,
                                "normalization_contract": "field1_confident_supervised_data_contract.yaml"}, checkpoint_path)
                else: stale += 1
                if args.mode != "smoke" and epoch >= int(training["minimum_epochs"]) and stale >= int(training["early_stopping_patience"]): break
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state"])
            val_loss, truth, predicted, _, validation_seconds = evaluate(model, val_loader, device, amp)
            precision, recall, class_f1, support = precision_recall_fscore_support(truth, predicted, labels=[0,1,2], zero_division=0)
            total_seconds = time.monotonic() - total_started
            results.append({"outer_fold": outer, "physical_gpu": args.physical_gpu, "variant": variant, "seed": seed,
                            "best_epoch": checkpoint["epoch"], "inner_validation_macro_f1": f1_score(truth,predicted,labels=[0,1,2],average="macro",zero_division=0),
                            "inner_validation_balanced_accuracy": balanced_accuracy_score(truth,predicted),
                            **{f"{name}_{metric}": float(values[index]) for metric, values in (("precision",precision),("recall",recall),("f1",class_f1)) for index,name in enumerate(CLASS_NAMES)},
                            "total_seconds": total_seconds, "samples_per_second": total_seen/total_seconds,
                            "data_loading_seconds": total_data, "gpu_compute_seconds": total_compute,
                            "peak_vram_gib": peak, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path)})
            report_dir = project / config["report_root"] / "training" / variant / f"outer{outer}"
            report_dir.mkdir(parents=True, exist_ok=True); pd.DataFrame(history).to_csv(report_dir / f"seed{seed}_history.csv", index=False)
            print(f"checkpoint save/reload PASS: {checkpoint_path}; total={total_seconds:.1f}s; peak={peak:.2f} GiB", flush=True)
            del train_loader, model, optimizer, scaler; gc.collect(); torch.cuda.empty_cache()
    report_dir = PROJECT_ROOT / yaml.safe_load(args.config.read_text())["report_root"] / "training"
    suffix = "smoke" if args.mode == "smoke" else f"outer{outer}"
    pd.DataFrame(results).to_csv(report_dir / f"{suffix}_gpu{args.physical_gpu}.csv", index=False)
    if args.mode == "smoke":
        seconds = float(pd.DataFrame(results).total_seconds.sum()); checkpoints = 15
        contract_root = PROJECT_ROOT / yaml.safe_load(args.config.read_text())["contract_root"]
        smoke_contract = {
            "status": "cuda_smoke_passed",
            "field": "Field 1",
            "field2_accessed": False,
            "physical_gpu": int(args.physical_gpu),
            "visible_gpu_count_in_worker": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0),
            "real_data": True,
            "batch_1_completed": True,
            "one_epoch_completed": True,
            "inner_validation_completed": True,
            "checkpoint_reload_passed": True,
            "results_sha256": sha256(report_dir / f"{suffix}_gpu{args.physical_gpu}.csv"),
        }
        (contract_root / "field1_confident_cuda_smoke_contract.yaml").write_text(
            yaml.safe_dump(smoke_contract, sort_keys=False)
        )
        print(f"Estimated 15-checkpoint runtime at observed one-epoch rate: {seconds * 15 / 2 / 3600:.2f} GPU-pair hours per epoch; "
              f"approximately {seconds * 15 * yaml.safe_load(args.config.read_text())['training']['epochs'] / 2 / 3600:.2f} hours at max epochs", flush=True)


def scheduler(args):
    config = yaml.safe_load(args.config.read_text()); gpus = [int(value) for value in config["training"]["gpu_indices"]]
    folds = [int(value) for value in config["training"]["outer_folds"]]
    pending = list(folds); running = {}; failed = []
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending: continue
            outer = pending.pop(0); env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [sys.executable, str(Path(__file__).resolve()), "--paths", str(args.paths), "--config", str(args.config),
                       "--bands", str(args.bands), "--mode", "worker", "--outer", str(outer), "--physical-gpu", str(gpu),
                       "--variants", args.variants]
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)
            running[gpu] = (process, outer); print(f"scheduled outer fold {outer} on physical cuda:{gpu}", flush=True)
        time.sleep(1)
        for gpu, (process, outer) in list(running.items()):
            code = process.poll()
            if code is not None:
                del running[gpu]
                if code: failed.append((outer, code))
                print(f"outer fold {outer} on cuda:{gpu} exited {code}", flush=True)
    if failed: raise RuntimeError(f"Training workers failed: {failed}")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/confident_labels_v1.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    parser.add_argument("--mode", choices=["benchmark","smoke","full","worker"], required=True)
    parser.add_argument("--outer", type=int, default=1); parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--variants", default="with_alley"); parser.add_argument("--smoke-seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true"); args = parser.parse_args()
    if args.mode == "benchmark": benchmark(args)
    elif args.mode == "smoke": train_worker(args)
    elif args.mode == "worker": train_worker(argparse.Namespace(**{**vars(args), "mode": "full"}))
    else: scheduler(args)


if __name__ == "__main__": main()
