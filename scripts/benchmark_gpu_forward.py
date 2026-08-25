#!/usr/bin/env python
"""Benchmark each visible GPU independently before selecting a training launcher."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import yaml

from chickpea_ssl.model import SupervisedClassifier


def timed_training_steps(
    device: torch.device, batch_size: int, deterministic: bool, iterations: int
) -> dict[str, object]:
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    model = SupervisedClassifier(in_channels=111, embedding_dim=128, classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    patches = torch.randn(batch_size, 111, 15, 15, device=device)
    labels = torch.arange(batch_size, device=device) % 3

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            loss = criterion(model(patches), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(
        f"GPU {device.index}: batch={batch_size}, deterministic={deterministic} — warm-up",
        flush=True,
    )
    step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for number in range(1, iterations + 1):
        step()
        torch.cuda.synchronize(device)
        print(f"  completed measured step {number}/{iterations}", flush=True)
    elapsed = time.monotonic() - started
    return {
        "gpu_index": int(device.index),
        "gpu_name": torch.cuda.get_device_name(device),
        "batch_size": batch_size,
        "deterministic": deterministic,
        "iterations": iterations,
        "seconds_per_training_step": elapsed / iterations,
        "samples_per_second": batch_size * iterations / elapsed,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--iterations", default=3, type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    reports = project / "metadata" / "local" / "reports" / "gpu_benchmark"
    reports.mkdir(parents=True, exist_ok=True)

    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
    print(f"Visible CUDA devices: {torch.cuda.device_count()}", flush=True)
    rows: list[dict[str, object]] = []
    for gpu_index in range(torch.cuda.device_count()):
        for deterministic in (True, False):
            for batch_size in (64, 256):
                rows.append(timed_training_steps(
                    torch.device(f"cuda:{gpu_index}"),
                    batch_size,
                    deterministic,
                    args.iterations,
                ))
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    csv_path = reports / "gpu_forward_benchmark.csv"
    frame.to_csv(csv_path, index=False)
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    labels = [
        f"GPU {row.gpu_index}\nB{row.batch_size}\n{'det' if row.deterministic else 'fast'}"
        for row in frame.itertuples()
    ]
    axis.bar(labels, frame["samples_per_second"], color=[
        "#4477AA" if value else "#228833" for value in frame["deterministic"]
    ])
    axis.set_ylabel("Training samples per second")
    axis.set_title("Independent single-GPU spectral-spatial throughput")
    axis.grid(axis="y", alpha=0.2)
    preview = reports / "gpu_forward_benchmark.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)
    print(frame.to_string(index=False), flush=True)
    print(f"Report: {csv_path}", flush=True)
    print(f"Visual QC: {preview}", flush=True)
    print(
        "Engineering benchmark only: generated tensors are never used for model fitting "
        "or scientific results.",
        flush=True,
    )


if __name__ == "__main__":
    main()
