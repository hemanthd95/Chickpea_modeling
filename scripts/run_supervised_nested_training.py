#!/usr/bin/env python
"""Train frozen supervised candidates with inner validation and unopened outer tests."""

from __future__ import annotations

import argparse
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
import torch
import yaml


DISPLAY_NAMES = {
    "center_spectrum_mlp": "Center spectrum",
    "spatial_average_cnn": "Spatial-average CNN",
    "center_context_fusion": "Center + context",
}
CLASS_ORDER = ["soil", "chickpea", "weed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fold_complete(
    report_dir: Path, architectures: list[str], seeds: list[int]
) -> bool:
    summary_path = report_dir / "architecture_ablation_summary.csv"
    contract_path = report_dir / "architecture_ablation_contract.yaml"
    if not summary_path.exists() or not contract_path.exists():
        return False
    try:
        summary = pd.read_csv(summary_path)
        contract = yaml.safe_load(contract_path.read_text())
    except Exception:
        return False
    expected = {(architecture, seed) for architecture in architectures for seed in seeds}
    observed = {
        (str(row.architecture), int(row.seed))
        for row in summary[["architecture", "seed"]].itertuples(index=False)
    }
    return (
        len(summary) == len(expected)
        and observed == expected
        and contract.get("status") == "nested_primary_training_complete"
        and contract.get("outer_test_accessed") is False
    )


def write_runtime_config(
    base: dict,
    destination: Path,
    outer_fold: int,
    gpu_index: int,
    config: dict,
) -> None:
    current = dict(base)
    current["diagnostic_only"] = False
    current["nested_primary_training"] = True
    current["heldout_fold"] = outer_fold
    current["run_title"] = (
        f"Outer fold {outer_fold}: nested fitting and inner validation"
    )
    current["report_stage"] = (
        f"{config['per_fold_report_stage']}/outer_fold{outer_fold}"
    )
    current["output_dir"] = (
        f"{config['per_fold_output_dir']}/outer_fold{outer_fold}"
    )
    current["architectures"] = list(config["architectures"])
    current["model_seeds"] = [int(seed) for seed in config["model_seeds"]]
    for key in (
        "nested_sample_file",
        "nested_sample_contract",
        "nested_normalization_file",
        "nested_normalization_contract",
    ):
        current[key] = config[key]
    current["training"] = dict(base["training"])
    current["training"]["gpu_index"] = gpu_index
    destination.write_text(yaml.safe_dump(current, sort_keys=False))


def run_outer_fold(
    outer_fold: int,
    gpu_index: int,
    project: Path,
    paths_file: Path,
    bands_file: Path,
    runtime_dir: Path,
    base_config: dict,
    config: dict,
) -> tuple[int, int, float, str]:
    report_dir = (
        project
        / "metadata"
        / "local"
        / "reports"
        / config["per_fold_report_stage"]
        / f"outer_fold{outer_fold}"
    )
    architectures = list(config["architectures"])
    seeds = [int(seed) for seed in config["model_seeds"]]
    if bool(config.get("skip_completed_folds", True)) and fold_complete(
        report_dir, architectures, seeds
    ):
        return outer_fold, gpu_index, 0.0, "already complete"

    runtime_config = runtime_dir / f"outer{outer_fold}_gpu{gpu_index}.yaml"
    write_runtime_config(
        base_config, runtime_config, outer_fold, gpu_index, config
    )
    command = [
        sys.executable,
        str(project / "scripts" / "run_supervised_architecture_ablation.py"),
        "--paths",
        str(paths_file),
        "--config",
        str(runtime_config),
        "--bands",
        str(bands_file),
    ]
    print(
        f"[scheduler] outer fold {outer_fold} nested training on cuda:{gpu_index}",
        flush=True,
    )
    started = time.monotonic()
    completed = subprocess.run(command, cwd=project, check=False)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Outer fold {outer_fold} failed on cuda:{gpu_index} "
            f"with exit code {completed.returncode}"
        )
    if not fold_complete(report_dir, architectures, seeds):
        raise RuntimeError(
            f"Outer fold {outer_fold} did not produce a complete nested contract"
        )
    return outer_fold, gpu_index, elapsed, "trained"


def read_fold_outputs(
    root: Path, folds: list[int], filename: str
) -> pd.DataFrame:
    frames = []
    for outer in folds:
        path = root / f"outer_fold{outer}" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "outer_fold", outer)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--config",
        default=Path("configs/supervised_nested_training.yaml"),
        type=Path,
    )
    parser.add_argument(
        "--bands", default=Path("configs/spectral_bands.yaml"), type=Path
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"

    required_contracts = {
        "primary_protocol": contracts / config["primary_protocol_contract"],
        "samples": contracts / config["nested_sample_contract"],
        "normalization": contracts / config["nested_normalization_contract"],
    }
    for path in required_contracts.values():
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = yaml.safe_load(required_contracts["primary_protocol"].read_text())
    sample_contract = yaml.safe_load(required_contracts["samples"].read_text())
    normalization_contract = yaml.safe_load(
        required_contracts["normalization"].read_text()
    )
    if protocol.get("status") != "supervised_primary_evaluation_protocol_frozen":
        raise ValueError("Primary evaluation protocol is not frozen")
    if sample_contract.get("status") != "frozen_nested_model_samples":
        raise ValueError("Nested model samples are not frozen")
    if (
        normalization_contract.get("status")
        != "frozen_nested_training_only_normalization"
    ):
        raise ValueError("Nested normalization is not frozen")
    if any(
        contract.get("field2_accessed") is not False
        for contract in (protocol, sample_contract, normalization_contract)
    ):
        raise ValueError("A source contract does not preserve the Field 2 lock")

    folds = [int(value) for value in config["outer_folds"]]
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError("Nested training requires outer folds [1, 2, 3, 4, 5]")
    architectures = list(config["architectures"])
    seeds = [int(seed) for seed in config["model_seeds"]]
    frozen_architectures = {
        protocol["primary_architecture"],
        *protocol["architectures"],
    }
    if set(architectures) != set(protocol["architectures"]):
        raise ValueError("Architectures differ from the frozen protocol")
    if seeds != [int(value) for value in protocol["model_seeds"]]:
        raise ValueError("Model seeds differ from the frozen protocol")

    gpu_indices = [int(value) for value in config["gpu_indices"]]
    if len(gpu_indices) != 2 or len(set(gpu_indices)) != 2:
        raise ValueError("Exactly two distinct GPU indices are required")
    if not torch.cuda.is_available() or max(gpu_indices) >= torch.cuda.device_count():
        raise RuntimeError("Requested CUDA devices are unavailable")

    base_path = project / config["base_training_config"]
    base_config = yaml.safe_load(base_path.read_text())
    runtime_dir = local / "runtime" / "supervised_nested_training"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    paths_file = args.paths.resolve()
    bands_file = args.bands.resolve()

    print(
        f"Nested supervised training: {len(folds)} outer folds × "
        f"{len(architectures)} architectures × {len(seeds)} seeds = "
        f"{len(folds) * len(architectures) * len(seeds)} checkpoint runs",
        flush=True,
    )
    print(
        "Only Fit and Stop folds are opened; outer Test labels remain unopened.",
        flush=True,
    )

    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for position, outer in enumerate(folds):
            gpu_index = gpu_indices[position % len(gpu_indices)]
            future = executor.submit(
                run_outer_fold,
                outer,
                gpu_index,
                project,
                paths_file,
                bands_file,
                runtime_dir,
                base_config,
                config,
            )
            futures[future] = outer
        for future in as_completed(futures):
            outer, gpu_index, elapsed, status = future.result()
            results.append((outer, gpu_index, elapsed, status))
            print(
                f"[scheduler] outer fold {outer} on cuda:{gpu_index}: {status}; "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    fold_root = local / "reports" / config["per_fold_report_stage"]
    reports = local / "reports" / config["aggregate_report_stage"]
    reports.mkdir(parents=True, exist_ok=True)
    history = read_fold_outputs(
        fold_root, folds, "architecture_ablation_history.csv"
    )
    summary = read_fold_outputs(
        fold_root, folds, "architecture_ablation_summary.csv"
    )
    class_metrics = read_fold_outputs(
        fold_root, folds, "architecture_ablation_class_metrics.csv"
    )
    cube_metrics = read_fold_outputs(
        fold_root, folds, "architecture_ablation_cube_metrics.csv"
    )
    history.to_csv(reports / "nested_training_history.csv", index=False)
    summary.to_csv(reports / "nested_inner_validation_summary.csv", index=False)
    class_metrics.to_csv(
        reports / "nested_inner_validation_class_metrics.csv", index=False
    )
    cube_metrics.to_csv(
        reports / "nested_inner_validation_cube_metrics.csv", index=False
    )

    fold_summary = (
        summary.groupby(["outer_fold", "architecture", "display_name"])
        .agg(
            runs=("seed", "size"),
            inner_macro_f1_mean=("validation_macro_f1", "mean"),
            inner_macro_f1_sd=("validation_macro_f1", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            best_epoch_min=("best_epoch", "min"),
            best_epoch_max=("best_epoch", "max"),
        )
        .reset_index()
    )
    fold_summary.to_csv(
        reports / "nested_training_by_outer_fold.csv", index=False
    )

    class_aggregate = (
        class_metrics.groupby(["architecture", "class_name"])["f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(19, 5.5), constrained_layout=True
    )
    colors = {
        "center_spectrum_mlp": "#4C78A8",
        "spatial_average_cnn": "#F58518",
        "center_context_fusion": "#54A24B",
    }
    for architecture in architectures:
        selected = fold_summary[fold_summary.architecture == architecture]
        axes[0].errorbar(
            selected["outer_fold"],
            selected["inner_macro_f1_mean"],
            yerr=selected["inner_macro_f1_sd"],
            marker="o",
            capsize=3,
            color=colors[architecture],
            label=DISPLAY_NAMES[architecture],
        )
    axes[0].set_xticks(folds)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Outer evaluation fold")
    axes[0].set_ylabel("Inner-validation macro-F1")
    axes[0].set_title("Early-stopping performance only")
    axes[0].legend()

    positions = np.arange(len(architectures))
    width = 0.24
    for offset, class_name in enumerate(CLASS_ORDER):
        selected = class_aggregate[
            class_aggregate.class_name == class_name
        ].set_index("architecture")
        axes[1].bar(
            positions + (offset - 1) * width,
            selected.loc[architectures, "mean"],
            width,
            yerr=selected.loc[architectures, "std"],
            capsize=3,
            label=class_name.capitalize(),
        )
    axes[1].set_xticks(
        positions,
        [DISPLAY_NAMES[value] for value in architectures],
        rotation=15,
        ha="right",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Inner-validation class F1")
    axes[1].set_title("Checkpoint-selection diagnostics")
    axes[1].legend()

    epoch_table = fold_summary.pivot(
        index="outer_fold", columns="architecture", values="best_epoch_mean"
    )
    epoch_table[architectures].plot.bar(
        ax=axes[2], color=[colors[value] for value in architectures]
    )
    axes[2].set_xlabel("Outer evaluation fold")
    axes[2].set_ylabel("Mean selected epoch")
    axes[2].set_title("Early-stopping epochs")
    axes[2].tick_params(axis="x", rotation=0)
    axes[2].legend([DISPLAY_NAMES[value] for value in architectures])

    figure.suptitle(
        "Nested supervised checkpoint training\n"
        "Orange Stop folds used; red outer Test folds remain unopened",
        fontsize=15,
    )
    preview = reports / "nested_training_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)

    checkpoint_root = project / config["per_fold_output_dir"]
    checkpoint_hashes = {}
    for outer in folds:
        for architecture in architectures:
            for seed in seeds:
                path = (
                    checkpoint_root
                    / f"outer_fold{outer}"
                    / f"{architecture}_seed{seed}_best.pt"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                checkpoint_hashes[
                    f"outer{outer}:{architecture}:seed{seed}"
                ] = sha256(path)

    output_files = {
        "history": reports / "nested_training_history.csv",
        "summary": reports / "nested_inner_validation_summary.csv",
        "fold_summary": reports / "nested_training_by_outer_fold.csv",
        "class_metrics": reports / "nested_inner_validation_class_metrics.csv",
        "visual_qc": preview,
    }
    contract = {
        "status": "nested_supervised_checkpoints_frozen",
        "field": "Field 1",
        "field2_accessed": False,
        "outer_test_accessed": False,
        "synthetic_scientific_observations": False,
        "outer_folds": folds,
        "architectures": architectures,
        "model_seeds": seeds,
        "total_checkpoints": len(checkpoint_hashes),
        "checkpoint_hashes": checkpoint_hashes,
        "output_hashes": {
            name: sha256(path) for name, path in output_files.items()
        },
        "source_contract_hashes": {
            name: sha256(path) for name, path in required_contracts.items()
        },
        "configuration_sha256": sha256(args.config),
        "notes": [
            "Checkpoints were selected using inner-validation folds only.",
            "Inner-validation metrics are model-selection diagnostics, not outer-test results.",
            "Outer-test labels and Field 2 remained unopened.",
        ],
    }
    contract_path = contracts / "field1_nested_supervised_checkpoints_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Nested checkpoints frozen: {len(checkpoint_hashes)}")
    print(f"Reports: {reports}")
    print(f"Visual QC: {preview}")
    print(f"Contract: {contract_path}")
    print(
        "Training complete; outer-test labels and Field 2 were not accessed."
    )


if __name__ == "__main__":
    main()
