#!/usr/bin/env python
"""Run and aggregate the frozen balanced five-fold supervised benchmark on two GPUs."""

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
    path = report_dir / "architecture_ablation_summary.csv"
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    expected = {(architecture, seed) for architecture in architectures for seed in seeds}
    observed = {
        (str(row.architecture), int(row.seed))
        for row in frame[["architecture", "seed"]].itertuples(index=False)
    }
    return len(frame) == len(expected) and observed == expected


def write_fold_config(
    base: dict, destination: Path, fold: int, gpu_index: int, config: dict
) -> None:
    current = dict(base)
    current["heldout_fold"] = fold
    current["run_title"] = (
        f"Fold-{fold} frozen supervised architecture benchmark "
        "(balanced diagnostic subset)"
    )
    current["report_stage"] = (
        f"{config['per_fold_report_stage']}/fold{fold}"
    )
    current["output_dir"] = f"{config['per_fold_output_dir']}/fold{fold}"
    current["training"] = dict(base["training"])
    current["training"]["gpu_index"] = gpu_index
    destination.write_text(yaml.safe_dump(current, sort_keys=False))


def run_fold(
    fold: int,
    gpu_index: int,
    project: Path,
    paths_file: Path,
    bands_file: Path,
    runtime_dir: Path,
    base_config: dict,
    config: dict,
) -> tuple[int, int, float, str]:
    report_dir = (
        project / "metadata" / "local" / "reports"
        / config["per_fold_report_stage"] / f"fold{fold}"
    )
    architectures = list(config["architectures"])
    seeds = [int(seed) for seed in config["model_seeds"]]
    if bool(config.get("skip_completed_folds", True)) and fold_complete(
        report_dir, architectures, seeds
    ):
        return fold, gpu_index, 0.0, "already complete"

    runtime_config = runtime_dir / f"fold{fold}_gpu{gpu_index}.yaml"
    write_fold_config(base_config, runtime_config, fold, gpu_index, config)
    command = [
        sys.executable,
        str(project / "scripts" / "run_supervised_architecture_ablation.py"),
        "--paths", str(paths_file),
        "--config", str(runtime_config),
        "--bands", str(bands_file),
    ]
    started = time.monotonic()
    print(
        f"[scheduler] starting fold {fold} on cuda:{gpu_index}; "
        f"runtime config={runtime_config}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=project, check=False)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Fold {fold} failed on cuda:{gpu_index} with exit code "
            f"{completed.returncode}"
        )
    if not fold_complete(report_dir, architectures, seeds):
        raise RuntimeError(f"Fold {fold} finished without a complete summary")
    return fold, gpu_index, elapsed, "trained"


def read_fold_outputs(
    reports_root: Path, folds: list[int], filename: str
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        path = reports_root / f"fold{fold}" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "heldout_fold", fold)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--config",
        default=Path("configs/supervised_fivefold_balanced.yaml"),
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
    architecture_contract_path = (
        local / "contracts" / config["architecture_contract"]
    )
    if not architecture_contract_path.exists():
        raise FileNotFoundError(
            "Frozen architecture contract is required: "
            f"{architecture_contract_path}"
        )
    architecture_contract = yaml.safe_load(architecture_contract_path.read_text())
    if architecture_contract.get("status") != "supervised_architecture_candidates_frozen":
        raise ValueError("Architecture contract status is not frozen")
    if architecture_contract.get("field2_accessed") is not False:
        raise ValueError("Architecture contract does not preserve the Field 2 lock")

    folds = [int(value) for value in config["heldout_folds"]]
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError("The benchmark requires frozen folds [1, 2, 3, 4, 5]")
    architectures = list(config["architectures"])
    seeds = [int(seed) for seed in config["model_seeds"]]
    policy = architecture_contract["selection_policy"]
    contracted = [
        policy["primary_architecture"], *policy["baseline_architectures"]
    ]
    if set(contracted) != set(architectures):
        raise ValueError("Benchmark architectures differ from the frozen contract")
    if [int(value) for value in architecture_contract["frozen_model_seeds"]] != seeds:
        raise ValueError("Benchmark seeds differ from the frozen contract")

    gpu_indices = [int(value) for value in config["gpu_indices"]]
    if len(gpu_indices) != 2 or len(set(gpu_indices)) != 2:
        raise ValueError("Exactly two distinct GPU indices are required")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if max(gpu_indices) >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPUs {gpu_indices}, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )

    base_config_path = project / config["base_training_config"]
    base_config = yaml.safe_load(base_config_path.read_text())
    for key in [
        "architectures", "model_seeds", "samples_per_training_class",
        "samples_per_validation_class",
    ]:
        if key in config:
            base_config[key] = config[key]

    runtime_dir = local / "runtime" / "supervised_fivefold_balanced"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    paths_file = args.paths.resolve()
    bands_file = args.bands.resolve()

    print(
        f"Frozen balanced benchmark: {len(folds)} folds × "
        f"{len(architectures)} architectures × {len(seeds)} seeds = "
        f"{len(folds) * len(architectures) * len(seeds)} runs",
        flush=True,
    )
    print(
        f"Independent workers will use cuda:{gpu_indices[0]} and "
        f"cuda:{gpu_indices[1]}; completed folds are resumable.",
        flush=True,
    )

    results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for position, fold in enumerate(folds):
            gpu_index = gpu_indices[position % len(gpu_indices)]
            future = executor.submit(
                run_fold, fold, gpu_index, project, paths_file, bands_file,
                runtime_dir, base_config, config,
            )
            futures[future] = fold
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            fold, gpu_index, seconds, status = result
            print(
                f"[scheduler] fold {fold} on cuda:{gpu_index}: {status}; "
                f"elapsed={seconds / 60:.1f} min",
                flush=True,
            )

    per_fold_root = local / "reports" / config["per_fold_report_stage"]
    final_reports = local / "reports" / config["aggregate_report_stage"]
    final_reports.mkdir(parents=True, exist_ok=True)

    history = read_fold_outputs(
        per_fold_root, folds, "architecture_ablation_history.csv"
    )
    summary = read_fold_outputs(
        per_fold_root, folds, "architecture_ablation_summary.csv"
    )
    class_metrics = read_fold_outputs(
        per_fold_root, folds, "architecture_ablation_class_metrics.csv"
    )
    cube_metrics = read_fold_outputs(
        per_fold_root, folds, "architecture_ablation_cube_metrics.csv"
    )
    history.to_csv(final_reports / "fivefold_training_history.csv", index=False)
    summary.to_csv(final_reports / "fivefold_run_summary.csv", index=False)
    class_metrics.to_csv(final_reports / "fivefold_class_metrics.csv", index=False)
    cube_metrics.to_csv(final_reports / "fivefold_cube_metrics.csv", index=False)

    fold_summary = (
        summary.groupby(["heldout_fold", "architecture", "display_name"])
        .agg(
            runs=("seed", "size"),
            macro_f1_mean=("validation_macro_f1", "mean"),
            macro_f1_sd=("validation_macro_f1", "std"),
            chickpea_f1_mean=("chickpea_f1", "mean"),
            weed_f1_mean=("weed_f1", "mean"),
            soil_f1_mean=("soil_f1", "mean"),
            worst_cube_macro_f1_mean=("worst_cube_macro_f1", "mean"),
        )
        .reset_index()
    )
    fold_summary.to_csv(
        final_reports / "fivefold_architecture_by_fold.csv", index=False
    )
    aggregate = (
        summary.groupby(["architecture", "display_name"])
        .agg(
            runs=("seed", "size"),
            folds=("heldout_fold", "nunique"),
            macro_f1_mean=("validation_macro_f1", "mean"),
            macro_f1_sd=("validation_macro_f1", "std"),
            balanced_accuracy_mean=("validation_balanced_accuracy", "mean"),
            chickpea_f1_mean=("chickpea_f1", "mean"),
            soil_f1_mean=("soil_f1", "mean"),
            weed_f1_mean=("weed_f1", "mean"),
            worst_cube_macro_f1_mean=("worst_cube_macro_f1", "mean"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    aggregate.to_csv(
        final_reports / "fivefold_architecture_aggregate.csv", index=False
    )

    class_aggregate = (
        class_metrics.groupby(["architecture", "class_name"])["f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    class_aggregate.to_csv(
        final_reports / "fivefold_class_aggregate.csv", index=False
    )

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    colors = {
        "center_spectrum_mlp": "#4C78A8",
        "spatial_average_cnn": "#F58518",
        "center_context_fusion": "#54A24B",
    }
    for architecture in architectures:
        selected = fold_summary[fold_summary.architecture == architecture]
        axes[0].errorbar(
            selected["heldout_fold"], selected["macro_f1_mean"],
            yerr=selected["macro_f1_sd"], marker="o", capsize=3,
            color=colors[architecture], label=DISPLAY_NAMES[architecture],
        )
    axes[0].set_xticks(folds)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Held-out spatial fold")
    axes[0].set_ylabel("Macro-F1 (three-seed mean ± SD)")
    axes[0].set_title("Spatial-fold generalization")
    axes[0].legend()

    positions = np.arange(len(architectures))
    width = 0.24
    for offset, class_name in enumerate(CLASS_ORDER):
        selected = class_aggregate[class_aggregate.class_name == class_name].set_index(
            "architecture"
        )
        axes[1].bar(
            positions + (offset - 1) * width,
            selected.loc[architectures, "mean"], width,
            yerr=selected.loc[architectures, "std"], capsize=3,
            label=class_name.capitalize(),
        )
    axes[1].set_xticks(
        positions, [DISPLAY_NAMES[value] for value in architectures],
        rotation=15, ha="right",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Class F1 across fold × seed runs")
    axes[1].set_title("Class-specific stability")
    axes[1].legend()

    values = [
        summary.loc[
            summary.architecture == architecture, "validation_macro_f1"
        ].to_numpy()
        for architecture in architectures
    ]
    axes[2].boxplot(values, tick_labels=[DISPLAY_NAMES[x] for x in architectures])
    for position, architecture_values in enumerate(values, start=1):
        jitter = np.linspace(-0.08, 0.08, len(architecture_values))
        axes[2].scatter(
            position + jitter, architecture_values, s=22,
            color=colors[architectures[position - 1]], alpha=0.75,
        )
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Validation macro-F1")
    axes[2].set_title("All 15 runs per architecture")

    figure.suptitle(
        "Field 1 frozen five-fold supervised benchmark\n"
        "Balanced observed diagnostic subsets; exhaustive evaluation remains separate",
        fontsize=15,
    )
    preview = final_reports / "supervised_fivefold_balanced_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)

    output_files = {
        "run_summary": final_reports / "fivefold_run_summary.csv",
        "fold_summary": final_reports / "fivefold_architecture_by_fold.csv",
        "aggregate": final_reports / "fivefold_architecture_aggregate.csv",
        "class_aggregate": final_reports / "fivefold_class_aggregate.csv",
        "visual_qc": preview,
    }
    benchmark_contract = {
        "status": "balanced_fivefold_supervised_benchmark_complete",
        "field": "Field 1",
        "field2_accessed": False,
        "synthetic_scientific_observations": False,
        "evaluation_scope": (
            "Balanced observed validation subsets for architecture stability. "
            "This is not exhaustive held-out-mask evaluation."
        ),
        "heldout_folds": folds,
        "architectures": architectures,
        "model_seeds": seeds,
        "total_completed_runs": int(len(summary)),
        "architecture_contract_sha256": sha256(architecture_contract_path),
        "benchmark_config_sha256": sha256(args.config),
        "output_hashes": {
            name: sha256(path) for name, path in output_files.items()
        },
    }
    benchmark_contract_path = (
        local / "contracts" / "field1_supervised_fivefold_balanced_contract.yaml"
    )
    benchmark_contract_path.write_text(
        yaml.safe_dump(benchmark_contract, sort_keys=False)
    )

    print("\nFive-fold aggregate:", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print(f"Reports: {final_reports}", flush=True)
    print(f"Visual QC: {preview}", flush=True)
    print(f"Contract: {benchmark_contract_path}", flush=True)
    print(
        "Balanced five-fold benchmark complete; exhaustive held-out-mask "
        "evaluation and Field 2 access did not occur.",
        flush=True,
    )


if __name__ == "__main__":
    main()
