#!/usr/bin/env python
"""Audit frozen supervised outer results with paired spatial-group uncertainty."""

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
import yaml


DISPLAY_NAMES = {
    "center_spectrum_mlp": "Center spectrum",
    "spatial_average_cnn": "Spatial-average CNN",
    "center_context_fusion": "Center + context",
}
METRICS = ["balanced_accuracy", "macro_f1", "soil_f1", "chickpea_f1", "weed_f1"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    support = matrix.sum(1); predicted = matrix.sum(0); diagonal = np.diag(matrix)
    precision = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(3), where=(precision + recall) > 0)
    return {
        "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
        "soil_f1": float(f1[0]), "chickpea_f1": float(f1[1]), "weed_f1": float(f1[2]),
    }


def matrix_from_rows(frame: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.float64)
    for row in frame.itertuples():
        matrix[int(row.true_class_id), int(row.predicted_class_id)] += float(row.count)
    return matrix


def group_matrices(counts: pd.DataFrame, architecture: str) -> tuple[list[tuple[int, str]], np.ndarray]:
    selected = counts[(counts.architecture == architecture) & (counts.level == "spatial_group")]
    keys, matrices = [], []
    for key, frame in selected.groupby(["outer_fold", "unit_id"], sort=True):
        matrix = matrix_from_rows(frame)
        keys.append((int(key[0]), str(key[1])))
        matrices.append(matrix / matrix.sum())
    return keys, np.stack(matrices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/outer_inference.yaml"), type=Path)
    parser.add_argument("--protocol", default=Path("configs/supervised_primary_evaluation.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text()); config = yaml.safe_load(args.config.read_text())
    protocol = yaml.safe_load(args.protocol.read_text()); project = Path(paths["project_root"])
    local = project / "metadata" / "local"; contracts = local / "contracts"
    reports = local / "reports" / "models" / "supervised_outer_evaluation"
    evaluation_contract_path = contracts / "field1_supervised_outer_evaluation_contract.yaml"
    evaluation_contract = yaml.safe_load(evaluation_contract_path.read_text())
    if evaluation_contract.get("status") != "field1_supervised_outer_evaluation_complete":
        raise ValueError("Supervised outer evaluation is not complete")
    if evaluation_contract.get("field2_accessed") is not False:
        raise ValueError("Evaluation contract violates Field 2 lock")
    counts_path = reports / "outer_confusion_counts.csv"
    groups_path = reports / "spatial_group_equal_weight_metrics.csv"
    folds_path = reports / "fold_metrics.csv"
    for path in (counts_path, groups_path, folds_path):
        if sha256(path) != evaluation_contract["output_hashes"][path.name]:
            raise ValueError(f"Frozen output hash mismatch: {path.name}")

    architectures = list(protocol["architectures"]); primary = str(protocol["primary_architecture"])
    comparators = [value for value in architectures if value != primary]
    counts = pd.read_csv(counts_path); groups = pd.read_csv(groups_path); folds = pd.read_csv(folds_path)
    key_by_architecture, matrices_by_architecture = {}, {}
    for architecture in architectures:
        keys, matrices = group_matrices(counts, architecture)
        key_by_architecture[architecture] = keys; matrices_by_architecture[architecture] = matrices
    if any(key_by_architecture[value] != key_by_architecture[primary] for value in comparators):
        raise ValueError("Architecture spatial-group keys are not exactly paired")
    group_count = len(key_by_architecture[primary])
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    bootstrap_indices = rng.integers(0, group_count, size=(replicates, group_count))

    point = {
        architecture: metrics_from_confusion(matrices_by_architecture[architecture].sum(0))
        for architecture in architectures
    }
    rows = []
    for comparator in comparators:
        differences = {metric: np.empty(replicates) for metric in METRICS}
        for index, sample in enumerate(bootstrap_indices):
            primary_metrics = metrics_from_confusion(matrices_by_architecture[primary][sample].sum(0))
            comparator_metrics = metrics_from_confusion(matrices_by_architecture[comparator][sample].sum(0))
            for metric in METRICS:
                differences[metric][index] = primary_metrics[metric] - comparator_metrics[metric]
        for metric in METRICS:
            values = differences[metric]
            rows.append({
                "primary_architecture": primary,
                "comparator_architecture": comparator,
                "metric": metric,
                "primary_point_estimate": point[primary][metric],
                "comparator_point_estimate": point[comparator][metric],
                "paired_difference": point[primary][metric] - point[comparator][metric],
                "paired_difference_lower_95": float(np.quantile(values, .025)),
                "paired_difference_upper_95": float(np.quantile(values, .975)),
                "bootstrap_probability_primary_greater": float((values > 0).mean()),
                "spatial_groups": group_count,
                "bootstrap_replicates": replicates,
            })
    paired = pd.DataFrame(rows)
    paired_path = reports / "architecture_paired_group_bootstrap.csv"
    paired.to_csv(paired_path, index=False)

    fold_rows = []
    primary_folds = folds[folds.architecture == primary].set_index("outer_fold")
    for comparator in comparators:
        comparator_folds = folds[folds.architecture == comparator].set_index("outer_fold")
        for outer_fold in sorted(primary_folds.index):
            fold_rows.append({
                "outer_fold": int(outer_fold), "primary_architecture": primary,
                "comparator_architecture": comparator,
                "primary_macro_f1": float(primary_folds.loc[outer_fold, "macro_f1"]),
                "comparator_macro_f1": float(comparator_folds.loc[outer_fold, "macro_f1"]),
                "paired_macro_f1_difference": float(primary_folds.loc[outer_fold, "macro_f1"] - comparator_folds.loc[outer_fold, "macro_f1"]),
            })
    fold_pairs = pd.DataFrame(fold_rows)
    fold_path = reports / "architecture_paired_fold_differences.csv"
    fold_pairs.to_csv(fold_path, index=False)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    metric_labels = ["Balanced\naccuracy", "Macro-F1", "Soil F1", "Chickpea F1", "Weed F1"]
    x = np.arange(len(METRICS)); width = .25
    for offset, architecture in enumerate(architectures):
        axes[0].bar(x + (offset-1)*width, [point[architecture][m] for m in METRICS], width,
                    label=DISPLAY_NAMES[architecture])
    axes[0].set_xticks(x, metric_labels); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Equal-spatial-group metric"); axes[0].set_title("Frozen outer performance")
    axes[0].legend(fontsize=8)
    positions = np.arange(len(comparators) * len(METRICS)); labels = []; y = []; lower = []; upper = []
    for comparator in comparators:
        selected = paired[paired.comparator_architecture == comparator].set_index("metric").loc[METRICS]
        for metric, row in selected.iterrows():
            labels.append(f"{DISPLAY_NAMES[comparator]}\n{metric.replace('_', ' ')}")
            y.append(row.paired_difference); lower.append(row.paired_difference-row.paired_difference_lower_95)
            upper.append(row.paired_difference_upper_95-row.paired_difference)
    axes[1].errorbar(positions, y, yerr=[lower, upper], fmt="o", capsize=4, color="#2C7FB8")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
    axes[1].set_ylabel("Center + context minus comparator"); axes[1].set_title("Paired 95% group-bootstrap intervals")
    for comparator in comparators:
        selected = fold_pairs[fold_pairs.comparator_architecture == comparator]
        axes[2].plot(selected.outer_fold, selected.paired_macro_f1_difference, marker="o", label=DISPLAY_NAMES[comparator])
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1); axes[2].set_xticks(sorted(folds.outer_fold.unique()))
    axes[2].set_xlabel("Outer fold"); axes[2].set_ylabel("Paired macro-F1 difference")
    axes[2].set_title("Geographic consistency of fusion gain"); axes[2].legend()
    figure.suptitle("Field 1 supervised architecture inference audit", fontsize=15)
    preview = reports / "supervised_outer_architecture_comparison.png"
    figure.savefig(preview, dpi=200); plt.close(figure)

    output_contract = {
        "status": "supervised_outer_architecture_comparison_complete",
        "field": "Field 1", "field2_accessed": False,
        "synthetic_scientific_observations": False,
        "primary_architecture": primary, "comparators": comparators,
        "paired_cluster_unit": "frozen_5m_spatial_group",
        "paired_spatial_groups": group_count,
        "bootstrap_replicates": replicates,
        "source_evaluation_contract_sha256": sha256(evaluation_contract_path),
        "output_hashes": {paired_path.name: sha256(paired_path), fold_path.name: sha256(fold_path), preview.name: sha256(preview)},
        "interpretation_rule": "A paired improvement is supported when its 95% interval excludes zero.",
    }
    contract_path = contracts / "field1_supervised_architecture_comparison_contract.yaml"
    contract_path.write_text(yaml.safe_dump(output_contract, sort_keys=False))
    print("Paired equal-spatial-group improvements for center + context:")
    print(paired[["comparator_architecture", "metric", "paired_difference",
                  "paired_difference_lower_95", "paired_difference_upper_95",
                  "bootstrap_probability_primary_greater"]].to_string(index=False))
    print(f"Fold comparison: {fold_path}"); print(f"Visual QC: {preview}"); print(f"Contract: {contract_path}")
    print("Paired audit complete; no new inference or Field 2 access occurred.")


if __name__ == "__main__": main()
