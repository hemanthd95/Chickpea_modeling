#!/usr/bin/env python
"""Analyze convergence-diagnostic errors against observed patch-label purity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from chickpea_ssl.data import authoritative_class_map, load_records


CLASS_NAMES = {0: "soil", 1: "chickpea", 2: "weed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--config", default=Path("configs/supervised_fold1_convergence.yaml"), type=Path
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    report_stage = Path(config["report_stage"])
    if report_stage.is_absolute() or ".." in report_stage.parts:
        raise ValueError("Unsafe report_stage")
    reports = local / "reports" / report_stage
    predictions_path = reports / "validation_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    predictions = pd.read_csv(predictions_path)
    manifest = local / "authoritative_manifest.csv"
    records = {record.cube_id: record for record in load_records(args.paths, manifest)}
    radius_by_size: dict[int, int] = {}
    enriched_parts: list[pd.DataFrame] = []

    for cube_number, (cube_id, group) in enumerate(predictions.groupby("cube_id"), start=1):
        print(
            f"Purity audit cube {cube_number}/{predictions['cube_id'].nunique()}: "
            f"{cube_id} ({len(group):,} patches)",
            flush=True,
        )
        labels = authoritative_class_map(records[str(cube_id)])
        rows: list[dict[str, float]] = []
        for sample in group.itertuples():
            size = int(sample.patch_size_pixels)
            radius = radius_by_size.setdefault(size, size // 2)
            window = labels[
                int(sample.row) - radius:int(sample.row) + radius + 1,
                int(sample.column) - radius:int(sample.column) + radius + 1,
            ]
            if window.shape != (size, size):
                raise ValueError(f"Patch shape mismatch for {sample.sample_id}")
            true_class = int(sample.true_class_id)
            if int(labels[int(sample.row), int(sample.column)]) != true_class:
                raise ValueError(f"Center-label mismatch for {sample.sample_id}")
            labeled = window >= 0
            labeled_count = int(labeled.sum())
            same_count = int((window == true_class).sum())
            rows.append({
                "center_class_fraction_all_pixels": same_count / window.size,
                "center_class_fraction_labeled_pixels": (
                    same_count / labeled_count if labeled_count else np.nan
                ),
                "unlabeled_fraction": float((~labeled).mean()),
                "soil_fraction": float((window == 0).mean()),
                "chickpea_fraction": float((window == 1).mean()),
                "weed_fraction": float((window == 2).mean()),
            })
        enriched_parts.append(pd.concat(
            [group.reset_index(drop=True), pd.DataFrame(rows)], axis=1
        ))
    enriched = pd.concat(enriched_parts, ignore_index=True)
    if len(enriched) != len(predictions):
        raise ValueError("Purity audit changed the prediction count")
    enriched["purity_bin"] = pd.cut(
        enriched["center_class_fraction_all_pixels"],
        bins=[-0.001, 0.25, 0.50, 0.75, 0.90, 1.001],
        labels=["≤0.25", "0.25–0.50", "0.50–0.75", "0.75–0.90", ">0.90"],
        include_lowest=True,
    )
    enriched.to_csv(reports / "validation_predictions_with_patch_purity.csv", index=False)
    purity_summary = (
        enriched.groupby(["class_name", "purity_bin"], observed=True)
        .agg(
            patches=("sample_id", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("prediction_confidence", "mean"),
            mean_unlabeled_fraction=("unlabeled_fraction", "mean"),
        )
        .reset_index()
    )
    purity_summary.to_csv(reports / "accuracy_by_patch_purity.csv", index=False)
    cube_summary = (
        enriched.groupby(["cube_id", "class_name"])
        .agg(
            patches=("sample_id", "size"),
            accuracy=("correct", "mean"),
            mean_center_class_fraction=("center_class_fraction_all_pixels", "mean"),
            mean_confidence=("prediction_confidence", "mean"),
        )
        .reset_index()
    )
    cube_summary.to_csv(reports / "accuracy_by_cube_and_class.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    sns.lineplot(
        data=purity_summary, x="purity_bin", y="accuracy", hue="class_name",
        marker="o", ax=axes[0],
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Center-class fraction in 15×15 patch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy versus patch-label purity")
    sns.boxplot(
        data=enriched, x="class_name", y="center_class_fraction_all_pixels",
        hue="class_name", legend=False, ax=axes[1],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("True center class")
    axes[1].set_ylabel("Center-class fraction")
    axes[1].set_title("Observed neighborhood purity")
    cube_plot = cube_summary.groupby("cube_id").agg(
        accuracy=("accuracy", "mean"),
        mean_purity=("mean_center_class_fraction", "mean"),
    ).reset_index()
    sns.scatterplot(
        data=cube_plot, x="mean_purity", y="accuracy", s=70, ax=axes[2]
    )
    for row in cube_plot.itertuples():
        axes[2].annotate(str(row.cube_id).replace("field1_", ""), (row.mean_purity, row.accuracy),
                         fontsize=7, alpha=0.75)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Mean center-class fraction")
    axes[2].set_ylabel("Mean class accuracy")
    axes[2].set_title("Cube-level purity and performance")
    figure.suptitle("Fold-1 supervised error analysis using observed labels", fontsize=15)
    preview = reports / "patch_purity_error_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)
    print(purity_summary.to_string(index=False), flush=True)
    print(f"Detailed predictions: {reports / 'validation_predictions_with_patch_purity.csv'}")
    print(f"Visual QC: {preview}")
    print("Field 2 was not accessed; no synthetic scientific observations were used.")


if __name__ == "__main__":
    main()
