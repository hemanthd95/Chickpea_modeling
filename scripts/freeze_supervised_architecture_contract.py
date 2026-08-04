#!/usr/bin/env python
"""Freeze the reviewed Fold-1 supervised architecture decision as a local contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--config",
        default=Path("configs/supervised_architecture_selection.yaml"),
        type=Path,
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    reports = local / "reports" / Path(config["source_report_stage"])
    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)

    required = {
        "aggregate": reports / "architecture_ablation_aggregate.csv",
        "summary": reports / "architecture_ablation_summary.csv",
        "class_metrics": reports / "architecture_ablation_class_metrics.csv",
        "cube_metrics": reports / "architecture_ablation_cube_metrics.csv",
        "visual_qc": reports / "supervised_architecture_ablation_overview.png",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ablation outputs:\n" + "\n".join(missing))

    aggregate = pd.read_csv(required["aggregate"])
    summary = pd.read_csv(required["summary"])
    class_metrics = pd.read_csv(required["class_metrics"])
    cube_metrics = pd.read_csv(required["cube_metrics"])

    architectures = list(config["architectures"])
    seeds = [int(seed) for seed in config["model_seeds"]]
    expected_runs = {(architecture, seed) for architecture in architectures for seed in seeds}
    observed_runs = {
        (str(row.architecture), int(row.seed))
        for row in summary[["architecture", "seed"]].itertuples(index=False)
    }
    if observed_runs != expected_runs:
        raise ValueError(
            f"Expected architecture/seed runs {sorted(expected_runs)}, "
            f"observed {sorted(observed_runs)}"
        )
    if len(summary) != len(expected_runs):
        raise ValueError("Architecture summary contains duplicate runs")
    if set(aggregate["architecture"]) != set(architectures):
        raise ValueError("Aggregate architecture set does not match frozen candidates")
    if set(class_metrics["class_name"]) != {"soil", "chickpea", "weed"}:
        raise ValueError("Class-metric output is incomplete")
    if cube_metrics["cube_id"].nunique() < int(config["minimum_validation_cubes"]):
        raise ValueError("Too few validation cubes for the architecture decision")

    primary = str(config["primary_architecture"])
    baselines = [str(value) for value in config["baseline_architectures"]]
    indexed = aggregate.set_index("architecture")
    if indexed.loc[primary, "macro_f1_mean"] != indexed["macro_f1_mean"].max():
        raise ValueError("Declared primary architecture is not the mean macro-F1 leader")

    comparator = str(config["primary_comparator"])
    paired = summary.pivot(
        index="seed", columns="architecture", values="validation_macro_f1"
    )
    paired_delta = paired[primary] - paired[comparator]
    if not (paired_delta > 0).all():
        raise ValueError("Primary architecture did not beat its comparator for every seed")

    class_summary = (
        class_metrics.groupby(["architecture", "class_name"])["f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    cube_summary = (
        cube_metrics.groupby(["architecture", "cube_id"])["macro_f1"]
        .mean()
        .reset_index()
    )
    cube_winners = (
        cube_summary.pivot(index="cube_id", columns="architecture", values="macro_f1")
        .idxmax(axis=1)
        .value_counts()
        .to_dict()
    )

    contract = {
        "status": "supervised_architecture_candidates_frozen",
        "field": "Field 1",
        "diagnostic_fold_used_for_selection": int(config["diagnostic_fold"]),
        "field2_accessed": False,
        "synthetic_scientific_observations": False,
        "selection_policy": {
            "primary_architecture": primary,
            "primary_role": "primary supervised spectral-spatial model",
            "baseline_architectures": baselines,
            "selection_metric": "three-seed mean validation macro-F1",
            "primary_comparator": comparator,
            "paired_seed_improvement_required": True,
            "warning": (
                "Fold 1 is architecture development only. Final scientific claims "
                "require all frozen spatial folds; architecture switching after the "
                "five-fold results is prohibited."
            ),
        },
        "frozen_model_seeds": seeds,
        "architectures": architectures,
        "descriptive_fold1_results": {
            row.architecture: {
                "macro_f1_mean": float(row.macro_f1_mean),
                "macro_f1_sd": float(row.macro_f1_sd),
                "chickpea_f1_mean": float(row.chickpea_f1_mean),
                "worst_cube_macro_f1_mean": float(row.worst_cube_macro_f1_mean),
                "parameter_count": int(row.parameter_count),
            }
            for row in aggregate.itertuples(index=False)
        },
        "primary_vs_comparator": {
            "absolute_macro_f1_improvement": float(
                indexed.loc[primary, "macro_f1_mean"]
                - indexed.loc[comparator, "macro_f1_mean"]
            ),
            "relative_macro_f1_improvement_percent": float(
                100
                * (
                    indexed.loc[primary, "macro_f1_mean"]
                    - indexed.loc[comparator, "macro_f1_mean"]
                )
                / indexed.loc[comparator, "macro_f1_mean"]
            ),
            "paired_seed_deltas": {
                int(seed): float(value) for seed, value in paired_delta.items()
            },
        },
        "cube_wins_by_architecture": {
            architecture: int(cube_winners.get(architecture, 0))
            for architecture in architectures
        },
        "class_f1_mean_and_sd": {
            f"{row.architecture}:{row.class_name}": {
                "mean": float(row["mean"]),
                "sd": float(row["std"]),
            }
            for _, row in class_summary.iterrows()
        },
        "source_hashes": {
            name: sha256(path) for name, path in required.items()
        },
        "configuration_hash": sha256(args.config),
    }

    contract_path = contracts / str(config["contract_filename"])
    visual_path = contracts / str(config["visual_filename"])
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(required["visual_qc"], visual_path)
    contract["frozen_visual_sha256"] = sha256(visual_path)
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Primary architecture: {primary}")
    print(f"Frozen baselines: {', '.join(baselines)}")
    print(
        "Primary macro-F1 improvement over comparator: "
        f"{contract['primary_vs_comparator']['absolute_macro_f1_improvement']:+.6f} "
        f"({contract['primary_vs_comparator']['relative_macro_f1_improvement_percent']:+.2f}%)"
    )
    print(f"Architecture/seed runs verified: {len(summary)}")
    print(f"Validation cubes verified: {cube_metrics['cube_id'].nunique()}")
    print(f"Contract: {contract_path}")
    print(f"Frozen visual QC: {visual_path}")
    print("Field 2 was not accessed; no five-fold model training occurred.")


if __name__ == "__main__":
    main()
