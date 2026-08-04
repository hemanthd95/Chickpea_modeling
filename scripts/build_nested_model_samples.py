#!/usr/bin/env python
"""Freeze exact nested fitting and inner-validation samples from observed candidates."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    return series if series.dtype == bool else series.astype(str).str.lower().eq("true")


def balanced_select(
    frame: pd.DataFrame, count_per_class: int, seed: int
) -> pd.DataFrame:
    parts = []
    for class_id in (0, 1, 2):
        group = frame[frame["class_id"] == class_id]
        if len(group) < count_per_class:
            raise ValueError(
                f"Class {class_id} has {len(group):,} eligible samples; "
                f"requested {count_per_class:,}"
            )
        parts.append(
            group.sample(
                n=count_per_class,
                random_state=seed + class_id,
                replace=False,
            )
        )
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=seed + 100)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        default=Path("configs/supervised_primary_evaluation.yaml"),
        type=Path,
    )
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    protocol_config = yaml.safe_load(args.protocol.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reports = local / "reports" / "nested_sampling"
    reports.mkdir(parents=True, exist_ok=True)

    base_index_path = contracts / "field1_training_candidate_centers.csv"
    base_contract_path = contracts / "field1_training_candidate_contract.yaml"
    protocol_path = contracts / protocol_config["contract_filename"]
    roles_path = contracts / protocol_config["role_table_filename"]
    normalization_path = contracts / "field1_nested_normalization.csv"
    normalization_contract_path = (
        contracts / "field1_nested_normalization_contract.yaml"
    )
    for required in (
        base_index_path,
        base_contract_path,
        protocol_path,
        roles_path,
        normalization_path,
        normalization_contract_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    base_contract = yaml.safe_load(base_contract_path.read_text())
    if sha256(base_index_path) != base_contract["center_index_sha256"]:
        raise ValueError("Base candidate-center hash mismatch")
    if base_contract.get("require_fully_observed_patch") is not True:
        raise ValueError("Base candidate pool does not require fully observed patches")

    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("status") != "supervised_primary_evaluation_protocol_frozen":
        raise ValueError("Primary evaluation protocol is not frozen")
    if sha256(args.protocol) != protocol["configuration_sha256"]:
        raise ValueError(
            "Primary evaluation configuration differs from its frozen contract"
        )
    if sha256(roles_path) != protocol["nested_role_table_sha256"]:
        raise ValueError("Nested role-table hash mismatch")

    normalization_contract = yaml.safe_load(
        normalization_contract_path.read_text()
    )
    if (
        normalization_contract.get("status")
        != "frozen_nested_training_only_normalization"
    ):
        raise ValueError("Nested normalization is not frozen")
    if (
        sha256(normalization_path)
        != normalization_contract["normalization_csv_sha256"]
    ):
        raise ValueError("Nested-normalization hash mismatch")

    centers = pd.read_csv(base_index_path)
    roles = pd.read_csv(roles_path)
    training = protocol["training"]
    training_count = int(training["samples_per_training_class"])
    validation_count = int(training["samples_per_inner_validation_class"])
    selection_seed = int(protocol_config.get("sample_selection_seed", 42))
    outer_folds = [int(value) for value in protocol["outer_folds"]]

    selected_frames = []
    summary_rows = []
    cube_rows = []
    for outer in outer_folds:
        outer_roles = roles[roles["outer_evaluation_fold"] == outer]
        inner = int(
            outer_roles.loc[
                outer_roles["role"] == "inner_validation", "spatial_fold"
            ].iloc[0]
        )
        train_folds = sorted(
            outer_roles.loc[outer_roles["role"] == "train", "spatial_fold"]
            .astype(int)
            .tolist()
        )

        safe_outer = as_bool(centers[f"train_eligible_holdout_{outer}"])
        safe_inner = as_bool(centers[f"train_eligible_holdout_{inner}"])
        training_pool = centers[
            centers["fold"].isin(train_folds) & safe_outer & safe_inner
        ].copy()
        inner_pool = centers[
            as_bool(centers[f"validation_eligible_fold_{inner}"])
        ].copy()

        if set(training_pool["fold"].unique()) != set(train_folds):
            raise ValueError(f"Outer {outer}: one or more fit folds have no samples")
        if set(inner_pool["fold"].unique()) != {inner}:
            raise ValueError(f"Outer {outer}: inner pool contains incorrect folds")

        selected_train = balanced_select(
            training_pool, training_count, selection_seed + outer * 10000
        )
        selected_inner = balanced_select(
            inner_pool, validation_count, selection_seed + outer * 10000 + 5000
        )
        selected_train.insert(0, "nested_role", "train")
        selected_inner.insert(0, "nested_role", "inner_validation")
        selected_train.insert(0, "inner_validation_fold", inner)
        selected_inner.insert(0, "inner_validation_fold", inner)
        selected_train.insert(0, "outer_fold", outer)
        selected_inner.insert(0, "outer_fold", outer)
        selected_frames.extend([selected_train, selected_inner])

        for class_id, class_name in (
            training_pool[["class_id", "class_name"]]
            .drop_duplicates()
            .sort_values("class_id")
            .itertuples(index=False)
        ):
            train_class = training_pool["class_id"] == class_id
            inner_class = inner_pool["class_id"] == class_id
            summary_rows.append(
                {
                    "outer_fold": outer,
                    "inner_validation_fold": inner,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "training_pool": int(train_class.sum()),
                    "selected_training": training_count,
                    "inner_validation_pool": int(inner_class.sum()),
                    "selected_inner_validation": validation_count,
                }
            )
        for role_name, frame in (
            ("training_pool", training_pool),
            ("inner_validation_pool", inner_pool),
        ):
            for cube_id, group in frame.groupby("cube_id"):
                cube_rows.append(
                    {
                        "outer_fold": outer,
                        "inner_validation_fold": inner,
                        "role": role_name,
                        "cube_id": cube_id,
                        "eligible_centers": len(group),
                        "classes_present": int(group["class_id"].nunique()),
                    }
                )

    selected = pd.concat(selected_frames, ignore_index=True)
    expected_rows = len(outer_folds) * 3 * (training_count + validation_count)
    if len(selected) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} selected rows, observed {len(selected):,}"
        )
    duplicate_keys = [
        "outer_fold", "nested_role", "sample_id"
    ]
    if selected.duplicated(duplicate_keys).any():
        raise ValueError("Duplicate sample within a nested role")

    selected_path = contracts / "field1_nested_model_samples.csv"
    summary_path = reports / "nested_sample_summary.csv"
    cube_path = reports / "nested_sample_cube_coverage.csv"
    selected.to_csv(selected_path, index=False)
    summary = pd.DataFrame(summary_rows)
    cube_coverage = pd.DataFrame(cube_rows)
    summary.to_csv(summary_path, index=False)
    cube_coverage.to_csv(cube_path, index=False)

    figure, axes = plt.subplots(
        1, 2, figsize=(15, 5.5), constrained_layout=True
    )
    class_order = ["soil", "chickpea", "weed"]
    train_pivot = summary.pivot(
        index="outer_fold", columns="class_name", values="training_pool"
    )[class_order]
    inner_pivot = summary.pivot(
        index="outer_fold", columns="class_name", values="inner_validation_pool"
    )[class_order]
    colors = {"soil": "#B07D42", "chickpea": "#2E8B57", "weed": "#7A4EAB"}
    train_pivot.plot.bar(
        ax=axes[0], color=[colors[value] for value in class_order]
    )
    inner_pivot.plot.bar(
        ax=axes[1], color=[colors[value] for value in class_order]
    )
    axes[0].axhline(training_count, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(validation_count, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Eligible fitting pool")
    axes[1].set_title("Eligible inner-validation pool")
    for axis in axes:
        axis.set_xlabel("Outer evaluation fold")
        axis.set_ylabel("Observed candidate centres")
        axis.tick_params(axis="x", rotation=0)
        axis.legend(title="Class")
    figure.suptitle(
        "Nested observed sample availability\n"
        "Dashed lines show frozen per-class selection counts",
        fontsize=14,
    )
    preview = reports / "nested_sample_overview.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)

    contract = {
        "status": "frozen_nested_model_samples",
        "field": "Field 1",
        "field2_accessed": False,
        "synthetic_scientific_observations": False,
        "outer_folds": outer_folds,
        "model_seeds": [int(value) for value in protocol["model_seeds"]],
        "samples_per_training_class": training_count,
        "samples_per_inner_validation_class": validation_count,
        "total_selected_rows": int(len(selected)),
        "sample_selection_seed": selection_seed,
        "require_fully_observed_patch": True,
        "sample_csv_sha256": sha256(selected_path),
        "summary_sha256": sha256(summary_path),
        "cube_coverage_sha256": sha256(cube_path),
        "visual_sha256": sha256(preview),
        "source_hashes": {
            "base_candidate_index": sha256(base_index_path),
            "base_candidate_contract": sha256(base_contract_path),
            "primary_protocol": sha256(protocol_path),
            "nested_roles": sha256(roles_path),
            "nested_normalization": sha256(normalization_path),
            "nested_normalization_contract": sha256(normalization_contract_path),
        },
        "notes": [
            "All selected samples are observed Field 1 pixels; none are synthetic.",
            "Training and inner-validation folds are disjoint for every outer fold.",
            "Outer-test labels are absent from this model-selection sample file.",
            "Primary outer evaluation remains exhaustive and is not defined by this balanced sample file.",
        ],
    }
    contract_path = contracts / "field1_nested_model_samples_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Nested model sample rows frozen: {len(selected):,}")
    print(
        summary[
            [
                "outer_fold", "inner_validation_fold", "class_name",
                "training_pool", "selected_training",
                "inner_validation_pool", "selected_inner_validation",
            ]
        ].to_string(index=False)
    )
    print(f"Contract: {contract_path}")
    print(f"Visual QC: {preview}")
    print(
        "Outer-test labels were not selected; Field 2 was not accessed."
    )


if __name__ == "__main__":
    main()
