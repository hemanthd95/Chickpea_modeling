#!/usr/bin/env python
"""Freeze the leakage-safe nested spatial protocol for primary supervised evaluation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import yaml


COLORS = {
    "train": "#4C78A8",
    "inner_validation": "#F58518",
    "outer_test": "#E45756",
}


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
        default=Path("configs/supervised_primary_evaluation.yaml"),
        type=Path,
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    paths = yaml.safe_load(args.paths.read_text())
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)

    if config.get("field2_locked") is not True:
        raise ValueError("Field 2 must remain locked")
    outer_folds = [int(value) for value in config["outer_folds"]]
    if outer_folds != [1, 2, 3, 4, 5]:
        raise ValueError("Outer folds must be the frozen folds [1, 2, 3, 4, 5]")
    inner_mapping = {
        int(key): int(value)
        for key, value in config["inner_validation_by_outer_fold"].items()
    }
    if set(inner_mapping) != set(outer_folds):
        raise ValueError("Every outer fold requires one inner validation fold")
    for outer, inner in inner_mapping.items():
        if inner == outer or inner not in outer_folds:
            raise ValueError(f"Invalid inner fold {inner} for outer fold {outer}")

    source_paths = {
        name: contracts / filename
        for name, filename in config["source_contracts"].items()
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source contracts:\n" + "\n".join(missing))
    source_contracts = {
        name: yaml.safe_load(path.read_text())
        for name, path in source_paths.items()
    }
    architecture = source_contracts["architecture_selection"]
    if architecture.get("status") != "supervised_architecture_candidates_frozen":
        raise ValueError("Architecture-selection contract is not frozen")
    if architecture.get("field2_accessed") is not False:
        raise ValueError("Architecture-selection contract violated Field 2 lock")
    development = source_contracts["balanced_development_benchmark"]
    if development.get("status") != "balanced_fivefold_supervised_benchmark_complete":
        raise ValueError("Balanced five-fold development benchmark is incomplete")
    if development.get("field2_accessed") is not False:
        raise ValueError("Development benchmark violated Field 2 lock")

    policy = architecture["selection_policy"]
    contracted_architectures = {
        policy["primary_architecture"], *policy["baseline_architectures"]
    }
    if set(config["architectures"]) != contracted_architectures:
        raise ValueError("Evaluation architectures differ from frozen selection")
    if config["primary_architecture"] != policy["primary_architecture"]:
        raise ValueError("Primary architecture differs from frozen selection")
    frozen_seeds = [int(value) for value in architecture["frozen_model_seeds"]]
    if [int(value) for value in config["model_seeds"]] != frozen_seeds:
        raise ValueError("Evaluation seeds differ from frozen selection")

    rows = []
    for outer in outer_folds:
        inner = inner_mapping[outer]
        for fold in outer_folds:
            if fold == outer:
                role = "outer_test"
            elif fold == inner:
                role = "inner_validation"
            else:
                role = "train"
            rows.append(
                {
                    "outer_evaluation_fold": outer,
                    "spatial_fold": fold,
                    "role": role,
                    "labels_visible_for_gradient_updates": role == "train",
                    "labels_visible_for_early_stopping": role == "inner_validation",
                    "labels_visible_only_after_model_freeze": role == "outer_test",
                }
            )
    roles = pd.DataFrame(rows)
    role_counts = roles.groupby(["outer_evaluation_fold", "role"]).size().unstack(
        fill_value=0
    )
    for outer in outer_folds:
        if (
            role_counts.loc[outer, "train"] != 3
            or role_counts.loc[outer, "inner_validation"] != 1
            or role_counts.loc[outer, "outer_test"] != 1
        ):
            raise ValueError(f"Invalid nested role allocation for outer fold {outer}")

    role_path = contracts / config["role_table_filename"]
    roles.to_csv(role_path, index=False)

    figure, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    for y, outer in enumerate(outer_folds):
        selected = roles[roles.outer_evaluation_fold == outer]
        for row in selected.itertuples(index=False):
            ax.add_patch(
                plt.Rectangle(
                    (row.spatial_fold - 0.45, y - 0.36),
                    0.9,
                    0.72,
                    facecolor=COLORS[row.role],
                    edgecolor="white",
                    linewidth=2,
                )
            )
            label = {
                "train": "Fit",
                "inner_validation": "Stop",
                "outer_test": "Test",
            }[row.role]
            ax.text(
                row.spatial_fold, y, label,
                ha="center", va="center", color="white", weight="bold",
            )
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.7, len(outer_folds) - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(outer_folds)
    ax.set_yticks(range(len(outer_folds)), [f"Outer fold {x}" for x in outer_folds])
    ax.set_xlabel("Frozen spatial fold")
    ax.set_title(
        "Leakage-safe nested spatial evaluation protocol\n"
        "Outer test labels remain hidden until training and early stopping finish"
    )
    ax.legend(
        handles=[
            Patch(color=COLORS["train"], label="Model fitting"),
            Patch(color=COLORS["inner_validation"], label="Early stopping only"),
            Patch(color=COLORS["outer_test"], label="Final outer evaluation"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
    )
    visual_path = contracts / config["visual_filename"]
    figure.savefig(visual_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    frozen = dict(config)
    frozen["status"] = "supervised_primary_evaluation_protocol_frozen"
    frozen["field2_accessed"] = False
    frozen["synthetic_scientific_observations"] = False
    frozen["nested_role_table_sha256"] = sha256(role_path)
    frozen["visual_sha256"] = sha256(visual_path)
    frozen["configuration_sha256"] = sha256(args.config)
    frozen["source_contract_hashes"] = {
        name: sha256(path) for name, path in source_paths.items()
    }
    frozen["scientific_rationale"] = {
        "development_checkpoint_reuse_for_primary_claims": False,
        "reason": (
            "The balanced development benchmark used each outer fold for early "
            "stopping. Its checkpoints are valid for architecture development "
            "but prohibited from primary outer-fold performance claims."
        ),
        "independent_uncertainty_unit": "outer spatial fold",
        "model_seed_role": "nested optimization stability, not independent field replication",
    }

    contract_path = contracts / config["contract_filename"]
    contract_path.write_text(yaml.safe_dump(frozen, sort_keys=False))
    print("Nested fold roles:")
    print(roles.pivot(
        index="outer_evaluation_fold", columns="spatial_fold", values="role"
    ).to_string())
    print(f"Role table: {role_path}")
    print(f"Visual QC: {visual_path}")
    print(f"Protocol contract: {contract_path}")
    print(
        "Protocol frozen; no model training, exhaustive inference, synthetic "
        "observations, or Field 2 access occurred."
    )


if __name__ == "__main__":
    main()
