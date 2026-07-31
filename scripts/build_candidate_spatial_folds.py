#!/usr/bin/env python
"""Search for contiguous, class-balanced candidate Field 1 spatial folds."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import yaml


METRICS = [
    "spatial_groups", "labeled_groups", "soil_pixel_observations",
    "chickpea_pixel_observations", "weed_pixel_observations",
    "labeled_pixel_observations",
]


def component_count(frame: pd.DataFrame) -> int:
    coordinates = set(map(tuple, frame[["block_x", "block_y"]].to_numpy()))
    count = 0
    while coordinates:
        count += 1
        stack = [coordinates.pop()]
        while stack:
            x, y = stack.pop()
            for neighbour in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbour in coordinates:
                    coordinates.remove(neighbour)
                    stack.append(neighbour)
    return count


def optimize_order(frame: pd.DataFrame, folds: int, weights: np.ndarray,
                   minimum_groups: int, minimum_labeled: int) -> tuple[float, list[int]] | None:
    values = frame[METRICS].to_numpy(dtype=float)
    totals = values.sum(axis=0)
    cumulative = np.vstack((np.zeros(len(METRICS)), np.cumsum(values, axis=0)))
    n = len(frame)
    valid_boundary = np.r_[True, np.diff(frame["projection"].to_numpy()) > 1e-8, True]
    infinity = float("inf")
    dp = np.full((folds + 1, n + 1), infinity)
    previous = np.full((folds + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    def segment_cost(start: int, end: int) -> float:
        segment = cumulative[end] - cumulative[start]
        shares = np.divide(segment, totals, out=np.zeros_like(segment), where=totals > 0)
        return float(np.sum(weights * ((shares - 1 / folds) / (1 / folds)) ** 2))

    for fold in range(1, folds + 1):
        end_min = fold * minimum_groups
        end_max = n - (folds - fold) * minimum_groups
        for end in range(end_min, end_max + 1):
            if end < n and not valid_boundary[end]:
                continue
            start_min = (fold - 1) * minimum_groups
            start_max = end - minimum_groups
            for start in range(start_min, start_max + 1):
                if not np.isfinite(dp[fold - 1, start]):
                    continue
                if start > 0 and not valid_boundary[start]:
                    continue
                labeled = cumulative[end, 1] - cumulative[start, 1]
                if labeled < minimum_labeled:
                    continue
                candidate = dp[fold - 1, start] + segment_cost(start, end)
                if candidate < dp[fold, end]:
                    dp[fold, end] = candidate
                    previous[fold, end] = start
    if not np.isfinite(dp[folds, n]):
        return None
    cuts = [n]
    end = n
    for fold in range(folds, 0, -1):
        end = int(previous[fold, end])
        cuts.append(end)
    return float(dp[folds, n]), sorted(cuts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/spatial_splits.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    configuration = yaml.safe_load(args.config.read_text())
    project = Path(paths["project_root"])
    reports = project / "metadata" / "local" / "reports" / "spatial_grouping"
    membership = pd.read_csv(reports / "spatial_group_membership.csv")
    labels = pd.read_csv(reports / "spatial_group_class_summary.csv")
    groups = membership.drop_duplicates("spatial_group_id")[[
        "spatial_group_id", "block_x", "block_y", "center_x_m", "center_y_m"
    ]].merge(labels, how="left", on=["spatial_group_id", "block_x", "block_y"])
    count_columns = [column for column in labels if column.endswith("observations")]
    groups[count_columns] = groups[count_columns].fillna(0)
    groups["spatial_groups"] = 1
    groups["labeled_groups"] = (groups["labeled_pixel_observations"] > 0).astype(int)

    candidate = configuration["candidate_split"]
    folds = int(candidate["folds"])
    angle_step = int(candidate["angle_step_degrees"])
    weights_by_name = candidate["balance_weights"]
    weights = np.array([
        weights_by_name["spatial_groups"], weights_by_name["labeled_groups"],
        weights_by_name["soil"], weights_by_name["chickpea"],
        weights_by_name["weed"], weights_by_name["all_labels"],
    ])
    center_x = groups["center_x_m"] - groups["center_x_m"].mean()
    center_y = groups["center_y_m"] - groups["center_y_m"].mean()
    best: tuple[float, float, pd.DataFrame, list[int]] | None = None
    for angle in range(0, 180, angle_step):
        radians = np.deg2rad(angle)
        ordered = groups.assign(
            projection=center_x * np.cos(radians) + center_y * np.sin(radians)
        ).sort_values(["projection", "block_y", "block_x"]).reset_index(drop=True)
        optimized = optimize_order(
            ordered, folds, weights, int(candidate["minimum_groups_per_fold"]),
            int(candidate["minimum_labeled_groups_per_fold"]),
        )
        if optimized is None:
            continue
        score, cuts = optimized
        ordered["fold"] = 0
        for fold, (start, end) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
            ordered.loc[start:end - 1, "fold"] = fold
        components = sum(component_count(part) - 1 for _, part in ordered.groupby("fold"))
        score += 25.0 * components
        if best is None or score < best[0]:
            best = (score, float(angle), ordered, cuts)
    if best is None:
        raise RuntimeError("No candidate split satisfied the configured constraints")
    score, angle, assignments, cuts = best
    if (assignments["fold"] == 0).any() or assignments["spatial_group_id"].duplicated().any():
        raise AssertionError("Invalid fold assignment")

    totals = assignments[METRICS].sum()
    rows = []
    for fold, part in assignments.groupby("fold"):
        row: dict[str, object] = {
            "fold": int(fold), "connected_components": component_count(part),
        }
        for metric in METRICS:
            row[metric] = int(part[metric].sum())
            row[f"{metric}_share"] = float(part[metric].sum() / totals[metric])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("fold")

    thresholds = []
    projection = assignments["projection"].to_numpy()
    for boundary_number, cut in enumerate(cuts[1:-1], start=1):
        thresholds.append({
            "boundary": boundary_number,
            "projection_threshold_m": float((projection[cut - 1] + projection[cut]) / 2),
            "angle_degrees": angle,
            "boundary_exclusion_m": float(candidate["boundary_exclusion_m"]),
        })
    metadata = pd.DataFrame([{
        "status": "candidate_requires_visual_and_numeric_review",
        "folds": folds, "angle_degrees": angle, "objective_score": score,
        "block_size_m": configuration["grouping"]["block_size_m"],
        "boundary_exclusion_m": candidate["boundary_exclusion_m"],
        "spatial_groups": len(assignments),
    }])
    assignments.to_csv(reports / "candidate_spatial_fold_assignments.csv", index=False)
    summary.to_csv(reports / "candidate_spatial_fold_summary.csv", index=False)
    pd.DataFrame(thresholds).to_csv(reports / "candidate_spatial_fold_boundaries.csv", index=False)
    metadata.to_csv(reports / "candidate_spatial_fold_metadata.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    x0, y0 = assignments["center_x_m"].min(), assignments["center_y_m"].min()
    colours = ListedColormap(["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"])
    image = axes[0].scatter(
        assignments["center_x_m"] - x0, assignments["center_y_m"] - y0,
        c=assignments["fold"], cmap=colours, vmin=0.5, vmax=5.5,
        marker="s", s=230, edgecolors="white", linewidths=0.35,
    )
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Candidate contiguous folds (search angle {angle:.0f}°)")
    axes[0].set_xlabel(f"Easting offset from {x0:.1f} m")
    axes[0].set_ylabel(f"Northing offset from {y0:.1f} m")
    colourbar = figure.colorbar(image, ax=axes[0], ticks=range(1, folds + 1), shrink=0.8)
    colourbar.set_label("Fold")

    share_columns = [
        "soil_pixel_observations_share", "chickpea_pixel_observations_share",
        "weed_pixel_observations_share", "spatial_groups_share",
    ]
    positions = np.arange(folds)
    width = 0.19
    for index, (column, label) in enumerate(zip(
        share_columns, ["Soil", "Chickpea", "Weed", "Spatial groups"]
    )):
        axes[1].bar(positions + (index - 1.5) * width, summary[column], width, label=label)
    axes[1].axhline(1 / folds, color="black", linestyle="--", linewidth=1, label="Ideal 20%")
    axes[1].set_xticks(positions, [f"Fold {fold}" for fold in range(1, folds + 1)])
    axes[1].set_ylim(0, max(0.35, summary[share_columns].to_numpy().max() * 1.15))
    axes[1].set_ylabel("Share of Field 1 total")
    axes[1].set_title("Class and geographic balance")
    axes[1].legend(ncol=2)
    figure.suptitle("Candidate leakage-safe Field 1 cross-validation folds", fontsize=15)
    figure.savefig(reports / "candidate_spatial_folds_overview.png", dpi=200)
    plt.close(figure)

    print(f"Candidate angle: {angle:.0f} degrees")
    print(f"Objective score: {score:.6f}")
    print(summary[["fold", "connected_components", *share_columns]].to_string(index=False))
    print(f"Reports written to: {reports}")
    print(f"Visual QC: {reports / 'candidate_spatial_folds_overview.png'}")
    print("Candidate only; no model training or Field 2 access occurred.")


if __name__ == "__main__":
    main()
