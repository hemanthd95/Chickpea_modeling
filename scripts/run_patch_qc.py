#!/usr/bin/env python
"""Verify real indexed patches and render an observed false-colour montage."""

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

from chickpea_ssl.data import EnviCube, authoritative_class_map, load_band_indices, load_records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    digest = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument("--sampling", default=Path("configs/sampling.yaml"), type=Path)
    parser.add_argument("--bands", default=Path("configs/spectral_bands.yaml"), type=Path)
    args = parser.parse_args()
    paths = yaml.safe_load(args.paths.read_text())
    sampling = yaml.safe_load(args.sampling.read_text())
    qc = sampling["patch_qc"]
    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    reports = local / "reports" / "patch_qc"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = local / "authoritative_manifest.csv"
    center_path = contracts / "field1_training_candidate_centers.csv"
    center_contract_path = contracts / "field1_training_candidate_contract.yaml"
    normalization_path = contracts / "field1_fold_normalization.csv"
    normalization_contract_path = contracts / "field1_fold_normalization_contract.yaml"
    required = [
        manifest, center_path, center_contract_path,
        normalization_path, normalization_contract_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Patch-QC inputs missing: {missing}")
    center_contract = yaml.safe_load(center_contract_path.read_text())
    normalization_contract = yaml.safe_load(normalization_contract_path.read_text())
    if sha256(center_path) != center_contract["center_index_sha256"]:
        raise ValueError("Training-centre index hash mismatch")
    if sha256(normalization_path) != normalization_contract["normalization_csv_sha256"]:
        raise ValueError("Normalization hash mismatch")

    centers = pd.read_csv(center_path)
    normalization = pd.read_csv(normalization_path)
    band_indices = load_band_indices(args.bands, normalization_contract["band_section"])
    records = {record.cube_id: record for record in load_records(args.paths, manifest)}
    selected_parts = []
    for fold in sorted(centers["fold"].unique()):
        validation_column = f"validation_eligible_fold_{fold}"
        validation_eligible = centers[validation_column]
        if validation_eligible.dtype == object:
            validation_eligible = validation_eligible.astype(str).str.lower().eq("true")
        for class_name in sorted(centers["class_name"].unique()):
            candidates = centers[
                (centers["fold"] == fold)
                & (centers["class_name"] == class_name)
                & validation_eligible
            ]
            count = min(int(qc["samples_per_class_fold"]), len(candidates))
            rng = stable_rng(int(sampling["seed"]), "patch_qc", fold, class_name)
            selected_parts.append(candidates.iloc[
                rng.choice(len(candidates), size=count, replace=False)
            ])
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["label_match"] = False
    selected["shape_ok"] = False
    selected["normalization_finite"] = False
    selected["normalized_abs_max"] = np.nan
    selected["raw_minimum"] = np.nan
    selected["raw_maximum"] = np.nan
    patch_cache: dict[str, np.ndarray] = {}

    for cube_id, cube_rows in selected.groupby("cube_id"):
        cube = EnviCube(records[cube_id], band_indices)
        class_map = authoritative_class_map(records[cube_id])
        for index, row in cube_rows.iterrows():
            patch = cube.patch(int(row["row"]), int(row["column"]), int(row["patch_size_pixels"]))
            fold_stats = normalization[normalization["heldout_fold"] == int(row["fold"])]
            fold_stats = fold_stats.set_index("band_index").loc[band_indices]
            mean = fold_stats["mean"].to_numpy()
            standard_deviation = fold_stats["standard_deviation"].to_numpy()
            normalized = (patch - mean[None, None, :]) / np.maximum(
                standard_deviation[None, None, :], 1e-6
            )
            selected.loc[index, "label_match"] = (
                int(class_map[int(row["row"]), int(row["column"])]) == int(row["class_id"])
            )
            selected.loc[index, "shape_ok"] = (
                patch.shape == (
                    int(row["patch_size_pixels"]), int(row["patch_size_pixels"]), len(band_indices)
                )
            )
            selected.loc[index, "normalization_finite"] = bool(np.isfinite(normalized).all())
            selected.loc[index, "normalized_abs_max"] = float(np.abs(normalized).max())
            selected.loc[index, "raw_minimum"] = float(patch.min())
            selected.loc[index, "raw_maximum"] = float(patch.max())
            patch_cache[row["sample_id"]] = patch

    summary = selected.groupby("class_name", as_index=False).agg(
        patches_checked=("sample_id", "size"),
        cubes_checked=("cube_id", "nunique"),
        folds_checked=("fold", "nunique"),
        label_mismatches=("label_match", lambda values: int((~values).sum())),
        shape_failures=("shape_ok", lambda values: int((~values).sum())),
        nonfinite_normalized=("normalization_finite", lambda values: int((~values).sum())),
        maximum_absolute_normalized_value=("normalized_abs_max", "max"),
        raw_minimum=("raw_minimum", "min"),
        raw_maximum=("raw_maximum", "max"),
    )
    selected.to_csv(reports / "patch_qc_details.csv", index=False)
    summary.to_csv(reports / "patch_qc_summary.csv", index=False)

    montage_rows = []
    for class_name in ("soil", "chickpea", "weed"):
        candidates = selected[selected["class_name"] == class_name].sort_values(
            ["fold", "cube_id", "sample_id"]
        )
        # Prefer different cubes before filling remaining montage positions.
        diverse = candidates.drop_duplicates("cube_id")
        if len(diverse) < int(qc["montage_per_class"]):
            diverse = pd.concat([
                diverse, candidates[~candidates["sample_id"].isin(diverse["sample_id"])]
            ])
        montage_rows.append(diverse.head(int(qc["montage_per_class"])))
    montage = pd.concat(montage_rows, ignore_index=True)
    target_wavelengths = np.asarray(qc["false_color_wavelengths_nm"], dtype=float)
    first_cube = EnviCube(records[montage.iloc[0]["cube_id"]], band_indices)
    selected_wavelengths = first_cube.wavelengths[band_indices]
    rgb_positions = [int(np.abs(selected_wavelengths - target).argmin()) for target in target_wavelengths]
    rgb_patches = [patch_cache[row.sample_id][..., rgb_positions] for row in montage.itertuples()]
    stacked = np.concatenate([patch.reshape(-1, 3) for patch in rgb_patches], axis=0)
    lower, upper = np.percentile(stacked, [2, 98], axis=0)
    columns = int(qc["montage_per_class"])
    figure, axes = plt.subplots(3, columns, figsize=(2.5 * columns, 7.8), constrained_layout=True)
    for axis, row, patch in zip(axes.flat, montage.itertuples(), rgb_patches):
        rgb = np.clip((patch - lower) / np.maximum(upper - lower, 1), 0, 1)
        axis.imshow(rgb)
        center = int(row.patch_size_pixels) // 2
        axis.plot(center, center, marker="+", color="white", markersize=8, markeredgewidth=1.5)
        axis.set_title(f"{row.class_name} | fold {row.fold}\n{row.cube_id}", fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(
        "Observed 15×15 pixel false-colour patches (NIR–red–green)\n"
        "White + marks the authoritative centre pixel; colours are visualization only",
        fontsize=13,
    )
    preview = reports / "observed_patch_montage.png"
    figure.savefig(preview, dpi=200)
    plt.close(figure)
    failures = int(summary[["label_mismatches", "shape_failures", "nonfinite_normalized"]].sum().sum())
    print(summary.to_string(index=False))
    print(f"Details: {reports / 'patch_qc_details.csv'}")
    print(f"Visual QC: {preview}")
    if failures:
        raise RuntimeError(f"Patch QC failed with {failures} issue(s)")
    print("Patch QC passed; only observed Field 1 data were opened.")


if __name__ == "__main__":
    main()
