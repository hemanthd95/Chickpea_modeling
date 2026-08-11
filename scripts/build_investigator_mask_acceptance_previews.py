#!/usr/bin/env python
"""Render imagery-backed, review-only mask acceptance panels.

The categorical review candidates are overlaid on the observed NIR-red-green
previews used for investigator annotation.  This stage is visual QC only: it
does not accept a candidate, alter an authoritative mask, or retrain a model.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
import yaml


DECISION_COLORS = {
    1: np.asarray([0.55, 0.35, 0.17], dtype=np.float32),  # soil
    2: np.asarray([0.49, 0.14, 0.83], dtype=np.float32),  # weed
    4: np.asarray([0.09, 0.64, 0.29], dtype=np.float32),  # chickpea
    6: np.asarray([0.96, 0.45, 0.05], dtype=np.float32),  # alley
}
UNCERTAINTY_COLORS = {
    3: np.asarray([0.64, 0.67, 0.72], dtype=np.float32),  # unresolved core
    5: np.asarray([0.99, 0.85, 0.32], dtype=np.float32),  # unresolved edge
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_hash(contract: dict, project: Path, path: Path) -> None:
    relative = str(path.relative_to(project))
    expected = contract.get("output_hashes", {}).get(relative)
    if expected != sha256(path):
        raise ValueError(f"Frozen artifact hash mismatch: {path}")


def overlay_classes(
    image: np.ndarray,
    classes: np.ndarray,
    colors: dict[int, np.ndarray],
    alpha: float,
) -> np.ndarray:
    """Return an RGB overlay without changing pixels outside requested codes."""
    if image.shape[:2] != classes.shape:
        raise ValueError("Image and categorical layer must have equal spatial shapes")
    result = np.asarray(image[..., :3], dtype=np.float32).copy()
    for code, color in colors.items():
        selected = classes == code
        result[selected] = (1.0 - alpha) * result[selected] + alpha * color
    return np.clip(result, 0, 1)


def preview_classes(candidate: np.ndarray, step: int, expected_shape: tuple[int, int]) -> np.ndarray:
    result = candidate[::step, ::step]
    if result.shape != expected_shape:
        raise ValueError(
            f"Candidate preview shape {result.shape} differs from imagery {expected_shape}"
        )
    return result


def legend_handles() -> list[Patch]:
    items = [
        ("High-confidence chickpea", DECISION_COLORS[4]),
        ("High-confidence weed", DECISION_COLORS[2]),
        ("Soil", DECISION_COLORS[1]),
        ("Explicit alley", DECISION_COLORS[6]),
        ("Unresolved core vegetation", UNCERTAINTY_COLORS[3]),
        ("Unresolved polygon edge", UNCERTAINTY_COLORS[5]),
    ]
    return [Patch(facecolor=color, edgecolor="none", label=label) for label, color in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--config", default=Path("configs/chickpea_mask_refinement.yaml"), type=Path
    )
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    policy = config["investigator_mask_acceptance_review"]
    if policy.get("status") != "enabled_after_review_candidate_materialization":
        raise ValueError("Mask acceptance preview stage is not enabled")
    for forbidden in (
        "automatically_accept_review_candidates",
        "automatically_replace_authoritative_masks",
        "automatically_retrain_models",
    ):
        if bool(policy.get(forbidden, False)):
            raise ValueError(f"{forbidden} must remain false")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    contracts = local / "contracts"
    candidate_contract_path = contracts / str(policy["candidate_contract_filename"])
    candidate_contract = yaml.safe_load(candidate_contract_path.read_text())
    if candidate_contract.get("status") != "field1_investigator_mask_review_candidates_materialized":
        raise ValueError("Investigator mask review candidates are not finalized")
    if bool(candidate_contract.get("candidate_geotiffs_are_authoritative", True)):
        raise ValueError("Review candidates must remain non-authoritative")

    candidate_reports = (
        local / "reports" / "mask_refinement" / "investigator_mask_review_candidates"
    )
    summary_path = candidate_reports / "investigator_mask_review_candidate_summary.csv"
    validate_output_hash(candidate_contract, project, summary_path)
    summary = pd.read_csv(summary_path)
    if len(summary) != 19 or summary["cube_id"].duplicated().any():
        raise ValueError("Candidate summary must contain 19 unique cubes")

    layer_manifest_path = project / str(policy["annotation_layer_manifest_relative"])
    layers = pd.read_csv(layer_manifest_path)
    if len(layers) != 19 or layers["cube_id"].duplicated().any():
        raise ValueError("Annotation-layer manifest must contain 19 unique cubes")
    merged = summary.merge(
        layers[["cube_id", "analysis_role", "preview_step", "false_colour_path"]],
        on="cube_id",
        suffixes=("", "_layer"),
        validate="one_to_one",
    )
    if not (merged["analysis_role"] == merged["analysis_role_layer"]).all():
        raise ValueError("Analysis roles differ between candidate and imagery manifests")

    reports = candidate_reports / "acceptance_review"
    individual = reports / "individual"
    reports.mkdir(parents=True, exist_ok=True)
    individual.mkdir(parents=True, exist_ok=True)
    decision_alpha = float(policy["decision_overlay_alpha"])
    uncertainty_alpha = float(policy["uncertainty_overlay_alpha"])
    trusted = {str(value) for value in policy["trusted_expansion_cubes"]}
    output_paths: list[Path] = []
    review_rows: list[dict] = []
    trusted_tiles: list[tuple] = []

    ordered = merged.assign(
        cube_number=lambda frame: frame["cube_id"].str.extract(r"(\d+)$")[0].astype(int)
    ).sort_values("cube_number")
    for number, row in enumerate(ordered.itertuples(index=False), 1):
        cube_id = str(row.cube_id)
        image_path = Path(str(row.false_colour_path))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = plt.imread(image_path)[..., :3].astype(np.float32)
        candidate_path = Path(str(row.review_candidate_path))
        validate_output_hash(candidate_contract, project, candidate_path)
        with rasterio.open(candidate_path) as dataset:
            candidate = dataset.read(1)
        classes = preview_classes(
            candidate, int(row.preview_step), tuple(int(v) for v in image.shape[:2])
        )
        decisions = overlay_classes(image, classes, DECISION_COLORS, decision_alpha)
        uncertainty = overlay_classes(image, classes, UNCERTAINTY_COLORS, uncertainty_alpha)

        core = int(row.core_vegetation_pixels)
        chickpea = int(row.high_confidence_chickpea_review_pixels)
        weed = int(row.high_confidence_weed_review_pixels)
        unresolved = int(row.unresolved_core_vegetation_pixels)
        status = "pending_investigator_visual_acceptance"
        review_rows.append({
            "cube_id": cube_id,
            "analysis_role": str(row.analysis_role),
            "candidate_disposition": str(row.candidate_disposition),
            "decision_eligible": bool(row.decision_eligible),
            "trusted_expansion_cube": cube_id in trusted,
            "core_vegetation_pixels": core,
            "high_confidence_chickpea_pixels": chickpea,
            "high_confidence_weed_pixels": weed,
            "unresolved_core_vegetation_pixels": unresolved,
            "high_confidence_chickpea_fraction_of_core": chickpea / max(core, 1),
            "high_confidence_weed_fraction_of_core": weed / max(core, 1),
            "unresolved_fraction_of_core": unresolved / max(core, 1),
            "visual_acceptance_status": status,
            "authoritative_mask_modified": False,
        })

        figure, axes = plt.subplots(1, 3, figsize=(15, 7), constrained_layout=True)
        panels = (
            ("Observed NIR-red-green", image),
            ("Proposed decisions", decisions),
            ("Unresolved support", uncertainty),
        )
        for axis, (title, panel) in zip(axes, panels):
            axis.imshow(panel, interpolation="nearest")
            axis.set_title(title)
            axis.axis("off")
        figure.suptitle(
            f"{cube_id} imagery-backed candidate acceptance review\n"
            f"C={chickpea:,} ({chickpea / max(core, 1):.1%}); "
            f"W={weed:,} ({weed / max(core, 1):.1%}); "
            f"unresolved={unresolved:,} ({unresolved / max(core, 1):.1%})\n"
            "PENDING REVIEW — no authoritative label change",
            fontsize=14,
        )
        figure.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=9)
        preview_path = individual / f"{cube_id}_mask_acceptance_review.png"
        figure.savefig(preview_path, dpi=220, facecolor="white", bbox_inches="tight")
        plt.close(figure)
        output_paths.append(preview_path)
        if cube_id in trusted:
            trusted_tiles.append((cube_id, image, decisions, uncertainty, chickpea, unresolved, core))
        print(
            f"Rendered {number}/{len(ordered)} {cube_id}: "
            f"chickpea={chickpea:,}; weed={weed:,}; unresolved={unresolved:,}",
            flush=True,
        )

    if {item[0] for item in trusted_tiles} != trusted:
        raise ValueError("Not every trusted expansion cube received an acceptance panel")
    trusted_tiles.sort(key=lambda item: int(item[0].split("cube")[-1]))
    figure, axes = plt.subplots(
        len(trusted_tiles), 3, figsize=(14, 6.0 * len(trusted_tiles)), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(len(trusted_tiles), 3)
    for row_axes, (cube_id, image, decisions, uncertainty, chickpea, unresolved, core) in zip(
        axes, trusted_tiles
    ):
        for axis, (title, panel) in zip(
            row_axes,
            (("Observed", image), ("Proposed chickpea", decisions), ("Unresolved", uncertainty)),
        ):
            axis.imshow(panel, interpolation="nearest")
            axis.set_title(f"{cube_id} — {title}")
            axis.axis("off")
        row_axes[1].text(
            0.5, -0.04,
            f"chickpea {chickpea:,} ({chickpea / max(core, 1):.1%}); "
            f"unresolved {unresolved:,} ({unresolved / max(core, 1):.1%})",
            transform=row_axes[1].transAxes, ha="center", va="top", fontsize=10,
        )
    figure.suptitle(
        "Trusted expansion cubes 12/14/15 — imagery-backed chickpea acceptance review\n"
        "Only P(chickpea) ≥ 0.80 is proposed; no weed labels are inferred",
        fontsize=16,
    )
    figure.legend(handles=legend_handles(), loc="lower center", ncol=3, fontsize=10)
    trusted_path = reports / "trusted_expansion_mask_acceptance_review.png"
    figure.savefig(trusted_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    output_paths.append(trusted_path)

    inventory = pd.DataFrame(review_rows)
    inventory_path = reports / "investigator_mask_acceptance_review_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    output_paths.append(inventory_path)

    contract = {
        "status": "field1_investigator_mask_acceptance_review_ready",
        "field": "Field 1",
        "visual_acceptance_status": "pending_investigator_review",
        "candidate_contract": str(candidate_contract_path),
        "candidate_contract_sha256": sha256(candidate_contract_path),
        "annotation_layer_manifest": str(layer_manifest_path),
        "annotation_layer_manifest_sha256": sha256(layer_manifest_path),
        "trusted_expansion_cubes": sorted(trusted),
        "trusted_expansion_cubes_are_chickpea_only": True,
        "outside_polygon_pixels_are_unknown_unlabeled": True,
        "unresolved_pixels_are_not_training_labels": True,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "output_hashes": {
            str(path.relative_to(project)): sha256(path) for path in output_paths
        },
    }
    contract_path = contracts / "field1_investigator_mask_acceptance_review_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Acceptance inventory: {inventory_path}")
    print(f"Trusted expansion visual QC: {trusted_path}")
    print(f"Individual imagery-backed panels: {individual}")
    print(f"Contract: {contract_path}")
    print(
        "Visual review only: every candidate remains pending; no authoritative mask "
        "changed, no model was retrained, and Field 2 remained locked."
    )


if __name__ == "__main__":
    main()
