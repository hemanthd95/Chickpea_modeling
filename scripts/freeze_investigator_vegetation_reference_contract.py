#!/usr/bin/env python
"""Freeze the accepted Field 1 investigator vegetation references."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import pandas as pd
import yaml


LABELS = ("confirmed_weed", "confirmed_chickpea")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values
    normalized = values.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "false"})
    if unknown.any():
        raise ValueError(
            "usable_reference contains non-boolean values: "
            f"{sorted(normalized[unknown].unique())}"
        )
    return normalized.eq("true")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, type=Path)
    parser.add_argument(
        "--config",
        default=Path("configs/chickpea_mask_refinement.yaml"),
        type=Path,
    )
    args = parser.parse_args()

    paths = yaml.safe_load(args.paths.read_text())
    config = yaml.safe_load(args.config.read_text())
    if not config["review"]["field2_locked"]:
        raise ValueError("Field 2 must remain locked")
    policy = config["investigator_vegetation_reference_freeze"]
    if policy["status"] != "accepted_pending_contract_materialization":
        raise ValueError("Investigator reference set has not been accepted for freezing")
    for forbidden in (
        "automatically_materialize_categorical_masks",
        "automatically_replace_authoritative_masks",
        "automatically_retrain_models",
    ):
        if bool(policy.get(forbidden, False)):
            raise ValueError(f"{forbidden} must remain false")

    project = Path(paths["project_root"])
    local = project / "metadata" / "local"
    reports = local / str(policy["source_report_relative"])
    contracts = local / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    annotations = (
        local / "annotations" / "vegetation_references"
        / "field1_vegetation_reference_points.csv"
    )
    audit_contract_path = contracts / str(policy["audit_contract_filename"])
    required = {
        "reference_points": annotations,
        "point_qc": reports / "investigator_vegetation_reference_qc.csv",
        "support": reports / "investigator_vegetation_reference_support_by_cube.csv",
        "loco": reports / "investigator_vegetation_reference_loco_validation.csv",
        "predictions": reports / "investigator_vegetation_reference_predictions.csv",
        "spectral_profile": reports / "investigator_vegetation_reference_spectral_profile.csv",
        "visual_qc": reports / "investigator_vegetation_reference_audit_overview.png",
        "audit_contract": audit_contract_path,
        "configuration": args.config,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Reference-freeze inputs are missing:\n" + "\n".join(missing))

    audit = yaml.safe_load(audit_contract_path.read_text())
    if not bool(audit.get("review_gate_passed", False)):
        raise ValueError("Investigator reference audit gate did not pass")
    if bool(audit.get("field2_accessed", True)):
        raise ValueError("Audit contract does not confirm that Field 2 stayed locked")
    if bool(audit.get("authoritative_masks_modified", True)):
        raise ValueError("Audit contract reports an authoritative mask modification")
    if bool(audit.get("models_retrained", True)):
        raise ValueError("Audit contract reports model retraining")

    expected_hashes = audit.get("output_hashes", {})
    report_outputs = {
        name: path for name, path in required.items()
        if name in {"point_qc", "support", "loco", "predictions", "spectral_profile", "visual_qc"}
    }
    mismatched = [
        name for name, path in report_outputs.items()
        if expected_hashes.get(path.name) != sha256(path)
    ]
    if mismatched:
        raise ValueError(f"Audit outputs differ from their recorded hashes: {mismatched}")
    if audit.get("source_hashes", {}).get("reference_points") != sha256(annotations):
        raise ValueError("Reference annotations differ from the audited point layer")

    points = pd.read_csv(annotations)
    points = points[points["kind"].isin(LABELS)].copy()
    point_qc = pd.read_csv(required["point_qc"])
    loco = pd.read_csv(required["loco"])
    spectral = pd.read_csv(required["spectral_profile"])
    if points["annotation_id"].duplicated().any():
        raise ValueError("Frozen point candidates contain duplicate annotation IDs")
    if point_qc["annotation_id"].duplicated().any():
        raise ValueError("Point QC contains duplicate annotation IDs")
    if set(points["annotation_id"]) != set(point_qc["annotation_id"]):
        raise ValueError("Annotation and QC point identities differ")
    if set(point_qc["kind"]) != set(LABELS):
        raise ValueError("Point QC does not contain both confirmed vegetation classes")
    if loco["heldout_cube"].duplicated().any():
        raise ValueError("LOCO validation contains duplicate held-out cubes")
    if set(spectral["class_name"]) != set(LABELS):
        raise ValueError("Spectral profile does not contain both confirmed classes")

    usable = point_qc[boolean_series(point_qc["usable_reference"])].copy()
    usable_counts = (
        usable.groupby(["cube_id", "kind"], observed=True).size()
        .unstack(fill_value=0)
        .reindex(
            index=sorted(point_qc["cube_id"].unique()),
            columns=list(LABELS),
            fill_value=0,
        )
        .fillna(0)
        .astype(int)
    )
    minimum_usable = int(policy["minimum_usable_points_per_class_per_cube"])
    if (usable_counts < minimum_usable).any().any():
        deficient = usable_counts[usable_counts < minimum_usable].stack()
        raise ValueError(f"Post-QC class support is insufficient: {deficient.to_dict()}")

    minimum_auc = float(policy["minimum_mean_loco_auc"])
    minimum_ba = float(policy["minimum_mean_loco_balanced_accuracy"])
    minimum_fraction = float(
        policy["minimum_fraction_heldout_cubes_passing_balanced_accuracy"]
    )
    mean_auc = float(loco["roc_auc"].mean())
    mean_ba = float(loco["balanced_accuracy"].mean())
    passing_fraction = float((loco["balanced_accuracy"] >= minimum_ba).mean())
    if mean_auc < minimum_auc or mean_ba < minimum_ba or passing_fraction < minimum_fraction:
        raise ValueError("Recomputed LOCO metrics do not pass the frozen acceptance gate")

    warning_ba = float(policy["per_cube_balanced_accuracy_warning_threshold"])
    warning_auc = float(policy["per_cube_auc_warning_threshold"])
    warnings = loco[
        (loco["balanced_accuracy"] < warning_ba) | (loco["roc_auc"] < warning_auc)
    ].copy()

    frozen = {
        "reference_points": contracts / str(policy["frozen_points_filename"]),
        "point_qc": contracts / str(policy["frozen_qc_filename"]),
        "loco": contracts / str(policy["frozen_loco_filename"]),
        "spectral_profile": contracts / str(policy["frozen_spectral_profile_filename"]),
        "visual_qc": contracts / str(policy["frozen_visual_filename"]),
    }
    copy_sources = {
        "reference_points": annotations,
        "point_qc": required["point_qc"],
        "loco": required["loco"],
        "spectral_profile": required["spectral_profile"],
        "visual_qc": required["visual_qc"],
    }
    points.to_csv(frozen["reference_points"], index=False)
    for name, destination in frozen.items():
        if name != "reference_points":
            shutil.copy2(copy_sources[name], destination)

    warning_records = [
        {
            "heldout_cube": str(row.heldout_cube),
            "roc_auc": float(row.roc_auc),
            "balanced_accuracy": float(row.balanced_accuracy),
        }
        for row in warnings.itertuples(index=False)
    ]
    contract = {
        "status": "field1_investigator_vegetation_references_frozen",
        "field": "Field 1",
        "reference_authority": "investigator_confirmed_points",
        "independent_observation_unit": "one_investigator_annotation_point",
        "neighbourhood_aggregation": "bandwise_median_after_soil_and_nodata_exclusion",
        "reference_points_total": int(len(point_qc)),
        "usable_reference_points": int(len(usable)),
        "reference_cubes": sorted(point_qc["cube_id"].unique()),
        "usable_points_by_cube_and_class": {
            f"{cube_id}:{kind}": int(value)
            for (cube_id, kind), value in usable_counts.stack().items()
        },
        "acceptance": {
            "mean_loco_auc": mean_auc,
            "mean_loco_balanced_accuracy": mean_ba,
            "heldout_cubes_passing_balanced_accuracy_fraction": passing_fraction,
            "gate_passed": True,
            "thresholds": {
                "minimum_mean_loco_auc": minimum_auc,
                "minimum_mean_loco_balanced_accuracy": minimum_ba,
                "minimum_fraction_heldout_cubes_passing_balanced_accuracy": minimum_fraction,
            },
        },
        "per_cube_transfer_warnings": warning_records,
        "downstream_use": str(policy["downstream_use"]),
        "historical_chickpea_and_weed_masks_are_reference_truth": False,
        "categorical_masks_materialized": False,
        "authoritative_masks_modified": False,
        "models_retrained": False,
        "field2_accessed": False,
        "source_hashes": {name: sha256(path) for name, path in required.items()},
        "frozen_artifact_hashes": {name: sha256(path) for name, path in frozen.items()},
    }
    contract_path = contracts / str(policy["frozen_contract_filename"])
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))

    print(f"Frozen reference points: {len(point_qc)} ({len(usable)} usable)")
    print(f"Reference cubes: {point_qc['cube_id'].nunique()}")
    print(f"Mean LOCO AUC: {mean_auc:.4f}")
    print(f"Mean LOCO balanced accuracy: {mean_ba:.4f}")
    if warning_records:
        print("Per-cube transfer warnings:")
        print(warnings[["heldout_cube", "roc_auc", "balanced_accuracy"]].to_string(index=False))
    else:
        print("Per-cube transfer warnings: none")
    print(f"Contract: {contract_path}")
    print(f"Frozen visual QC: {frozen['visual_qc']}")
    print(
        "References frozen for probability review only; no categorical mask was "
        "materialized, no model was retrained, and Field 2 remained locked."
    )


if __name__ == "__main__":
    main()
