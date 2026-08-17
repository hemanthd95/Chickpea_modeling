"""Prediction-free Field 2 role-review and blind-sampling contracts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml

from chickpea_ssl.field2_readiness import sha256, yaml_safe


PRIMARY_ROLES = (
    "primary_three_class",
    "chickpea_absent_negative_control",
    "challenge_only",
    "sensitivity_only",
    "exclude_with_reason",
    "unreviewed",
)

REVIEW_FLAGS = (
    "chickpea_visible",
    "chickpea_very_small_or_immature",
    "chickpea_absent_confirmed",
    "ordinary_weed_visible",
    "tall_grass_weed_visible",
    "soil_visible",
    "mixed_pixels_common",
    "annotation_difficult",
    "georectification_concern",
    "other_visual_concern",
)

CONFIDENCE_VALUES = ("high", "medium", "low")

POINT_LABELS = (
    "soil",
    "weed",
    "tall_grass_weed",
    "chickpea",
    "chickpea_soil_mixed",
    "chickpea_weed_mixed",
    "uncertain",
    "nodata_invalid",
)

PREVIEW_LAYERS = (
    "review_sheet",
    "false_colour",
    "pca",
    "stored_index",
    "valid_support",
    "support_outline",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace one output after a complete same-directory write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode())


def atomic_write_yaml(path: Path, payload: dict) -> None:
    buffer = io.StringIO()
    yaml.safe_dump(yaml_safe(payload), buffer, sort_keys=False)
    atomic_write_bytes(path, buffer.getvalue().encode())


def default_review_record(cube_id: str, preview_hashes: dict[str, str]) -> dict:
    """Return a neutral record; no biological role or flag is inferred."""
    return {
        "cube_id": cube_id,
        "primary_role": "unreviewed",
        "flags": {name: False for name in REVIEW_FLAGS},
        "investigator_notes": "",
        "exclusion_reason": "",
        "confidence": "",
        "reviewed": False,
        "review_timestamp": "",
        "reviewer_identifier": "",
        "source_preview_checksums": dict(preview_hashes),
    }


def default_review_payload(manifest: pd.DataFrame) -> dict:
    reviews = {}
    for row in manifest.sort_values("cube_id").itertuples(index=False):
        hashes = json.loads(str(row.preview_sha256_json))
        reviews[str(row.cube_id)] = default_review_record(str(row.cube_id), hashes)
    return {
        "version": "field2_cube_role_reviews_v1",
        "revision": 0,
        "automatic_metadata": {
            "biological_roles_assigned_automatically": False,
            "predictions_or_probabilities_used": False,
        },
        "reviews": reviews,
    }


def validate_review_record(record: dict, expected_cube_id: str) -> list[str]:
    issues: list[str] = []
    if not isinstance(record, dict):
        return ["review_not_mapping"]
    if str(record.get("cube_id", "")) != expected_cube_id:
        issues.append("cube_id_mismatch")
    role = str(record.get("primary_role", ""))
    if role not in PRIMARY_ROLES:
        issues.append("unknown_primary_role")
    flags = record.get("flags")
    if not isinstance(flags, dict) or set(flags) != set(REVIEW_FLAGS):
        issues.append("flags_schema_mismatch")
        flags = {name: False for name in REVIEW_FLAGS}
    elif any(not isinstance(flags[name], bool) for name in REVIEW_FLAGS):
        issues.append("flags_must_be_boolean")
    visible = bool(flags.get("chickpea_visible"))
    absent = bool(flags.get("chickpea_absent_confirmed"))
    weed = bool(flags.get("ordinary_weed_visible") or flags.get("tall_grass_weed_visible"))
    soil = bool(flags.get("soil_visible"))
    if visible and absent:
        issues.append("chickpea_visible_conflicts_with_absent_confirmed")
    if role == "primary_three_class" and not (visible and weed and soil):
        issues.append("primary_three_class_requires_chickpea_weed_soil_visible")
    if role == "chickpea_absent_negative_control" and not (absent and weed and soil):
        issues.append("negative_control_requires_absent_weed_soil")
    if role == "exclude_with_reason" and not str(record.get("exclusion_reason", "")).strip():
        issues.append("exclude_with_reason_requires_exclusion_reason")
    confidence = str(record.get("confidence", ""))
    reviewed = record.get("reviewed")
    if not isinstance(reviewed, bool):
        issues.append("reviewed_must_be_boolean")
    if reviewed:
        if role == "unreviewed":
            issues.append("reviewed_cube_cannot_have_unreviewed_role")
        if confidence not in CONFIDENCE_VALUES:
            issues.append("reviewed_cube_requires_confidence")
        if not str(record.get("review_timestamp", "")).strip():
            issues.append("reviewed_cube_requires_timestamp")
    elif confidence and confidence not in CONFIDENCE_VALUES:
        issues.append("unknown_confidence")
    if not isinstance(record.get("source_preview_checksums"), dict):
        issues.append("source_preview_checksums_missing")
    for key in ("investigator_notes", "exclusion_reason", "review_timestamp", "reviewer_identifier"):
        if not isinstance(record.get(key, ""), str):
            issues.append(f"{key}_must_be_string")
    return issues


def validate_review_payload(payload: dict, expected_cube_ids: Iterable[str]) -> dict[str, list[str]]:
    expected = list(expected_cube_ids)
    if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), dict):
        raise ValueError("Expected an object with a reviews mapping")
    reviews = payload["reviews"]
    if set(reviews) != set(expected):
        missing = sorted(set(expected) - set(reviews))
        extra = sorted(set(reviews) - set(expected))
        raise ValueError(f"Review inventory mismatch; missing={missing}, extra={extra}")
    return {cube_id: validate_review_record(reviews[cube_id], cube_id) for cube_id in expected}


def reject_prediction_provenance(value) -> None:
    """Reject model-derived input references from a future frozen contract."""
    forbidden_keys = (
        "prediction_path", "probability_path", "checkpoint", "embedding_path",
        "field1_label", "field1_mask", "model_output",
    )
    forbidden_suffixes = (".ckpt", ".pth", ".pt", ".joblib")
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in forbidden_keys) and item not in (None, "", False, [], {}):
                raise ValueError(f"Prohibited prediction/model provenance key: {key}")
            reject_prediction_provenance(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_prediction_provenance(item)
    elif isinstance(value, str) and value.lower().endswith(forbidden_suffixes):
        raise ValueError(f"Prohibited model artifact in provenance: {value}")


def manifest_preview_hashes(row) -> dict[str, str]:
    return json.loads(str(row.preview_sha256_json))


def verify_preview_hashes(manifest: pd.DataFrame, project: Path) -> list[str]:
    issues: list[str] = []
    for row in manifest.itertuples(index=False):
        hashes = manifest_preview_hashes(row)
        for layer in PREVIEW_LAYERS:
            path = project / str(getattr(row, f"{layer}_path"))
            expected = hashes.get(layer, "")
            if not path.is_file():
                issues.append(f"{row.cube_id}:{layer}:missing")
            elif sha256(path) != expected:
                issues.append(f"{row.cube_id}:{layer}:sha256_mismatch")
    return issues


def _review_csv_bytes(payload: dict) -> bytes:
    columns = [
        "cube_id", "primary_role", *REVIEW_FLAGS, "investigator_notes", "exclusion_reason",
        "confidence", "reviewed", "review_timestamp", "reviewer_identifier",
        "source_preview_checksums",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for cube_id in sorted(payload["reviews"]):
        record = payload["reviews"][cube_id]
        row = {key: record.get(key, "") for key in columns if key not in REVIEW_FLAGS}
        row.update({name: bool(record["flags"][name]) for name in REVIEW_FLAGS})
        row["source_preview_checksums"] = json.dumps(record["source_preview_checksums"], sort_keys=True)
        writer.writerow(row)
    return buffer.getvalue().encode()


def _audit_csv_bytes(payload: dict, issues: dict[str, list[str]], hash_matches: dict[str, bool]) -> bytes:
    columns = ["cube_id", "reviewed", "primary_role", "logical_validation", "preview_hashes_match", "issues"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for cube_id in sorted(payload["reviews"]):
        record = payload["reviews"][cube_id]
        writer.writerow({
            "cube_id": cube_id, "reviewed": bool(record["reviewed"]),
            "primary_role": record["primary_role"],
            "logical_validation": "pass" if not issues[cube_id] else "fail",
            "preview_hashes_match": bool(hash_matches[cube_id]),
            "issues": "|".join(issues[cube_id]),
        })
    return buffer.getvalue().encode()


def csv_boolean(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Unrecognized CSV boolean value: {value!r}")


def review_totals(payload: dict) -> dict:
    records = list(payload["reviews"].values())
    return {
        "primary_roles": {
            role: sum(record["primary_role"] == role for record in records)
            for role in PRIMARY_ROLES
        },
        "biological_flags": {
            flag: sum(record["flags"][flag] is True for record in records)
            for flag in REVIEW_FLAGS
        },
        "confidence": {
            confidence: sum(record["confidence"] == confidence for record in records)
            for confidence in CONFIDENCE_VALUES
        },
        "reviewed": sum(record["reviewed"] is True for record in records),
    }


def validate_saved_review_products(
    payload: dict,
    csv_path: Path,
    audit_path: Path,
    expected_cube_ids: Iterable[str],
) -> dict:
    """Require JSON, CSV, and audit products to describe one identical review."""
    expected = list(expected_cube_ids)
    logical = validate_review_payload(payload, expected)
    invalid = {cube_id: values for cube_id, values in logical.items() if values}
    if invalid:
        raise ValueError(f"Logical review validation failed: {invalid}")

    frame = pd.read_csv(csv_path, keep_default_na=False)
    if len(frame) != len(expected) or frame.cube_id.nunique() != len(expected):
        raise ValueError("Review CSV has duplicate or missing cube IDs")
    if frame.cube_id.astype(str).tolist() != expected:
        raise ValueError("Review CSV cube inventory or order differs from the frozen inventory")
    scalar_fields = (
        "primary_role", "investigator_notes", "exclusion_reason", "confidence",
        "review_timestamp", "reviewer_identifier",
    )
    mismatches: list[str] = []
    for row in frame.itertuples(index=False):
        cube_id = str(row.cube_id)
        record = payload["reviews"][cube_id]
        for field in scalar_fields:
            if str(getattr(row, field)) != record[field]:
                mismatches.append(f"{cube_id}:{field}")
        if csv_boolean(row.reviewed) != record["reviewed"]:
            mismatches.append(f"{cube_id}:reviewed")
        for flag in REVIEW_FLAGS:
            if csv_boolean(getattr(row, flag)) != record["flags"][flag]:
                mismatches.append(f"{cube_id}:{flag}")
        if json.loads(str(row.source_preview_checksums)) != record["source_preview_checksums"]:
            mismatches.append(f"{cube_id}:source_preview_checksums")
    if mismatches:
        raise ValueError(f"Review CSV and JSON disagree: {mismatches}")

    audit = pd.read_csv(audit_path, keep_default_na=False)
    if len(audit) != len(expected) or audit.cube_id.nunique() != len(expected):
        raise ValueError("Review audit has duplicate or missing cube IDs")
    if audit.cube_id.astype(str).tolist() != expected:
        raise ValueError("Review audit cube inventory or order differs from the frozen inventory")
    audit_issues: list[str] = []
    for row in audit.itertuples(index=False):
        record = payload["reviews"][str(row.cube_id)]
        if str(row.logical_validation) != "pass":
            audit_issues.append(f"{row.cube_id}:logical_validation")
        if not csv_boolean(row.preview_hashes_match):
            audit_issues.append(f"{row.cube_id}:preview_hashes_match")
        if str(row.issues).strip():
            audit_issues.append(f"{row.cube_id}:issues={row.issues}")
        if csv_boolean(row.reviewed) != record["reviewed"]:
            audit_issues.append(f"{row.cube_id}:reviewed")
        if str(row.primary_role) != record["primary_role"]:
            audit_issues.append(f"{row.cube_id}:primary_role")
    if audit_issues:
        raise ValueError(f"Unresolved review audit issues: {audit_issues}")
    return review_totals(payload)


def frozen_role_table_bytes(payload: dict) -> bytes:
    return _review_csv_bytes(payload)


def role_summary_csv_bytes(totals: dict) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["summary_type", "name", "count"])
    writer.writeheader()
    for section in ("primary_roles", "biological_flags", "confidence"):
        for name, count in totals[section].items():
            writer.writerow({"summary_type": section, "name": name, "count": count})
    writer.writerow({"summary_type": "review_status", "name": "reviewed", "count": totals["reviewed"]})
    return buffer.getvalue().encode()


class CubeRoleReviewStore:
    """Resumable local store with validation, optimistic revision, and atomic exports."""

    def __init__(self, project: Path, manifest_path: Path, output_root: Path):
        self.project = project.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = pd.read_csv(manifest_path).fillna("").sort_values("cube_id")
        self.expected_cube_ids = self.manifest.cube_id.astype(str).tolist()
        self.by_cube = {str(row.cube_id): row for row in self.manifest.itertuples(index=False)}
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_root / "field2_cube_role_reviews.json"
        self.csv_path = self.output_root / "field2_cube_role_reviews.csv"
        self.audit_path = self.output_root / "field2_cube_role_review_audit.csv"
        self.overview_path = self.output_root / "field2_cube_role_review_overview.png"
        self.lock = threading.Lock()

    def public_manifest(self) -> list[dict]:
        keys = [
            "cube_id", "width", "height", "gsd_m", "valid_fraction", "bounds",
            "footprint_overlap_warning", "spectral_qc_status", "alignment_status",
            "preview_width", "preview_height", "preview_step", "preview_sha256_json",
        ]
        return [{key: row[key] for key in keys} for row in self.manifest.to_dict("records")]

    def load(self) -> dict:
        if not self.json_path.exists():
            return default_review_payload(self.manifest)
        payload = json.loads(self.json_path.read_text())
        validate_review_payload(payload, self.expected_cube_ids)
        return payload

    def layer_path(self, cube_id: str, layer: str) -> Path:
        if cube_id not in self.by_cube or layer not in PREVIEW_LAYERS:
            raise KeyError((cube_id, layer))
        return self.project / str(getattr(self.by_cube[cube_id], f"{layer}_path"))

    def _preview_hash_matches(self, cube_id: str, record: dict) -> bool:
        expected = manifest_preview_hashes(self.by_cube[cube_id])
        supplied = record.get("source_preview_checksums", {})
        if supplied != expected:
            return False
        return all(sha256(self.layer_path(cube_id, layer)) == expected[layer] for layer in PREVIEW_LAYERS)

    def write_overview(self, payload: dict, target: Path | None = None) -> None:
        target = (target or self.overview_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        colors = {
            "primary_three_class": "#2ca25f", "chickpea_absent_negative_control": "#3182bd",
            "challenge_only": "#e6550d", "sensitivity_only": "#756bb1",
            "exclude_with_reason": "#de2d26", "unreviewed": "#7f8c8d",
        }
        fig, axes = plt.subplots(5, 8, figsize=(20, 13), constrained_layout=True)
        for axis, cube_id in zip(axes.flat, self.expected_cube_ids):
            image = plt.imread(self.layer_path(cube_id, "false_colour"))
            record = payload["reviews"][cube_id]
            role = record["primary_role"]
            axis.imshow(image)
            for spine in axis.spines.values():
                spine.set_visible(True); spine.set_linewidth(5); spine.set_color(colors[role])
            status = "reviewed" if record["reviewed"] else "UNREVIEWED"
            active_flags = sum(bool(value) for value in record["flags"].values())
            axis.set_title(f"{cube_id}\n{role} · {status} · flags {active_flags}", fontsize=7)
            axis.set_xticks([]); axis.set_yticks([])
        fig.suptitle("Field 2 investigator cube-role review (prediction-free)")
        descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".png", dir=target.parent)
        os.close(descriptor)
        temporary = Path(name)
        try:
            fig.savefig(temporary, dpi=160)
            plt.close(fig)
            temporary.replace(target)
        except Exception:
            plt.close(fig); temporary.unlink(missing_ok=True); raise

    def save(self, payload: dict) -> tuple[int, int]:
        with self.lock:
            existing = self.load()
            if int(payload.get("revision", -1)) != int(existing.get("revision", 0)):
                raise ValueError("Review revision conflict; reload before saving")
            issues = validate_review_payload(payload, self.expected_cube_ids)
            invalid = {cube_id: values for cube_id, values in issues.items() if values}
            if invalid:
                raise ValueError("Logical review validation failed: " + json.dumps(invalid, sort_keys=True))
            hash_matches = {
                cube_id: self._preview_hash_matches(cube_id, payload["reviews"][cube_id])
                for cube_id in self.expected_cube_ids
            }
            failed_hashes = [cube_id for cube_id, passed in hash_matches.items() if not passed]
            if failed_hashes:
                raise ValueError(f"Preview checksum validation failed: {failed_hashes}")
            saved = json.loads(json.dumps(payload))
            saved["version"] = "field2_cube_role_reviews_v1"
            saved["revision"] = int(existing.get("revision", 0)) + 1
            saved["updated_utc"] = utc_now()
            saved["automatic_metadata"] = {
                "review_package_manifest_path": str(self.manifest_path.relative_to(self.project)),
                "review_package_manifest_sha256": sha256(self.manifest_path),
                "biological_roles_assigned_automatically": False,
                "predictions_or_probabilities_used": False,
            }
            atomic_write_json(self.json_path, saved)
            atomic_write_bytes(self.csv_path, _review_csv_bytes(saved))
            atomic_write_bytes(self.audit_path, _audit_csv_bytes(saved, issues, hash_matches))
            self.write_overview(saved)
            reviewed_count = sum(bool(record["reviewed"]) for record in saved["reviews"].values())
            return len(saved["reviews"]), reviewed_count


def stable_selection_rank(seed: int, *values: object) -> int:
    digest = hashlib.sha256(":".join([str(seed), *(str(value) for value in values)]).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def spatial_group_ids(crs: str, x: np.ndarray, y: np.ndarray, block_size_m: float) -> np.ndarray:
    if block_size_m <= 0:
        raise ValueError("Spatial block size must be positive")
    gx = np.floor(np.asarray(x, dtype=float) / block_size_m).astype(np.int64)
    gy = np.floor(np.asarray(y, dtype=float) / block_size_m).astype(np.int64)
    return np.asarray([f"{crs}:{a}:{b}" for a, b in zip(gx, gy)], dtype=object)


def geographic_ground_cell_ids(crs: str, x: np.ndarray, y: np.ndarray, cell_size_m: float) -> np.ndarray:
    if cell_size_m <= 0:
        raise ValueError("Ground-cell size must be positive")
    gx = np.floor(np.asarray(x, dtype=float) / cell_size_m).astype(np.int64)
    gy = np.floor(np.asarray(y, dtype=float) / cell_size_m).astype(np.int64)
    return np.asarray([f"{crs}:{a}:{b}" for a, b in zip(gx, gy)], dtype=object)


def scalar_index_rank_strata(values: np.ndarray, labels: Iterable[str]) -> np.ndarray:
    """Assign empirical rank quintiles; these are not biological classes."""
    values = np.asarray(values, dtype=float)
    names = list(labels)
    if len(names) != 5 or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Rank strata require five labels and finite one-dimensional values")
    ranks = rankdata(values, method="average")
    bins = np.minimum(((ranks - 1) * 5 / max(len(values), 1)).astype(int), 4)
    return np.asarray([names[index] for index in bins], dtype=object)


def deduplicate_geographic_views(frame: pd.DataFrame, seed: int, repeated_view_count: int = 0) -> pd.DataFrame:
    required = {"geographic_ground_cell_id", "cube_id", "row", "column"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Candidate frame missing columns: {sorted(required - set(frame.columns))}")
    work = frame.copy()
    work["deterministic_selection_rank"] = [
        stable_selection_rank(seed, row.geographic_ground_cell_id, row.cube_id, row.row, row.column)
        for row in work.itertuples(index=False)
    ]
    work = work.sort_values(
        ["geographic_ground_cell_id", "deterministic_selection_rank", "cube_id", "row", "column"]
    )
    group_sizes = work.groupby("geographic_ground_cell_id")["cube_id"].transform("size")
    work["overlap_status"] = np.where(group_sizes > 1, "overlapping_view", "unique_view")
    primary = work.groupby("geographic_ground_cell_id", sort=False).head(1).copy()
    primary.loc[primary.overlap_status == "overlapping_view", "overlap_status"] = "overlapping_view_selected"
    if repeated_view_count > 0:
        repeated = work.groupby("geographic_ground_cell_id", sort=False).nth(1).dropna().reset_index()
        repeated = repeated.sort_values("deterministic_selection_rank").head(repeated_view_count)
        repeated["overlap_status"] = "repeated_view_reliability"
        primary = pd.concat([primary, repeated], ignore_index=True)
    return primary.sort_values(["cube_id", "deterministic_selection_rank"]).reset_index(drop=True)


def minimum_separation_thin(frame: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    """Greedily thin in deterministic-rank order before probability sampling."""
    if distance_m <= 0:
        raise ValueError("Minimum separation must be positive")
    accepted: list[int] = []
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    ordered = frame.sort_values(["deterministic_selection_rank", "row", "column"])
    for index, row in ordered.iterrows():
        bx, by = int(np.floor(float(row.x) / distance_m)), int(np.floor(float(row.y) / distance_m))
        neighbors = [point for dx in (-1, 0, 1) for dy in (-1, 0, 1) for point in buckets.get((bx + dx, by + dy), [])]
        if any((float(row.x) - x) ** 2 + (float(row.y) - y) ** 2 < distance_m ** 2 for x, y in neighbors):
            continue
        accepted.append(index)
        buckets.setdefault((bx, by), []).append((float(row.x), float(row.y)))
    return frame.loc[accepted].copy()


def stratified_deterministic_sample(
    frame: pd.DataFrame,
    proposed_counts: dict[str, int],
    strata: Iterable[str],
    minimum_separation_m: float,
) -> pd.DataFrame:
    """Select a deterministic, spatially thinned equal-allocation frame."""
    required = {"cube_id", "cube_evaluation_role", "scalar_index_rank_stratum", "deterministic_selection_rank", "x", "y"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Candidate frame missing columns: {sorted(required - set(frame.columns))}")
    strata_names = list(strata)
    selected: list[pd.DataFrame] = []
    for cube_id, cube in frame.groupby("cube_id", sort=True):
        roles = cube.cube_evaluation_role.unique()
        if len(roles) != 1 or roles[0] not in proposed_counts:
            raise ValueError(f"{cube_id}: unknown or ambiguous evaluation role")
        total_quota = int(proposed_counts[roles[0]])
        if total_quota <= 0:
            continue
        thinned = minimum_separation_thin(cube, minimum_separation_m)
        base, remainder = divmod(total_quota, len(strata_names))
        for position, stratum in enumerate(strata_names):
            population = thinned[thinned.scalar_index_rank_stratum == stratum].sort_values(
                ["deterministic_selection_rank", "row", "column"]
            )
            quota = min(len(population), base + int(position < remainder))
            if not quota:
                continue
            sample = population.head(quota).copy()
            probability = float(quota / len(population))
            sample["inclusion_probability"] = probability
            sample["design_weight"] = 1.0 / probability
            sample["sampling_stratum_population"] = len(population)
            selected.append(sample)
    if not selected:
        return frame.head(0).assign(
            inclusion_probability=pd.Series(dtype=float), design_weight=pd.Series(dtype=float),
            sampling_stratum_population=pd.Series(dtype=int),
        )
    return pd.concat(selected, ignore_index=True)


def require_frozen_role_contract(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError("Frozen Field 2 cube-role contract is required before sampling")
    contract = yaml.safe_load(path.read_text())
    if contract.get("status") != "field2_cube_roles_frozen":
        raise RuntimeError("Cube-role contract is unfinished; sampling is prohibited")
    reject_prediction_provenance(contract.get("provenance", {}))
    if contract.get("all_cubes_reviewed") is not True:
        raise RuntimeError("Cube-role contract does not freeze all reviewed cubes")
    return contract


def _verified_contract_file(project: Path, reference: dict, label: str) -> Path:
    relative = Path(str(reference.get("path", "")))
    if not str(relative) or relative.is_absolute():
        raise ValueError(f"{label} must use a project-relative path")
    path = (project / relative).resolve()
    try:
        path.relative_to(project.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the project directory") from error
    if not path.is_file() or sha256(path) != str(reference.get("sha256", "")):
        raise ValueError(f"{label} is missing or its SHA-256 changed")
    return path


def validate_frozen_role_contract(project: Path, config: dict, contract_path: Path) -> dict:
    """Validate a frozen role contract and every hashed input/output dependency."""
    contract = yaml.safe_load(contract_path.read_text())
    if contract.get("version") != "field2_cube_role_contract_v1":
        raise ValueError("Unknown frozen role-contract version")
    if contract.get("status") != "field2_cube_roles_frozen":
        raise ValueError("Field 2 cube roles are not frozen")
    expected = list(config["expected_cube_ids"])
    rows = contract.get("cube_roles")
    if not isinstance(rows, list) or [row.get("cube_id") for row in rows] != expected:
        raise ValueError("Frozen role contract does not contain the exact ordered 40-cube inventory")
    if len({row["cube_id"] for row in rows}) != len(expected):
        raise ValueError("Frozen role contract contains duplicate cube IDs")
    payload = {
        "version": contract.get("review_schema_version"),
        "reviews": {row["cube_id"]: row for row in rows},
    }
    issues = validate_review_payload(payload, expected)
    invalid = {cube_id: values for cube_id, values in issues.items() if values}
    if invalid:
        raise ValueError(f"Frozen role contract has invalid reviews: {invalid}")
    totals = review_totals(payload)
    if totals["reviewed"] != len(expected):
        raise ValueError("Frozen role contract does not mark all 40 cubes reviewed")
    if totals["primary_roles"].get("unreviewed"):
        raise ValueError("Frozen role contract contains an unreviewed primary role")
    if totals["primary_roles"] != contract.get("role_totals"):
        raise ValueError("Frozen role totals differ from the contract summary")
    if totals["biological_flags"] != contract.get("biological_flag_totals"):
        raise ValueError("Frozen biological-flag totals differ from the contract summary")
    for name, expected_count in config["freeze"]["expected_role_totals"].items():
        if totals["primary_roles"].get(name) != int(expected_count):
            raise ValueError(f"Unexpected frozen role total: {name}")
    for name, expected_count in config["freeze"]["expected_flag_totals"].items():
        if totals["biological_flags"].get(name) != int(expected_count):
            raise ValueError(f"Unexpected frozen biological-flag total: {name}")
    if not str(contract.get("freeze_timestamp_utc", "")).strip():
        raise ValueError("Frozen role contract has no freeze timestamp")
    commit = str(contract.get("freeze_git_commit", ""))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Frozen role contract has no exact Git commit")
    if contract.get("review_schema_version") != "field2_cube_role_reviews_v1":
        raise ValueError("Frozen role contract has an unexpected review schema")
    provenance = contract.get("provenance", {})
    reject_prediction_provenance(provenance)
    if provenance.get("prediction_free_review") is not True:
        raise ValueError("Frozen role contract does not declare prediction-free review")
    if provenance.get("supervised_checkpoint_loaded") is not False:
        raise ValueError("Frozen role contract does not declare that checkpoints were not loaded")
    if provenance.get("biological_roles_assigned_automatically") is not False:
        raise ValueError("Frozen role contract permits automatic biological roles")

    inputs = contract.get("frozen_input_products", {})
    verified_inputs = {}
    for name in ("review_json", "review_csv", "review_audit", "review_overview", "review_package_manifest"):
        verified_inputs[name] = _verified_contract_file(
            project, inputs.get(name, {}), f"frozen input {name}",
        )
    outputs = contract.get("frozen_output_products", {})
    verified_outputs = {}
    for name in ("role_table", "role_summary", "frozen_overview"):
        verified_outputs[name] = _verified_contract_file(
            project, outputs.get(name, {}), f"frozen output {name}",
        )
    source_payload = json.loads(verified_inputs["review_json"].read_text())
    source_totals = validate_saved_review_products(
        source_payload, verified_inputs["review_csv"], verified_inputs["review_audit"], expected,
    )
    if source_payload.get("version") != contract["review_schema_version"]:
        raise ValueError("Frozen source review schema differs from the role contract")
    if source_payload.get("revision") != contract.get("review_revision"):
        raise ValueError("Frozen source review revision differs from the role contract")
    if source_payload["reviews"] != payload["reviews"] or source_totals != totals:
        raise ValueError("Frozen source reviews differ from the role contract")
    table_totals = validate_saved_review_products(
        payload, verified_outputs["role_table"], verified_inputs["review_audit"], expected,
    )
    if table_totals != totals:
        raise ValueError("Frozen role table differs from the role contract")
    summary = pd.read_csv(verified_outputs["role_summary"], keep_default_na=False)
    expected_summary = pd.read_csv(io.BytesIO(role_summary_csv_bytes(totals)), keep_default_na=False)
    if not summary.equals(expected_summary):
        raise ValueError("Frozen role summary differs from the role contract")
    if inputs["review_package_manifest"]["sha256"] != sha256(project / config["review"]["package_manifest"]):
        raise ValueError("Review-package manifest differs from the configured package")
    if contract.get("field2_source_manifest_sha256") != config["source_manifest_sha256"]:
        raise ValueError("Field 2 source-manifest hash differs from the frozen configuration")
    support_path = project / config["inputs"]["valid_support_manifest"]
    if contract.get("field2_valid_support_manifest_sha256") != sha256(support_path):
        raise ValueError("Field 2 valid-support manifest changed after role freeze")

    manifest = pd.read_csv(project / config["review"]["package_manifest"]).fillna("")
    preview_by_cube = {
        str(row.cube_id): manifest_preview_hashes(row)
        for row in manifest.itertuples(index=False)
    }
    for row in rows:
        if row["source_preview_checksums"] != preview_by_cube.get(row["cube_id"]):
            raise ValueError(f"Frozen preview-checksum reference changed: {row['cube_id']}")
    return {"contract": contract, "totals": totals}
