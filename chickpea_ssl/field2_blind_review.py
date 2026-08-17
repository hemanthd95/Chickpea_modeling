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

    def _write_overview(self, payload: dict) -> None:
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
        descriptor, name = tempfile.mkstemp(prefix=f".{self.overview_path.name}.", suffix=".png", dir=self.output_root)
        os.close(descriptor)
        temporary = Path(name)
        try:
            fig.savefig(temporary, dpi=160)
            plt.close(fig)
            temporary.replace(self.overview_path)
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
            self._write_overview(saved)
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
