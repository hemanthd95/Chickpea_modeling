"""Deterministic utilities for frozen-model spectral reliance audits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ("soil", "chickpea", "weed")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wavelength_groups(wavelengths: np.ndarray, target_width_nm: float) -> pd.DataFrame:
    """Return contiguous physical-wavelength groups, merging a short last bin."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.ndim != 1 or len(wavelengths) < 2 or not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Wavelengths must be a strictly increasing vector")
    if not 10 <= target_width_nm <= 15:
        raise ValueError("Target wavelength-group width must be 10–15 nm")
    identifiers = np.floor((wavelengths - wavelengths[0]) / target_width_nm).astype(int)
    if identifiers[-1] != identifiers[-2]:
        last = wavelengths[identifiers == identifiers[-1]]
        if last[-1] - last[0] < target_width_nm / 2:
            identifiers[identifiers == identifiers[-1]] = identifiers[-2]
    unique = pd.unique(identifiers)
    remap = {old: new for new, old in enumerate(unique)}
    rows = []
    for old in unique:
        indices = np.flatnonzero(identifiers == old)
        rows.append({
            "wavelength_group_id": f"wg{remap[old] + 1:02d}",
            "first_input_channel": int(indices[0]),
            "last_input_channel": int(indices[-1]),
            "band_count": int(len(indices)),
            "wavelength_min_nm": float(wavelengths[indices[0]]),
            "wavelength_max_nm": float(wavelengths[indices[-1]]),
            "wavelength_center_nm": float(wavelengths[indices].mean()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(truth, probabilities, bins: int = 15, weights=None) -> float:
    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    weights = np.ones(len(truth), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    confidence = probability.max(axis=1)
    predicted = probability.argmax(axis=1)
    result = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for number in range(bins):
        selected = (confidence >= edges[number]) & (
            confidence <= edges[number + 1] if number == bins - 1 else confidence < edges[number + 1]
        )
        if not selected.any():
            continue
        mass = weights[selected].sum()
        result += mass * abs(
            np.average((predicted[selected] == truth[selected]).astype(float), weights=weights[selected])
            - np.average(confidence[selected], weights=weights[selected])
        )
    return float(result)


def prediction_metrics(truth, probabilities, groups, bins: int = 15) -> dict[str, float]:
    """Equal-spatial-group classification and probability metrics."""
    truth = np.asarray(truth, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups).astype(str)
    predicted = probability.argmax(axis=1)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(unique) * counts[inverse])
    matrix = np.zeros((3, 3), float)
    for group in unique:
        selected = groups == group
        local = confusion_matrix(truth[selected], predicted[selected], labels=[0, 1, 2]).astype(float)
        matrix += local / max(local.sum(), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1e-12)
    precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    selected_probability = np.clip(probability[np.arange(len(truth)), truth], 1e-12, 1)
    one_hot = np.eye(3)[truth]
    result = {
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "negative_log_likelihood": float(np.sum(weights * -np.log(selected_probability))),
        "brier_score": float(np.sum(weights * np.sum((probability - one_hot) ** 2, axis=1))),
        "expected_calibration_error": expected_calibration_error(truth, probability, bins, weights),
    }
    for index, name in enumerate(CLASS_NAMES):
        result[f"{name}_f1"] = float(f1[index])
        result[f"{name}_recall"] = float(recall[index])
    return result


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def deterministic_audit_support(frame: pd.DataFrame, per_class_group: int, maximum: int, seed: int) -> pd.DataFrame:
    """Round-robin support across spatial group/class, without pixel duplication."""
    one = frame.copy()
    one["_audit_hash"] = [
        int.from_bytes(hashlib.sha256(f"{seed}|{r.cube_id}|{r.row}|{r.column}".encode()).digest()[:8], "little")
        for r in one.itertuples()
    ]
    one = one.sort_values(["_audit_hash", "cube_id", "row", "column"])
    one = one.drop_duplicates(["ground_x", "ground_y"], keep="first")
    one["_audit_rank"] = one.groupby(["spatial_group_id", "class_id"], sort=True).cumcount()
    one = one[one._audit_rank < per_class_group]
    one = one.sort_values(["_audit_rank", "spatial_group_id", "class_id", "_audit_hash"]).head(maximum)
    return one.drop(columns=["_audit_hash", "_audit_rank"]).sort_values(
        ["cube_id", "spatial_group_id", "class_id", "row", "column"]
    ).reset_index(drop=True)
