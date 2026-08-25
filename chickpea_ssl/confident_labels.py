"""Pure rules for the versioned Field 1 confident-label product."""

from __future__ import annotations

import hashlib

import numpy as np


UNRESOLVED = np.uint8(255)
SOIL, CHICKPEA, WEED = np.uint8(0), np.uint8(1), np.uint8(2)

PROVENANCE = {
    0: "unresolved",
    1: "authoritative_soil_mask",
    2: "validated_scalar_index_expansion_soil",
    3: "alley_core_authoritative_soil",
    4: "alley_core_expansion_soil",
    5: "alley_core_valid_nonsoil_weed",
    6: "polygon_core_probability_chickpea",
    7: "polygon_core_probability_weed",
    8: "investigator_point_chickpea",
    9: "investigator_point_weed",
}


def stable_uint64(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def compose_confident_labels(
    *,
    valid: np.ndarray,
    polygon_core: np.ndarray,
    polygon_edge: np.ndarray,
    alley_full: np.ndarray,
    alley_core: np.ndarray,
    soil: np.ndarray,
    soil_evidence_valid: np.ndarray,
    probability: np.ndarray,
    authoritative_soil: bool,
    weed_maximum_probability: float = 0.15,
    chickpea_minimum_probability: float = 0.80,
    allow_dense_probability: bool = True,
    enrich_alleys: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen precedence and return labels, provenance, conflicts.

    Conflict bits are: 1=probable chickpea in an alley, 2=soil/probable
    vegetation disagreement, and 4=an alley-core pixel lacked usable soil
    evidence.  Conflicts never override the declared precedence.
    """
    arrays = (
        polygon_core, polygon_edge, alley_full, alley_core, soil,
        soil_evidence_valid, probability,
    )
    if any(value.shape != valid.shape for value in arrays):
        raise ValueError("All confident-label inputs must have one raster shape")
    if not 0 <= weed_maximum_probability < chickpea_minimum_probability <= 1:
        raise ValueError("Probability thresholds must satisfy 0 <= weed < chickpea <= 1")
    if np.any(alley_core & ~alley_full):
        raise ValueError("The inward alley core must be a subset of the full alley")

    labels = np.full(valid.shape, UNRESOLVED, dtype=np.uint8)
    provenance = np.zeros(valid.shape, dtype=np.uint8)
    conflicts = np.zeros(valid.shape, dtype=np.uint8)
    finite_probability = np.isfinite(probability) & (probability >= 0) & (probability <= 1)
    probable_chickpea = finite_probability & (probability >= chickpea_minimum_probability)
    probable_weed = finite_probability & (probability <= weed_maximum_probability)

    # Authoritative/provisionally validated soil is the only class allowed to
    # resolve polygon edges and the uncertain outer ring of an alley.
    resolved_soil = valid & soil_evidence_valid & soil
    labels[resolved_soil] = SOIL
    provenance[resolved_soil] = 1 if authoritative_soil else 2

    conflicts[valid & alley_full & probable_chickpea] |= np.uint8(1)
    conflicts[resolved_soil & finite_probability & (probable_chickpea | probable_weed)] |= np.uint8(2)
    conflicts[valid & alley_core & ~soil_evidence_valid] |= np.uint8(4)

    if enrich_alleys:
        core_soil = valid & alley_core & soil_evidence_valid & soil
        core_weed = valid & alley_core & soil_evidence_valid & ~soil
        labels[core_soil] = SOIL
        provenance[core_soil] = 3 if authoritative_soil else 4
        labels[core_weed] = WEED
        provenance[core_weed] = 5

    if allow_dense_probability:
        probability_scope = valid & polygon_core & ~polygon_edge & ~alley_full
        chickpea = probability_scope & ~soil & finite_probability & probable_chickpea
        weed = probability_scope & ~soil & finite_probability & probable_weed
        labels[chickpea] = CHICKPEA
        provenance[chickpea] = 6
        labels[weed] = WEED
        provenance[weed] = 7

    # Full alleys always prohibit chickpea, including when enrichment is off.
    forbidden = valid & alley_full & (labels == CHICKPEA)
    labels[forbidden] = UNRESOLVED
    provenance[forbidden] = 0
    labels[~valid] = UNRESOLVED
    provenance[~valid] = 0
    return labels, provenance, conflicts


def apply_confirmed_points(
    labels: np.ndarray,
    provenance: np.ndarray,
    points: list[tuple[int, int, int]],
    *,
    valid: np.ndarray,
    alley_full: np.ndarray,
    soil: np.ndarray,
) -> list[dict[str, int | str]]:
    """Apply sparse investigator points without violating stronger evidence."""
    conflicts: list[dict[str, int | str]] = []
    height, width = labels.shape
    for row, column, class_id in points:
        if not (0 <= row < height and 0 <= column < width) or not valid[row, column]:
            conflicts.append({"row": row, "column": column, "reason": "invalid_or_outside"})
            continue
        if class_id == 1 and alley_full[row, column]:
            conflicts.append({"row": row, "column": column, "reason": "chickpea_point_in_alley"})
            continue
        if soil[row, column] and class_id != 0:
            conflicts.append({"row": row, "column": column, "reason": "vegetation_point_on_soil"})
            continue
        if labels[row, column] != UNRESOLVED and labels[row, column] != class_id:
            conflicts.append({"row": row, "column": column, "reason": "existing_label_conflict"})
            continue
        labels[row, column] = np.uint8(class_id)
        provenance[row, column] = np.uint8(8 if class_id == 1 else 9)
    return conflicts


def deterministic_balanced_selection(
    frame,
    count_per_class: int,
    seed: int,
):
    """Balance cube/group strata within class without replacement."""
    import pandas as pd

    selected = []
    for class_id, group in frame.groupby("class_id", sort=True):
        # A boundary pixel can be associated with more than one spatial group.
        # Resolve that ambiguity deterministically before stratified selection so
        # no cube pixel is sampled twice under different group identifiers.
        group = (
            group.sort_values(["cube_id", "row", "column", "spatial_group_id"])
            .drop_duplicates(["cube_id", "row", "column"], keep="first")
        )
        if len(group) < count_per_class:
            raise ValueError(
                f"Class {class_id} has {len(group)} unique candidates; "
                f"{count_per_class} requested"
            )
        one = group.copy()
        one["_stable"] = [
            stable_uint64(seed, class_id, row.cube_id, row.row, row.column)
            for row in one.itertuples()
        ]
        one = one.sort_values(
            ["cube_id", "spatial_group_id", "_stable", "row", "column"]
        )
        one["_stratum_rank"] = one.groupby(
            ["cube_id", "spatial_group_id"], sort=True
        ).cumcount()
        one = one.sort_values(
            ["_stratum_rank", "cube_id", "spatial_group_id", "_stable"]
        ).head(count_per_class)
        selected.append(one.drop(columns=["_stable", "_stratum_rank"]))
    return pd.concat(selected, ignore_index=True).sort_values(
        ["class_id", "cube_id", "spatial_group_id", "row", "column"]
    ).reset_index(drop=True)
