"""Map-coordinate spatial groups shared by overlapping hyperspectral cubes."""

from __future__ import annotations

import numpy as np


def map_block_indices(
    x: np.ndarray,
    y: np.ndarray,
    block_size_m: float,
    origin_x_m: float = 0.0,
    origin_y_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return globally anchored integer block columns and rows."""
    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive")
    block_x = np.floor((np.asarray(x) - origin_x_m) / block_size_m).astype(np.int64)
    block_y = np.floor((np.asarray(y) - origin_y_m) / block_size_m).astype(np.int64)
    return block_x, block_y


def spatial_group_id(crs_token: str, block_x: int, block_y: int) -> str:
    """Create a stable group identifier without encoding cube identity."""
    token = "".join(character if character.isalnum() else "_" for character in crs_token)
    return f"{token}_x{block_x:+08d}_y{block_y:+08d}"


def neighbour_boundary_safe_mask(
    x: np.ndarray,
    y: np.ndarray,
    block_x: np.ndarray,
    block_y: np.ndarray,
    fold_lookup: dict[tuple[int, int], int],
    forbidden_neighbour_folds: set[int],
    block_size_m: float,
    origin_x_m: float,
    origin_y_m: float,
    buffer_m: float,
) -> np.ndarray:
    """Flag centres farther than ``buffer_m`` from forbidden neighbouring folds."""
    safe = np.ones(len(x), dtype=bool)
    local_x = x - (origin_x_m + block_x * block_size_m)
    local_y = y - (origin_y_m + block_y * block_size_m)
    for offset_x in (-1, 0, 1):
        for offset_y in (-1, 0, 1):
            if offset_x == 0 and offset_y == 0:
                continue
            neighbour_fold = np.fromiter((
                fold_lookup.get((int(one_x + offset_x), int(one_y + offset_y)), 0)
                for one_x, one_y in zip(block_x, block_y)
            ), dtype=np.int8, count=len(block_x))
            forbidden = np.isin(neighbour_fold, list(forbidden_neighbour_folds))
            if not forbidden.any():
                continue
            distance_x = (
                local_x if offset_x < 0 else
                block_size_m - local_x if offset_x > 0 else np.zeros_like(local_x)
            )
            distance_y = (
                local_y if offset_y < 0 else
                block_size_m - local_y if offset_y > 0 else np.zeros_like(local_y)
            )
            safe &= ~(forbidden & (np.hypot(distance_x, distance_y) <= buffer_m))
    return safe
