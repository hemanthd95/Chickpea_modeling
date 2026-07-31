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

