import numpy as np
import pytest

from chickpea_ssl.spatial import map_block_indices, spatial_group_id


def test_overlapping_observations_share_global_group() -> None:
    x = np.array([500001.2, 500001.2, 500006.2])
    y = np.array([3699001.2, 3699001.2, 3699001.2])
    block_x, block_y = map_block_indices(x, y, block_size_m=5.0)
    assert (block_x[0], block_y[0]) == (block_x[1], block_y[1])
    assert block_x[2] == block_x[0] + 1


def test_nonpositive_block_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        map_block_indices(np.array([1.0]), np.array([2.0]), block_size_m=0)


def test_group_id_does_not_include_cube_identity() -> None:
    assert spatial_group_id("EPSG:32617", 3, 4) == "EPSG_32617_x+0000003_y+0000004"
