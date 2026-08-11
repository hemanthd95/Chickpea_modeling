import numpy as np
from affine import Affine

from scripts.build_investigator_chickpea_probability_maps import (
    dense_neighbourhood_features,
    map_disk_offsets,
)


def test_map_disk_offsets_respects_metric_radius():
    transform = Affine(0.015, 0, 0, 0, -0.015, 0)
    offsets = map_disk_offsets(transform, 0.03)
    assert [0, 0] in offsets.tolist()
    assert [0, 2] in offsets.tolist()
    assert [2, 0] in offsets.tolist()
    assert [2, 2] not in offsets.tolist()


def test_dense_features_exclude_soil_and_use_bandwise_median():
    cube = np.zeros((3, 3, 2), dtype=np.float32)
    cube[..., 0] = np.arange(9).reshape(3, 3) + 1
    cube[..., 1] = (np.arange(9).reshape(3, 3) + 1) * 10
    soil = np.zeros((3, 3), dtype=bool)
    soil[1, 0] = True
    allowed = np.ones((3, 3), dtype=bool)
    offsets = np.asarray([[0, 0], [0, -1], [0, 1]], dtype=np.int32)
    features, counts = dense_neighbourhood_features(
        cube, np.asarray([[1, 1]], dtype=np.int32), np.asarray([0, 1]),
        soil, allowed, offsets, minimum_usable_pixels=2,
    )
    # Values 4 (soil), 5, 6 become observed usable values 5 and 6.
    assert counts.tolist() == [2]
    np.testing.assert_allclose(features[0], [5.5, 55.0])


def test_dense_features_marks_insufficient_neighbourhood_invalid():
    cube = np.ones((2, 2, 1), dtype=np.float32)
    soil = np.zeros((2, 2), dtype=bool)
    allowed = np.ones((2, 2), dtype=bool)
    features, counts = dense_neighbourhood_features(
        cube, np.asarray([[0, 0]], dtype=np.int32), np.asarray([0]),
        soil, allowed, np.asarray([[0, 0]], dtype=np.int32),
        minimum_usable_pixels=2,
    )
    assert counts.tolist() == [1]
    assert np.isnan(features[0, 0])
