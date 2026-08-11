import numpy as np
from affine import Affine

from scripts.audit_investigator_vegetation_references import (
    aggregate_reference_spectrum,
    disk_indices,
)


def test_disk_indices_uses_map_space_radius():
    transform = Affine(0.01, 0, 100, 0, -0.01, 200)
    rows, columns = disk_indices((20, 20), transform, 100.105, 199.895, 0.011)
    assert set(zip(rows, columns)) == {
        (9, 10), (10, 9), (10, 10), (10, 11), (11, 10)
    }


def test_aggregate_excludes_soil_and_nodata_and_returns_one_median():
    values = np.asarray([
        [1, 10], [3, 30], [9, 90], [0, 0], [np.nan, 4],
    ], dtype=np.float32)
    soil = np.asarray([False, False, True, False, False])
    spectrum, qc = aggregate_reference_spectrum(values, soil)
    np.testing.assert_allclose(spectrum, [2, 20])
    assert qc["sampled_pixels"] == 5
    assert qc["usable_nonsoil_pixels"] == 2
    assert qc["soil_pixels"] == 1
    assert qc["nodata_or_nonfinite_pixels"] == 2
