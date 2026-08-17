from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from chickpea_ssl.field2_readiness import (
    ReadOnlySourceGuard,
    compare_snapshots,
    dump_yaml,
    sha256,
    source_snapshot,
)
from chickpea_ssl.field2_valid_support import (
    component_metrics,
    disagreement_metrics,
    grid_identity,
    model_band_indices,
    support_masks_windowed,
    verify_mask_geotiff,
    write_mask_geotiff,
)


def synthetic_spectra() -> np.ndarray:
    array = np.zeros((2, 3, 5), dtype=np.float32)
    array[0, 1, 1:4] = [0, 2, 3]       # partial zero, otherwise supported
    array[0, 2, 1:4] = [-2, -3, -4]    # negative, finite, supported
    array[1, 0, 1:4] = [1, 1, 1]       # low amplitude sensitivity case
    array[1, 1, 1:4] = [np.nan, 2, 3]  # nonfinite, unsupported
    array[1, 2, 0] = 9                  # full-spectrum-only support
    return array


def test_exact_zero_partial_zero_nonfinite_and_negative_spectra():
    masks, diagnostics = support_masks_windowed(
        synthetic_spectra(), [1, 2, 3], [0, 1, 5, 10], window_rows=1
    )
    primary = masks["finite_positive_norm_gt_0"]
    assert not primary[0, 0]
    assert primary[0, 1]
    assert not masks["finite_all_nonzero"][0, 1]
    assert primary[0, 2]
    assert not primary[1, 1]
    assert not primary[1, 2]
    assert masks["full_spectrum_any_nonzero"][1, 2]
    assert diagnostics["partial_zero_spectrum_count"] == 1
    assert diagnostics["negative_value_spectrum_count"] == 1
    assert diagnostics["finite_failure_count"] == 1
    assert diagnostics["exact_all_zero_spectrum_count"] == 2


def test_threshold_sensitivity_and_windowed_processing():
    array = synthetic_spectra()
    windowed, diagnostics = support_masks_windowed(array, [1, 2, 3], [0, 1, 5, 10], 1)
    whole, _ = support_masks_windowed(array, [1, 2, 3], [0, 1, 5, 10], 20)
    assert diagnostics["window_count"] == 2
    assert all(np.array_equal(windowed[name], whole[name]) for name in windowed)
    assert windowed["finite_positive_norm_gt_0"][1, 0]
    assert not windowed["finite_positive_norm_gt_1"][1, 0]
    assert not windowed["finite_positive_norm_gt_5"].any()
    assert disagreement_metrics(
        windowed["finite_positive_norm_gt_0"], windowed["finite_positive_norm_gt_1"]
    )["disagreement_count"] == 1


def test_envi_and_python_band_index_conversion_is_explicit():
    indices = model_band_indices(3, 113, 150)
    assert len(indices) == 111
    assert (indices[0], indices[-1]) == (3, 113)
    assert (indices[0] + 1, indices[-1] + 1) == (4, 114)
    for invalid in ((-1, 3), (5, 4), (3, 150)):
        try:
            model_band_indices(*invalid, 150)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid band range accepted: {invalid}")


def test_connected_components_holes_and_boundary_are_reported():
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:8, 2:8] = True
    mask[4, 4] = False
    mask[0, 0] = True
    metrics = component_metrics(mask, Affine(2, 0, 10, 0, -2, 20), 8, 1)
    assert metrics["connected_component_count"] == 2
    assert metrics["small_component_count"] == 1
    assert metrics["small_component_pixels"] == 1
    assert metrics["non_small_component_count"] == 1
    assert metrics["non_small_interior_component_count"] == 1
    assert metrics["interior_hole_count"] == 1
    assert metrics["interior_hole_pixels"] == 1
    assert metrics["boundary_pixel_count"] > 0
    assert metrics["geographic_left"] == 10
    assert metrics["geographic_top"] == 20


def test_grid_identity_detects_transform_and_dimension_changes():
    reference = {
        "width": 3, "height": 2, "count": 1, "crs": "EPSG:32617",
        "transform": Affine(0.02, 0, 1, 0, -0.02, 2), "bounds": (1, 1.96, 1.06, 2),
    }
    assert grid_identity(reference, dict(reference))[0]
    shifted = dict(reference, transform=Affine(0.02, 0, 1.02, 0, -0.02, 2))
    assert "transform_mismatch" in grid_identity(reference, shifted)[1]
    resized = dict(reference, width=4)
    assert "width_mismatch" in grid_identity(reference, resized)[1]


def test_output_path_source_rejection_and_read_only_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    item = source / "cube.bin"
    item.write_bytes(b"immutable")
    guard = ReadOnlySourceGuard((source,))
    assert guard.source(item, "rb") == item.resolve()
    try:
        guard.output(source / "mask.tif")
    except ValueError as error:
        assert "inside" in str(error)
    else:
        raise AssertionError("Mask output inside source directory was accepted")


def test_deterministic_uint8_geotiff_codes_and_grid(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "reflectance.bin"
    source_file.write_bytes(b"source")
    guard = ReadOnlySourceGuard((source,))
    transform = Affine(0.02, 0, 487000, 0, -0.02, 3669000)
    reference = {
        "width": 4, "height": 3, "count": 1, "crs": rasterio.crs.CRS.from_epsg(32617),
        "transform": transform,
        "bounds": rasterio.transform.array_bounds(3, 4, transform),
    }
    mask = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool)
    first, second = tmp_path / "first.tif", tmp_path / "second.tif"
    for path in (first, second):
        write_mask_geotiff(path, mask, reference, guard, "a" * 64,
                           "finite_positive_norm_gt_0", "b" * 64, [3, 4])
        passed, issues, codes = verify_mask_geotiff(path, reference)
        assert passed, issues
        assert codes == {0, 1}
    assert sha256(first) == sha256(second)


def test_numpy_yaml_and_before_after_checksums(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    item = source / "cube.bin"
    item.write_bytes(b"before")
    guard = ReadOnlySourceGuard((source,))
    before = source_snapshot(guard, tmp_path)
    assert compare_snapshots(before, source_snapshot(guard, tmp_path)) == []
    item.write_bytes(b"after")
    assert any(value.startswith("changed_sha256:") for value in compare_snapshots(
        before, source_snapshot(guard, tmp_path)
    ))
    yaml_path = tmp_path / "contract.yaml"
    dump_yaml({"count": np.int64(2), "fraction": np.float64(0.5)}, yaml_path)
    assert "count: 2" in yaml_path.read_text()
