from pathlib import Path

import numpy as np
from affine import Affine

from chickpea_ssl.field2_readiness import (
    ReadOnlySourceGuard,
    affine_comparison,
    compare_snapshots,
    deterministic_coordinates,
    dump_yaml,
    expected_payload_bytes,
    match_wavelengths,
    normalized_cube_id,
    open_envi_memmap,
    parse_envi_header,
    parse_wavelengths,
    resolve_envi_payload,
    source_snapshot,
)
from scripts.audit_field2_georectified_inventory import sampled_band_rows


def write_envi_pair(root: Path, interleave: str = "bip") -> tuple[Path, Path]:
    header = root / "kusi_field2_polygon_Pika_L-GigE_7.bil-Georectify.bip.hdr"
    payload = Path(str(header)[:-4])
    header.write_text(
        "ENVI\n"
        "samples = 3\nlines = 2\nbands = 2\n"
        "header offset = 0\ndata type = 12\nbyte order = 0\n"
        f"interleave = {interleave}\n"
        "wavelength = {401.84, 405.91}\n"
        "data ignore value = 65535\n"
    )
    canonical = np.arange(12, dtype="<u2").reshape(2, 3, 2)
    encoded = canonical if interleave == "bip" else (
        np.moveaxis(canonical, 2, 1) if interleave == "bil" else np.moveaxis(canonical, 2, 0)
    )
    payload.write_bytes(encoded.tobytes())
    return header, payload


def test_writable_source_is_permitted_for_read_only_audit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "writable.bin"
    path.write_bytes(b"science")
    guard = ReadOnlySourceGuard((source,))
    assert guard.source(path, "rb") == path.resolve()


def test_output_inside_source_root_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    guard = ReadOnlySourceGuard((source,))
    try:
        guard.output(source / "report.csv")
    except ValueError as error:
        assert "inside" in str(error)
    else:
        raise AssertionError("Output inside source root was accepted")


def test_update_source_modes_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "cube.bip"
    path.write_bytes(b"1234")
    guard = ReadOnlySourceGuard((source,))
    for mode in ("r+", "w", "wb", "a", "x"):
        try:
            guard.source(path, mode)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Update mode accepted: {mode}")


def test_read_only_memmap_and_all_interleaves(tmp_path):
    for interleave in ("bip", "bil", "bsq"):
        root = tmp_path / interleave
        root.mkdir()
        header, payload = write_envi_pair(root, interleave)
        guard = ReadOnlySourceGuard((root,))
        array, metadata = open_envi_memmap(header, payload, guard)
        assert array.shape == (2, 3, 2)
        assert not array.flags.writeable
        assert np.array_equal(array, np.arange(12, dtype=np.uint16).reshape(2, 3, 2))
        assert expected_payload_bytes(metadata) == payload.stat().st_size


def test_header_payload_resolution_and_ambiguous_pair(tmp_path):
    header, payload = write_envi_pair(tmp_path)
    guard = ReadOnlySourceGuard((tmp_path,))
    assert resolve_envi_payload(header, guard) == payload.resolve()
    metadata = parse_envi_header(header, guard)
    assert np.array_equal(parse_wavelengths(metadata), [401.84, 405.91])
    extra = Path(str(payload) + ".bip")
    extra.write_bytes(b"duplicate")
    try:
        resolve_envi_payload(header, guard)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("Ambiguous payload resolution was accepted")


def test_wavelength_matching_and_cube_normalization():
    matches = match_wavelengths(np.array([401.84, 405.91]), np.array([401.84, 405.92]), 0.02)
    assert [row["match_status"] for row in matches] == ["matched", "matched"]
    assert all(row["ordering_status"] == "strictly_increasing" for row in matches)
    assert normalized_cube_id("Pika_L-GigE_7.bil-Georectify.bip") == "field2_cube07"


def test_affine_comparison_and_deterministic_sampling():
    reference = Affine(0.02, 0, 1, 0, -0.02, 2)
    assert affine_comparison(reference, reference, 1e-9, 1e-7) == "exact_alignment"
    numerical = Affine(0.02, 0, 1 + 5e-8, 0, -0.02, 2)
    assert affine_comparison(reference, numerical, 1e-9, 1e-7) == "numerically_equivalent"
    first = deterministic_coordinates(20, 30, 100, 42, "field2_cube07")
    second = deterministic_coordinates(20, 30, 100, 42, "field2_cube07")
    assert np.array_equal(first, second)
    assert len(np.unique(first, axis=0)) == 100


def test_nodata_handling_excludes_declared_value():
    array = np.array([[[1.0], [65535.0]], [[3.0], [np.nan]]])
    coordinates = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    row = sampled_band_rows("field2_cube07", "stored_index", array,
                            {"data ignore value": "65535"}, coordinates,
                            np.ones(4, dtype=bool))[0]
    assert row["valid_observation_count"] == 2
    assert row["nodata_fraction"] == 0.25
    assert row["nan_fraction"] == 0.25


def test_snapshot_checksum_change_is_detected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "cube.bip"
    path.write_bytes(b"before")
    guard = ReadOnlySourceGuard((source,))
    before = source_snapshot(guard, tmp_path)
    path.write_bytes(b"after")
    after = source_snapshot(guard, tmp_path)
    issues = compare_snapshots(before, after)
    assert any(issue.startswith("changed_sha256:") for issue in issues)


def test_numpy_scalars_serialize_to_yaml(tmp_path):
    output = tmp_path / "contract.yaml"
    dump_yaml({"integer": np.int64(7), "floating": np.float32(1.5), "flag": np.bool_(True)}, output)
    text = output.read_text()
    assert "integer: 7" in text
    assert "floating: 1.5" in text
    assert "flag: true" in text
