import json
from pathlib import Path

import pandas as pd

from scripts.run_vegetation_reference_annotator import Store


def test_store_saves_synchronized_point_exports(tmp_path: Path):
    layer = tmp_path / "layer.png"
    layer.write_bytes(b"png")
    manifest_path = tmp_path / "manifest.csv"
    layer_names = (
        "false_colour", "current_chickpea", "pca",
        "first_difference", "second_difference", "ndvi",
    )
    pd.DataFrame([{
        "cube_id": "field1_cube20",
        "analysis_role": "primary_candidate",
        "preview_width": 100,
        "preview_height": 80,
        "preview_step": 2,
        "transform_a": 0.015,
        "transform_b": 0.0,
        "transform_c": 487000.0,
        "transform_d": 0.0,
        "transform_e": -0.015,
        "transform_f": 3669000.0,
        "crs": "EPSG:32617",
        **{f"{name}_path": str(layer) for name in layer_names},
    }]).to_csv(manifest_path, index=False)
    references = tmp_path / "references.json"
    references.write_text(json.dumps({"version": 1, "cubes": {}}))
    store = Store(
        manifest_path, tmp_path / "output", references, references
    )

    payload = {"version": 1, "cubes": {"field1_cube20": [{
        "id": "point-1",
        "kind": "confirmed_chickpea",
        "layer": "false_colour",
        "radius_m": 0.03,
        "note": "clean plant interior",
        "x": 10.0,
        "y": 20.0,
    }]}}
    assert store.save(payload) == (1, 1)

    row = pd.read_csv(store.csv_path).iloc[0]
    assert row.kind == "confirmed_chickpea"
    assert abs(row.map_x - 487000.315) < 1e-6
    assert abs(row.map_y - 3668999.385) < 1e-6
    feature = json.loads(store.geojson_path.read_text())["features"][0]
    assert feature["geometry"]["type"] == "Point"
    assert feature["properties"]["radius_m"] == 0.03
    assert store.public_manifest()[0]["preview_pixel_size_m"] == 0.03


def test_store_rejects_out_of_bounds_point(tmp_path: Path):
    layer = tmp_path / "layer.png"
    layer.write_bytes(b"png")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "cube_id": "field1_cube20",
        "analysis_role": "primary_candidate",
        "preview_width": 10,
        "preview_height": 10,
        "preview_step": 1,
        "transform_a": 1.0,
        "transform_b": 0.0,
        "transform_c": 0.0,
        "transform_d": 0.0,
        "transform_e": -1.0,
        "transform_f": 0.0,
        "crs": "EPSG:32617",
        **{f"{name}_path": str(layer) for name in (
            "false_colour", "current_chickpea", "pca",
            "first_difference", "second_difference", "ndvi",
        )},
    }]).to_csv(manifest_path, index=False)
    references = tmp_path / "references.json"
    references.write_text(json.dumps({"version": 1, "cubes": {}}))
    store = Store(manifest_path, tmp_path / "output", references, references)
    payload = {"version": 1, "cubes": {"field1_cube20": [{
        "id": "point-1", "kind": "confirmed_weed", "layer": "pca",
        "radius_m": 0.03, "note": "", "x": 11.0, "y": 5.0,
    }]}}

    try:
        store.validate(payload)
    except ValueError as error:
        assert "outside preview bounds" in str(error)
    else:
        raise AssertionError("Out-of-bounds point was accepted")
