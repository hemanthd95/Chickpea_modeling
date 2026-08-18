import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine

from chickpea_ssl.field2_area_review import (
    COVERAGE_MODES,
    ZONE_CODES,
    Field2AreaAnnotationStore,
    clip_polygon_to_bounds,
    compile_zone_mask,
    incompatible_overlap_rows,
    map_vertices_to_pixel,
    pixel_vertices_to_map,
    polygon_audit,
    polygon_self_intersects,
)
from chickpea_ssl.field2_blind_review import PREVIEW_LAYERS, atomic_write_bytes, natural_rgb_png_bytes
from chickpea_ssl.field2_readiness import sha256
from scripts.freeze_field2_area_contract import membership_rows, validate_ready_payload
from scripts.run_field2_area_annotator import HTML


def make_area_store(tmp_path: Path) -> Field2AreaAnnotationStore:
    image_path = tmp_path / "shared.png"
    atomic_write_bytes(image_path, natural_rgb_png_bytes(np.full((8, 10, 3), 90, dtype=np.uint8)))
    image_hash = sha256(image_path)
    support_path = tmp_path / "support.tif"
    support = np.ones((8, 10), dtype=np.uint8)
    support[:, :2] = 0
    with rasterio.open(
        support_path, "w", driver="GTiff", width=10, height=8, count=1, dtype="uint8",
        crs="EPSG:32617", transform=Affine(1, 0, 100, 0, -1, 200), nodata=0,
    ) as dataset:
        dataset.write(support, 1)
    support_hash = sha256(support_path)
    transform = json.dumps([1, 0, 100, 0, -1, 200, 0, 0, 1])
    previews = json.dumps({name: image_hash for name in PREVIEW_LAYERS}, sort_keys=True)
    review_rows, rgb_rows, support_rows = [], [], []
    for number in range(40):
        cube_id = f"field2_cube{number + 1:02d}"
        review_rows.append({
            "cube_id": cube_id, "width": 10, "height": 8, "preview_width": 10,
            "preview_height": 8, "preview_step": 1, "crs": "EPSG:32617",
            "transform": transform, "preview_sha256_json": previews,
            **{f"{name}_path": image_path.name for name in PREVIEW_LAYERS},
        })
        rgb_rows.append({
            "cube_id": cube_id, "width": 10, "height": 8, "preview_step": 1,
            "natural_rgb_path": image_path.name, "natural_rgb_sha256": image_hash,
        })
        support_rows.append({
            "cube_id": cube_id, "width": 10, "height": 8, "crs": "EPSG:32617",
            "transform": transform, "mask_path": support_path.name, "mask_sha256": support_hash,
        })
    review_path, rgb_path, support_manifest_path = tmp_path / "review.csv", tmp_path / "rgb.csv", tmp_path / "support.csv"
    pd.DataFrame(review_rows).to_csv(review_path, index=False)
    pd.DataFrame(rgb_rows).to_csv(rgb_path, index=False)
    pd.DataFrame(support_rows).to_csv(support_manifest_path, index=False)
    return Field2AreaAnnotationStore(tmp_path, review_path, rgb_path, support_manifest_path, tmp_path / "annotations")


def test_area_editor_polygon_create_edit_delete_and_undo_redo():
    source = Path("scripts/field2_area_annotator.js").resolve()
    javascript = f"""
const assert = require('assert'); const model = require({json.dumps(str(source))});
const polygon = {{polygon_id:'p1',zone_type:'alley',vertices_pixel:[{{x:1,y:1}},{{x:5,y:1}},{{x:5,y:5}}]}};
let polygons = model.addPolygon([], polygon); assert.strictEqual(polygons.length, 1);
polygons = model.moveVertex(polygons, 'p1', 1, {{x:6,y:2}}); assert.deepStrictEqual(polygons[0].vertices_pixel[1], {{x:6,y:2}});
polygons = model.insertVertex(polygons, 'p1', 1, {{x:6,y:4}}); assert.strictEqual(polygons[0].vertices_pixel.length, 4);
polygons = model.deleteVertex(polygons, 'p1', 2); assert.strictEqual(polygons[0].vertices_pixel.length, 3);
polygons = model.movePolygon(polygons, 'p1', {{x:2,y:-1}}); assert.deepStrictEqual(polygons[0].vertices_pixel[0], {{x:3,y:0}});
let history = model.history([]); history = model.historyApply(history, polygons); history = model.historyUndo(history); assert.strictEqual(history.present.length, 0);
history = model.historyRedo(history); assert.strictEqual(history.present.length, 1);
assert.strictEqual(model.deletePolygon(history.present, 'p1').length, 0);
"""
    result = subprocess.run(["node", "-e", javascript], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_area_store_save_resume_is_atomic_and_prediction_free(tmp_path):
    store = make_area_store(tmp_path)
    payload = store.load()
    record = payload["annotations"][store.cube_order[0]]
    record.update(
        coverage_mode="entire_support_research_field", confidence="high", reviewed=True,
        review_timestamp="2026-08-18T12:00:00+00:00", investigator_notes="confirmed",
    )
    reviewed, polygons, vertices, revision = store.save(payload)
    assert (reviewed, polygons, vertices, revision) == (1, 0, 0, 1)
    resumed = store.load()
    assert resumed["annotations"][store.cube_order[0]]["investigator_notes"] == "confirmed"
    assert resumed["automatic_metadata"] == {
        "biological_labels_assigned_automatically": False,
        "predictions_or_probabilities_used": False,
        "geometry_inferred_automatically": False,
        "reserve_frame_exposed": False,
        "area_geometry_frozen": False,
    }
    for path in (store.json_path, store.geojson_path, store.vertex_csv_path, store.audit_csv_path, store.overview_path):
        assert path.is_file()


def test_area_freeze_refuses_until_all_40_cubes_are_reviewed(tmp_path):
    store = make_area_store(tmp_path)
    with pytest.raises(ValueError, match="all 40 cubes reviewed"):
        validate_ready_payload(store, store.load())


def test_pixel_and_crs_vertices_round_trip_without_changing_originals():
    transform = Affine(0.02, 0, 487000, 0, -0.02, 3669000)
    pixel = [{"x": 2.25, "y": 3.75}, {"x": 9.5, "y": 3.75}, {"x": 5, "y": 7}]
    original = json.loads(json.dumps(pixel))
    mapped = pixel_vertices_to_map(pixel, transform)
    restored = map_vertices_to_pixel(mapped, transform)
    assert pixel == original
    assert np.allclose([[x["x"], x["y"]] for x in restored], [[x["x"], x["y"]] for x in pixel])


def test_valid_support_clipping_and_outside_raster_fraction_are_reported():
    support = np.ones((8, 10), dtype=bool); support[:, :2] = False
    polygon = {"polygon_id": "p", "zone_type": "research_crop_area", "vertices_pixel": [
        {"x": -1, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 7}, {"x": -1, "y": 7},
    ]}
    audit = polygon_audit(polygon, support)
    assert 0 < audit["outside_raster_fraction"] < 1
    assert 0 < audit["outside_valid_support_fraction"] < 1
    assert clip_polygon_to_bounds(polygon["vertices_pixel"], 10, 8)


def test_self_intersection_and_incompatible_zone_overlap_are_detected():
    bow = [{"x": 1, "y": 1}, {"x": 7, "y": 7}, {"x": 1, "y": 7}, {"x": 7, "y": 1}]
    assert polygon_self_intersects(bow)
    polygons = [
        {"polygon_id": "a", "zone_type": "research_crop_area", "vertices_pixel": [{"x": 1, "y": 1}, {"x": 7, "y": 1}, {"x": 7, "y": 6}, {"x": 1, "y": 6}]},
        {"polygon_id": "b", "zone_type": "alley", "vertices_pixel": [{"x": 4, "y": 2}, {"x": 9, "y": 2}, {"x": 9, "y": 7}, {"x": 4, "y": 7}]},
    ]
    overlap = incompatible_overlap_rows(polygons, np.ones((8, 10), dtype=bool))
    assert len(overlap) == 1 and overlap[0]["overlap_support_pixels"] > 0


def test_entire_support_modes_and_explicit_unassigned_handling():
    support = np.ones((8, 10), dtype=bool); support[0, :] = False
    for mode, zone in (
        ("entire_support_research_field", "research_crop_area"),
        ("entire_support_outside_research_field", "outside_research_field"),
        ("uncertain_requires_review", "uncertain_boundary"),
    ):
        mask = compile_zone_mask({"coverage_mode": mode, "polygons": []}, support)
        assert np.all(mask[support] == ZONE_CODES[zone]) and np.all(mask[~support] == 0)
    polygon = {"polygon_id": "a", "zone_type": "alley", "vertices_pixel": [{"x": 2, "y": 2}, {"x": 5, "y": 2}, {"x": 5, "y": 6}, {"x": 2, "y": 6}]}
    draft = {"coverage_mode": "mixed_manual_boundaries", "polygons": [polygon], "treat_unassigned_valid_support_as_outside": False}
    unassigned = compile_zone_mask(draft, support)
    assert np.any(unassigned[support] == 0)
    draft["treat_unassigned_valid_support_as_outside"] = True
    assigned = compile_zone_mask(draft, support)
    assert not np.any(assigned[support] == 0)
    assert np.any(assigned == ZONE_CODES["outside_research_field"])


def test_exact_deterministic_point_zone_membership_omits_coordinates():
    mask = np.zeros((8, 10), dtype=np.uint8); mask[:, :5] = ZONE_CODES["research_crop_area"]; mask[:, 5:] = ZONE_CODES["outside_research_field"]
    frame = pd.DataFrame([
        {"sample_id": "one", "cube_id": "cube", "cube_evaluation_role": "primary_three_class", "row": 2, "column": 3},
        {"sample_id": "two", "cube_id": "cube", "cube_evaluation_role": "primary_three_class", "row": 2, "column": 7},
    ])
    first = membership_rows(frame, {"cube": mask}, "main")
    assert first == membership_rows(frame, {"cube": mask}, "main")
    assert [item["zone_type"] for item in first] == ["research_crop_area", "outside_research_field"]
    assert all("row" not in item and "column" not in item for item in first)
    assert first[0]["primary_external_validation_eligible"] is True
    assert first[1]["supplementary_or_domain_shift"] is True


def test_area_ui_is_sticky_complete_and_has_no_biological_or_reserve_controls():
    javascript = Path("scripts/field2_area_annotator.js").read_text()
    assert ".side{position:sticky;top:0;height:100vh" in HTML
    assert ".side-bottom" in HTML and 'id="save"' in HTML and 'id="review"' in HTML
    for control in ("draw", "finish", "edit", "insert", "deleteVertex", "movePolygon", "deletePolygon", "undo", "redo", "clearSelected", "opacity", "resetView", "previous", "next", "cube", "notes", "confidence"):
        assert f'id="{control}"' in HTML
    for layer in ("natural_rgb", "false_colour", "pca", "stored_index", "valid_support", "support_outline"):
        assert f'value="{layer}"' in HTML
    assert "biological label" in HTML.lower()
    assert "/api/reserve" not in javascript and "prediction" not in "".join(
        line for line in javascript.splitlines() if "fetch(" in line
    ).lower()


def test_area_coverage_modes_are_exact_and_no_automatic_inference_tools_are_imported():
    assert set(COVERAGE_MODES) == {
        "entire_support_research_field", "entire_support_outside_research_field",
        "mixed_manual_boundaries", "uncertain_requires_review",
    }
    prohibited = {"torch", "tensorflow", "xgboost", "sklearn", "cv2", "shapely"}
    for path in (
        Path("chickpea_ssl/field2_area_review.py"), Path("scripts/run_field2_area_annotator.py"),
        Path("scripts/freeze_field2_area_contract.py"), Path("scripts/validate_field2_source_support_contracts.py"),
    ):
        tree = __import__("ast").parse(path.read_text())
        imports = {node.names[0].name.split(".")[0] for node in __import__("ast").walk(tree) if isinstance(node, __import__("ast").Import)}
        imports |= {node.module.split(".")[0] for node in __import__("ast").walk(tree) if isinstance(node, __import__("ast").ImportFrom) and node.module}
        assert imports.isdisjoint(prohibited)
