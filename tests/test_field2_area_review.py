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
    ZONE_DEFINITIONS,
    Field2AreaAnnotationStore,
    clip_polygon_to_bounds,
    compile_zone_mask,
    distinct_vertex_count,
    incompatible_overlap_rows,
    map_vertices_to_pixel,
    pixel_vertices_to_map,
    polygon_audit,
    polygon_self_intersects,
    polygon_stage_counts,
)
from chickpea_ssl.field2_area_operational import (
    PRECEDENCE_RULE,
    compile_operational_membership,
    effective_coverage_mode,
    operationalize_polygon,
    rasterize_operational_components,
)
from chickpea_ssl.field2_blind_review import PREVIEW_LAYERS, atomic_write_bytes, natural_rgb_png_bytes
from chickpea_ssl.field2_readiness import sha256
from scripts.freeze_field2_area_contract import (
    directory_hashes, materialize_freeze_products, membership_rows, validate_ready_payload,
)
from scripts.build_field2_area_reconciliation import reconciliation_rows
from scripts.build_field2_area_operational_geometry import materialize
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


def test_cube12_seven_vertex_crop_finalizes_identically_for_all_completion_paths():
    source = Path("scripts/field2_area_annotator.js").resolve()
    javascript = f"""
const assert = require('assert'); const model = require({json.dumps(str(source))});
const vertices = [
  {{x:222.5,y:482.2}},{{x:240,y:451}},{{x:275,y:390}},{{x:320,y:320}},
  {{x:353.1,y:266.6}},{{x:348.7,y:264.7}},{{x:230,y:470}}
];
for (const trigger of ['finish_button','double_click','enter_key']) {{
  const result = model.polygonFromDraft(vertices, 'research_crop_area', 'field2_cube12', `crop-${{trigger}}`, 900, 1000);
  assert.strictEqual(result.counts.displayed_vertex_count, 7);
  assert.strictEqual(result.counts.submitted_vertex_count, 7);
  assert.strictEqual(result.counts.distinct_pixel_vertex_count, 7);
  assert.deepStrictEqual(result.polygon.vertices_pixel, vertices);
  assert.strictEqual(result.polygon.zone_type, 'research_crop_area');
}}
"""
    result = subprocess.run(["node", "-e", javascript], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_crop_draft_pan_zoom_partial_bounds_multiple_polygons_zone_switch_and_cancel():
    source = Path("scripts/field2_area_annotator.js").resolve()
    javascript = f"""
const assert = require('assert'); const model = require({json.dumps(str(source))});
const point = model.imagePositionFromCanvas({{x:700,y:250}}, {{center:{{x:410,y:520}}}}, 4, {{width:1100,height:780}});
assert.deepStrictEqual(point, {{x:447.5,y:485}});
const outside = [{{x:-5,y:2}},{{x:6,y:1}},{{x:8,y:7}},{{x:1,y:9}}];
const original = JSON.parse(JSON.stringify(outside));
const first = model.polygonFromDraft(outside,'research_crop_area','field2_cube12','crop-one',10,8);
assert.deepStrictEqual(first.polygon.vertices_pixel, original);
assert.ok(first.counts.clipped_vertex_count >= 3);
const second = model.polygonFromDraft([{{x:2,y:2}},{{x:4,y:2}},{{x:3,y:5}}],'research_crop_area','field2_cube12','crop-two',10,8);
const polygons = model.addPolygon(model.addPolygon([], first.polygon), second.polygon);
assert.strictEqual(polygons.filter(p => p.zone_type === 'research_crop_area').length, 2);
assert.strictEqual(model.canSwitchDraftZone('alley', 3, 'research_crop_area'), false);
assert.strictEqual(model.canSwitchDraftZone('alley', 0, 'research_crop_area'), true);
const cancelled = model.cancelledDraftState();
assert.deepStrictEqual(cancelled, {{zone_type:'',vertices_pixel:[]}});
const afterCancel = model.polygonFromDraft([{{x:1,y:1}},{{x:5,y:1}},{{x:3,y:4}}],'research_crop_area','field2_cube12','crop-new',10,8);
assert.strictEqual(afterCancel.polygon.zone_type, 'research_crop_area');
for (const zone of ['alley','outside_research_field','uncertain_boundary']) {{
  assert.strictEqual(model.polygonFromDraft([{{x:1,y:1}},{{x:5,y:1}},{{x:3,y:4}}],zone,'field2_cube12',zone,10,8).polygon.zone_type, zone);
}}
"""
    result = subprocess.run(["node", "-e", javascript], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_polygon_validation_counts_do_not_deduplicate_or_clip_original_vertices():
    vertices = [
        {"x": -1.0, "y": 1.0}, {"x": 6.0, "y": 1.0}, {"x": 6.0, "y": 7.0},
        {"x": 0.0000001, "y": 1.0000001},
    ]
    original = json.loads(json.dumps(vertices))
    counts = polygon_stage_counts(vertices, 10, 8)
    assert counts == {
        "displayed_vertex_count": 4,
        "submitted_vertex_count": 4,
        "distinct_pixel_vertex_count": 4,
        "clipped_vertex_count": 5,
    }
    assert distinct_vertex_count(vertices) == 4
    assert vertices == original


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


def test_crop_save_reload_preserves_exact_original_pixel_and_map_vertices(tmp_path):
    store = make_area_store(tmp_path)
    payload = store.load()
    record = payload["annotations"]["field2_cube12"]
    original_pixel = [
        {"x": -0.25, "y": 1.125}, {"x": 4.5, "y": 0.75}, {"x": 8.25, "y": 3.5},
        {"x": 7.0, "y": 7.5}, {"x": 2.0, "y": 7.25}, {"x": 0.25, "y": 5.0},
        {"x": -0.1, "y": 2.0},
    ]
    record["coverage_mode"] = "mixed_manual_boundaries"
    record["polygons"] = [{
        "polygon_id": "field2_cube12-crop-seven",
        "zone_type": "research_crop_area",
        "vertices_pixel": json.loads(json.dumps(original_pixel)),
    }]
    store.save(payload)
    resumed = store.load()["annotations"]["field2_cube12"]["polygons"][0]
    assert resumed["vertices_pixel"] == original_pixel
    expected_map = pixel_vertices_to_map(original_pixel, Affine(1, 0, 100, 0, -1, 200))
    assert resumed["vertices_geospatial"] == expected_map


def test_no_edit_save_keeps_all_annotation_artifacts_byte_identical(tmp_path):
    store = make_area_store(tmp_path)
    payload = store.load()
    payload["annotations"]["field2_cube12"].update(
        coverage_mode="mixed_manual_boundaries",
        polygons=[{
            "polygon_id": "field2_cube12-crop",
            "zone_type": "research_crop_area",
            "vertices_pixel": [{"x": 2, "y": 2}, {"x": 7, "y": 2}, {"x": 5, "y": 6}],
        }],
    )
    store.save(payload)
    paths = (store.json_path, store.geojson_path, store.vertex_csv_path, store.audit_csv_path, store.overview_path)
    before = {path.name: sha256(path) for path in paths}
    result = store.save(store.load())
    after = {path.name: sha256(path) for path in paths}
    assert result[3] == 1
    assert after == before


def test_operational_products_are_materialized_separately_from_raw_annotations(tmp_path):
    store = make_area_store(tmp_path)
    payload = store.load()
    record = payload["annotations"]["field2_cube12"]
    record.update(
        coverage_mode="", polygons=[{
            "polygon_id": "field2_cube12-crop", "zone_type": "research_crop_area",
            "vertices_pixel": [{"x": 2, "y": 2}, {"x": 7, "y": 2}, {"x": 5, "y": 6}],
        }],
    )
    store.save(payload)
    raw_hash = sha256(store.json_path)
    output = tmp_path / "derived-operational"
    result = materialize(store, store.load(), output)
    assert result["raw_polygon_count"] == 1 and result["source_annotation_sha256"] == raw_hash
    assert sha256(store.json_path) == raw_hash
    assert (output / "field2_area_operational_geometry.geojson").is_file()
    assert (output / "field2_area_operational_geometry_audit.csv").is_file()
    summary = pd.read_csv(output / "field2_area_operational_membership_summary.csv").fillna("")
    cube12 = summary.loc[summary.cube_id == "field2_cube12"].iloc[0]
    assert cube12.effective_operational_mode == "mixed_manual_boundaries"
    assert cube12.raw_coverage_mode == ""


def test_polygons_use_effective_mixed_mode_and_geometry_warnings_do_not_block_review(tmp_path):
    store = make_area_store(tmp_path)
    payload = store.normalize_payload(store.load())
    record = payload["annotations"]["field2_cube12"]
    record.update(
        coverage_mode="entire_support_research_field", confidence="high", reviewed=True,
        review_timestamp="2026-08-18T12:00:00+00:00",
        polygons=[{
            "polygon_id": "field2_cube12-crop", "zone_type": "research_crop_area",
            "vertices_pixel": [{"x": 2, "y": 2}, {"x": 7, "y": 2}, {"x": 5, "y": 6}],
        }],
    )
    payload = store.normalize_payload(payload)
    issues = store.validate(payload)["field2_cube12"]
    assert issues == []
    assert effective_coverage_mode(record) == "entire_support_research_field"
    record = payload["annotations"]["field2_cube12"]
    record["coverage_mode"] = ""
    assert effective_coverage_mode(record) == "mixed_manual_boundaries"
    assert store.validate(payload)["field2_cube12"] == []


def test_self_intersection_and_overlap_are_review_warnings_not_validation_errors(tmp_path):
    store = make_area_store(tmp_path)
    payload = store.load(); record = payload["annotations"]["field2_cube12"]
    record.update(
        coverage_mode="", confidence="high", reviewed=True,
        review_timestamp="2026-08-18T12:00:00+00:00",
        polygons=[
            {
                "polygon_id": "bow", "zone_type": "research_crop_area",
                "vertices_pixel": [{"x": 2, "y": 2}, {"x": 7, "y": 6}, {"x": 2, "y": 6}, {"x": 7, "y": 2}],
            },
            {
                "polygon_id": "alley", "zone_type": "alley",
                "vertices_pixel": [{"x": 3, "y": 2}, {"x": 8, "y": 2}, {"x": 8, "y": 7}, {"x": 3, "y": 7}],
            },
        ],
    )
    normalized = store.normalize_payload(payload)
    assert store.validate(normalized)["field2_cube12"] == []
    compiled = compile_operational_membership(normalized["annotations"]["field2_cube12"], store.support_mask("field2_cube12"))
    assert compiled["precedence_applied"].any()


def test_terminal_vertex_reconciliation_is_review_only_and_never_changes_source():
    source = {
        "cube_order": ["field2_cube12"],
        "annotations": {"field2_cube12": {"polygons": [
            {
                "polygon_id": "candidate", "zone_type": "research_crop_area",
                "vertices_pixel": [
                    {"x": 0, "y": 0}, {"x": 4, "y": 0}, {"x": 4, "y": 4},
                    {"x": 0, "y": 4}, {"x": 2, "y": -1},
                ],
            },
            {
                "polygon_id": "manual", "zone_type": "alley",
                "vertices_pixel": [
                    {"x": 0, "y": 0}, {"x": 4, "y": 4}, {"x": 0, "y": 4},
                    {"x": 4, "y": 0}, {"x": 0.1, "y": 0.1},
                ],
            },
        ]}},
    }
    original = json.loads(json.dumps(source))
    rows = reconciliation_rows(source, "a" * 64)
    assert len(rows) == 2
    assert rows[0]["proposed_action"] == "remove_terminal_operational_vertex"
    assert rows[0]["proposed_operational_vertex_count"] == 4
    assert rows[0]["proposal_applied"] is False and rows[0]["investigator_decision"] == ""
    assert rows[1]["proposed_action"] == "manual_review_no_change"
    assert source == original


def test_area_freeze_refuses_until_all_40_cubes_are_reviewed(tmp_path):
    store = make_area_store(tmp_path)
    with pytest.raises(ValueError, match="all 40 cubes reviewed"):
        validate_ready_payload(store, store.load())


def test_freeze_readiness_accepts_recoverable_geometry_warnings_after_all_reviews(tmp_path):
    store = make_area_store(tmp_path); payload = store.load()
    for record in payload["annotations"].values():
        record.update(
            coverage_mode="entire_support_research_field", confidence="high", reviewed=True,
            review_timestamp="2026-08-18T12:00:00+00:00",
        )
    record = payload["annotations"]["field2_cube12"]
    record.update(coverage_mode="", polygons=[
        {
            "polygon_id": "bow", "zone_type": "research_crop_area",
            "vertices_pixel": [{"x": 2, "y": 2}, {"x": 7, "y": 6}, {"x": 2, "y": 6}, {"x": 7, "y": 2}],
        },
        {
            "polygon_id": "alley", "zone_type": "alley",
            "vertices_pixel": [{"x": 3, "y": 2}, {"x": 8, "y": 2}, {"x": 8, "y": 7}, {"x": 3, "y": 7}],
        },
    ])
    normalized = store.normalize_payload(payload)
    validate_ready_payload(store, normalized)


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


def test_operational_terminal_and_validity_repairs_preserve_raw_vertices():
    shape = (12, 12)
    terminal = {
        "polygon_id": "terminal", "zone_type": "research_crop_area",
        "vertices_pixel": [
            {"x": 1, "y": 1}, {"x": 9, "y": 1}, {"x": 9, "y": 9},
            {"x": 1, "y": 9}, {"x": 3, "y": 0},
        ],
    }
    remaining = {
        "polygon_id": "remaining", "zone_type": "alley",
        "vertices_pixel": [
            {"x": 1, "y": 1}, {"x": 9, "y": 9}, {"x": 1, "y": 9},
            {"x": 9, "y": 1}, {"x": 1.1, "y": 1.1},
        ],
    }
    originals = json.loads(json.dumps([terminal, remaining]))
    first = operationalize_polygon(terminal, shape)
    second = operationalize_polygon(remaining, shape)
    assert first["repair_method"] == "terminal_vertex_removed_operational_only"
    assert second["repair_method"] == "raster_polygonize_make_valid"
    assert second["component_count"] >= 1 and second["absolute_area_change_pixels2"] >= 0
    for item in (first, second):
        assert item["source_vertices_preserved"] is True
        assert not item["operational_self_intersection"]
        assert rasterize_operational_components(item["components"], shape).any()
    assert [terminal, remaining] == originals


def test_overlap_precedence_is_deterministic_and_records_all_raw_memberships():
    support = np.ones((10, 10), dtype=bool)
    vertices = [{"x": 1, "y": 1}, {"x": 8, "y": 1}, {"x": 8, "y": 8}, {"x": 1, "y": 8}]
    polygons = [
        {"polygon_id": zone, "zone_type": zone, "vertices_pixel": vertices}
        for zone in ("alley", "research_crop_area", "outside_research_field", "uncertain_boundary")
    ]
    record = {"coverage_mode": "mixed_manual_boundaries", "polygons": polygons}
    first = compile_operational_membership(record, support)
    second = compile_operational_membership({**record, "polygons": list(reversed(polygons))}, support)
    assert np.array_equal(first["winning_membership"], second["winning_membership"])
    overlap = first["precedence_applied"]
    assert overlap.any()
    assert np.all(first["winning_membership"][overlap] == ZONE_CODES["outside_research_field"])
    assert np.all(first["raw_membership_bitmask"][overlap] == 15)
    assert first["precedence_rule"] == PRECEDENCE_RULE


def test_current_438_polygons_all_operationalize_without_changing_raw_package():
    source_path = Path("metadata/local/annotations/field2_area_zones/field2_area_annotations.json")
    before_bytes = source_path.read_bytes(); payload = json.loads(before_bytes)
    original = json.loads(json.dumps(payload))
    support = pd.read_csv(
        "metadata/local/reports/field2_valid_support/field2_valid_support_mask_manifest.csv"
    ).set_index("cube_id")
    methods = {}
    count = 0
    for cube_id in payload["cube_order"]:
        shape = (int(support.loc[cube_id, "height"]), int(support.loc[cube_id, "width"]))
        for polygon in payload["annotations"][cube_id]["polygons"]:
            item = operationalize_polygon(polygon, shape)
            methods[item["repair_method"]] = methods.get(item["repair_method"], 0) + 1
            mask = rasterize_operational_components(item["components"], shape)
            assert mask.any() and not item["operational_self_intersection"]
            count += 1
    assert count == 438
    assert methods == {
        "none": 334,
        "terminal_vertex_removed_operational_only": 103,
        "raster_polygonize_make_valid": 1,
    }
    assert payload == original and source_path.read_bytes() == before_bytes
    assert sha256(source_path) == "9e45943b2b8817fdbea63e28d94aea6bcd46852cbe2750985631a39b8da560d0"


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
    assert np.any(unassigned[support] == ZONE_CODES["unassigned_valid_support"])
    draft["treat_unassigned_valid_support_as_outside"] = True
    assigned = compile_zone_mask(draft, support)
    assert not np.any(assigned[support] == ZONE_CODES["unassigned_valid_support"])
    assert np.any(assigned == ZONE_CODES["outside_research_field"])


def test_exact_deterministic_point_zone_membership_omits_coordinates():
    mask = np.zeros((8, 10), dtype=np.uint8); mask[:, :5] = ZONE_CODES["research_crop_area"]; mask[:, 5:] = ZONE_CODES["outside_research_field"]
    raw_bits = np.zeros((8, 10), dtype=np.uint8); raw_bits[:, :5] = 1; raw_bits[:, 5:] = 8
    compiled = {"cube": {
        "winning_membership": mask, "raw_membership_bitmask": raw_bits,
        "raw_polygon_membership_bitmask": raw_bits,
        "precedence_applied": np.zeros((8, 10), dtype=bool),
        "effective_coverage_mode": "mixed_manual_boundaries", "raw_coverage_mode": "mixed_manual_boundaries",
    }}
    frame = pd.DataFrame([
        {"sample_id": "one", "cube_id": "cube", "cube_evaluation_role": "primary_three_class", "row": 2, "column": 3, "scalar_index_rank_stratum": "rank_00_20", "spatial_group_id": "block-a"},
        {"sample_id": "two", "cube_id": "cube", "cube_evaluation_role": "primary_three_class", "row": 2, "column": 7, "scalar_index_rank_stratum": "rank_20_40", "spatial_group_id": "block-b"},
    ])
    first = membership_rows(frame, compiled, "main")
    assert first == membership_rows(frame, compiled, "main")
    assert [item["domain_name"] for item in first] == ["research_crop_area", "outside_research_field"]
    assert all("row" not in item and "column" not in item for item in first)
    assert first[0]["raw_zone_memberships"] == "research_crop_area"
    assert first[1]["winning_operational_membership"] == "outside_research_field"
    assert first[0]["primary_external_validation_eligible"] is True
    assert first[1]["domain_reporting_stratum"] == "outside_domain_ood"


def test_freeze_products_are_deterministic_and_explicit_all_outside_wins_preserved_alley(tmp_path):
    store = make_area_store(tmp_path); payload = store.load()
    for record in payload["annotations"].values():
        record.update(coverage_mode="entire_support_research_field", confidence="high", reviewed=True,
                      review_timestamp="2026-08-19T12:00:00+00:00")
    payload["annotations"]["field2_cube11"].update(coverage_mode="", polygons=[{
        "polygon_id": "blank-crop", "zone_type": "research_crop_area",
        "vertices_pixel": [{"x": 2, "y": 1}, {"x": 8, "y": 1}, {"x": 8, "y": 7}, {"x": 2, "y": 7}],
    }])
    payload["annotations"]["field2_cube38"].update(
        coverage_mode="entire_support_outside_research_field", polygons=[{
            "polygon_id": "preserved-alley", "zone_type": "alley",
            "vertices_pixel": [{"x": 3, "y": 2}, {"x": 7, "y": 2}, {"x": 7, "y": 6}, {"x": 3, "y": 6}],
        }],
    )
    payload = store.normalize_payload(payload); validate_ready_payload(store, payload)
    store.save(payload); payload = store.load()
    cube_ids = store.cube_order
    def frame(count, kind):
        return pd.DataFrame([{
            "sample_id": f"{kind}-{index}", "cube_id": cube_ids[index % 40],
            "cube_evaluation_role": "primary_three_class", "row": 3, "column": 4,
            "scalar_index_rank_stratum": "rank_00_20", "spatial_group_id": f"block-{index % 5}",
        } for index in range(count)])
    config = __import__("yaml").safe_load(Path("configs/field2_area_annotation.yaml").read_text())
    first, second = tmp_path / "freeze-a", tmp_path / "freeze-b"
    result = materialize_freeze_products(tmp_path, config, store, payload, frame(800, "main"), frame(396, "reserve"), first)
    materialize_freeze_products(tmp_path, config, store, payload, frame(800, "main"), frame(396, "reserve"), second)
    assert directory_hashes(first) == directory_hashes(second)
    cube38 = result["compiled"]["field2_cube38"]
    support = store.support_mask("field2_cube38")
    assert np.all(cube38["winning_membership"][support] == ZONE_CODES["outside_research_field"])
    assert np.any(cube38["raw_polygon_membership_bitmask"][support] != 0)
    reconciled = {row["cube_id"]: row for row in result["reconciliation_rows"]}
    assert reconciled["field2_cube11"]["effective_coverage_mode"] == "mixed_manual_boundaries"
    assert reconciled["field2_cube11"]["reconciliation_reason"] == "reviewed_cube_with_investigator_polygons"
    reserve = pd.read_csv(first / config["freeze"]["reserve_membership_locked"])
    assert len(reserve) == 396 and {"row", "column", "x", "y", "longitude", "latitude"}.isdisjoint(reserve.columns)


def test_area_ui_is_sticky_complete_and_has_no_biological_or_reserve_controls():
    javascript = Path("scripts/field2_area_annotator.js").read_text()
    assert ".side{position:sticky;top:0;height:100vh" in HTML
    assert ".side-bottom" in HTML and 'id="save"' in HTML and 'id="review"' in HTML
    for control in ("draw", "finish", "cancelDrawing", "edit", "insert", "deleteVertex", "movePolygon", "deletePolygon", "undo", "redo", "clearSelected", "opacity", "resetView", "previous", "next", "cube", "notes", "confidence"):
        assert f'id="{control}"' in HTML
    for layer in ("natural_rgb", "false_colour", "pca", "stored_index", "valid_support", "support_outline"):
        assert f'value="{layer}"' in HTML
    assert "biological label" in HTML.lower()
    assert "investigator-identified planted-row/research-plot domain; not a biological chickpea label" in HTML
    assert "Small overlaps are resolved automatically: outside &gt; alley &gt; uncertain &gt; research crop." in HTML
    assert 'id="effectiveMode"' in HTML
    assert "operational pixels use outside > alley > uncertain > research crop" in javascript
    assert 'canvas.addEventListener("click"' in javascript
    assert 'event.detail === 2' in javascript
    assert "displayed_vertex_count" in javascript and "clipped_vertex_count" in javascript
    assert "/api/reserve" not in javascript and "prediction" not in "".join(
        line for line in javascript.splitlines() if "fetch(" in line
    ).lower()


def test_area_coverage_modes_are_exact_and_no_automatic_inference_tools_are_imported():
    assert set(COVERAGE_MODES) == {
        "entire_support_research_field", "entire_support_outside_research_field",
        "mixed_manual_boundaries", "uncertain_requires_review",
    }
    config = __import__("yaml").safe_load(Path("configs/field2_area_annotation.yaml").read_text())
    assert config["annotation"]["zone_definitions"] == ZONE_DEFINITIONS
    assert config["operational_geometry"]["precedence_rule"] == PRECEDENCE_RULE
    assert config["operational_geometry"]["blank_mode_with_polygons"] == "mixed_manual_boundaries"
    prohibited = {"torch", "tensorflow", "xgboost", "sklearn", "cv2", "shapely"}
    for path in (
        Path("chickpea_ssl/field2_area_review.py"), Path("scripts/run_field2_area_annotator.py"),
        Path("chickpea_ssl/field2_area_operational.py"), Path("scripts/build_field2_area_operational_geometry.py"),
        Path("scripts/freeze_field2_area_contract.py"), Path("scripts/validate_field2_source_support_contracts.py"),
    ):
        tree = __import__("ast").parse(path.read_text())
        imports = {node.names[0].name.split(".")[0] for node in __import__("ast").walk(tree) if isinstance(node, __import__("ast").Import)}
        imports |= {node.module.split(".")[0] for node in __import__("ast").walk(tree) if isinstance(node, __import__("ast").ImportFrom) and node.module}
        assert imports.isdisjoint(prohibited)
