import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from chickpea_ssl.field2_blind_review import (
    POINT_LABELS,
    PREVIEW_LAYERS,
    PRIMARY_ROLES,
    REVIEW_FLAGS,
    CubeRoleReviewStore,
    atomic_write_json,
    atomic_write_yaml,
    deduplicate_geographic_views,
    default_review_payload,
    geographic_ground_cell_ids,
    require_frozen_role_contract,
    scalar_index_rank_strata,
    spatial_group_ids,
    stratified_deterministic_sample,
    reject_prediction_provenance,
    validate_review_record,
    verify_preview_hashes,
)
from chickpea_ssl.field2_readiness import sha256


def test_configuration_freezes_exact_40_cube_inventory():
    config = yaml.safe_load(Path("configs/field2_blind_evaluation.yaml").read_text())
    expected_numbers = [2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,22,23,
                        24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
    assert config["expected_cube_ids"] == [f"field2_cube{value:02d}" for value in expected_numbers]
    assert len(config["expected_cube_ids"]) == 40
    assert config["valid_support_manifest_sha256"] == "1f48ffe67d7ad2cbf95cca5c2e6e98d15245843435c519bbb8a67841c2bbf3e3"


def test_frozen_manifest_checksum_detects_changes(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("cube_id,status\nfield2_cube02,ready\n")
    frozen = sha256(manifest)
    assert sha256(manifest) == frozen
    manifest.write_text("cube_id,status\nfield2_cube02,changed\n")
    assert sha256(manifest) != frozen


def neutral_record(cube_id="field2_cube02"):
    return {
        "cube_id": cube_id, "primary_role": "unreviewed",
        "flags": {name: False for name in REVIEW_FLAGS},
        "investigator_notes": "", "exclusion_reason": "", "confidence": "",
        "reviewed": False, "review_timestamp": "", "reviewer_identifier": "",
        "source_preview_checksums": {name: "a" * 64 for name in PREVIEW_LAYERS},
    }


def test_role_schema_and_mutually_inconsistent_flags():
    record = neutral_record()
    assert validate_review_record(record, record["cube_id"]) == []
    record["flags"]["chickpea_visible"] = True
    record["flags"]["chickpea_absent_confirmed"] = True
    assert "chickpea_visible_conflicts_with_absent_confirmed" in validate_review_record(record, record["cube_id"])


def test_role_requirements_and_low_confidence_allowed():
    record = neutral_record()
    record.update(primary_role="primary_three_class", reviewed=True,
                  review_timestamp="2026-08-17T00:00:00+00:00", confidence="low")
    assert "primary_three_class_requires_chickpea_weed_soil_visible" in validate_review_record(record, record["cube_id"])
    record["flags"].update(chickpea_visible=True, ordinary_weed_visible=True, soil_visible=True)
    assert validate_review_record(record, record["cube_id"]) == []
    record["primary_role"] = "chickpea_absent_negative_control"
    record["flags"]["chickpea_visible"] = False
    record["flags"]["chickpea_absent_confirmed"] = True
    assert validate_review_record(record, record["cube_id"]) == []


def test_exclusion_requires_reason():
    record = neutral_record()
    record["primary_role"] = "exclude_with_reason"
    assert "exclude_with_reason_requires_exclusion_reason" in validate_review_record(record, record["cube_id"])
    record["exclusion_reason"] = "unusable acquisition geometry"
    assert validate_review_record(record, record["cube_id"]) == []


def make_review_manifest(tmp_path: Path, cube_ids=("field2_cube02", "field2_cube03")) -> Path:
    rows = []
    for cube_id in cube_ids:
        hashes, paths = {}, {}
        for layer in PREVIEW_LAYERS:
            path = tmp_path / f"{cube_id}_{layer}.png"
            plt.imsave(path, np.zeros((3, 4, 3), dtype=np.float32))
            paths[f"{layer}_path"] = str(path.relative_to(tmp_path))
            hashes[layer] = sha256(path)
        rows.append({
            "cube_id": cube_id, "width": 8, "height": 6, "gsd_m": .02,
            "valid_fraction": .5, "bounds": "[0,0,1,1]", "footprint_overlap_warning": False,
            "spectral_qc_status": "pass", "alignment_status": "exact_alignment",
            "preview_width": 4, "preview_height": 3, "preview_step": 2,
            "preview_sha256_json": json.dumps(hashes, sort_keys=True), **paths,
        })
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_atomic_save_resume_and_no_automatic_biological_roles(tmp_path):
    manifest_path = make_review_manifest(tmp_path)
    store = CubeRoleReviewStore(tmp_path, manifest_path, tmp_path / "annotations")
    payload = store.load()
    assert all(record["primary_role"] == "unreviewed" for record in payload["reviews"].values())
    assert all(not any(record["flags"].values()) for record in payload["reviews"].values())
    records, reviewed = store.save(payload)
    assert (records, reviewed) == (2, 0)
    assert store.load()["revision"] == 1
    assert store.csv_path.is_file() and store.audit_path.is_file() and store.overview_path.is_file()
    assert not list(store.output_root.glob("*.tmp"))
    stale = payload
    try:
        store.save(stale)
    except ValueError as error:
        assert "revision conflict" in str(error)
    else:
        raise AssertionError("Stale review payload overwrote saved work")


def test_preview_checksum_verification_and_save_rejection(tmp_path):
    manifest_path = make_review_manifest(tmp_path, ("field2_cube02",))
    store = CubeRoleReviewStore(tmp_path, manifest_path, tmp_path / "annotations")
    assert verify_preview_hashes(store.manifest, tmp_path) == []
    payload = default_review_payload(store.manifest)
    store.layer_path("field2_cube02", "pca").write_bytes(b"changed")
    assert verify_preview_hashes(store.manifest, tmp_path) == ["field2_cube02:pca:sha256_mismatch"]
    try:
        store.save(payload)
    except ValueError as error:
        assert "Preview checksum" in str(error)
    else:
        raise AssertionError("Changed preview was accepted")


def test_prediction_probability_and_checkpoint_provenance_rejected():
    for payload in (
        {"prediction_path": "predictions.tif"},
        {"probability_path": "probabilities.tif"},
        {"nested": {"checkpoint": "model.ckpt"}},
        {"allowed_inputs": ["weights.pth"]},
    ):
        try:
            reject_prediction_provenance(payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Prohibited provenance accepted: {payload}")
    reject_prediction_provenance({"prediction_path": "", "allowed_inputs": ["reflectance", "support"]})


def test_spatial_blocks_and_ground_cells_are_deterministic():
    x = np.array([0.1, 4.99, 5.01]); y = np.array([10.1, 10.2, 10.2])
    first = spatial_group_ids("EPSG:32617", x, y, 5.0)
    second = spatial_group_ids("EPSG:32617", x, y, 5.0)
    assert np.array_equal(first, second)
    assert first[0] == first[1] and first[1] != first[2]
    cells = geographic_ground_cell_ids("EPSG:32617", np.array([1.01, 1.09, 1.11]), np.ones(3), .1)
    assert cells[0] == cells[1] and cells[1] != cells[2]


def test_scalar_index_rank_stratification():
    labels = ["q1", "q2", "q3", "q4", "q5"]
    strata = scalar_index_rank_strata(np.arange(100, dtype=float), labels)
    assert [int(np.count_nonzero(strata == label)) for label in labels] == [20] * 5
    tied = scalar_index_rank_strata(np.ones(10), labels)
    assert len(set(tied)) == 1


def test_geographic_duplicate_prevention_is_deterministic():
    frame = pd.DataFrame([
        {"geographic_ground_cell_id": "cell-a", "cube_id": "field2_cube02", "row": 1, "column": 1},
        {"geographic_ground_cell_id": "cell-a", "cube_id": "field2_cube03", "row": 2, "column": 2},
        {"geographic_ground_cell_id": "cell-b", "cube_id": "field2_cube03", "row": 3, "column": 3},
    ])
    first = deduplicate_geographic_views(frame, 20260817)
    second = deduplicate_geographic_views(frame.sample(frac=1, random_state=4), 20260817)
    assert first[["cube_id", "row", "column"]].to_dict("records") == second[["cube_id", "row", "column"]].to_dict("records")
    assert len(first) == 2
    assert first.geographic_ground_cell_id.nunique() == len(first)


def test_deterministic_sampling_inclusion_probability_and_weight():
    labels = ["q1", "q2", "q3", "q4", "q5"]
    rows = []
    for stratum_index, stratum in enumerate(labels):
        for number in range(5):
            rows.append({
                "cube_id": "field2_cube02", "cube_evaluation_role": "primary_three_class",
                "scalar_index_rank_stratum": stratum,
                "deterministic_selection_rank": stratum_index * 10 + number,
                "row": stratum_index * 10 + number, "column": number,
                "x": stratum_index * 10.0 + number, "y": number * 2.0,
            })
    frame = pd.DataFrame(rows)
    first = stratified_deterministic_sample(frame, {"primary_three_class": 10}, labels, .25)
    second = stratified_deterministic_sample(frame.sample(frac=1, random_state=8), {"primary_three_class": 10}, labels, .25)
    assert first[["row", "column"]].to_dict("records") == second[["row", "column"]].to_dict("records")
    assert len(first) == 10
    assert np.allclose(first.inclusion_probability, .4)
    assert np.allclose(first.design_weight, 2.5)


def test_sampling_refuses_missing_or_unfinished_role_contract(tmp_path):
    missing = tmp_path / "missing.yaml"
    for path in (missing, tmp_path / "unfinished.yaml"):
        if path.name == "unfinished.yaml":
            path.write_text("status: review_in_progress\nall_cubes_reviewed: false\n")
        try:
            require_frozen_role_contract(path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Sampling accepted a missing or unfinished role contract")


def test_atomic_json_and_numpy_yaml_serialization(tmp_path):
    json_path, yaml_path = tmp_path / "state.json", tmp_path / "contract.yaml"
    atomic_write_json(json_path, {"reviewed": False})
    atomic_write_yaml(yaml_path, {"count": np.int64(40), "fraction": np.float32(.5)})
    assert json.loads(json_path.read_text()) == {"reviewed": False}
    loaded = yaml.safe_load(yaml_path.read_text())
    assert loaded == {"count": 40, "fraction": .5}
    assert not list(tmp_path.glob("*.tmp"))


def test_future_point_labels_preserve_tall_grass():
    assert "tall_grass_weed" in POINT_LABELS
    assert set(PRIMARY_ROLES) == {
        "primary_three_class", "chickpea_absent_negative_control", "challenge_only",
        "sensitivity_only", "exclude_with_reason", "unreviewed",
    }


def test_review_entrypoints_do_not_import_model_frameworks():
    prohibited = {"torch", "torchvision", "xgboost", "sklearn", "tensorflow"}
    paths = [
        Path("scripts/prepare_field2_cube_role_review.py"),
        Path("scripts/run_field2_cube_role_reviewer.py"),
        Path("scripts/freeze_field2_cube_role_contract.py"),
        Path("scripts/build_field2_blind_annotation_sampling_frame.py"),
        Path("scripts/run_field2_point_annotator.py"),
    ]
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = {
            node.names[0].name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        }
        imports.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        assert imports.isdisjoint(prohibited), path
