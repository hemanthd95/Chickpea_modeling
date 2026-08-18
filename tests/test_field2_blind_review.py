import ast
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.transform import Affine
import yaml

from chickpea_ssl.field2_blind_review import (
    POINT_LABELS,
    PREVIEW_LAYERS,
    PRIMARY_ROLES,
    REVIEW_FLAGS,
    CubeRoleReviewStore,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_yaml,
    deduplicate_geographic_views,
    default_review_payload,
    frozen_role_table_bytes,
    geographic_ground_cell_ids,
    minimum_separation_thin,
    natural_rgb_png_bytes,
    require_frozen_role_contract,
    balanced_stratum_quotas,
    review_totals,
    role_summary_csv_bytes,
    scalar_index_rank_strata,
    select_natural_rgb_bands,
    spatial_group_ids,
    stratified_deterministic_sample,
    select_main_reserve_frames,
    reject_prediction_provenance,
    render_natural_rgb_uint8,
    validate_frozen_role_contract,
    validate_frozen_sampling_contract,
    validate_review_record,
    validate_saved_review_products,
    verify_preview_hashes,
)
from chickpea_ssl.field2_readiness import sha256
from scripts.build_field2_blind_annotation_sampling_frame import collapsed_cube_candidates
from scripts.run_field2_point_annotator import HTML, PointAnnotationStore


def test_configuration_freezes_exact_40_cube_inventory():
    config = yaml.safe_load(Path("configs/field2_blind_evaluation.yaml").read_text())
    expected_numbers = [2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,22,23,
                        24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
    assert config["expected_cube_ids"] == [f"field2_cube{value:02d}" for value in expected_numbers]
    assert len(config["expected_cube_ids"]) == 40
    assert config["valid_support_manifest_sha256"] == "1f48ffe67d7ad2cbf95cca5c2e6e98d15245843435c519bbb8a67841c2bbf3e3"
    assert config["freeze"]["expected_role_totals"] == {
        "primary_three_class": 20, "challenge_only": 8,
        "chickpea_absent_negative_control": 7, "sensitivity_only": 5,
        "exclude_with_reason": 0, "unreviewed": 0,
    }


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


def completed_review_store(tmp_path: Path) -> tuple[CubeRoleReviewStore, dict]:
    manifest_path = make_review_manifest(tmp_path)
    store = CubeRoleReviewStore(tmp_path, manifest_path, tmp_path / "annotations")
    payload = store.load()
    first = payload["reviews"]["field2_cube02"]
    first.update(primary_role="primary_three_class", confidence="high", reviewed=True,
                 review_timestamp="2026-08-17T12:00:00+00:00")
    first["flags"].update(chickpea_visible=True, ordinary_weed_visible=True, soil_visible=True)
    second = payload["reviews"]["field2_cube03"]
    second.update(primary_role="chickpea_absent_negative_control", confidence="low", reviewed=True,
                  review_timestamp="2026-08-17T12:01:00+00:00")
    second["flags"].update(chickpea_absent_confirmed=True, ordinary_weed_visible=True, soil_visible=True)
    store.save(payload)
    return store, store.load()


def test_saved_review_products_require_exact_json_csv_audit_agreement(tmp_path):
    store, payload = completed_review_store(tmp_path)
    totals = validate_saved_review_products(
        payload, store.csv_path, store.audit_path, store.expected_cube_ids,
    )
    assert totals["reviewed"] == 2
    assert totals["primary_roles"]["primary_three_class"] == 1
    store.csv_path.write_text(store.csv_path.read_text().replace("primary_three_class", "challenge_only"))
    try:
        validate_saved_review_products(payload, store.csv_path, store.audit_path, store.expected_cube_ids)
    except ValueError as error:
        assert "CSV and JSON disagree" in str(error)
    else:
        raise AssertionError("Mutated review CSV passed JSON agreement validation")


def test_frozen_role_contract_detects_later_input_mutation(tmp_path):
    store, payload = completed_review_store(tmp_path)
    totals = review_totals(payload)
    support_manifest = tmp_path / "support.csv"
    support_manifest.write_text("cube_id\nfield2_cube02\nfield2_cube03\n")
    role_table = tmp_path / "frozen" / "roles.csv"
    summary = tmp_path / "frozen" / "summary.csv"
    overview = tmp_path / "frozen" / "overview.png"
    atomic_write_bytes(role_table, frozen_role_table_bytes(payload))
    atomic_write_bytes(summary, role_summary_csv_bytes(totals))
    atomic_write_bytes(overview, store.overview_path.read_bytes())
    inputs = {
        "review_json": store.json_path, "review_csv": store.csv_path,
        "review_audit": store.audit_path, "review_overview": store.overview_path,
        "review_package_manifest": store.manifest_path,
    }
    outputs = {"role_table": role_table, "role_summary": summary, "frozen_overview": overview}
    contract = {
        "version": "field2_cube_role_contract_v1", "status": "field2_cube_roles_frozen",
        "freeze_timestamp_utc": "2026-08-17T13:00:00+00:00", "freeze_git_commit": "a" * 40,
        "review_schema_version": "field2_cube_role_reviews_v1", "review_revision": payload["revision"],
        "cube_roles": [payload["reviews"][cube_id] for cube_id in store.expected_cube_ids],
        "role_totals": totals["primary_roles"], "biological_flag_totals": totals["biological_flags"],
        "frozen_input_products": {
            name: {"path": str(path.relative_to(tmp_path)), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "frozen_output_products": {
            name: {"path": str(path.relative_to(tmp_path)), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "field2_source_manifest_sha256": "source-hash",
        "field2_valid_support_manifest_sha256": sha256(support_manifest),
        "provenance": {
            "prediction_free_review": True, "supervised_checkpoint_loaded": False,
            "biological_roles_assigned_automatically": False,
        },
    }
    contract_path = tmp_path / "contract.yaml"
    atomic_write_yaml(contract_path, contract)
    config = {
        "expected_cube_ids": store.expected_cube_ids,
        "source_manifest_sha256": "source-hash",
        "review": {"package_manifest": str(store.manifest_path.relative_to(tmp_path))},
        "inputs": {"valid_support_manifest": str(support_manifest.relative_to(tmp_path))},
        "freeze": {
            "expected_role_totals": totals["primary_roles"],
            "expected_flag_totals": totals["biological_flags"],
        },
    }
    assert validate_frozen_role_contract(tmp_path, config, contract_path)["totals"] == totals
    source_review_text = store.json_path.read_text()
    store.json_path.write_text(source_review_text.replace('"revision": 1', '"revision": 2'))
    try:
        validate_frozen_role_contract(tmp_path, config, contract_path)
    except ValueError as error:
        assert "SHA-256 changed" in str(error)
    else:
        raise AssertionError("Mutated frozen review JSON passed contract validation")
    store.json_path.write_text(source_review_text)
    mutated_contract = yaml.safe_load(contract_path.read_text())
    mutated_contract["cube_roles"][0]["investigator_notes"] = "later mutation"
    atomic_write_yaml(contract_path, mutated_contract)
    try:
        validate_frozen_role_contract(tmp_path, config, contract_path)
    except ValueError as error:
        assert "source reviews differ" in str(error)
    else:
        raise AssertionError("Mutated frozen role contract passed cross-product validation")


def make_frozen_role_fixture(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    store, payload = completed_review_store(tmp_path)
    totals = review_totals(payload)
    support = tmp_path / "support.csv"; support.write_text("cube_id\nfield2_cube02\nfield2_cube03\n")
    role_table, summary, overview = tmp_path / "roles.csv", tmp_path / "summary.csv", tmp_path / "frozen.png"
    atomic_write_bytes(role_table, frozen_role_table_bytes(payload)); atomic_write_bytes(summary, role_summary_csv_bytes(totals)); atomic_write_bytes(overview, store.overview_path.read_bytes())
    inputs = {"review_json": store.json_path, "review_csv": store.csv_path, "review_audit": store.audit_path, "review_overview": store.overview_path, "review_package_manifest": store.manifest_path}
    outputs = {"role_table": role_table, "role_summary": summary, "frozen_overview": overview}
    contract = {
        "version": "field2_cube_role_contract_v1", "status": "field2_cube_roles_frozen",
        "freeze_timestamp_utc": "2026-08-17T13:00:00+00:00", "freeze_git_commit": "a" * 40,
        "review_schema_version": "field2_cube_role_reviews_v1", "review_revision": payload["revision"],
        "cube_roles": [payload["reviews"][cube] for cube in store.expected_cube_ids],
        "role_totals": totals["primary_roles"], "biological_flag_totals": totals["biological_flags"],
        "frozen_input_products": {name: {"path": str(path.relative_to(tmp_path)), "sha256": sha256(path)} for name, path in inputs.items()},
        "frozen_output_products": {name: {"path": str(path.relative_to(tmp_path)), "sha256": sha256(path)} for name, path in outputs.items()},
        "field2_source_manifest_sha256": "source-hash", "field2_valid_support_manifest_sha256": sha256(support),
        "provenance": {"prediction_free_review": True, "supervised_checkpoint_loaded": False, "biological_roles_assigned_automatically": False},
    }
    role_contract = tmp_path / "role_contract.yaml"; atomic_write_yaml(role_contract, contract)
    config = {
        "expected_cube_ids": store.expected_cube_ids, "source_manifest_sha256": "source-hash",
        "review": {"package_manifest": str(store.manifest_path.relative_to(tmp_path))},
        "inputs": {"valid_support_manifest": str(support.relative_to(tmp_path))},
        "freeze": {"expected_role_totals": totals["primary_roles"], "expected_flag_totals": totals["biological_flags"]},
    }
    return config, role_contract, support, store.manifest_path


def test_frozen_sampling_contract_and_reserve_policy_detect_mutation(tmp_path):
    config, role_contract, support, review_manifest = make_frozen_role_fixture(tmp_path)
    seed = 20260817
    def sample(frame, cube, row, role, ground):
        return {
            "sample_id": "f2-" + hashlib.sha256(f"{seed}:{cube}:{row}:0".encode()).hexdigest()[:20],
            "sampling_frame": frame, "cube_id": cube, "row": row, "column": 0,
            "x": float(row), "y": 0.0, "crs": "EPSG:32617", "geographic_ground_cell_id": ground,
            "first_stage_inclusion_probability": .5, "final_inclusion_probability": .5,
            "overall_inclusion_probability": .25, "design_weight": 4.0,
            "reserve_release_status": "released_for_main_annotation" if frame == "main" else "locked_pending_separate_authorization",
        }
    main = pd.DataFrame([
        sample("main", "field2_cube02", 1, "primary_three_class", "g1"),
        sample("main", "field2_cube02", 2, "primary_three_class", "g2"),
        sample("main", "field2_cube03", 3, "chickpea_absent_negative_control", "g3"),
        sample("main", "field2_cube03", 4, "chickpea_absent_negative_control", "g4"),
    ])
    reserve = pd.DataFrame([
        sample("reserve", "field2_cube02", 5, "primary_three_class", "g5"),
        sample("reserve", "field2_cube03", 6, "chickpea_absent_negative_control", "g6"),
    ])
    combined = pd.concat([main, reserve], ignore_index=True)
    paths = {name: tmp_path / f"{name}.csv" for name in ("main", "reserve", "combined", "allocation_audit", "stratum_block_audit")}
    main.to_csv(paths["main"], index=False); reserve.to_csv(paths["reserve"], index=False); combined.to_csv(paths["combined"], index=False)
    allocation = pd.DataFrame([
        {"cube_id": "field2_cube02", "main_requested": 2, "main_achieved": 2, "reserve_requested": 1, "reserve_achieved": 1},
        {"cube_id": "field2_cube03", "main_requested": 2, "main_achieved": 2, "reserve_requested": 1, "reserve_achieved": 1},
    ])
    allocation.to_csv(paths["allocation_audit"], index=False); pd.DataFrame({"points": [6]}).to_csv(paths["stratum_block_audit"], index=False)
    audit = {"valid_support_violations": 0, "invalid_context_or_border_violations": 0, "duplicate_ground_cell_count": 0, "main_reserve_intersection": 0}
    audit_path, map_path = tmp_path / "audit.yaml", tmp_path / "map.png"; atomic_write_yaml(audit_path, audit); map_path.write_bytes(b"map")
    outputs = {**paths, "sampling_audit": audit_path, "sampling_map": map_path}
    gates = {"confident_chickpea": 75}; prohibited = ["supervised_predictions"]
    config["sampling"] = {
        "seed": seed, "expected_main_total": 4, "expected_reserve_total": 2,
        "minimum_separation_m": .25,
        "main_points_per_cube_by_role": {"primary_three_class": 2, "chickpea_absent_negative_control": 2},
        "reserve_points_per_cube_by_role": {"primary_three_class": 1, "chickpea_absent_negative_control": 1},
        "reserve_release_gates_after_800_main_reviews": gates,
        "reserve_release_prohibited_evidence": prohibited,
    }
    contract = {
        "version": "field2_blind_sampling_frame_contract_v1", "status": "field2_blind_sampling_frame_frozen",
        "materialization_git_commit": "b" * 40, "seed": seed,
        "input_contracts": {
            "cube_role_contract": {"path": str(role_contract.relative_to(tmp_path)), "sha256": sha256(role_contract)},
            "valid_support_manifest": {"path": str(support.relative_to(tmp_path)), "sha256": sha256(support)},
            "review_package_manifest": {"path": str(review_manifest.relative_to(tmp_path)), "sha256": sha256(review_manifest)},
            "field2_source_manifest_sha256": "source-hash",
        },
        "frozen_outputs": {name: {"path": str(path.relative_to(tmp_path)), "sha256": sha256(path)} for name, path in outputs.items()},
        "reserve_release_policy": {"status": "locked_not_released_for_annotation", "authorization_required": "separate immutable reserve-release contract", "release_gates_after_800_main_reviews": gates, "may_not_depend_on": prohibited},
        "audit": audit,
        "provenance": {"prediction_free": True, "supervised_checkpoint_loaded": False, "predictions_or_probabilities_used": False, "pseudo_labels_used": False, "embeddings_ssl_features_or_clusters_used": False, "existing_biological_masks_used": False, "investigator_point_labels_used": False, "biological_classes_inferred_automatically": False},
    }
    contract_path = tmp_path / "sampling_contract.yaml"; atomic_write_yaml(contract_path, contract)
    assert len(validate_frozen_sampling_contract(tmp_path, config, contract_path)["combined"]) == 6
    paths["reserve"].write_text(paths["reserve"].read_text().replace("locked_pending", "released"))
    try:
        validate_frozen_sampling_contract(tmp_path, config, contract_path)
    except ValueError as error:
        assert "SHA-256 changed" in str(error)
    else:
        raise AssertionError("Mutated reserve frame passed frozen sampling validation")


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


def test_global_minimum_separation_thinning():
    frame = pd.DataFrame([
        {"cube_id": "a", "row": 0, "column": 0, "x": 0.0, "y": 0.0, "crs": "EPSG:32617", "deterministic_selection_rank": 1},
        {"cube_id": "b", "row": 0, "column": 0, "x": 0.1, "y": 0.0, "crs": "EPSG:32617", "deterministic_selection_rank": 2},
        {"cube_id": "b", "row": 1, "column": 0, "x": 0.3, "y": 0.0, "crs": "EPSG:32617", "deterministic_selection_rank": 3},
    ])
    thinned = minimum_separation_thin(frame, .25)
    assert thinned.x.tolist() == [0.0, 0.3]


def test_candidate_frame_requires_valid_support_and_full_visual_context():
    support = np.zeros((500, 500), dtype=bool)
    support[10, 10] = True
    support[250, 250] = True
    scalar = np.ones((500, 500), dtype=np.float32)
    role = {"primary_role": "challenge_only", "flags": {name: False for name in REVIEW_FLAGS}, "investigator_notes": ""}
    sampling = {
        "annotation_context_radius_preview_pixels": 110,
        "scalar_index_rank_strata": ["q1", "q2", "q3", "q4", "q5"],
        "geographic_ground_cell_size_m": .1, "spatial_group_size_m": 5.0, "seed": 20260817,
    }
    frame = collapsed_cube_candidates(
        "field2_cube02", support, scalar, Affine(.02, 0, 0, 0, -.02, 10), "EPSG:32617",
        role, "a" * 64, "b" * 64, json.dumps({name: "c" * 64 for name in PREVIEW_LAYERS}),
        1, 500, 500, sampling,
    )
    assert frame[["row", "column"]].to_dict("records") == [{"row": 250, "column": 250}]
    assert frame.valid_support_status.all() and frame.valid_context_status.all()


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


def test_exact_deterministic_main_reserve_allocation_and_disjointness():
    labels = ["q1", "q2", "q3", "q4", "q5"]
    rows = []
    for cube_position, (cube_id, role) in enumerate((
        ("field2_cube02", "primary_three_class"),
        ("field2_cube03", "sensitivity_only"),
    )):
        for stratum_position, stratum in enumerate(labels):
            for number in range(8):
                base = cube_position * 10_000 + stratum_position * 1_000 + number * 10
                rows.append({
                    "cube_id": cube_id, "cube_evaluation_role": role,
                    "scalar_index_rank_stratum": stratum,
                    "spatial_group_id": f"{cube_id}:block:{stratum}:{number}",
                    "deterministic_selection_rank": base, "row": base, "column": number,
                    "x": float(base), "y": float(number), "crs": "EPSG:32617",
                    "first_stage_inclusion_probability": .5,
                })
    eligible = pd.DataFrame(rows)
    args = (
        eligible,
        {"field2_cube02": "primary_three_class", "field2_cube03": "sensitivity_only"},
        {"primary_three_class": 10, "sensitivity_only": 5},
        {"primary_three_class": 5, "sensitivity_only": 5},
        labels, 20260817,
    )
    main, reserve = select_main_reserve_frames(*args)
    repeated_main, repeated_reserve = select_main_reserve_frames(*args)
    assert main.sample_id.tolist() == repeated_main.sample_id.tolist()
    assert reserve.sample_id.tolist() == repeated_reserve.sample_id.tolist()
    assert len(main) == 15 and len(reserve) == 10
    assert not set(main.sample_id) & set(reserve.sample_id)
    assert main.groupby("cube_id").size().to_dict() == {"field2_cube02": 10, "field2_cube03": 5}
    assert reserve.groupby("cube_id").size().to_dict() == {"field2_cube02": 5, "field2_cube03": 5}
    assert main.groupby(["cube_id", "scalar_index_rank_stratum"]).size().min() >= 1
    assert np.allclose(main.overall_inclusion_probability, main.first_stage_inclusion_probability * main.final_inclusion_probability)
    assert np.allclose(main.design_weight, 1 / main.overall_inclusion_probability)
    assert balanced_stratum_quotas(12, labels) == {"q1": 3, "q2": 3, "q3": 2, "q4": 2, "q5": 2}


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
    assert set(POINT_LABELS) == {
        "chickpea", "ordinary_weed", "tall_grass_weed", "soil",
        "chickpea_soil_mixed", "chickpea_weed_mixed", "weed_soil_mixed",
        "uncertain", "nodata_invalid",
    }
    config = yaml.safe_load(Path("configs/field2_blind_evaluation.yaml").read_text())
    assert config["point_annotation"]["labels"] == list(POINT_LABELS)


def make_main_annotation_frame(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path).set_index("cube_id")
    rows = []
    for number in range(800):
        cube_id = "field2_cube02" if number < 400 else "field2_cube03"
        rows.append({
            "sample_id": f"sample-{number:04d}", "sampling_frame": "main",
            "cube_id": cube_id, "cube_evaluation_role": (
                "chickpea_absent_negative_control" if cube_id == "field2_cube02" else "primary_three_class"
            ),
            "source_preview_checksums": manifest.loc[cube_id, "preview_sha256_json"],
            "row": number % 6, "column": number % 8,
        })
    return pd.DataFrame(rows)


def make_natural_rgb_manifest(tmp_path: Path, cube_ids=("field2_cube02", "field2_cube03")) -> pd.DataFrame:
    selected = {
        "red": {"target_wavelength_nm": 670.0, "selected_wavelength_nm": 671.0,
                "python_band_index_0_based": 2, "envi_band_number_1_based": 3,
                "absolute_difference_nm": 1.0},
        "green": {"target_wavelength_nm": 550.0, "selected_wavelength_nm": 551.0,
                  "python_band_index_0_based": 1, "envi_band_number_1_based": 2,
                  "absolute_difference_nm": 1.0},
        "blue": {"target_wavelength_nm": 480.0, "selected_wavelength_nm": 481.0,
                 "python_band_index_0_based": 0, "envi_band_number_1_based": 1,
                 "absolute_difference_nm": 1.0},
    }
    rows = []
    for cube_id in cube_ids:
        path = tmp_path / f"{cube_id}_natural_rgb.png"
        atomic_write_bytes(path, natural_rgb_png_bytes(np.zeros((6, 8, 3), dtype=np.uint8)))
        rows.append({
            "cube_id": cube_id, "width": 8, "height": 6, "preview_step": 1,
            "natural_rgb_path": path.name, "natural_rgb_sha256": sha256(path),
            "selected_bands_json": json.dumps(selected, sort_keys=True),
        })
    return pd.DataFrame(rows)


def make_point_store(
    tmp_path: Path, frame: pd.DataFrame, manifest_path: Path, area_contract_hash: str = "",
) -> PointAnnotationStore:
    return PointAnnotationStore(
        tmp_path, frame, "b" * 64, manifest_path, make_natural_rgb_manifest(tmp_path),
        "field2_annotation_display_natural_rgb_v1", "c" * 64, tmp_path / "points",
        area_contract_hash,
    )


def test_point_viewer_model_freezes_sample_and_separates_inspection_pan_zoom_reset_and_layer():
    source = Path("scripts/field2_point_viewer.js").resolve()
    javascript = f"""
const assert = require('assert');
const viewer = require({json.dumps(str(source))});
const initial = viewer.createState({{row: 17, column: 23}}, 100, 80);
const clicked = viewer.inspect(initial, 42.8, 51.2);
assert.deepStrictEqual(clicked.frozen, {{row: 17, column: 23}});
assert.deepStrictEqual(clicked.inspection, {{row: 42, column: 51}});
assert.deepStrictEqual(initial.inspection, {{row: 17, column: 23}});
const zoomed = viewer.setZoom(clicked, 8);
assert.strictEqual(zoomed.zoom, 8);
const panned = viewer.pan(zoomed, 80, -40, zoomed.zoom);
assert.deepStrictEqual(panned.center, {{x: 40, y: 45}});
assert.deepStrictEqual(panned.frozen, initial.frozen);
const switched = viewer.switchLayer(panned, 'pca');
assert.strictEqual(switched.layer, 'pca');
assert.deepStrictEqual(switched.frozen, initial.frozen);
assert.deepStrictEqual(switched.inspection, clicked.inspection);
const reset = viewer.reset(switched);
assert.strictEqual(reset.zoom, 0);
assert.deepStrictEqual(reset.center, {{x: 50, y: 40}});
assert.deepStrictEqual(reset.inspection, reset.frozen);
assert.deepStrictEqual(reset.frozen, {{row: 17, column: 23}});
"""
    result = subprocess.run(["node", "-e", javascript], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_point_viewer_exposes_required_controls_and_nonblocking_overlay_contract():
    viewer_js = Path("scripts/field2_point_viewer.js").read_text()
    assert "Sample location is frozen; clicking only changes the inspection view." in HTML
    for element_id in (
        "viewer", "frozenMagnifier", "inspectionMagnifier", "resetView", "layer",
        "prev", "next", "save", "review", "reviewer", "note", "labelButtons",
        "confidenceButtons",
    ):
        assert f'id="{element_id}"' in HTML
    for layer in ("natural_rgb", "false_colour", "pca", "stored_index", "support_outline"):
        assert f'value="{layer}"' in HTML
    for zoom in (2, 4, 8, 16):
        assert f'data-zoom="{zoom}"' in HTML
    assert ".controls-shield{pointer-events:none" in HTML
    assert 'viewer.addEventListener("pointerdown"' in viewer_js
    assert 'viewer.addEventListener("pointermove"' in viewer_js
    assert 'viewer.addEventListener("wheel"' in viewer_js
    assert 'button.addEventListener("click", () => setChoice' in viewer_js
    assert '$("save").addEventListener("click", save)' in viewer_js
    assert '$("review").addEventListener("click"' in viewer_js


def test_point_schema_adds_weed_soil_mixed_and_enforces_alley_labels_and_boundary_workflow(tmp_path):
    manifest_path = make_review_manifest(tmp_path)
    frame = make_main_annotation_frame(manifest_path)
    frame["zone_type"] = "alley"
    frame["domain"] = "alley"
    store = make_point_store(tmp_path, frame, manifest_path, "d" * 64)
    payload = store.load()
    record = payload["annotations"]["sample-0000"]
    assert record["selected_label"] == "" and record["boundary_needs_correction"] is False
    assert record["zone_type"] == "alley" and record["area_zone_contract_sha256"] == "d" * 64
    record["selected_label"] = "chickpea"
    assert "label_not_allowed_in_alley" in store.validate(payload)["sample-0000"]
    record["selected_label"] = "weed_soil_mixed"
    record["boundary_needs_correction"] = True
    assert "boundary_correction_must_not_force_label" in store.validate(payload)["sample-0000"]
    record["selected_label"] = ""
    assert store.validate(payload)["sample-0000"] == []
    viewer_js = Path("scripts/field2_point_viewer.js").read_text()
    assert "schema.alley_labels" in viewer_js
    assert '$("boundaryCorrection")' in viewer_js
    assert "Frozen zone:" in viewer_js


def test_frozen_sampling_and_annotation_display_hashes_are_still_exact():
    sampling_contract = yaml.safe_load(
        Path("metadata/local/contracts/field2_blind_sampling_frame_contract.yaml").read_text()
    )
    assert sampling_contract["main_count"] == 800
    assert sampling_contract["reserve_count"] == 396
    for item in ("main", "reserve", "combined"):
        frozen = sampling_contract["frozen_outputs"][item]
        assert sha256(Path(frozen["path"])) == frozen["sha256"]
    display_contract = Path("metadata/local/contracts/field2_annotation_display_addendum.yaml")
    assert sha256(display_contract) == "1854458fb428b7c4b7eb1ec15bbc5b354dc21573f35944ef2cefbbb36ef08a7f"
    display = yaml.safe_load(display_contract.read_text())
    assert display["cube_count"] == 40
    assert display["input_contracts"]["sampling_contract"]["sha256"] == sha256(
        Path(display["input_contracts"]["sampling_contract"]["path"])
    )


def test_natural_rgb_band_selection_stretch_mask_and_png_are_deterministic():
    selected = select_natural_rgb_bands(
        [479.0, 552.0, 668.0, 801.0],
        {"red": 670.0, "green": 550.0, "blue": 480.0}, 5.0,
    )
    assert [selected[name]["python_band_index_0_based"] for name in ("red", "green", "blue")] == [2, 1, 0]
    reflectance = np.arange(4 * 5 * 4, dtype=np.float32).reshape(4, 5, 4)
    support = np.ones((4, 5), dtype=bool)
    support[0, :] = False
    rgb, stretch = render_natural_rgb_uint8(reflectance, support, selected, (2.0, 98.0))
    assert rgb.dtype == np.uint8 and rgb.shape == (4, 5, 3)
    assert np.all(rgb[~support] == 0)
    assert all(stretch[channel]["low_percentile"] == 2.0 for channel in stretch)
    first = natural_rgb_png_bytes(rgb)
    assert first == natural_rgb_png_bytes(rgb)
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(natural_rgb_png_bytes(rgb)).hexdigest()


def test_natural_rgb_band_selection_rejects_out_of_tolerance():
    try:
        select_natural_rgb_bands(
            [400.0, 500.0, 600.0],
            {"red": 670.0, "green": 550.0, "blue": 480.0}, 10.0,
        )
    except ValueError as error:
        assert "within 10.0 nm" in str(error)
    else:
        raise AssertionError("Out-of-tolerance natural RGB band selection passed")


def test_main_annotation_save_resume_no_default_and_role_contradiction(tmp_path):
    manifest_path = make_review_manifest(tmp_path)
    frame = make_main_annotation_frame(manifest_path)
    store = make_point_store(tmp_path, frame, manifest_path)
    payload = store.load()
    assert len(payload["annotations"]) == 800
    assert not any(record["selected_label"] for record in payload["annotations"].values())
    record = payload["annotations"]["sample-0000"]
    record.update(selected_label="chickpea", confidence="low", reviewed=True,
                  review_timestamp="2026-08-17T14:00:00+00:00", role_contradiction=True)
    records, reviewed, revision = store.save(payload)
    assert (records, reviewed, revision) == (800, 1, 1)
    resumed = store.load()
    assert resumed["annotations"]["sample-0000"]["role_contradiction"] is True
    csv_frame = pd.read_csv(store.csv_path, keep_default_na=False)
    assert len(csv_frame) == 800 and csv_frame.loc[0, "selected_label"] == "chickpea"
    assert store.audit_path.is_file() and store.overview_path.is_file()


def test_display_version_migration_preserves_labels_and_flags_only_reviewed_records(tmp_path):
    manifest_path = make_review_manifest(tmp_path)
    store = make_point_store(tmp_path, make_main_annotation_frame(manifest_path), manifest_path)
    payload = store.initial_payload()
    payload["annotations"]["sample-0000"].update(
        selected_label="soil", confidence="high", reviewed=True,
        review_timestamp="2026-08-17T14:00:00+00:00",
    )
    payload.pop("annotation_display_version")
    payload.pop("annotation_display_contract_sha256")
    for record in payload["annotations"].values():
        for key in ("annotation_display_version", "annotation_display_contract_sha256",
                    "natural_rgb_preview_sha256", "requires_visual_rereview"):
            record.pop(key)
    migrated, count = store.migrate_display_version(payload)
    assert count == 1
    assert migrated["annotations"]["sample-0000"]["selected_label"] == "soil"
    assert migrated["annotations"]["sample-0000"]["requires_visual_rereview"] is True
    assert migrated["annotations"]["sample-0001"]["selected_label"] == ""
    assert migrated["annotations"]["sample-0001"]["requires_visual_rereview"] is False
    assert migrated["automatic_metadata"]["display_version_migration_changed_biological_labels"] is False


def test_new_display_version_with_zero_reviews_requires_no_rereview(tmp_path):
    manifest_path = make_review_manifest(tmp_path)
    store = make_point_store(tmp_path, make_main_annotation_frame(manifest_path), manifest_path)
    payload, count = store.migrate_display_version(store.initial_payload())
    assert count == 0
    assert not any(record["requires_visual_rereview"] for record in payload["annotations"].values())
    assert not any(record["selected_label"] for record in payload["annotations"].values())


def test_point_annotator_rejects_any_reserve_exposure(tmp_path):
    manifest_path = make_review_manifest(tmp_path)
    frame = make_main_annotation_frame(manifest_path)
    frame.loc[0, "sampling_frame"] = "reserve"
    try:
        make_point_store(tmp_path, frame, manifest_path)
    except ValueError as error:
        assert "main frame" in str(error)
    else:
        raise AssertionError("Point annotator accepted a reserve-frame sample")


def test_review_entrypoints_do_not_import_model_frameworks():
    prohibited = {"torch", "torchvision", "xgboost", "sklearn", "tensorflow"}
    paths = [
        Path("scripts/prepare_field2_cube_role_review.py"),
        Path("scripts/run_field2_cube_role_reviewer.py"),
        Path("scripts/freeze_field2_cube_role_contract.py"),
        Path("scripts/validate_field2_cube_role_contract.py"),
        Path("scripts/build_field2_blind_annotation_sampling_frame.py"),
        Path("scripts/validate_field2_blind_sampling_contract.py"),
        Path("scripts/run_field2_point_annotator.py"),
        Path("scripts/prepare_field2_annotation_rgb.py"),
        Path("scripts/validate_field2_annotation_display.py"),
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
