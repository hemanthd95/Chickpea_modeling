import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from chickpea_ssl.field2_blind_review import validate_annotation_display_contract
from chickpea_ssl.field2_chickpea_review import (
    MANUAL_LABEL_PROVENANCE,
    ndvi_status,
    provisional_reference,
    validate_policy,
)
from chickpea_ssl.field2_readiness import sha256
from scripts.run_field2_point_annotator_v2 import ChickpeaReviewStore


PROJECT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT / "configs/field2_chickpea_review_policy.yaml"


def load_configs():
    policy = yaml.safe_load(POLICY_PATH.read_text())
    v2 = yaml.safe_load((PROJECT / policy["v2_sampling_config"]).read_text())
    base = yaml.safe_load((PROJECT / v2["base_config"]).read_text())
    area = yaml.safe_load((PROJECT / v2["area_config"]).read_text())
    return policy, v2, base, area


def test_predeclared_ndvi_rule_boundaries_and_invalid_values():
    assert ndvi_status(7.0, 13.0) == (0.3, "vegetation")
    assert ndvi_status(8.0, 12.0) == (0.2, "soil")
    assert ndvi_status(0.0, 0.0) == (None, "invalid")
    assert ndvi_status(float("nan"), 1.0) == (None, "invalid")


def test_area_constrained_provisional_references_never_create_chickpea():
    policy, *_ = load_configs()
    rules = policy["provisional_reference_rules"]
    assert provisional_reference("alley", "soil", rules) == ("soil", "rule_based_soil_ndvi_0p30", False)
    assert provisional_reference("alley", "vegetation", rules) == ("field_weed", "investigator_area_constrained_field_weed", False)
    assert provisional_reference("outside_research_field", "vegetation", rules) == ("outside_weed_ood", "investigator_area_constrained_outside_weed_ood", False)
    assert provisional_reference("research_crop_area", "vegetation", rules) == ("pending_investigator_chickpea_review", "", True)
    assert provisional_reference("unassigned_valid_support", "vegetation", rules) == ("pending_investigator_chickpea_review", "", True)
    assert provisional_reference("research_crop_area", "invalid", rules) == ("nodata_invalid", "nodata_invalid", False)


def test_manual_decisions_have_explicit_investigator_provenance():
    assert MANUAL_LABEL_PROVENANCE == {
        "chickpea": "investigator_chickpea",
        "weed_unspecified": "investigator_nonchickpea_vegetation",
        "chickpea_weed_mixed": "investigator_chickpea",
        "uncertain": "investigator_uncertain",
    }


def test_local_policy_has_exact_workload_and_field1_audit_when_available():
    policy, *_ = load_configs()
    contract_path = PROJECT / policy["outputs"]["contract"]
    if not contract_path.exists():
        pytest.skip("machine-local frozen review policy is not present")
    result = validate_policy(PROJECT, policy)
    assert result["contract"]["counts"] == {
        "v2_main_total": 800, "rule_based_soil": 99, "alley_field_weed": 146,
        "outside_weed_ood": 78, "manual_chickpea_review": 477, "invalid": 0,
    }
    audit = pd.read_csv(PROJECT / policy["outputs"]["report_root"] / policy["outputs"]["field1_ndvi_audit"])
    assert audit.groupby("frozen_field1_label").pixel_count.sum().to_dict() == {
        "chickpea": 441995, "soil": 6930741, "weed": 1440422,
    }
    assert audit[audit.ndvi_rule_status == "soil"].pixel_count.sum() == 0
    assert not audit.threshold_optimized_on_field2.astype(bool).any()


def make_temp_store(tmp_path: Path):
    policy, v2, base, area = load_configs()
    result = validate_policy(PROJECT, policy)
    main = result["main"].copy()
    main["zone_type"] = main["domain_name"]
    main["domain"] = main["domain_reporting_stratum"]
    display_path = PROJECT / base["annotation_display"]["display_contract"]
    display = validate_annotation_display_contract(PROJECT, base, display_path)
    store = ChickpeaReviewStore(
        PROJECT, main, result["contract"]["inputs"]["v2_main_frame_sha256"],
        PROJECT / base["review"]["package_manifest"], display["manifest"],
        base["annotation_display"]["version"], sha256(display_path), tmp_path,
        sha256(PROJECT / area["freeze"]["contract"]), references=result["references"],
        policy_contract_hash=sha256(PROJECT / policy["outputs"]["contract"]),
    )
    return store


def test_temporary_save_reload_preserves_production_and_restricts_nonqueue(tmp_path):
    policy, v2, *_ = load_configs()
    production = PROJECT / v2["point_annotation"]["output_root"] / "field2_blind_main_point_annotations.json"
    before = sha256(production)
    store = make_temp_store(tmp_path)
    payload = store.load()
    assert len(payload["annotations"]) == 800 and len(store.queue_ids) == 477
    sample_id = store.queue_ids[0]
    record = payload["annotations"][sample_id]
    record.update(
        selected_label="weed_unspecified", manual_decision="weed_unspecified",
        optional_weed_subtype="tall_grass_weed",
        reference_provenance="investigator_nonchickpea_vegetation",
        confidence="high", reviewed=True,
        review_timestamp="2026-08-25T12:00:00+00:00",
        boundary_needs_correction=True,
    )
    assert store.save(payload) == (800, 1, 1)
    resumed = store.load()["annotations"][sample_id]
    assert resumed["selected_label"] == "weed_unspecified"
    assert resumed["optional_weed_subtype"] == "tall_grass_weed"
    assert resumed["boundary_needs_correction"] is True
    nonqueue = next(value for value in store.sample_order if value not in store.queue_set)
    resumed_payload = store.load()
    resumed_payload["annotations"][nonqueue].update(
        selected_label="chickpea", manual_decision="chickpea",
        reference_provenance="investigator_chickpea", confidence="high",
    )
    issues = store.validate(resumed_payload)
    assert "rule_or_area_constrained_record_not_manually_editable" in issues[nonqueue]
    assert sha256(production) == before


def test_interface_is_chickpea_only_and_resumes_first_unreviewed():
    html = (PROJECT / "scripts/run_field2_point_annotator.py").read_text()
    viewer = (PROJECT / "scripts/field2_point_viewer.js").read_text()
    server = (PROJECT / "scripts/run_field2_point_annotator_v2.py").read_text()
    assert "Label the frozen yellow center pixel—not the surrounding row." in html
    assert 'type="checkbox"> Boundary needs correction' in html
    assert '"manual_review_count": 477' in server
    assert '"reserve_exposed": False' in server
    assert '"workflow": "chickpea_review_v1"' in server
    assert "firstUnreviewed" in viewer
    assert '$("next").textContent = "Skip"' in viewer
    assert "Save & Next" in viewer
