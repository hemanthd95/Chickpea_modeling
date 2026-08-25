from pathlib import Path

import pandas as pd
import pytest
import yaml

from chickpea_ssl.field2_area_sampling_v2 import (
    DOMAINS,
    initial_quotas,
    load_verified_v2,
    reconcile_quotas,
)
from chickpea_ssl.field2_readiness import sha256


PROJECT = Path(__file__).resolve().parents[1]
ZERO_CROP_PRIMARY = {
    "field2_cube09", "field2_cube11", "field2_cube17",
    "field2_cube18", "field2_cube19", "field2_cube20",
}
ALL_OUTSIDE = {"field2_cube38", "field2_cube41", "field2_cube42"}


def configs():
    v2 = yaml.safe_load((PROJECT / "configs/field2_area_sampling_v2.yaml").read_text())
    base = yaml.safe_load((PROJECT / v2["base_config"]).read_text())
    return v2, base


def synthetic_capacity():
    v2, base = configs()
    cubes = base["expected_cube_ids"]
    primary = {
        "field2_cube07", "field2_cube08", "field2_cube09", "field2_cube11",
        "field2_cube12", "field2_cube13", "field2_cube17", "field2_cube18",
        "field2_cube19", "field2_cube20", "field2_cube21", "field2_cube22",
        "field2_cube25", "field2_cube26", "field2_cube27", "field2_cube28",
        "field2_cube29", "field2_cube30", "field2_cube33", "field2_cube34",
    }
    roles = {cube: "primary_three_class" if cube in primary else "challenge_only" for cube in cubes}
    eligible_crop = primary - ZERO_CROP_PRIMARY
    rows = []
    for position, cube in enumerate(cubes):
        for domain in DOMAINS:
            feasible = 500 + position
            if domain == "research_crop_area" and cube not in eligible_crop:
                feasible = 0
            if domain == "alley" and cube in ALL_OUTSIDE:
                feasible = 0
            rows.append({
                "cube_id": cube, "sampling_domain": domain,
                "operational_pixel_count": feasible * 25,
                "feasible_capacity": feasible,
            })
    return v2, cubes, roles, pd.DataFrame(rows)


def test_capacity_aware_initial_allocation_matches_approved_global_design():
    v2, cubes, roles, capacity = synthetic_capacity()
    quotas = initial_quotas(cubes, roles, capacity, v2)
    expected = {
        "main": {"research_crop_area": 400, "alley": 160, "outside_research_field": 120, "unassigned_valid_support": 120},
        "reserve": {"research_crop_area": 200, "alley": 80, "outside_research_field": 60, "unassigned_valid_support": 60},
    }
    for frame in ("main", "reserve"):
        for domain in DOMAINS:
            assert sum(quotas[frame][domain].values()) == expected[frame][domain]
        assert all(quotas[frame]["research_crop_area"][cube] == 0 for cube in ZERO_CROP_PRIMARY)
        assert all(quotas[frame]["alley"][cube] == 0 for cube in ALL_OUTSIDE)
    assert all(quotas["main"]["outside_research_field"][cube] == 10 for cube in ALL_OUTSIDE)
    assert all(quotas["reserve"]["outside_research_field"][cube] == 5 for cube in ALL_OUTSIDE)
    assert all(quotas["main"]["unassigned_valid_support"][cube] == 10 for cube in ZERO_CROP_PRIMARY)
    assert all(quotas["reserve"]["unassigned_valid_support"][cube] == 5 for cube in ZERO_CROP_PRIMARY)


def test_capacity_reconciliation_records_zero_shortfall_rows_and_keeps_domains_separate():
    v2, cubes, roles, capacity = synthetic_capacity()
    requested = initial_quotas(cubes, roles, capacity, v2)
    final, audit = reconcile_quotas(cubes, roles, capacity, requested, v2)
    assert len(audit) == 2 * 4 * 40
    assert not audit.shortfall_exported.any()
    assert not audit.redistribution_received.any()
    assert final == requested


def test_v2_configuration_is_prediction_free_and_uses_separate_namespace():
    v2, _ = configs()
    assert v2["outputs"]["root"] != "metadata/local/annotations/field2_blind_sampling"
    assert v2["point_annotation"]["output_root"] != "metadata/local/annotations/field2_blind_points"
    assert v2["sampling"]["expected_main_total"] == 800
    assert v2["sampling"]["expected_reserve_total"] == 400
    assert v2["safety"] == {
        "source_open_mode": "read_only",
        "reserve_api_exposure": False,
        "biological_labels_preselected": False,
        "predictions_prohibited": True,
        "probabilities_from_models_prohibited": True,
        "checkpoints_prohibited": True,
        "field1_outputs_prohibited": True,
    }


def test_local_frozen_v2_contract_and_preserved_v1_hashes_when_available():
    v2, base = configs()
    contract_path = PROJECT / v2["outputs"]["contract"]
    if not contract_path.exists():
        pytest.skip("machine-local frozen v2 products are not present")
    result = load_verified_v2(PROJECT, v2)
    assert len(result["main"]) == 800
    assert len(result["reserve"]) == 400
    assert sha256(PROJECT / base["sampling"]["sampling_contract"]) == "78c0378105e23cb7cc42da0871f144f06532bf7b5ed83ea51c262b4094e229d2"
    assert sha256(PROJECT / base["sampling"]["main_frame"]) == "355980b7dddc302a2b82f0193495bca5ce1b2bc9e31a4c8f72a95a20af8ffe0a"
    assert sha256(PROJECT / base["sampling"]["reserve_frame"]) == "37db9d20e86f40d3fe330cf352c10ded1463ebdbe0226bc7ff09ec1204d287fc"
    old_points = PROJECT / base["point_annotation"]["output_root"] / "field2_blind_main_point_annotations.json"
    assert sha256(old_points) == "22d9f4cf701f3e5576686e7e1328c18c0e46c12917501cb6dba3774e0e14460c"
    assert not set(result["main"].sample_id) & set(result["reserve"].sample_id)
    assert set(result["reserve"].reserve_release_status) == {"locked_pending_separate_authorization"}
