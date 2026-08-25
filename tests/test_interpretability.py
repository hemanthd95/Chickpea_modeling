import numpy as np
import pandas as pd
import pytest
import torch

from chickpea_ssl.interpretability import (
    benjamini_hochberg,
    deterministic_audit_support,
    expected_calibration_error,
    wavelength_groups,
)
from scripts import run_grouped_importance_audit as audit_script
from scripts import train_field1_deployment_ensembles as deployment_script


def test_wavelength_groups_are_contiguous_and_physical():
    wavelengths = np.linspace(401.84, 869.34, 111)
    groups = wavelength_groups(wavelengths, 12.5)
    assert groups.first_input_channel.iloc[0] == 0
    assert groups.last_input_channel.iloc[-1] == 110
    assert (groups.first_input_channel.iloc[1:].to_numpy() == groups.last_input_channel.iloc[:-1].to_numpy() + 1).all()
    assert groups.band_count.sum() == 111


def test_bh_is_monotone_in_rank_and_bounded():
    values = np.array([.01, .04, .03, .9])
    adjusted = benjamini_hochberg(values)
    order = np.argsort(values)
    assert np.all(np.diff(adjusted[order]) >= 0)
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_ece_is_zero_for_correct_certain_predictions():
    truth = np.array([0, 1, 2])
    probabilities = np.eye(3)
    assert expected_calibration_error(truth, probabilities) == 0


def test_audit_support_is_deterministic_and_ground_unique():
    frame = pd.DataFrame([
        {"cube_id": "c", "row": i, "column": i, "ground_x": i, "ground_y": i,
         "spatial_group_id": f"g{i % 2}", "class_id": i % 3}
        for i in range(30)
    ])
    first = deterministic_audit_support(frame, 3, 12, 42)
    second = deterministic_audit_support(frame, 3, 12, 42)
    pd.testing.assert_frame_equal(first, second)
    assert not first.duplicated(["ground_x", "ground_y"]).any()


def test_occlusion_routing_changes_only_requested_branch(monkeypatch):
    captured = []

    def fake_forward(_model, center, context):
        captured.append((center.clone(), context.clone()))
        return torch.zeros((len(center), 3))

    monkeypatch.setattr(audit_script, "routed_forward", fake_forward)
    patches = torch.ones((2, 4, 3, 3))
    condition = {
        "audit_type": "occlusion",
        "occlusion_method": "fold_training_mean",
        "route": "center_branch_only",
        "first_input_channel": 1,
        "last_input_channel": 2,
    }
    audit_script.condition_probabilities(
        [object()], patches, condition, np.arange(4), torch.zeros(4), torch.ones(4), torch.arange(9)
    )
    center, context = captured[0]
    assert torch.count_nonzero(center[:, 1:3]) == 0
    assert torch.equal(context, patches)


def test_branch_neutralization_uses_fold_mean_in_normalized_space(monkeypatch):
    captured = []

    def fake_forward(_model, center, context):
        captured.append((center.clone(), context.clone()))
        return torch.zeros((len(center), 3))

    monkeypatch.setattr(audit_script, "routed_forward", fake_forward)
    patches = torch.ones((1, 3, 3, 3))
    condition = {"audit_type": "branch", "branch_ablation": "both_neutralized"}
    audit_script.condition_probabilities(
        [object()], patches, condition, np.arange(3), torch.zeros(3), torch.ones(3), torch.arange(9)
    )
    center, context = captured[0]
    assert torch.count_nonzero(center[:, :, 1, 1]) == 0
    assert torch.count_nonzero(context) == 0
    assert torch.count_nonzero(center[:, :, 0, 0]) == 3


def test_deployment_worker_refuses_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is prohibited"):
        deployment_script.require_gpu()
