import numpy as np
import pandas as pd
import pytest

from chickpea_ssl.confident_labels import (
    UNRESOLVED, apply_confirmed_points, compose_confident_labels,
    deterministic_balanced_selection,
)


def inputs():
    shape = (4, 6)
    return dict(
        valid=np.ones(shape, bool), polygon_core=np.ones(shape, bool),
        polygon_edge=np.zeros(shape, bool), alley_full=np.zeros(shape, bool),
        alley_core=np.zeros(shape, bool), soil=np.zeros(shape, bool),
        soil_evidence_valid=np.ones(shape, bool),
        probability=np.full(shape, 0.5, np.float32),
        authoritative_soil=True,
    )


def test_precedence_alley_prohibits_chickpea_and_core_enriches_weed():
    values = inputs()
    values["alley_full"][:, :3] = True
    values["alley_core"][:, 1:2] = True
    values["probability"][:, :3] = 0.99
    labels, source, conflicts = compose_confident_labels(**values)
    assert not np.any(labels[:, :3] == 1)
    assert np.all(labels[:, 1] == 2)
    assert np.all(source[:, 1] == 5)
    assert np.all(conflicts[:, :3] & 1)


def test_authoritative_soil_resolves_edge_and_alley_ring():
    values = inputs()
    values["polygon_edge"][:, 3] = True
    values["alley_full"][:, 4] = True
    values["soil"][:, 3:5] = True
    labels, source, _ = compose_confident_labels(**values)
    assert np.all(labels[:, 3:5] == 0)
    assert np.all(source[:, 3:5] == 1)


def test_middle_probability_and_invalid_are_unresolved():
    values = inputs()
    values["valid"][0, 0] = False
    labels, source, _ = compose_confident_labels(**values)
    assert labels[0, 0] == UNRESOLVED
    assert np.all(labels[values["valid"]] == UNRESOLVED)
    assert np.all(source == 0)


def test_cube32_dense_policy_is_enforced_by_switch():
    values = inputs()
    values["probability"][:] = 0.99
    labels, _, _ = compose_confident_labels(**values, allow_dense_probability=False)
    assert not np.any(labels == 1)


def test_confirmed_chickpea_point_cannot_override_alley_or_soil():
    values = inputs()
    values["alley_full"][1, 1] = True
    values["soil"][2, 2] = True
    labels, source, _ = compose_confident_labels(**values)
    conflicts = apply_confirmed_points(
        labels, source, [(1, 1, 1), (2, 2, 1), (3, 3, 1)],
        valid=values["valid"], alley_full=values["alley_full"], soil=values["soil"],
    )
    assert labels[1, 1] == UNRESOLVED and labels[2, 2] == 0
    assert labels[3, 3] == 1 and source[3, 3] == 8
    assert len(conflicts) == 2


def test_deterministic_sampling_balances_and_does_not_duplicate():
    frame = pd.DataFrame([
        {"class_id": cls, "cube_id": f"cube{cube}", "spatial_group_id": f"g{group}",
         "row": row, "column": row}
        for cls in range(3) for cube in range(2) for group in range(2) for row in range(10)
    ])
    first = deterministic_balanced_selection(frame, 12, 42)
    second = deterministic_balanced_selection(frame, 12, 42)
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("class_id").size().to_dict() == {0: 12, 1: 12, 2: 12}
    assert not first.duplicated(["class_id", "cube_id", "row", "column"]).any()


def test_bad_thresholds_fail():
    with pytest.raises(ValueError):
        compose_confident_labels(**inputs(), weed_maximum_probability=.9,
                                 chickpea_minimum_probability=.8)


def test_gpu_worker_fails_fast_without_cuda(monkeypatch):
    import torch
    from scripts.run_confident_supervised_training import require_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        require_device(False)
