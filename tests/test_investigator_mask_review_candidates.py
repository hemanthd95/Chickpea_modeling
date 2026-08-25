import numpy as np

from scripts.materialize_investigator_mask_review_candidates import (
    build_review_candidate,
)


def test_review_candidate_preserves_precedence_and_thresholds():
    support = np.asarray([[0, 1, 2, 2, 2, 3, 4, 5, 255]], dtype=np.uint8)
    probability = np.asarray([[np.nan, np.nan, 0.10, 0.50, 0.90, 0.1, 0.9, np.nan, np.nan]])
    result = build_review_candidate(support, probability, 0.15, 0.80, True)
    assert result.tolist() == [[0, 1, 2, 3, 4, 5, 5, 6, 255]]


def test_warning_cube_forces_core_vegetation_unresolved():
    support = np.asarray([[2, 2, 2]], dtype=np.uint8)
    probability = np.asarray([[0.01, 0.50, 0.99]], dtype=float)
    result = build_review_candidate(support, probability, 0.15, 0.80, False)
    assert result.tolist() == [[3, 3, 3]]


def test_trusted_expansion_cube_materializes_chickpea_only():
    support = np.asarray([[2, 2, 2]], dtype=np.uint8)
    probability = np.asarray([[0.01, 0.50, 0.99]], dtype=float)
    result = build_review_candidate(
        support, probability, 0.15, 0.80,
        weed_decision_eligible=False,
        chickpea_decision_eligible=True,
    )
    assert result.tolist() == [[3, 3, 4]]


def test_invalid_threshold_order_is_rejected():
    support = np.asarray([[2]], dtype=np.uint8)
    probability = np.asarray([[0.5]], dtype=float)
    try:
        build_review_candidate(support, probability, 0.8, 0.2, True)
    except ValueError as error:
        assert "Thresholds" in str(error)
    else:
        raise AssertionError("Expected invalid threshold ordering to fail")
