import numpy as np

from scripts.materialize_polygon_guided_candidates import (
    classify_candidate_support,
    predict_soil,
)


def test_lower_scalar_values_predict_soil() -> None:
    values = np.array([[0.70, 0.73, np.nan]], dtype=np.float32)

    soil = predict_soil(
        values,
        direction="lower_values_indicate_soil",
        threshold=0.720602,
    )

    assert soil[0, :2].tolist() == [True, False]
    assert not bool(soil[0, 2])


def test_candidate_classes_and_alley_precedence() -> None:
    core = np.array([[True, True, False, False]])
    edge = np.array([[False, False, True, True]])
    observed = np.ones((1, 4), dtype=bool)
    alley = np.array([[False, True, False, False]])
    soil = np.array([[True, False, True, False]])
    source_valid = np.array([[True, False, True, False]])

    result = classify_candidate_support(
        core=core,
        edge=edge,
        observed=observed,
        alley=alley,
        soil=soil,
        source_valid=source_valid,
    )

    # Alley evidence overrides the missing scalar source in column 1.
    assert result.tolist() == [[1, 5, 3, 255]]
