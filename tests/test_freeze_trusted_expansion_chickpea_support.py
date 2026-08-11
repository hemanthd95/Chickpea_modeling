import numpy as np

from scripts.freeze_trusted_expansion_chickpea_support import (
    positive_support,
    true_value,
)


def test_positive_support_accepts_only_chickpea_review_code():
    candidate = np.asarray([[0, 1, 2, 3], [4, 5, 6, 255]], dtype=np.uint8)
    expected = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(positive_support(candidate), expected)


def test_true_value_does_not_treat_false_string_as_true():
    assert true_value("True")
    assert not true_value("False")
