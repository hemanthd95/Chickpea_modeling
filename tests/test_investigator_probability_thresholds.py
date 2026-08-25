import numpy as np

from scripts.audit_investigator_probability_thresholds import (
    threshold_metrics,
    triage,
    wilson_lower,
)


def test_wilson_lower_is_below_observed_precision():
    assert 0.85 < wilson_lower(95, 100) < 0.95


def test_threshold_metrics_respect_class_direction():
    truth = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.8, 0.9, 0.2])
    chickpea = threshold_metrics(truth, probability, "chickpea", 0.8)
    weed = threshold_metrics(truth, probability, "weed", 0.2)
    assert chickpea["true_positive"] == 1
    assert chickpea["false_positive"] == 1
    assert weed["true_positive"] == 1
    assert weed["false_positive"] == 1


def test_triage_preserves_unscored_and_uncertain():
    probability = np.asarray([[np.nan, 0.10, 0.50, 0.90]], dtype=float)
    assert triage(probability, 0.15, 0.80).tolist() == [[0, 1, 2, 3]]
