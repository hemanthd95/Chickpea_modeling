import numpy as np

from scripts.audit_polygon_guided_vegetation_separability import (
    classification_metrics,
    predict_probability,
    sample_coordinates,
)


def test_sample_coordinates_is_deterministic_and_bounded():
    mask = np.ones((10, 12), dtype=bool)
    first = sample_coordinates(mask, 15, np.random.default_rng(42))
    second = sample_coordinates(mask, 15, np.random.default_rng(42))
    assert first.shape == (15, 2)
    assert np.array_equal(first, second)
    assert np.all(mask[first[:, 0], first[:, 1]])


def test_probability_chunking_and_metrics():
    class IdentityScaler:
        def transform(self, values):
            return values

    class FirstColumnModel:
        def predict_proba(self, values):
            positive = values[:, 0]
            return np.column_stack([1 - positive, positive])

    features = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float32)
    probability = predict_probability(
        features, IdentityScaler(), FirstColumnModel(), 2
    )
    assert np.allclose(probability, features[:, 0])
    result = classification_metrics(np.array([0, 0, 1, 1]), probability)
    assert result["balanced_accuracy"] == 1.0
    assert result["roc_auc"] == 1.0
