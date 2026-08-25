import numpy as np

from scripts.build_investigator_mask_acceptance_previews import (
    overlay_classes,
    preview_classes,
)


def test_overlay_changes_only_requested_codes():
    image = np.full((1, 3, 3), 0.5, dtype=np.float32)
    classes = np.asarray([[0, 4, 3]], dtype=np.uint8)
    result = overlay_classes(
        image, classes, {4: np.asarray([0.0, 1.0, 0.0], dtype=np.float32)}, 0.8
    )
    np.testing.assert_allclose(result[0, 0], image[0, 0])
    np.testing.assert_allclose(result[0, 2], image[0, 2])
    np.testing.assert_allclose(result[0, 1], [0.1, 0.9, 0.1])


def test_preview_classes_uses_frozen_sampling_step():
    candidate = np.arange(36, dtype=np.uint8).reshape(6, 6)
    result = preview_classes(candidate, 2, (3, 3))
    np.testing.assert_array_equal(result, candidate[::2, ::2])


def test_preview_classes_rejects_misalignment():
    try:
        preview_classes(np.zeros((6, 6), dtype=np.uint8), 2, (4, 4))
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("Expected preview misalignment to fail")
