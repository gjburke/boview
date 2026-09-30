"""Tests the postprocessing stage of the pipeline"""

import numpy as np

from boview.pipeline.postprocess.filter import get_mask_iou, get_percentage_zero

# Filtering


def test_get_percentage_zero():
    frame_mask_pairs = [
        (np.array([1, 0, 1]), np.array([False, True, False])),
        (np.array([1, 0, 1]), np.array([True, True, True])),
        (np.array([1, 0, 1]), np.array([True, False, True])),
    ]
    expecteds = [100, 100 * (1 / 3), 0.0]

    for (frame, mask), expected in zip(frame_mask_pairs, expecteds):
        assert np.isclose(
            get_percentage_zero(frame=frame, mask=mask), expected, rtol=1e-6, atol=1e-7
        )


def test_get_mask_iou():
    mask_pairs = [
        (np.array([False, True]), np.array([True, False])),
        (np.array([False, True]), np.array([True, True])),
        (np.array([False, True]), np.array([False, True])),
    ]
    expecteds = [0, 1 / 2, 1]

    for (mask1, mask2), expected in zip(mask_pairs, expecteds):
        assert np.isclose(
            get_mask_iou(mask1=mask1, mask2=mask2), expected, rtol=1e-6, atol=1e-7
        )


def test_filtering():
    assert len(np.array([0]) == 0)
