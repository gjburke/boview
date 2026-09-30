"""Tests the postprocessing stage of the pipeline"""

import numpy as np

from boview.pipeline.postprocess.filter import (
    get_mask_iou,
    get_percentage_zero,
    is_filtered,
)

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


def test_is_filtered():
    frames = [
        np.array([0, 0]),
        np.array([1, 1]),
        np.array([1, 0]),
    ]
    mask_pairs = [
        (np.array([True, True]), np.array([True, False])),
        (np.array([False, True]), np.array([True, True])),
        (np.array([True, False]), np.array([True, False])),
    ]
    max_percentage_zero = 50
    min_iou = 0.75

    expecteds = [True, True, False]

    for frame, (mask1, mask2), expected in zip(frames, mask_pairs, expecteds):
        filtered = is_filtered(
            height_frame=frame,
            color_frame=None,
            height_mask=mask1,
            color_mask=mask2,
            max_percentage_zero=max_percentage_zero,
            min_iou=min_iou,
        )
        assert filtered == expected
