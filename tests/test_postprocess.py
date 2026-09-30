"""Tests the postprocessing stage of the pipeline"""

import numpy as np

from boview.pipeline.postprocess.filter import get_percentage_zero

# Filtering


def test_filtering():
    frame_mask_pairs = [
        (np.array([1, 0, 1]), np.array([False, True, False])),
        (np.array([1, 0, 1]), np.array([True, True, True])),
        (np.array([1, 0, 1]), np.array([True, False, True])),
    ]
    expecteds = [0, 2 / 3, 1]

    for (frame, mask), expected in zip(frame_mask_pairs, expecteds):
        assert get_percentage_zero(depth_frame=frame, mask=mask) == expected


def test_filtering():
    assert len(np.array([0]) == 0)
