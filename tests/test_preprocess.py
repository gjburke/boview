"""Tests the preprocessing stage of the pipeline"""

import numpy as np

from boview.pipeline.preprocess import depth_to_height, normalize_height


def test_depth_to_height():
    camera_height = 2

    depth_frames = [
        np.array([[1, 2], [2, 1]], dtype=np.uint16),  # Normal conditions
        np.array([[1, 3], [3, 1]], dtype=np.uint16),  # Clipping negative values
    ]
    expecteds = [
        np.array([[1, 0], [0, 1]], dtype=np.uint16),
        np.array([[1, 0], [0, 1]], dtype=np.uint16),
    ]

    for depth_frame, expected in zip(depth_frames, expecteds):
        result = depth_to_height(depth_frame=depth_frame, camera_height=camera_height)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


def test_normalize_height():
    height_frames = [
        np.array([[1, 0], [0, 1]], dtype=np.uint16),  # Extremes
        np.array([[2, 1], [1, 2]], dtype=np.uint16),  # Extremes shifted
        np.array([[0, 1], [2, 3]], dtype=np.uint16),  # Intermediates
    ]
    expecteds = [
        np.array(
            [[[255, 255, 255], [0, 0, 0]], [[0, 0, 0], [255, 255, 255]]], dtype=np.uint8
        ),
        np.array(
            [[[255, 255, 255], [0, 0, 0]], [[0, 0, 0], [255, 255, 255]]], dtype=np.uint8
        ),
        np.array(
            [[[0, 0, 0], [85, 85, 85]], [[170, 170, 170], [255, 255, 255]]],
            dtype=np.uint8,
        ),
    ]

    for height_frame, expected in zip(height_frames, expecteds):
        result = normalize_height(height_frame=height_frame)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
