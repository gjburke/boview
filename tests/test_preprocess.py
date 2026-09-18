"""Tests the preprocessing stage of the pipeline"""

import numpy as np

from boview.pipeline.preprocess import depth_to_height


def test_depth_to_height():
    camera_height = 2

    depth_frames = [
        np.array([[1, 2], [2, 1]], dtype=np.uint16),  # Normal Conditions
        np.array([[1, 3], [3, 1]], dtype=np.uint16),  # Clipping negative values
    ]
    expecteds = [
        np.array([[1, 0], [0, 1]], dtype=np.uint16),
        np.array([[1, 0], [0, 1]], dtype=np.uint16),
    ]
    assert len(depth_frames) == len(expecteds)

    for depth_frame, expected in zip(depth_frames, expecteds):
        result = depth_to_height(depth_frame=depth_frame, camera_height=camera_height)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
