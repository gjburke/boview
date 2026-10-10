"""Tests the postprocessing stage of the pipeline"""

import numpy as np

from boview.pipeline.postprocess.filter import (
    get_mask_iou,
    get_percentage_zero,
    is_filtered,
)
from boview.pipeline.postprocess.rotate import (
    get_rotation_matrix,
    rotate_frame,
    rotate_horizontal,
    rotate_points,
)
from boview.pipeline.postprocess.utils import mask_frame

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


# Rotation


def test_get_rotation_matrix():
    masks = [
        np.array([[False, False, True], [False, True, False], [False, False, False]]),
        np.array([[False, False, True], [False, True, False], [True, False, False]]),
        np.array([[False, True, False], [False, True, False], [False, True, False]]),
    ]
    rads = [np.deg2rad(-45), np.deg2rad(-45), np.deg2rad(-90)]
    centers = [(1.5, 0.5), (1, 1), (1, 1)]

    for mask, rad, (cx, cy) in zip(masks, rads, centers):
        expected = np.array(
            [
                [np.cos(rad), np.sin(rad), (1 - np.cos(rad)) * cx - np.sin(rad) * cy],
                [-np.sin(rad), np.cos(rad), np.sin(rad) * cx + (1 - np.cos(rad)) * cy],
            ]
        )

        result = get_rotation_matrix(mask=mask)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


def test_rotate_frame():
    masks = [
        np.array([[False, False, False], [True, True, True], [False, False, False]]),
        np.array([[True, False, False], [False, True, False], [False, False, True]]),
        np.array([[False, False, True], [False, True, False], [True, False, False]]),
        np.array([[False, True, False], [False, True, False], [False, True, False]]),
    ]
    expected = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])

    for mask in masks:
        rotation_matrix = get_rotation_matrix(mask)
        result = rotate_frame(
            frame=mask.astype(np.uint8), rotation_matrix=rotation_matrix
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


def test_rotate_points():
    masks = [
        np.array([[False, False, False], [True, True, True], [False, False, False]]),
        np.array([[True, False, False], [False, True, False], [False, False, True]]),
        np.array([[False, False, True], [False, True, False], [True, False, False]]),
    ]
    pointss = [
        np.array([[0, 1], [1, 1], [2, 1]]),
        np.array([[0, 0], [1, 1], [2, 2]]),
        np.array([[0, 2], [1, 1], [2, 0]]),
    ]
    expected = np.array([[0, 1], [1, 1], [2, 1]])

    for mask, points in zip(masks, pointss):
        rotation_matrix = get_rotation_matrix(mask)
        result = rotate_points(points=points, rotation_matrix=rotation_matrix)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


def test_rotate_horizontal():
    masks = [
        np.array([[False, False, False], [True, True, True], [False, False, False]]),
        np.array([[True, False, False], [False, True, False], [False, False, True]]),
        np.array([[False, False, True], [False, True, False], [True, False, False]]),
    ]
    pointss = [
        np.array([[0, 1], [1, 1], [2, 1]]),
        np.array([[0, 0], [1, 1], [2, 2]]),
        np.array([[0, 2], [1, 1], [2, 0]]),
    ]
    expected_frame = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    expected_mask = np.array(
        [[False, False, False], [True, True, True], [False, False, False]]
    )
    expected_points = np.array([[0, 1], [1, 1], [2, 1]])

    for mask, points in zip(masks, pointss):
        result_frame, result_mask, result_points = rotate_horizontal(
            frame=mask.astype(np.uint8), mask=mask, points=points
        )

        np.testing.assert_allclose(result_frame, expected_frame, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(result_mask, expected_mask, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(result_points, expected_points, rtol=1e-6, atol=1e-7)


# Utils


def test_mask_frame():
    masks = [
        np.array([[False, False], [False, False]]),
        np.array([[True, False], [False, True]]),
        np.array([[True, True], [True, True]]),
        np.array([[True, False], [False, True]]),
    ]
    frames = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[[1, 1], [0, 0]], [[1, 1], [0, 0]]]),
    ]
    expecteds = [
        np.array([[0, 0], [0, 0]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[[1, 1], [0, 0]], [[0, 0], [0, 0]]]),
    ]

    for mask, frame, expected in zip(masks, frames, expecteds):
        result = mask_frame(frame=frame, mask=mask)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
