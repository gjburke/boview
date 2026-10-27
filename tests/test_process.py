"""Testing frame orchestration, quality rejection, and aligned outputs"""

from dataclasses import FrozenInstanceError, fields
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from boview.pipeline import PipelineFrame, process_frame
from boview.pipeline.model import get_features, model_frame
from boview.pipeline.postprocess import align_frame, filter_frame, postprocess_frame
from boview.pipeline.preprocess import preprocess_frame
from boview.pipeline.vision import keypoint_frame, segment_frame

# Setup


@pytest.fixture
def frame_inputs():
    mask = np.zeros((160, 200), dtype=bool)
    mask[50:110, 40:160] = True

    color = np.zeros((160, 200, 3), dtype=np.uint8)
    color[mask] = [200, 100, 50]

    depth = np.full(mask.shape, 3000, dtype=np.uint16)
    depth[mask] = 2000

    prediction = SimpleNamespace(
        masks=SimpleNamespace(data=torch.tensor(mask[None].astype(np.float32))),
        orig_shape=mask.shape,
    )

    points = torch.tensor([[[100.0, 80.0, 0.9]] * 15])
    pose = SimpleNamespace(keypoints=SimpleNamespace(data=points))

    return {
        "color_frame": color,
        "depth_frame": depth,
        "camera_height": 3000.0,
        "segmentation_model": Mock(predict=Mock(return_value=[prediction])),
        "keypoint_model": Mock(predict=Mock(return_value=[pose])),
        "weight_model": Mock(predict=Mock(return_value=np.array([450.0]))),
    }


def prepare_predictor(inputs):
    # The predictor must receive the same named columns as the trained pipeline
    mask = inputs["depth_frame"] == 2000
    points = np.full((15, 2), (100, 80))
    features = get_features(mask.astype(float) * 1000, mask, points, 3000)
    inputs["weight_model"].feature_names_in_ = np.array(list(features))


# Frame processing


def test_process_frame(frame_inputs):
    prepare_predictor(frame_inputs)
    frame_inputs["color_frame"][20, 20] = [10, 20, 30]
    color_before = frame_inputs["color_frame"].copy()
    depth_before = frame_inputs["depth_frame"].copy()

    result = process_frame(**frame_inputs)

    assert isinstance(result, PipelineFrame)
    assert result.status == "accepted"
    assert result.rejection_reason is None
    assert result.weight == 450
    assert len(result.features) == 20
    assert result.mask_iou == 1
    assert result.percentage_zero == 0
    assert result.features["AvgHeight"] == 100

    assert result.height_mask.dtype == bool
    assert result.keypoints.shape == (15, 2)
    assert np.all(result.height_frame[~result.height_mask] == 0)

    np.testing.assert_array_equal(frame_inputs["color_frame"], color_before)
    np.testing.assert_array_equal(frame_inputs["depth_frame"], depth_before)

    # RGB input becomes BGR for inference, while the result stays RGB
    inference_color = (
        frame_inputs["segmentation_model"].predict.call_args_list[0].args[0]
    )
    np.testing.assert_array_equal(inference_color[80, 100], [50, 100, 200])
    np.testing.assert_array_equal(result.color_frame[80, 100], [200, 100, 50])

    # Background removal affects only the temporary pose input
    pose_color = frame_inputs["keypoint_model"].predict.call_args.args[0]
    np.testing.assert_array_equal(pose_color[20, 20], [0, 0, 0])
    np.testing.assert_array_equal(result.color_frame[20, 20], [10, 20, 30])


# Detection rejection


@pytest.mark.parametrize("count", [0, 2])
def test_reject_segmentation_count(frame_inputs, count):
    prediction = frame_inputs["segmentation_model"].predict.return_value[0]
    prediction.masks.data = prediction.masks.data.repeat(count, 1, 1)

    result = process_frame(**frame_inputs)

    assert result.status == "rejected"
    assert result.stage == "segment"
    assert result.rejection_reason
    frame_inputs["keypoint_model"].predict.assert_not_called()
    frame_inputs["weight_model"].predict.assert_not_called()


@pytest.mark.parametrize("count", [0, 2])
def test_reject_pose_count(frame_inputs, count):
    pose = frame_inputs["keypoint_model"].predict.return_value[0]
    pose.keypoints.data = pose.keypoints.data.repeat(count, 1, 1)

    result = process_frame(**frame_inputs)

    assert result.status == "rejected"
    assert result.stage == "keypoint"
    assert result.color_frame is not None
    frame_inputs["weight_model"].predict.assert_not_called()


# Quality filtering


def test_missing_height_threshold(frame_inputs):
    prepare_predictor(frame_inputs)

    # 180 missing pixels out of 7200 = 2.5%, above the default 2%
    frame_inputs["depth_frame"][60:63, 60:120] = 0

    result = process_frame(**frame_inputs)

    assert result.status == "rejected"
    assert result.stage == "filter"
    assert result.percentage_zero == pytest.approx(2.5)
    assert result.color_mask is not None
    assert result.weight is None
    frame_inputs["weight_model"].predict.assert_not_called()

    result = process_frame(**frame_inputs, max_percentage_zero=3)

    assert result.status == "accepted"


def test_iou_threshold(frame_inputs):
    prepare_predictor(frame_inputs)

    prediction = frame_inputs["segmentation_model"].predict.return_value[0]
    smaller_mask = prediction.masks.data.clone()
    smaller_mask[:, :, 140:] = 0
    smaller = SimpleNamespace(
        masks=SimpleNamespace(data=smaller_mask), orig_shape=prediction.orig_shape
    )

    frame_inputs["segmentation_model"].predict.side_effect = [[prediction], [smaller]]

    result = process_frame(**frame_inputs)

    assert result.status == "rejected"
    assert result.stage == "filter"
    assert result.mask_iou == pytest.approx(5 / 6)
    frame_inputs["weight_model"].predict.assert_not_called()

    frame_inputs["segmentation_model"].predict.side_effect = [[prediction], [smaller]]
    result = process_frame(**frame_inputs, min_iou=0.8)

    assert result.status == "accepted"
    assert result.mask_iou == pytest.approx(5 / 6)


def test_empty_height(frame_inputs):
    frame_inputs["depth_frame"][:] = 0

    result = process_frame(**frame_inputs)

    assert result.status == "rejected"
    assert result.stage == "preprocess"
    assert result.height_frame is not None
    frame_inputs["segmentation_model"].predict.assert_not_called()


# Invalid inputs and errors


def test_invalid_inputs_and_model_errors(frame_inputs):
    with pytest.raises(ValueError, match="min_iou"):
        process_frame(**frame_inputs, min_iou=1.1)

    frame_inputs["segmentation_model"].predict.side_effect = RuntimeError("model error")

    with pytest.raises(RuntimeError, match="model error"):
        process_frame(**frame_inputs)

    frame_inputs["depth_frame"] = np.zeros((10, 10))

    with pytest.raises(ValueError, match="aligned"):
        process_frame(**frame_inputs)


def test_reject_invalid_hips(frame_inputs):
    pose = frame_inputs["keypoint_model"].predict.return_value[0]
    pose.keypoints.data[0, 10, :2] = -1000

    result = process_frame(**frame_inputs)

    assert result.status == "rejected"
    assert result.stage == "align"
    assert result.keypoints is not None
    frame_inputs["weight_model"].predict.assert_not_called()


# Stage contracts


def test_stage_snapshots(frame_inputs):
    prepare_predictor(frame_inputs)
    frame = preprocess_frame(
        frame_inputs["color_frame"], frame_inputs["depth_frame"], 3000.0
    )
    stages = [
        partial(segment_frame, model=frame_inputs["segmentation_model"]),
        filter_frame,
        postprocess_frame,
        partial(keypoint_frame, model=frame_inputs["keypoint_model"]),
        align_frame,
        partial(model_frame, model=frame_inputs["weight_model"]),
    ]
    snapshots = []

    for stage in stages:
        arrays = {
            field.name: value.copy()
            for field in fields(frame)
            if isinstance(value := getattr(frame, field.name), np.ndarray)
        }
        snapshots.append((frame, arrays, frame.stage, frame.status))

        result = stage(frame)

        assert result is not frame
        frame = result

    assert frame.status == "accepted"

    # Later stages must not modify any earlier container or array
    for previous, arrays, stage, status in snapshots:
        assert previous.stage == stage
        assert previous.status == status

        for name, expected in arrays.items():
            np.testing.assert_array_equal(getattr(previous, name), expected)

    with pytest.raises(FrozenInstanceError):
        frame.weight = 0


def test_stage_order(frame_inputs):
    frame = preprocess_frame(
        frame_inputs["color_frame"], frame_inputs["depth_frame"], 3000.0
    )

    with pytest.raises(ValueError, match="pending align"):
        model_frame(frame, frame_inputs["weight_model"])
