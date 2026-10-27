"""Functions for processing aligned color and depth frames into bodyweight"""

import numpy as np
from sklearn.pipeline import Pipeline
from ultralytics import YOLO

from .frame import PipelineFrame
from .model.frame import model_frame
from .postprocess.filter import filter_frame
from .postprocess.frame import align_frame, postprocess_frame
from .preprocess.frame import preprocess_frame
from .vision.frame import keypoint_frame, segment_frame


def process_frame(
    color_frame: np.ndarray,
    depth_frame: np.ndarray,
    camera_height: float,
    segmentation_model: YOLO,
    keypoint_model: YOLO,
    weight_model: Pipeline,
    min_iou: float = 0.90,
    max_percentage_zero: float = 2.0,
) -> PipelineFrame:
    """
    Processes one aligned color/depth pair using already-loaded models

    Parameters:
    - color_frame: HxWx3 uint8 RGB image, aligned with the depth frame
    - depth_frame: HxW nonnegative depth values in millimeters, zero where missing
    - camera_height: positive camera height above the floor in millimeters
    - segmentation_model: model used for both color and height segmentation
    - keypoint_model: pose model with hip keypoints at indices 10 and 14
    - weight_model: fitted pipeline using the same conversions as get_features
    - min_iou: minimum accepted mask overlap, between 0 and 1
    - max_percentage_zero: maximum accepted missing-height percentage, from 0 to 100

    Returns:
    - result: new frame with prediction and visualization arrays on success, or
      rejection status, reason, and intermediate outputs when a stage rejects it

    Raises:
    - ValueError: invalid frame shapes, values, or thresholds; unexpected model
      and processing errors propagate to the caller
    """
    frame = preprocess_frame(color_frame, depth_frame, camera_height)
    if frame.status == "rejected":
        return frame

    frame = segment_frame(frame, segmentation_model)
    if frame.status == "rejected":
        return frame

    frame = filter_frame(frame, min_iou, max_percentage_zero)
    if frame.status == "rejected":
        return frame

    frame = postprocess_frame(frame)
    if frame.status == "rejected":
        return frame

    frame = keypoint_frame(frame, keypoint_model)
    if frame.status == "rejected":
        return frame

    frame = align_frame(frame)
    if frame.status == "rejected":
        return frame

    return model_frame(frame, weight_model)
