"""Frame-level segmentation and keypoint detection"""

from dataclasses import replace

import cv2
import numpy as np
from ultralytics import YOLO

from ..frame import PipelineFrame, reject_frame, require_stage
from ..postprocess.utils import mask_frame
from .keypoint import keypoint_color_image, keypoints_from_result
from .segment import mask_from_result, segment_color_image, segment_height_image


def segment_frame(frame: PipelineFrame, model: YOLO) -> PipelineFrame:
    """
    Segments the original color and height images without changing the input

    Parameters:
    - frame: pending preprocessed frame with RGB color and raw height in millimeters
    - model: loaded segmentation model used for both images

    Returns:
    - segmented: new frame with original-coordinate boolean masks, or a rejected
      frame if either image does not contain exactly one nonempty segmentation
    """
    require_stage(frame, "preprocess")

    # Ultralytics numpy images use BGR channel order
    color_bgr = cv2.cvtColor(frame.color_frame, cv2.COLOR_RGB2BGR)
    color_result = segment_color_image(model, color_bgr)
    height_result = segment_height_image(model, frame.height_frame)

    color_mask = mask_from_result(color_result)
    height_mask = mask_from_result(height_result)
    segmented = replace(
        frame, color_mask=color_mask, height_mask=height_mask, stage="segment"
    )

    for name, prediction in (("Color", color_result), ("Height", height_result)):
        if prediction is None or prediction.masks is None:
            return reject_frame(segmented, "segment", f"{name} segmentation is missing")
        if len(prediction.masks.data) != 1:
            return reject_frame(
                segmented, "segment", f"{name} segmentation must contain one animal"
            )

    if not color_mask.any() or not height_mask.any():
        return reject_frame(segmented, "segment", "Segmentation mask is empty")

    return segmented


def keypoint_frame(frame: PipelineFrame, model: YOLO) -> PipelineFrame:
    """
    Detects keypoints after mask cleanup without changing the input

    Parameters:
    - frame: pending postprocessed frame with RGB color and a refined height mask
    - model: loaded pose model with hip keypoints at indices 10 and 14

    Returns:
    - detected: new frame with original-coordinate Kx2 (x, y) keypoints,
      or a rejected frame for missing, ambiguous, or unusable keypoints
    """
    require_stage(frame, "postprocess")

    color_bgr = cv2.cvtColor(
        mask_frame(frame.color_frame, frame.height_mask), cv2.COLOR_RGB2BGR
    )
    result = keypoint_color_image(model, color_bgr)

    if result is None or result.keypoints is None:
        return reject_frame(frame, "keypoint", "Keypoints are missing")
    if len(result.keypoints.data) != 1:
        return reject_frame(frame, "keypoint", "Pose detection must contain one animal")

    keypoints = keypoints_from_result(result)
    detected = replace(frame, keypoints=keypoints, stage="keypoint")

    if keypoints is None or len(keypoints) < 15:
        return reject_frame(detected, "keypoint", "At least 15 keypoints are required")

    keypoints = keypoints[:, :2].copy()
    detected = replace(detected, keypoints=keypoints)

    if not np.isfinite(keypoints).all():
        return reject_frame(detected, "keypoint", "Keypoints contain nonfinite values")
    if np.any(np.all(keypoints[[10, 14]] == 0, axis=1)):
        return reject_frame(detected, "keypoint", "Hip keypoints are missing")

    return detected
