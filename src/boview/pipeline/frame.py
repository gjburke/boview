"""Shared inputs and outputs for the frame processing stages"""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class PipelineFrame:
    """Stores the current images, masks, keypoints, and prediction

    Attributes:
        color_frame: HxWx3 uint8 RGB image, rotated during alignment
        height_frame: aligned HxW height image in millimeters
        camera_height: camera height above the floor in millimeters
        color_mask: current HxW boolean color segmentation mask
        height_mask: current HxW boolean height mask; after cleanup this is the
            refined union of both segmentations used for measurements and keypoints
        keypoints: current Kx2 (x, y) pixel coordinates
        mask_iou: quality measurements taken before cleanup
        percentage_zero: quality measurements taken before cleanup
        features: 20 unscaled model inputs and prediction in training units
        weight: 20 unscaled model inputs and prediction in training units
        stage: processing progress and rejection details
        status: processing progress and rejection details
        rejection_reason: processing progress and rejection details

    Stages replace current values in a new object without modifying input arrays.
    Unchanged arrays are shared and should be treated as read-only. Keep an earlier
    frame separately if its pre-cleanup or pre-rotation values are needed.
    Background removal is temporary and depth is not stored.
    """

    color_frame: np.ndarray
    height_frame: np.ndarray
    camera_height: float

    color_mask: np.ndarray | None = None
    height_mask: np.ndarray | None = None
    keypoints: np.ndarray | None = None

    mask_iou: float | None = None
    percentage_zero: float | None = None
    features: dict | None = None
    weight: float | None = None

    stage: str = "preprocess"
    status: Literal["pending", "accepted", "rejected"] = "pending"
    rejection_reason: str | None = None


def require_stage(frame: PipelineFrame, stage: str) -> None:
    """Checks that a pending frame is ready for the next processing stage

    Args:
        frame: input frame, unchanged by this check
        stage: required last completed stage

    Raises:
        ValueError: the frame has already finished or its stages are out of order
    """
    if frame.status != "pending" or frame.stage != stage:
        raise ValueError(
            f"Expected a pending {stage} frame, got {frame.status}/{frame.stage}"
        )


def reject_frame(frame: PipelineFrame, stage: str, reason: str) -> PipelineFrame:
    """Returns a rejected copy while preserving the available intermediate results

    Args:
        frame: input frame, unchanged by rejection
        stage: stage that rejected the frame
        reason: explanation of why the frame could not continue

    Returns:
        rejected: new frame with rejection status and reason
    """
    return replace(frame, stage=stage, status="rejected", rejection_reason=reason)
