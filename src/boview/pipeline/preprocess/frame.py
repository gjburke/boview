"""Frame-level validation and depth preprocessing"""

import numpy as np

from ..frame import PipelineFrame, reject_frame
from .height import depth_to_height


def preprocess_frame(
    color_frame: np.ndarray, depth_frame: np.ndarray, camera_height: float
) -> PipelineFrame:
    """Validates the original images and calculates height without changing the input

    Args:
        color_frame: HxWx3 uint8 RGB image
        depth_frame: aligned HxW depth image in millimeters, zero where missing
        camera_height: positive camera height above the floor in millimeters

    Returns:
        processed: new preprocessed frame, or rejected frame for constant height

    Raises:
        ValueError: invalid image shapes, values, camera height
    """
    color = color_frame
    depth = depth_frame

    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color_frame must be a HxWx3 RGB image")
    if color.dtype != np.uint8:
        raise ValueError("color_frame must contain uint8 RGB values")

    if depth.ndim != 2 or depth.shape != color.shape[:2]:
        raise ValueError("depth_frame must be HxW and aligned with color_frame")
    if depth.size == 0 or not np.isfinite(depth).all():
        raise ValueError("depth_frame must be nonempty and finite")
    if np.any(depth < 0):
        raise ValueError("depth_frame must contain nonnegative depths")

    if not np.isfinite(camera_height) or not 0 < camera_height <= 65535:
        raise ValueError("camera_height must be positive and fit uint16 millimeters")

    height = depth_to_height(depth, camera_height)
    processed = PipelineFrame(color_frame, height, camera_height)

    if np.ptp(height) == 0:
        return reject_frame(processed, "preprocess", "Height image has no variation")

    return processed
