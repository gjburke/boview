"""Utilities for transforming depth frames"""

import numpy as np


def depth_to_height(
    depth_frame: np.ndarray[np.uint16], camera_height: int
) -> np.ndarray[np.uint16]:
    """
    Converts depth frame values to height based on given camera height, persisting zero-valued depths

    Parameters:
    - depth_frame: HxW frame of depth values (usually in millimeters)
    - camera_height: height from the ground to the capturing camera (usually in millimeters, same unit as depth camera)

    Returns:
    - height_frame: HxW frame of height values (usually in millimeters)

    """

    zero_inds = np.where(depth_frame == 0)

    height_frame = camera_height - depth_frame.astype(np.float32)  # Prevent overflow

    height_frame = np.clip(height_frame, a_min=0, a_max=height_frame.max() + 1)
    height_frame = height_frame.astype(np.uint16)
    height_frame[zero_inds] = 0

    return height_frame


def normalize_height(height_frame: np.ndarray[np.uint16]) -> np.ndarray[np.uint8]:
    """
    Normalizes a height measurement frame to relative gray pixel values, for model input

    Parameters:
    - height_frame: HxW frame of height values (uaully in millimeters)

    Returns:
    - normalized_height_frame: HxWx3 frame of grayscale color values (0-255)
    """

    shifted_height = height_frame - height_frame.min()
    scaled_height = shifted_height * 255 / shifted_height.max()
    cast_height = scaled_height.astype(np.uint8)

    normalized_height = np.repeat(
        cast_height[..., None], 3, axis=-1
    )  # Expand normalized values to 3 channels
    return normalized_height
