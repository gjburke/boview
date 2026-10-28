"""Frame-level mask cleanup and alignment"""

from dataclasses import replace

import numpy as np

from ..frame import PipelineFrame, reject_frame, require_stage
from .refine import clean_mask, interpolate_height
from .rotate import get_rotation_matrix, rotate_frame, rotate_points
from .utils import mask_frame


def postprocess_frame(frame: PipelineFrame) -> PipelineFrame:
    """Combines masks and cleans height without changing the input

    Args:
        frame: pending filtered frame with original masks and raw height

    Returns:
        processed: new frame with cleaned millimeter height and refined height mask
            in original coordinates, or rejected if no usable silhouette remains
    """
    require_stage(frame, "filter")

    mask = clean_mask(frame.color_mask | frame.height_mask)
    processed = replace(frame, height_mask=mask, stage="postprocess")

    if not mask.any() or not np.any(frame.height_frame[mask] > 0):
        return reject_frame(
            processed, "postprocess", "Cleaned mask has no valid heights"
        )

    height, mask = interpolate_height(frame.height_frame, mask)
    mask = mask & (height > 0)
    processed = replace(
        processed,
        height_frame=mask_frame(height, mask),
        height_mask=mask,
    )

    if not mask.any():
        return reject_frame(
            processed, "postprocess", "Height cleanup removed the silhouette"
        )

    return processed


def align_frame(frame: PipelineFrame) -> PipelineFrame:
    """Rotates cleaned images and keypoints together without changing the input

    Args:
        frame: pending keypoint frame with cleaned images and original keypoints

    Returns:
        aligned: new frame with rotated RGB, millimeter height, boolean mask,
            and Kx2 (x, y) keypoints; rejects an empty mask or out-of-bounds hips
    """
    require_stage(frame, "keypoint")

    rotation = get_rotation_matrix(frame.height_mask)
    color = rotate_frame(frame.color_frame, rotation)
    height = rotate_frame(frame.height_frame, rotation)
    mask = rotate_frame(frame.height_mask.astype(np.uint8), rotation).astype(bool)
    color_mask = rotate_frame(frame.color_mask.astype(np.uint8), rotation).astype(bool)
    keypoints = rotate_points(frame.keypoints, rotation)
    aligned = replace(
        frame,
        color_frame=color,
        height_frame=height,
        height_mask=mask,
        color_mask=color_mask,
        keypoints=keypoints,
        stage="align",
    )

    if not mask.any():
        return reject_frame(aligned, "align", "Rotated mask is empty")

    hips = keypoints[[10, 14]]

    if (hips < 0).any() or (hips[:, 0] >= mask.shape[1]).any():
        return reject_frame(aligned, "align", "Rotated hips are outside the image")
    if (hips[:, 1] >= mask.shape[0]).any():
        return reject_frame(aligned, "align", "Rotated hips are outside the image")

    return aligned
