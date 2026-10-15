"""Functions for filtering out low-quality frames"""

import numpy as np


def get_percentage_zero(frame: np.ndarray, mask: np.ndarray[np.bool]) -> np.float64:
    """
    Finds the percentage of "zero" pixels (with no data) within a masked frame

    Parameters:
    - frame: a HxW frame of height values
    - mask: a HxW boolean array, True where the mask is, False where it isn't

    Returns:
    - percentage_zero: percentage of the values that are zero within the mask
    """
    assert frame.shape == mask.shape

    masked = frame[mask]
    assert len(masked) > 0

    num_zero = len(np.where(masked == 0)[0])
    percentage_zero = (100 * num_zero) / len(masked)

    return percentage_zero


def get_mask_iou(mask1: np.ndarray[np.bool], mask2: np.ndarray[np.bool]) -> np.float64:
    """
    Find the IOU between two masks

    Parameters:
    - mask1: a HxW boolean array
    - mask2: a HxW boolean array

    Returns:
    - iou: the intersection over union of the True values
    """
    assert mask1.shape == mask2.shape

    overlap_count = np.sum(np.bitwise_and(mask1, mask2))
    total_count = np.sum(np.bitwise_or(mask1, mask2))
    assert total_count > 0

    iou = overlap_count / total_count
    return iou


def is_filtered(
    height_frame: np.ndarray,
    color_frame: np.ndarray,
    height_mask: np.ndarray[np.bool],
    color_mask: np.ndarray[np.bool],
    max_percentage_zero: np.float64,
    min_iou: np.float64,
) -> bool:
    """
    Takes relevant frames, masks, and cutoffs to determine if the frames should be filtered out

    Parameters:
    - height_frame: HxW frame of height values
    - color_frame: HxWx3 frame of RGB color values
    - height_mask: HxW boolean mask from height segmentation
    - color_mask: HxW boolean mask from color segmentation
    - max_percentage_zero: the maximum allowed percentage zero
    - min_iou: the minimum allowed Intersection Over Union for a frame

    Returns:
    - filtered: true if it should be filtered, false if it sohuld pass the filter
    """
    percentage_zero = get_percentage_zero(frame=height_frame, mask=height_mask)
    if percentage_zero > max_percentage_zero:
        return True

    iou = get_mask_iou(mask1=height_mask, mask2=color_mask)
    if iou < min_iou:
        return True

    return False
