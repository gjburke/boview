"""Functions for filtering out low-quality frames"""

import numpy as np


def get_percentage_zero(frame: np.ndarray, mask: np.ndarray[np.bool]) -> np.float64:
    """
    Finds the percentage of "zero" pixels (with no data) within a masked frame

    Parameters:
    - image: a HxW frame of height values
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

    return 0
