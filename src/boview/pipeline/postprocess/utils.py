"""Utility functions for postprocessing"""

import numpy as np


def mask_frame(frame: np.ndarray, mask: np.ndarray[np.bool]) -> np.ndarray:
    """
    Returns a frame that keeps values within the True values of the mask

    Parameters:
    - frame: a HxWx... array (could be color or depth values)
    - mask: a HxW boolean array that has 'True's on pixels that should be kept

    Returns:
    - masked_frame: a HxWx... frame with zeroed out values outside of the mask
    """
    masked_frame = frame.copy()
    masked_frame[~mask] = 0
    return masked_frame
