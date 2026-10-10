"""Functions for smoothing noise in height values and masks"""

import cv2
import numpy as np


def clean_mask(mask: np.ndarray[np.bool]) -> np.ndarray[np.bool]:
    """
    Cleans model output's mask, filling holes, isolating the main mask, and smoothing jagged edges

    Parameters:
    - mask: a HxW boolean array that has 'True's on pixels that should be kept

    Returns:
    - cleaned_mask: a HxW boolean array filled and smoothed from original mask
    """
    working_mask = 255 * mask.astype(np.uint8)

    # Closing bridges in gaps in the mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    working_mask = cv2.morphologyEx(
        working_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1
    )

    # Find the largest contour
    contours, _ = cv2.findContours(
        working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(mask, dtype=bool)

    largest = max(contours, key=cv2.contourArea)

    # Fill the contour in to eliminate any holes
    solid_mask = np.zeros_like(working_mask)
    cv2.drawContours(solid_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Open operation smoothes hard edges
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleaned_mask = cv2.morphologyEx(
        solid_mask, cv2.MORPH_OPEN, kernel_open, iterations=1
    )

    return cleaned_mask.astype(bool)
