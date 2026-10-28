"""Functions for smoothing noise in height values and masks"""

import cv2
import numpy as np


def clean_mask(mask: np.ndarray[np.bool]) -> np.ndarray[np.bool]:
    """Cleans model output's mask, filling holes, isolating the main mask, and smoothing jagged edges

    Args:
        mask: a HxW boolean array that has 'True's on pixels that should be kept

    Returns:
        cleaned_mask: a HxW boolean array filled and smoothed from original mask
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


def interpolate_height(
    frame: np.ndarray, mask: np.ndarray[np.bool]
) -> (np.ndarray, np.ndarray[np.bool]):
    """Interpolates missing height inside a validated mask, trims statistical
    outliers from the mask (leaving the image untouched), smooths the final
    silhouette, and returns the full image and final mask

    Args:
        frame: a HxW array of height values
        mask: a HxW boolean array that has 'True's on pixels that should be kept

    Returns:
        cleaned_mask: a HxW boolean array filled and smoothed from original mask
    """
    working_frame = frame.copy().astype(np.uint16)
    working_mask = 255 * mask.astype(np.uint8)

    # Interpolate any missing height values within mask
    mask_inpaint = 255 * ((working_frame == 0) & (working_mask > 0)).astype(np.uint8)
    interpolated_frame = cv2.inpaint(working_frame, mask_inpaint, 3, cv2.INPAINT_TELEA)

    # Disclude any outlying height values (ramps on the edge)
    valid_heights = interpolated_frame[working_mask > 0]

    q1, q3 = np.quantile(valid_heights, [0.25, 0.75])
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 3.0 * iqr, q3 + 3.0 * iqr

    outliers = (interpolated_frame < lower_bound) | (interpolated_frame > upper_bound)
    working_mask[outliers] = 0

    # Final smoothing of the mask
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    smoothed_mask = cv2.morphologyEx(
        working_mask, cv2.MORPH_OPEN, kernel_open, iterations=1
    ).astype(np.bool)

    return interpolated_frame, smoothed_mask
