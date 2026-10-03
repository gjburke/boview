"""Functions for rotating an image"""

import cv2
import numpy as np
from skimage.measure import regionprops


def get_rotation_matrix(mask: np.ndarray[np.bool]) -> np.ndarray:
    """
    Returns the matrix to rotate an image's object back to horizontal

    Parameters:
    - mask: the HxW boolean segmentation mask for that particular image

    Returns:
    - rotation_matrix: a 2x3 array with a 2x2 rotation/scaling matrix and scaling column
    """
    binary = mask.astype(np.uint8)
    props = regionprops(binary)[0]

    orientation_degree = np.rad2deg(props.orientation)
    angle = (180 - orientation_degree) % 180 - 90
    center = (props.centroid[1], props.centroid[0])

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return rotation_matrix
