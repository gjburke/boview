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


def rotate_frame(frame: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Rotates a HxW frame, interpolating pixels when needed and keeping a hard edge

    Parameters:
    - frame: a HxW(x...) frame to rotate
    - rotation_matrix: a 2x3 array with a 2x2 rotation/scaling matrix and scaling column

    Returns:
    - frame_rotated: a HxW(x...) rotated frame
    """
    frame_rotated = cv2.warpAffine(
        frame,
        rotation_matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return frame_rotated


def rotate_points(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Rotates a set of (N, 2) points in (x, y) format and rounds to nearest pixel

    Parameters:
    - points: an (N, 2) array of (x, y) points
    - rotation_matrix: a 2x3 array with a 2x2 rotation/scaling matrix and scaling column

    Returns:
    - points_rotated: an (N, 2) array of rotated (x, y) points
    """
    points_rotated = cv2.transform(
        points[:, :2].reshape(-1, 1, 2),
        rotation_matrix,
    ).reshape(-1, 2)
    points_rotated = np.round(points_rotated).astype(int)

    return points_rotated
