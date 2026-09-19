import cv2
import numpy as np


def load_color_frame(frame_path: str) -> np.ndarray[np.uint8]:
    """
    Loads a color frame from a given path

    Parameters:
    - frame_path: a path to a color frame

    Returns:
    - rgb_frame: loaded HxWx3 numpy array containing RGB triples
    """
    bgr_frame = cv2.imread(frame_path, cv2.IMREAD_ANYCOLOR)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    return rgb_frame


def load_depth_frame(frame_path: str) -> np.ndarray[np.uint16]:
    """
    Loads a depth frame from a given path

    Parameters:
    - frame_path: a path to a depth frame

    Returns:
    - rgb_frame: loaded HxW numpy array containing depth information
    """
    depth_frame = cv2.imread(frame_path, cv2.IMREAD_ANYDEPTH)

    return depth_frame
