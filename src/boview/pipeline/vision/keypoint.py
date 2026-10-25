"""Functions for detecting keypoints with an Ultralytics YOLO11 pose model"""

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


def load_keypoint_model(path: str) -> YOLO:
    """
    Loads a keypoint model from a given path

    Parameters:
    - path: a path to an Ultralytics YOLO11 pose checkpoint

    Returns:
    - model: loaded keypoint model
    """
    model = YOLO(path)

    return model


# Color


def keypoint_color_image(model: YOLO, image: np.ndarray) -> Results:
    """
    Detects keypoints in a color image

    Parameters:
    - model: loaded Ultralytics YOLO11 pose model
    - image: a HxWx3 color image

    Returns:
    - result: keypoint prediction for the image
    """
    result = model.predict(image)[0]

    return result


# Keypoints


def keypoints_from_result(result: Results | None) -> np.ndarray | None:
    """
    Extracts keypoints from a prediction containing one detected animal

    Parameters:
    - result: keypoint prediction for one animal, or None

    Returns:
    - keypoints: a Kx2 or Kx3 array of (x, y) coordinates and optional confidence,
      or None if the result or its keypoints are unavailable
    """
    if result is None or result.keypoints is None:
        return None

    keypoints = result.keypoints.data
    keypoints = keypoints.reshape(keypoints.shape[1:]).cpu().numpy()

    return keypoints
