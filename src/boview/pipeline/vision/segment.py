"""Functions for segmenting images with Ultralytics YOLO11 segmentation models"""

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


def load_segmentation_model(path: str) -> YOLO:
    """
    Loads a segmentation model from a given path

    Parameters:
    - path: a path to an Ultralytics YOLO11 segmentation checkpoint

    Returns:
    - model: loaded segmentation model
    """
    model = YOLO(path)

    return model


# Color


def segment_color_image(model: YOLO, image: np.ndarray) -> Results:
    """
    Segments a color image

    Parameters:
    - model: loaded Ultralytics YOLO11 segmentation model
    - image: a HxWx3 color image

    Returns:
    - result: segmentation prediction for the image
    """
    result = model.predict(image)[0]

    return result


# Depth


def preprocess_height(height: np.ndarray) -> np.ndarray:
    """
    Clips and scales a height frame to a three-channel image

    Parameters:
    - height: a HxW height frame with a nonzero range after clipping

    Returns:
    - rgb: a HxWx3 uint8 image with matching grayscale channels
    """
    clipped = np.clip(height, a_min=0, a_max=height.max() + 1)
    normalized = (clipped / (clipped.max() - clipped.min())) * 255
    rgb = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return rgb


def segment_height_image(model: YOLO, image: np.ndarray) -> Results:
    """
    Preprocesses and segments a height image

    Parameters:
    - model: loaded Ultralytics YOLO11 segmentation model
    - image: a HxW height frame with a nonzero range after clipping

    Returns:
    - result: segmentation prediction for the preprocessed image
    """
    image = preprocess_height(image)
    result = model.predict(image)[0]

    return result


# Masks


def mask_from_result(result: Results | None) -> np.ndarray | None:
    """
    Extracts the first segmentation mask and resizes it to the original image

    Parameters:
    - result: segmentation prediction for an image, or None

    Returns:
    - mask: a HxW boolean array matching the original image dimensions,
      or None if the result or its masks are unavailable
    """
    if result is None or result.masks is None or len(result.masks.data) == 0:
        return None

    mask_numpy = result.masks.data[0].cpu().numpy()

    # Resize the mask to the original image dimensions
    original_height, original_width = result.orig_shape
    mask_resized = cv2.resize(
        mask_numpy,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = mask_resized.astype(bool)

    return mask
