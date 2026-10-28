"""Frame-level feature extraction and bodyweight prediction"""

from dataclasses import replace

import numpy as np
from sklearn.pipeline import Pipeline

from ..frame import PipelineFrame, reject_frame, require_stage
from .features import get_features
from .predictor import predict_weight


def model_frame(frame: PipelineFrame, model: Pipeline) -> PipelineFrame:
    """Extracts model inputs and predicts bodyweight without changing the input

    Args:
        frame: pending aligned frame with rotated height, mask, and keypoints
        model: fitted weight pipeline using the same conversions as get_features

    Returns:
        predicted: accepted new frame with 20 unscaled features and weight, or
            rejected frame for invalid measurements or nonfinite predictions
    """
    require_stage(frame, "align")

    features = get_features(
        frame.height_frame, frame.height_mask, frame.keypoints, frame.camera_height
    )
    predicted = replace(frame, features=features, stage="model")

    if not np.isfinite(list(features.values())).all():
        return reject_frame(predicted, "model", "Features contain nonfinite values")
    if features["AverageHipHeight"] <= 0 or features["HipWidth"] <= 0:
        return reject_frame(predicted, "model", "Hip measurements are unusable")

    weight = predict_weight(model, features)

    if not np.isfinite(weight):
        return reject_frame(predicted, "model", "Prediction is not finite")

    return replace(predicted, weight=weight, status="accepted")
