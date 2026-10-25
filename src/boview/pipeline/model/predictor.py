"""Functions for loading fitted pipelines and predicting bodyweight"""

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_weight_model(path: str) -> Pipeline:
    """
    Loads a fitted weight prediction pipeline from a given path

    Parameters:
    - path: a path to a trusted joblib file containing the fitted scaler,
      feature selector, and regression model

    Returns:
    - model: loaded weight prediction pipeline
    """
    model = joblib.load(path)

    return model


def predict_weight(model: Pipeline, features: dict) -> float:
    """
    Predicts bodyweight from one frame's extracted measurements

    Parameters:
    - model: fitted pipeline trained with named DataFrame columns and the same
      feature conversions used by get_features
    - features: dictionary of unscaled measurements returned by get_features

    Returns:
    - weight: predicted bodyweight in the training target's units
    """
    # Arrange measurements in the saved training column order
    frame = pd.DataFrame([features])
    frame = frame.loc[:, model.feature_names_in_]
    weight = model.predict(frame)[0]

    return float(weight)
