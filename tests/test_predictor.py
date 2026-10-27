"""Testing saved weight pipelines and prediction input ordering"""

import joblib
import pandas as pd
import pytest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boview.pipeline.model import load_weight_model, predict_weight


def test_saved_pipeline_prediction(tmp_path):
    frame = pd.DataFrame({"Length": [1, 2, 3, 4], "Width": [3, 1, 4, 2]})
    target = 10 * frame["Length"] + 2 * frame["Width"]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("feature_selection", RFE(LinearRegression(), n_features_to_select=2)),
            ("model", LinearRegression()),
        ]
    )
    model.fit(frame, target)

    path = tmp_path / "weight_pipeline.joblib"
    joblib.dump(model, path)

    loaded = load_weight_model(str(path))

    # Dictionary order must not swap the inputs passed to the fitted scaler
    assert predict_weight(loaded, {"Width": 5, "Length": 6}) == pytest.approx(70)

    with pytest.raises(KeyError):
        predict_weight(loaded, {"Length": 6})
