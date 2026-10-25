"""Testing the model input measurements and distance-based conversions"""

import numpy as np
import pytest

from boview.pipeline.model import get_features


def test_model_features():
    # A 4x6 silhouette at 1000 mm height, with background excluded by the mask
    height = np.full((6, 8), 1500.0)
    mask = np.zeros((6, 8), dtype=bool)
    mask[1:5, 1:7] = True
    height[mask] = 1000.0
    keypoints = np.full((15, 2), (3, 2))

    # FL = 2000 mm and AvgHeight = 100 cm give Z = 1000 mm
    features = get_features(height, mask, keypoints, fl=2000.0)

    assert list(features) == [
        "Length",
        "Width",
        "DorsalArea",
        "Volume",
        "CentroidHeight",
        "MaxHeight",
        "AvgHeight",
        "SurfaceArea",
        "HipWidth",
        "AverageHipHeight",
        "Length_cm",
        "Width_cm",
        "DorsalArea_cm2",
        "Volume_cm3",
        "CentroidHeight_cm",
        "MaxHeight_cm",
        "AvgHeight_cm",
        "SurfaceArea_cm2",
        "HipWidth_cm",
        "AverageHipHeight_cm",
    ]
    assert features["Length"] == 6
    assert features["Width"] == 4
    assert features["DorsalArea"] == 24
    assert features["Volume"] == 2400
    assert features["AvgHeight"] == 100
    assert features["HipWidth"] == 3
    assert features["Length_cm"] == pytest.approx(0.858)
    assert features["DorsalArea_cm2"] == pytest.approx(0.552)
    assert features["Volume_cm3"] == pytest.approx(55.2)
    for name in (
        "CentroidHeight_cm",
        "MaxHeight_cm",
        "AvgHeight_cm",
        "AverageHipHeight_cm",
    ):
        assert features[name] == pytest.approx(14.3)

    # Changing camera height changes the conversion factors, not raw measurements
    farther = get_features(height, mask, keypoints, fl=3000.0)
    assert farther["Length"] == features["Length"]
    assert farther["Length_cm"] == pytest.approx(0.858 * 2**0.862)
    assert farther["Volume_cm3"] == pytest.approx(55.2 * 2**1.866)
