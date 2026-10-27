"""Test the full pipeline with sample data"""

from pathlib import Path

from boview.pipeline import process_frame
from boview.pipeline.model import load_weight_model
from boview.pipeline.setup import load_color_frame, load_depth_frame
from boview.pipeline.vision import load_keypoint_model, load_segmentation_model

DATA_DIR = Path(__file__).resolve().parent / "data"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "src" / "boview" / "models"

# Full Pipeline


def test_full_pipeline():

    color_frame = load_color_frame(str(DATA_DIR / "color.jpg"))
    depth_frame = load_depth_frame(str(DATA_DIR / "depth.png"))
    camera_height = 3154

    seg_model_path = Path(WEIGHTS_PATH / "segmentation/best.pt")
    key_model_path = Path(WEIGHTS_PATH / "keypoint/best.pt")
    pred_model_path = Path(WEIGHTS_PATH / "predictor/Random_Forest.joblib")

    segmentation_model = load_segmentation_model(seg_model_path)
    keypoint_model = load_keypoint_model(key_model_path)
    predictor = load_weight_model(pred_model_path)

    result = process_frame(
        color_frame=color_frame,
        depth_frame=depth_frame,
        camera_height=camera_height,
        segmentation_model=segmentation_model,
        keypoint_model=keypoint_model,
        weight_model=predictor,
    )
    assert result.status == "accepted", result.rejection_reason

    weight = result.weight

    assert abs(336 - weight) < 5
