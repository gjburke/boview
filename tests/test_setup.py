"""Testing the setup stage of the pipeline"""

from pathlib import Path

from boview.pipeline.setup import load_color_frame, load_depth_frame

DATA_DIR = Path(__file__).resolve().parent / "data"


def test_frame_loading():

    color_frame = load_color_frame(str(DATA_DIR / "color.jpg"))
    depth_frame = load_depth_frame(str(DATA_DIR / "depth.png"))

    assert color_frame is not None
    assert depth_frame is not None
