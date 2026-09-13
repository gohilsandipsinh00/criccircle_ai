from app.ml.models.yolo_model import CricketYOLOModel


def test_class_names_mapping():
    assert CricketYOLOModel.CLASS_NAMES[0] == "cricket_ball"
    assert CricketYOLOModel.CLASS_NAMES[4] == "stumps"


def test_detect_missing_frame_handled_gracefully():
    model = CricketYOLOModel()
    result = model.detect("nonexistent_frame.jpg")
    assert "objects" in result
