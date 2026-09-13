from app.core.video_processor import CricketVideoProcessor


def test_deduplicate_removes_close_timestamps():
    processor = CricketVideoProcessor.__new__(CricketVideoProcessor)
    detections = [
        {"timestamp": 10.0, "confidence": 0.9},
        {"timestamp": 11.5, "confidence": 0.6},
        {"timestamp": 30.0, "confidence": 0.8},
    ]
    result = processor._deduplicate(detections, min_gap=4.0)
    timestamps = [d["timestamp"] for d in result]
    assert timestamps == [10.0, 30.0]


def test_deduplicate_empty_input():
    processor = CricketVideoProcessor.__new__(CricketVideoProcessor)
    assert processor._deduplicate([], min_gap=4.0) == []
