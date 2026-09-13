from typing import List, Dict
from app.ml.models.yolo_model import CricketYOLOModel
from app.utils.logger import log


class ObjectDetector:
    """Thin facade over CricketYOLOModel for the video processing pipeline."""

    def __init__(self):
        self.model = CricketYOLOModel()

    def detect_frame(self, frame_path: str) -> Dict:
        return self.model.detect(frame_path)

    def detect_frames(
        self, frame_paths: List[str], batch_size: int = 16
    ) -> List[Dict]:
        log.info(f"Running YOLO detection on {len(frame_paths)} frames")
        return self.model.detect_batch(frame_paths, batch_size=batch_size)
