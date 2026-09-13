from typing import List, Dict
from app.ml.models.classifier_model import CricketEventClassifier
from app.utils.logger import log


class EventClassifier:
    """Thin facade over CricketEventClassifier for the video processing pipeline."""

    def __init__(self):
        self.model = CricketEventClassifier()

    def classify_frame(self, frame_path: str) -> Dict:
        return self.model.classify_frame(frame_path)

    def classify_clip(self, frame_paths: List[str]) -> Dict:
        log.info(f"Classifying clip from {len(frame_paths)} frames")
        return self.model.classify_clip_frames(frame_paths)
