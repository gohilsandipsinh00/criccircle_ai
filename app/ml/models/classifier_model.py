import os
from typing import Dict, List

import numpy as np
from PIL import Image

from app.config import settings
from app.utils.logger import log


class CricketEventClassifier:
    """
    Event classifier, run via onnxruntime rather than torch/
    torchvision (see yolo_model.py for why -- torch's import cost
    alone is more memory than an entire free-tier host's budget).

    Classifies cricket moments into:
      0: SIX
      1: FOUR
      2: WICKET
      3: CATCH
      4: CELEBRATION
      5: NORMAL_PLAY

    No trained classifier exists yet (training produces a .pth
    checkpoint today -- export it with torch.onnx.export() once
    trained, see app/ml/training/train_classifier.py). Until then,
    this class has nothing to load and every call returns "unknown"
    with 0 confidence, same as its previous random-weight-ResNet50
    fallback effectively did (random weights = meaningless output) --
    the difference is this version doesn't spend ~100MB of RAM to
    produce that same meaningless output.
    """

    CLASS_NAMES = {
        0: "six",
        1: "four",
        2: "wicket",
        3: "catch",
        4: "celebration",
        5: "normal_play",
    }

    DISPLAY_NAMES = {
        "six": "SIX \U0001F3CF",
        "four": "FOUR \U0001F3C3",
        "wicket": "WICKET \U0001F3AF",
        "catch": "CATCH \U0001F64C",
        "celebration": "CELEBRATION \U0001F389",
        "normal_play": "Normal Play",
    }

    INPUT_SIZE = 224
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self):
        self.session = None
        self.model_path = settings.classifier_model_path
        self._load_model()

    def _load_model(self):
        """Load an ONNX classifier if a trained one exists."""
        try:
            if not os.path.exists(self.model_path):
                log.warning(
                    f"Classifier model not found at {self.model_path}. "
                    f"Skipping event classification until trained + "
                    f"exported (train_classifier.py, then "
                    f"torch.onnx.export)."
                )
                return

            import onnxruntime as ort

            self.session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            log.info(f"Classifier loaded successfully from {self.model_path}")

        except Exception as e:
            log.error(f"Failed to load classifier: {e}")
            self.session = None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize(
            (self.INPUT_SIZE, self.INPUT_SIZE), Image.BILINEAR
        )
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - self.IMAGENET_MEAN) / self.IMAGENET_STD
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def classify_frame(self, frame_path: str) -> Dict:
        """Classify a single frame."""
        if self.session is None:
            return {
                "event_type": "unknown",
                "confidence": 0.0,
                "all_scores": {},
            }

        try:
            image = Image.open(frame_path)
            tensor = self._preprocess(image)
            logits = self.session.run(
                None, {self.input_name: tensor}
            )[0][0]
            probabilities = self._softmax(logits)

            class_id = int(np.argmax(probabilities))
            conf = float(probabilities[class_id])
            all_scores = {
                self.CLASS_NAMES[i]: float(probabilities[i])
                for i in range(6)
            }

            return {
                "event_type": self.CLASS_NAMES[class_id],
                "confidence": conf,
                "all_scores": all_scores,
                "display_name": self.DISPLAY_NAMES.get(
                    self.CLASS_NAMES[class_id], "Unknown"
                ),
            }

        except Exception as e:
            log.error(f"Classification error on {frame_path}: {e}")
            return {
                "event_type": "unknown",
                "confidence": 0.0,
                "all_scores": {},
            }

    def classify_clip_frames(self, frame_paths: List[str]) -> Dict:
        """
        Classify multiple frames from a clip.
        Returns aggregated prediction for the clip.
        """
        if not frame_paths or self.session is None:
            return {"event_type": "unknown", "confidence": 0.0}

        predictions = [
            self.classify_frame(fp) for fp in frame_paths
        ]
        predictions = [
            p for p in predictions if p["event_type"] != "unknown"
        ]

        if not predictions:
            return {"event_type": "unknown", "confidence": 0.0}

        score_sums = {cls: 0.0 for cls in self.CLASS_NAMES.values()}
        for pred in predictions:
            for cls, score in pred["all_scores"].items():
                score_sums[cls] = score_sums.get(cls, 0.0) + score

        total = sum(score_sums.values())
        if total > 0:
            score_sums = {k: v / total for k, v in score_sums.items()}

        best_class = max(score_sums, key=score_sums.get)
        best_confidence = score_sums[best_class]

        if best_class == "normal_play" and best_confidence < 0.7:
            sorted_scores = sorted(
                score_sums.items(), key=lambda x: x[1], reverse=True
            )
            if len(sorted_scores) > 1:
                best_class = sorted_scores[1][0]
                best_confidence = sorted_scores[1][1]

        return {
            "event_type": best_class,
            "confidence": best_confidence,
            "all_scores": score_sums,
            "frames_analyzed": len(predictions),
            "display_name": self.DISPLAY_NAMES.get(best_class, "Unknown"),
        }
