import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, List
from app.config import settings
from app.utils.logger import log


class CricketEventClassifier:
    """
    ResNet50-based event classifier.

    Classifies cricket moments into:
      0: SIX
      1: FOUR
      2: WICKET
      3: CATCH
      4: CELEBRATION
      5: NORMAL_PLAY
    """

    CLASS_NAMES = {
        0: "six",
        1: "four",
        2: "wicket",
        3: "catch",
        4: "celebration",
        5: "normal_play"
    }

    DISPLAY_NAMES = {
        "six": "SIX \U0001F3CF",
        "four": "FOUR \U0001F3C3",
        "wicket": "WICKET \U0001F3AF",
        "catch": "CATCH \U0001F64C",
        "celebration": "CELEBRATION \U0001F389",
        "normal_play": "Normal Play"
    }

    # Standard ImageNet normalization
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def __init__(self):
        self.model = None
        self.device = torch.device(
            "cuda" if (
                settings.use_gpu and
                torch.cuda.is_available()
            )
            else "cpu"
        )
        self.model_path = settings.classifier_model_path
        self._load_model()

    def _build_model(self) -> nn.Module:
        """Build ResNet50 with custom head"""
        model = models.resnet50(weights=None)

        # Custom classification head
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 6)  # 6 event classes
        )
        return model

    def _load_model(self):
        """Load trained ResNet50 classifier"""
        try:
            self.model = self._build_model()

            if os.path.exists(self.model_path):
                log.info(
                    f"Loading classifier: {self.model_path}"
                )
                state_dict = torch.load(
                    self.model_path,
                    map_location=self.device
                )
                self.model.load_state_dict(state_dict)
                log.info("Classifier loaded successfully")
            else:
                log.warning(
                    "Classifier model not found. "
                    "Using random weights (train first!)"
                )

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            log.error(f"Failed to load classifier: {e}")
            self.model = None

    def classify_frame(
        self,
        frame_path: str
    ) -> Dict:
        """Classify a single frame"""
        if self.model is None:
            return {
                "event_type": "unknown",
                "confidence": 0.0,
                "all_scores": {}
            }

        try:
            image = Image.open(frame_path).convert("RGB")
            tensor = self.TRANSFORM(image).unsqueeze(0)
            tensor = tensor.to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(
                    probabilities, 1
                )

            class_id = predicted.item()
            conf = confidence.item()

            # All class scores
            all_scores = {
                self.CLASS_NAMES[i]: float(
                    probabilities[0][i]
                )
                for i in range(6)
            }

            return {
                "event_type": self.CLASS_NAMES[class_id],
                "confidence": conf,
                "all_scores": all_scores,
                "display_name": self.DISPLAY_NAMES.get(
                    self.CLASS_NAMES[class_id], "Unknown"
                )
            }

        except Exception as e:
            log.error(
                f"Classification error on {frame_path}: {e}"
            )
            return {
                "event_type": "unknown",
                "confidence": 0.0,
                "all_scores": {}
            }

    def classify_clip_frames(
        self,
        frame_paths: List[str]
    ) -> Dict:
        """
        Classify multiple frames from a clip.
        Returns aggregated prediction for the clip.
        """
        if not frame_paths:
            return {
                "event_type": "unknown",
                "confidence": 0.0
            }

        predictions = []
        for fp in frame_paths:
            pred = self.classify_frame(fp)
            if pred["event_type"] != "unknown":
                predictions.append(pred)

        if not predictions:
            return {
                "event_type": "unknown",
                "confidence": 0.0
            }

        # Aggregate: sum scores across all frames
        score_sums = {cls: 0.0 for cls in self.CLASS_NAMES.values()}
        for pred in predictions:
            for cls, score in pred["all_scores"].items():
                score_sums[cls] = (
                    score_sums.get(cls, 0.0) + score
                )

        # Normalize
        total = sum(score_sums.values())
        if total > 0:
            score_sums = {
                k: v / total
                for k, v in score_sums.items()
            }

        # Get best class
        best_class = max(score_sums, key=score_sums.get)
        best_confidence = score_sums[best_class]

        # Ignore normal_play results < 0.7 threshold
        if (
            best_class == "normal_play" and
            best_confidence < 0.7
        ):
            # Get second best
            sorted_scores = sorted(
                score_sums.items(),
                key=lambda x: x[1],
                reverse=True
            )
            if len(sorted_scores) > 1:
                best_class = sorted_scores[1][0]
                best_confidence = sorted_scores[1][1]

        return {
            "event_type": best_class,
            "confidence": best_confidence,
            "all_scores": score_sums,
            "frames_analyzed": len(predictions),
            "display_name": self.DISPLAY_NAMES.get(
                best_class, "Unknown"
            )
        }
