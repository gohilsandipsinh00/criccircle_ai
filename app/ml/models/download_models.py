"""
Download baseline pretrained models so the service has a
working fallback before the custom cricket models are trained.

Usage:
  python -m app.ml.models.download_models
"""
import os
from pathlib import Path
from app.config import settings
from app.utils.logger import log


def download_yolo_baseline():
    """Downloads the stock YOLOv8n weights via ultralytics
    (used as a fallback until models/yolo/cricket_yolo_v1.pt
    exists)."""
    from ultralytics import YOLO

    Path(settings.yolo_model_path).parent.mkdir(
        parents=True, exist_ok=True
    )
    log.info("Downloading baseline YOLOv8n weights...")
    YOLO("yolov8n.pt")
    log.info(
        "Baseline YOLO ready. Place your trained cricket "
        f"model at {settings.yolo_model_path} when available."
    )


def ensure_classifier_dir():
    Path(settings.classifier_model_path).parent.mkdir(
        parents=True, exist_ok=True
    )
    if not os.path.exists(settings.classifier_model_path):
        log.warning(
            "No trained classifier found at "
            f"{settings.classifier_model_path}. "
            "Run app/ml/training/train_classifier.py first."
        )


if __name__ == "__main__":
    download_yolo_baseline()
    ensure_classifier_dir()
