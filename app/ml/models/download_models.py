"""
Export a baseline YOLOv8n ONNX model so the service has a working
fallback before the custom cricket-trained model exists, and check
whether a trained classifier is already in place.

The exported yolov8n_fallback.onnx is committed to the repo already
(models/yolo/yolov8n_fallback.onnx) -- this script only needs to be
re-run if that file is ever missing or you want to regenerate it.
Requires ultralytics+torch, which are NOT part of the deployed
service's requirements.txt (see requirements-train.txt) -- this is a
one-off local/dev step, not something the running service does.

Usage:
  pip install -r requirements-train.txt
  python -m app.ml.models.download_models
"""
import os
from pathlib import Path
from app.utils.logger import log

FALLBACK_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "models", "yolo", "yolov8n_fallback.onnx",
    )
)


def export_yolo_fallback():
    """(Re-)generate the bundled COCO-pretrained YOLOv8n ONNX fallback."""
    from ultralytics import YOLO

    Path(FALLBACK_PATH).parent.mkdir(parents=True, exist_ok=True)
    log.info("Exporting baseline YOLOv8n to ONNX...")
    model = YOLO("yolov8n.pt")
    exported_path = model.export(format="onnx", imgsz=640, simplify=True)
    os.replace(exported_path, FALLBACK_PATH)
    log.info(f"Fallback ONNX model ready at {FALLBACK_PATH}")
    log.info(
        "Once a custom cricket model is trained (train_yolo.py), "
        "export it the same way and place it at the path configured "
        "by YOLO_MODEL_PATH -- the service prefers that over this "
        "generic fallback automatically."
    )


def check_classifier():
    from app.config import settings

    Path(settings.classifier_model_path).parent.mkdir(
        parents=True, exist_ok=True
    )
    if not os.path.exists(settings.classifier_model_path):
        log.warning(
            "No trained classifier found at "
            f"{settings.classifier_model_path}. Run "
            "train_classifier.py, then torch.onnx.export() the "
            "result to that path. Until then the service skips "
            "event classification and relies on YOLO + audio "
            "heuristics alone."
        )


if __name__ == "__main__":
    export_yolo_fallback()
    check_classifier()
