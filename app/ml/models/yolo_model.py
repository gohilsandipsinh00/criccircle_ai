import os
from typing import List, Dict
from app.config import settings
from app.utils.logger import log


class CricketYOLOModel:
    """
    YOLOv8 wrapper for cricket object detection.

    Detects:
      - cricket_ball (class 0)
      - batsman (class 1)
      - bowler (class 2)
      - fielder (class 3)
      - stumps (class 4)
      - boundary_rope (class 5)
      - umpire (class 6)
    """

    CLASS_NAMES = {
        0: "cricket_ball",
        1: "batsman",
        2: "bowler",
        3: "fielder",
        4: "stumps",
        5: "boundary_rope",
        6: "umpire",
    }

    def __init__(self):
        self.model = None
        self.model_path = settings.yolo_model_path
        self.confidence_threshold = (
            settings.confidence_threshold
        )
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO

            if os.path.exists(self.model_path):
                log.info(
                    f"Loading custom YOLO model: "
                    f"{self.model_path}"
                )
                self.model = YOLO(self.model_path)
            else:
                # Use pretrained YOLOv8 as fallback
                # until custom model is trained
                log.warning(
                    "Custom model not found. "
                    "Using pretrained YOLOv8n as fallback."
                )
                self.model = YOLO("yolov8n.pt")

            log.info("YOLO model loaded successfully")

        except Exception as e:
            log.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(
        self,
        frame_path: str
    ) -> Dict:
        """
        Run object detection on a single frame.
        Returns detected objects with positions.
        """
        if self.model is None:
            return {"objects": [], "error": "Model not loaded"}

        try:
            results = self.model(
                frame_path,
                conf=self.confidence_threshold,
                verbose=False
            )

            detected_objects = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()

                    detected_objects.append({
                        "class_id": class_id,
                        "class_name": self.CLASS_NAMES.get(
                            class_id, "unknown"
                        ),
                        "confidence": confidence,
                        "bbox": {
                            "x1": bbox[0],
                            "y1": bbox[1],
                            "x2": bbox[2],
                            "y2": bbox[3],
                        }
                    })

            return {"objects": detected_objects}

        except Exception as e:
            log.error(f"Detection error on {frame_path}: {e}")
            return {"objects": [], "error": str(e)}

    def detect_batch(
        self,
        frame_paths: List[str],
        batch_size: int = 16
    ) -> List[Dict]:
        """Run detection on multiple frames efficiently"""
        all_results = []

        if self.model is None:
            return [
                {"objects": [], "error": "Model not loaded"}
                for _ in frame_paths
            ]

        for i in range(0, len(frame_paths), batch_size):
            batch = frame_paths[i:i + batch_size]

            try:
                results = self.model(
                    batch,
                    conf=self.confidence_threshold,
                    verbose=False
                )

                for result in results:
                    detected_objects = []
                    if result.boxes:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            detected_objects.append({
                                "class_id": class_id,
                                "class_name": (
                                    self.CLASS_NAMES.get(
                                        class_id, "unknown"
                                    )
                                ),
                                "confidence": float(
                                    box.conf[0]
                                ),
                                "bbox": {
                                    "x1": float(
                                        box.xyxy[0][0]
                                    ),
                                    "y1": float(
                                        box.xyxy[0][1]
                                    ),
                                    "x2": float(
                                        box.xyxy[0][2]
                                    ),
                                    "y2": float(
                                        box.xyxy[0][3]
                                    ),
                                }
                            })
                    all_results.append(
                        {"objects": detected_objects}
                    )

            except Exception as e:
                log.error(f"Batch detection error: {e}")
                all_results.extend(
                    [{"objects": [], "error": str(e)}]
                    * len(batch)
                )

        return all_results
