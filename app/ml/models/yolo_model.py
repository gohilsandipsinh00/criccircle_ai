import os
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

from app.config import settings
from app.utils.logger import log


class CricketYOLOModel:
    """
    YOLOv8 object detector, run via onnxruntime rather than the full
    ultralytics/torch stack. torch alone costs 300-500MB of resident
    memory just to import -- more than an entire free-tier host's RAM
    budget -- so this wrapper does its own letterbox preprocessing and
    NMS postprocessing instead of relying on ultralytics' convenience
    API, in exchange for a CPU inference runtime with a much smaller
    footprint.

    Detects (once a custom-trained cricket_yolo_v1.onnx exists):
      - cricket_ball (class 0)
      - batsman (class 1)
      - bowler (class 2)
      - fielder (class 3)
      - stumps (class 4)
      - boundary_rope (class 5)
      - umpire (class 6)

    Until that custom model is trained, this falls back to a plain
    COCO-pretrained yolov8n.onnx (bundled in models/yolo/) -- its
    class IDs are COCO's 80 classes, not the cricket ones above, so
    the CLASS_NAMES mapping below is only meaningful once the real
    cricket-trained model is in place. This mirrors the exact same
    fallback caveat the previous torch-based version had; switching
    to ONNX doesn't change that.
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

    INPUT_SIZE = 640
    DEFAULT_FALLBACK_PATH = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "models", "yolo", "yolov8n_fallback.onnx",
    )

    def __init__(self):
        self.session = None
        self.model_path = settings.yolo_model_path
        self.confidence_threshold = settings.confidence_threshold
        self.iou_threshold = 0.45
        self._load_model()

    def _load_model(self):
        """Load an ONNX YOLOv8 model via onnxruntime."""
        try:
            import onnxruntime as ort

            path = self.model_path
            if not os.path.exists(path):
                fallback = os.path.normpath(self.DEFAULT_FALLBACK_PATH)
                log.warning(
                    f"Custom model not found at {path}. "
                    f"Using bundled pretrained YOLOv8n (COCO classes) "
                    f"as fallback: {fallback}"
                )
                path = fallback

            self.session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            log.info(f"YOLO ONNX model loaded successfully from {path}")

        except Exception as e:
            log.error(f"Failed to load YOLO ONNX model: {e}")
            self.session = None

    def _preprocess(
        self, image: Image.Image
    ) -> Tuple[np.ndarray, float, float, float]:
        """Letterbox-resize to INPUT_SIZE x INPUT_SIZE, return the
        model input tensor plus the scale/padding needed to map
        predicted boxes back to the original image's coordinates."""
        orig_w, orig_h = image.size
        scale = min(
            self.INPUT_SIZE / orig_w, self.INPUT_SIZE / orig_h
        )
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (self.INPUT_SIZE - new_w) / 2
        pad_y = (self.INPUT_SIZE - new_h) / 2

        resized = image.convert("RGB").resize(
            (new_w, new_h), Image.BILINEAR
        )
        canvas = Image.new(
            "RGB", (self.INPUT_SIZE, self.INPUT_SIZE), (114, 114, 114)
        )
        canvas.paste(resized, (int(pad_x), int(pad_y)))

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # HWC -> NCHW
        return np.ascontiguousarray(arr), scale, pad_x, pad_y

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> List[Dict]:
        """Raw YOLOv8 ONNX output is (1, 4 + num_classes, num_boxes).
        Filter by confidence, undo letterbox, then NMS."""
        predictions = output[0].T  # (num_boxes, 4 + num_classes)
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        keep = confidences >= self.confidence_threshold
        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # center-xywh (model space) -> x1y1x2y2 (original image space)
        cx, cy, w, h = (
            boxes_xywh[:, 0], boxes_xywh[:, 1],
            boxes_xywh[:, 2], boxes_xywh[:, 3],
        )
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep_idx = self._nms(boxes_xyxy, confidences, self.iou_threshold)

        detected_objects = []
        for i in keep_idx:
            class_id = int(class_ids[i])
            detected_objects.append({
                "class_id": class_id,
                "class_name": self.CLASS_NAMES.get(class_id, "unknown"),
                "confidence": float(confidences[i]),
                "bbox": {
                    "x1": float(boxes_xyxy[i, 0]),
                    "y1": float(boxes_xyxy[i, 1]),
                    "x2": float(boxes_xyxy[i, 2]),
                    "y2": float(boxes_xyxy[i, 3]),
                },
            })
        return detected_objects

    @staticmethod
    def _nms(
        boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> List[int]:
        """Plain numpy NMS (greedy, class-agnostic) -- avoids pulling
        in opencv just for cv2.dnn.NMSBoxes."""
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = (
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        )
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = np.where(union > 0, inter / union, 0)

            order = order[1:][iou <= iou_threshold]

        return keep

    def detect(self, frame_path: str) -> Dict:
        """Run object detection on a single frame."""
        if self.session is None:
            return {"objects": [], "error": "Model not loaded"}

        try:
            image = Image.open(frame_path)
            tensor, scale, pad_x, pad_y = self._preprocess(image)
            output = self.session.run(None, {self.input_name: tensor})[0]
            objects = self._postprocess(output, scale, pad_x, pad_y)
            return {"objects": objects}

        except Exception as e:
            log.error(f"Detection error on {frame_path}: {e}")
            return {"objects": [], "error": str(e)}

    def detect_batch(
        self, frame_paths: List[str], batch_size: int = 16
    ) -> List[Dict]:
        """Run detection on multiple frames.

        onnxruntime's CPUExecutionProvider doesn't gain much from
        batching the way a GPU would, and keeping this a simple
        per-frame loop avoids holding `batch_size` full-resolution
        images in memory at once on an already memory-constrained
        host. batch_size is kept as a parameter for API compatibility
        with callers, but no longer changes the actual execution
        strategy.
        """
        return [self.detect(frame_path) for frame_path in frame_paths]
