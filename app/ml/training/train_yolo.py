"""
Train YOLOv8 on cricket dataset.

Run on Google Colab (free GPU) or Runpod.

Usage:
  python app/ml/training/train_yolo.py \
    --data dataset/cricket.yaml \
    --epochs 100 \
    --model yolov8m
"""

import argparse


def train_yolo(
    data_yaml: str,
    epochs: int = 100,
    model_size: str = "yolov8m",
    imgsz: int = 640,
    batch: int = 16,
    name: str = "cricket_yolo_v1"
):
    """
    Train YOLOv8 on cricket dataset.

    Model sizes:
      yolov8n -> nano (fastest, less accurate)
      yolov8s -> small
      yolov8m -> medium (recommended)
      yolov8l -> large
      yolov8x -> xlarge (most accurate, slowest)

    Dataset YAML format:
      path: /path/to/dataset
      train: images/train
      val: images/val
      nc: 7
      names:
        0: cricket_ball
        1: batsman
        2: bowler
        3: fielder
        4: stumps
        5: boundary_rope
        6: umpire
    """
    from ultralytics import YOLO

    print(f"Loading {model_size} pretrained model...")
    model = YOLO(f"{model_size}.pt")

    print(f"Starting training on {data_yaml}...")
    print(f"Epochs: {epochs}, Image size: {imgsz}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,

        # Augmentation (helps with cricket scenarios)
        hsv_h=0.015,      # Hue variation
        hsv_s=0.7,        # Saturation variation
        hsv_v=0.4,        # Brightness variation
        degrees=5.0,      # Rotation
        translate=0.1,    # Translation
        scale=0.5,        # Scale
        fliplr=0.5,       # Horizontal flip
        mosaic=1.0,       # Mosaic augmentation

        # Training settings
        patience=20,      # Early stopping
        save=True,
        plots=True,
        device=0,         # GPU 0, or 'cpu'
    )

    print("\nTraining complete!")
    print(f"Best model: runs/detect/{name}/weights/best.pt")
    print(
        "mAP50:",
        results.results_dict.get("metrics/mAP50(B)", "N/A")
    )
    print(
        "mAP50-95:",
        results.results_dict.get("metrics/mAP50-95(B)", "N/A")
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolov8m")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="cricket_yolo_v1")
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        model_size=args.model,
        batch=args.batch,
        name=args.name
    )
