"""
Benchmarks per-frame YOLO inference time and per-clip classifier
inference time on the current machine/GPU config.

Usage:
  python scripts/benchmark.py --frames data/frames
"""
import argparse
import time
from pathlib import Path

from app.ml.models.yolo_model import CricketYOLOModel
from app.ml.models.classifier_model import CricketEventClassifier


def benchmark(frames_dir: str, sample_size: int = 50):
    frames = sorted(Path(frames_dir).glob("*.jpg"))[:sample_size]
    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    frame_paths = [str(f) for f in frames]

    yolo = CricketYOLOModel()
    start = time.time()
    yolo.detect_batch(frame_paths, batch_size=16)
    yolo_time = time.time() - start
    print(
        f"YOLO: {len(frame_paths)} frames in {yolo_time:.2f}s "
        f"({yolo_time / len(frame_paths) * 1000:.1f} ms/frame)"
    )

    classifier = CricketEventClassifier()
    start = time.time()
    for fp in frame_paths:
        classifier.classify_frame(fp)
    clf_time = time.time() - start
    print(
        f"Classifier: {len(frame_paths)} frames in {clf_time:.2f}s "
        f"({clf_time / len(frame_paths) * 1000:.1f} ms/frame)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()
    benchmark(args.frames, args.sample_size)
