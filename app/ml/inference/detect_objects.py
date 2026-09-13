"""Thin CLI wrapper around CricketYOLOModel for ad-hoc inference."""
import argparse
import json
from app.ml.models.yolo_model import CricketYOLOModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame", help="Path to a frame image")
    args = parser.parse_args()

    model = CricketYOLOModel()
    result = model.detect(args.frame)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
