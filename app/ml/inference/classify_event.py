"""Thin CLI wrapper around CricketEventClassifier for ad-hoc inference."""
import argparse
import json
from app.ml.models.classifier_model import CricketEventClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame", help="Path to a frame image")
    args = parser.parse_args()

    classifier = CricketEventClassifier()
    result = classifier.classify_frame(args.frame)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
