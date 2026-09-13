"""
Interactive CLI helper for sorting extracted frames into
class folders (data/annotated/classifier/<class_name>/) for
the ResNet50 classifier dataset.

Usage:
  python scripts/label_data.py --frames data/frames --output data/annotated/classifier
"""
import argparse
import shutil
from pathlib import Path

CLASSES = ["six", "four", "wicket", "catch", "celebration", "normal_play"]


def label_frames(frames_dir: str, output_dir: str):
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Found {len(frames)} frames. Classes: {', '.join(CLASSES)}")
    print("Enter class number, 's' to skip, or 'q' to quit.\n")

    for i, frame in enumerate(frames):
        for idx, cls in enumerate(CLASSES):
            print(f"  {idx}: {cls}")
        choice = input(f"[{i + 1}/{len(frames)}] {frame.name} > ").strip()

        if choice.lower() == "q":
            break
        if choice.lower() == "s":
            continue
        if not choice.isdigit() or int(choice) >= len(CLASSES):
            print("Invalid choice, skipping.")
            continue

        cls = CLASSES[int(choice)]
        dest_dir = Path(output_dir) / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(frame, dest_dir / frame.name)

    print("\nLabeling session complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    label_frames(args.frames, args.output)
