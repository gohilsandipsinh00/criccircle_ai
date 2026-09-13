"""
Splits a folder of labeled frames/clips into train/val sets
for either the YOLO detector or the ResNet50 classifier.

Usage:
  python app/ml/training/prepare_dataset.py \
    --source data/annotated/classifier \
    --output dataset/classifier \
    --val-split 0.2
"""
import argparse
import random
import shutil
from pathlib import Path


def prepare_classifier_dataset(
    source_dir: str,
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42
):
    """
    Expects source_dir/<class_name>/*.jpg and produces
    output_dir/train/<class_name>/*.jpg and
    output_dir/val/<class_name>/*.jpg
    """
    random.seed(seed)
    source = Path(source_dir)
    output = Path(output_dir)

    class_dirs = [d for d in source.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in {source_dir}")

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg")) + list(
            class_dir.glob("*.png")
        )
        random.shuffle(images)

        val_count = max(1, int(len(images) * val_split))
        val_images = images[:val_count]
        train_images = images[val_count:]

        train_out = output / "train" / class_dir.name
        val_out = output / "val" / class_dir.name
        train_out.mkdir(parents=True, exist_ok=True)
        val_out.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy(img, train_out / img.name)
        for img in val_images:
            shutil.copy(img, val_out / img.name)

        print(
            f"{class_dir.name}: "
            f"{len(train_images)} train / {len(val_images)} val"
        )

    print(f"\nDataset prepared at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    prepare_classifier_dataset(
        source_dir=args.source,
        output_dir=args.output,
        val_split=args.val_split
    )
