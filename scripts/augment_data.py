"""
Applies basic augmentation (flip, brightness, rotation) to a
folder of labeled frames to grow a small training dataset.

Usage:
  python scripts/augment_data.py --source data/annotated/classifier --factor 3
"""
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance
import random


def augment_image(image: Image.Image) -> Image.Image:
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    brightness = ImageEnhance.Brightness(image)
    image = brightness.enhance(random.uniform(0.8, 1.2))

    angle = random.uniform(-5, 5)
    image = image.rotate(angle, expand=False, fillcolor=(0, 0, 0))

    return image


def augment_dataset(source_dir: str, factor: int = 3):
    source = Path(source_dir)
    class_dirs = [d for d in source.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg"))
        print(f"{class_dir.name}: augmenting {len(images)} images x{factor}")

        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            for i in range(factor):
                augmented = augment_image(image)
                out_path = class_dir / f"{img_path.stem}_aug{i}.jpg"
                augmented.save(out_path)

    print("\nAugmentation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--factor", type=int, default=3)
    args = parser.parse_args()
    augment_dataset(args.source, args.factor)
