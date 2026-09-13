from PIL import Image
import numpy as np


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.array(image)


def resize_image(
    image: Image.Image,
    width: int,
    height: int
) -> Image.Image:
    return image.resize((width, height))


def crop_bbox(
    image: Image.Image,
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> Image.Image:
    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def bbox_center(bbox: dict) -> tuple:
    center_x = (bbox["x1"] + bbox["x2"]) / 2
    center_y = (bbox["y1"] + bbox["y2"]) / 2
    return center_x, center_y


def bbox_distance(bbox_a: dict, bbox_b: dict) -> float:
    ax, ay = bbox_center(bbox_a)
    bx, by = bbox_center(bbox_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
