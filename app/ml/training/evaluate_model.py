"""
Evaluate a trained classifier checkpoint against a validation
folder and print per-class accuracy.

Usage:
  python app/ml/training/evaluate_model.py \
    --data dataset/classifier/val \
    --model models/classifier/cricket_classifier_v1.pth
"""
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from app.ml.models.classifier_model import CricketEventClassifier


def evaluate(data_dir: str, model_path: str, batch_size: int = 32):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    classifier = CricketEventClassifier()
    classifier.model_path = model_path
    classifier._load_model()
    model = classifier.model.to(device)
    model.eval()

    class_names = dataset.classes
    correct_per_class = {c: 0 for c in class_names}
    total_per_class = {c: 0 for c in class_names}
    total_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for pred, label in zip(predicted, labels):
                cls = class_names[label.item()]
                total_per_class[cls] += 1
                if pred == label:
                    correct_per_class[cls] += 1
                    total_correct += 1

    print(f"\nOverall accuracy: {total_correct / len(dataset):.3f}\n")
    for cls in class_names:
        total = total_per_class[cls]
        acc = correct_per_class[cls] / total if total else 0.0
        print(f"  {cls:15s}: {acc:.3f} ({total} samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    evaluate(args.data, args.model)
