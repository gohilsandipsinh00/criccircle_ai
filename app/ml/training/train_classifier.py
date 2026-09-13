"""
Train ResNet50 event classifier on cricket clips.

Classes:
  0: six
  1: four
  2: wicket
  3: catch
  4: celebration
  5: normal_play

Dataset structure:
  dataset/
    train/
      six/      -> images of six moments
      four/     -> images of four moments
      wicket/   -> images of wicket moments
      catch/    -> images of catch moments
      celebration/
      normal_play/
    val/
      (same structure)

Usage:
  python app/ml/training/train_classifier.py \
    --data dataset/classifier \
    --epochs 50
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
import argparse

CLASS_NAMES = [
    "six", "four", "wicket",
    "catch", "celebration", "normal_play"
]


def train_classifier(
    data_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    output_path: str = "models/classifier/cricket_classifier_v1.pth"
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Datasets
    train_dataset = ImageFolder(
        f"{data_dir}/train",
        transform=train_transform
    )
    val_dataset = ImageFolder(
        f"{data_dir}/val",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Model (transfer learning from ImageNet)
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )

    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 6)
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            # Higher weight for rarer events
            [2.0, 2.0, 2.0, 2.0, 1.5, 0.5]
        ).to(device)
    )
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = (
                images.to(device), labels.to(device)
            )

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (
                predicted == labels
            ).sum().item()

        train_acc = train_correct / len(train_dataset)

        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = (
                    images.to(device), labels.to(device)
                )
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (
                    predicted == labels
                ).sum().item()

        val_acc = val_correct / len(val_dataset)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {train_acc:.3f} | "
            f"Val: {val_acc:.3f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(output_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            torch.save(model.state_dict(), output_path)
            print(f"  -> Best model saved! ({val_acc:.3f})")

    print("\nTraining complete!")
    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--output",
        default="models/classifier/cricket_classifier_v1.pth"
    )
    args = parser.parse_args()

    train_classifier(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        output_path=args.output
    )
