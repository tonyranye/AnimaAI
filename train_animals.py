import os
import random

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from animal_dataset import AnimalDataset



DATA_ROOT = "Animals"  
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-4
VAL_SPLIT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 1) Scan local folders and build image_paths + labels
def list_image_paths_and_labels(root_dir):
    exts = (".jpg", ".jpeg", ".png")
    image_paths = []
    labels = []

    for label in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(exts):
                continue
            full_path = os.path.join(class_dir, fname)
            image_paths.append(full_path)
            labels.append(label)

    print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")
    print("Classes:", sorted(set(labels)))
    return image_paths, labels


# 2) Create train/val datasets and loaders
def create_dataloaders():
    image_paths, labels = list_image_paths_and_labels(DATA_ROOT)

    # Map class name -> integer index
    class_names = sorted(set(labels))
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    print("label_to_idx:", label_to_idx)

    # Shuffle and split indices
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    split = int(len(indices) * (1.0 - VAL_SPLIT))
    train_idx = indices[:split]
    val_idx = indices[split:]

    def select(idxs):
        return [image_paths[i] for i in idxs], [labels[i] for i in idxs]

    train_paths, train_labels = select(train_idx)
    val_paths, val_labels = select(val_idx)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = AnimalDataset(train_paths, train_labels, label_to_idx, transform=train_transform)
    val_dataset = AnimalDataset(val_paths, val_labels, label_to_idx, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, label_to_idx, class_names


# 3) Build model
def create_model(num_classes):
    # Start from ImageNet pre-trained ResNet18 
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model.to(DEVICE)


# 4) Training loop
def train():
    train_loader, val_loader, label_to_idx, class_names = create_dataloaders()
    model = create_model(num_classes=len(class_names))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        # ----- Train -----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
        )

    # Save model + label mapping
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "class_names": class_names,
        },
        "animal_model_local.pth",
    )
    print("Saved model to animal_model_local.pth")


if __name__ == "__main__":
    train()
