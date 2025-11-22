import torch
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms

from gcs_dataset import GCSImageDataset


MANIFEST_PATH = "manifest.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-4
VAL_SPLIT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_dataloaders():
    """
    Create train/validation DataLoaders using the GCSImageDataset,
    which streams images directly from a private GCS bucket.
    """

    # Augmentation + normalization for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Only resize + normalize for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Build two copies of the dataset with different transforms
    full_dataset_for_train = GCSImageDataset(MANIFEST_PATH, transform=train_transform)
    full_dataset_for_val = GCSImageDataset(MANIFEST_PATH, transform=val_transform)

    label_to_idx = full_dataset_for_train.label_to_idx
    class_names = sorted(label_to_idx.keys())

    dataset_size = len(full_dataset_for_train)
    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size

    # Ensure at least one sample in each split (for tiny datasets)
    if train_size == 0:
        train_size = 1
        val_size = dataset_size - 1
    if val_size == 0 and dataset_size > 1:
        val_size = 1
        train_size = dataset_size - 1

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_dataset_for_train,
        [train_size, val_size],
        generator=generator,
    )

    # Validation subset uses validation transforms
    val_subset.dataset = full_dataset_for_val

    # IMPORTANT on Windows: num_workers=0 to avoid pickling GCS client issues
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, label_to_idx, class_names


def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model.to(DEVICE)


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
