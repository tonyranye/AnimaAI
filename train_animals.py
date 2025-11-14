import random
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from google.cloud import storage
from animals_data_stream import GCSImageDataset

PROJECT_ID = "poised-gateway-478017-a4"
BUCKET_NAME = "animal-ai-images"


# -------------------------------------------------------------
# Function: list_blobs_and_labels
# -------------------------------------------------------------
# Lists all image blobs in the GCS bucket and extracts:
#   - The blob objects themselves (for downloading/streaming)
#   - Labels (taken from the folder name)
# Only valid image formats (.jpg/.jpeg/.png) are accepted.
# Results are printed and returned.
# -------------------------------------------------------------
def list_blobs_and_labels():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    blobs = []
    labels = []

    # Loop through every file stored in the bucket
    for blob in client.list_blobs(bucket):

        # blob.name is something like "images/cat/00001.jpg"
        parts = blob.name.split("/")

        # We expect structure: parent_folder/class_name/filename
        if len(parts) < 3 or parts[-1] == "":
            continue

        filename = parts[-1]

        # Only process images
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Class label is taken from the 2nd last folder name
        label = parts[-2]

        blobs.append(blob)
        labels.append(label)

    print("Loaded", len(blobs), "images from GCS")
    print("Found classes:", sorted(set(labels)))
    return blobs, labels


# -------------------------------------------------------------
# Function: train
# -------------------------------------------------------------
# Main training function:
#   - Loads blob names & labels from GCS
#   - Randomizes them
#   - Creates a mapping from class name -> numeric index
#   - Splits into train/val sets
#   - Creates PyTorch datasets and dataloaders
#   - Builds a pretrained ResNet18 model
#   - Trains for several epochs
#   - Saves model checkpoint
# -------------------------------------------------------------
def train():

    # Fetch (blob, label) pairs from bucket
    blobs, labels = list_blobs_and_labels()

    # Shuffle to randomize order
    combined = list(zip(blobs, labels))
    random.shuffle(combined)
    blobs, labels = zip(*combined)

    # Use first 100 images for quick testing
    blobs, labels = blobs[:100], labels[:100]

    # Build a class-index mapping
    classes = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    print("Classes used in this run:", classes)

    # 80/20 train/validation split
    split = int(0.8 * len(blobs))
    train_blobs = blobs[:split]
    val_blobs   = blobs[split:]
    train_labels = labels[:split]
    val_labels   = labels[split:]

    # ---------------------------------------------------------
    # Image preprocessing for ResNet:
    #   - Resize to 224x224
    #   - Convert to tensor
    #   - Normalize using ImageNet mean/std
    # ---------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Custom streaming dataset (loads images directly from GCS)
    train_dataset = GCSImageDataset(train_blobs, train_labels, label_to_idx, transform=transform)
    val_dataset   = GCSImageDataset(val_blobs,   val_labels,   label_to_idx, transform=transform)

    # PyTorch loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # ---------------------------------------------------------
    # Model setup:
    #   - Load pretrained ResNet18
    #   - Replace final layer with correct number of classes
    # ---------------------------------------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    # GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer + loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total = 0
        correct = 0
        all_expected = []
        all_actual = []

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            opt.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()

            # Accuracy calculation
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_expected.extend(targets.cpu().tolist())
            all_actual.extend(preds.cpu().tolist())

        # Print training accuracy
        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train accuracy = {train_acc:.3f}")
        print("Example expected labels (first 100):", all_expected[:100])
        print("Example predicted labels (first 100):", all_actual[:100])

        # -----------------------------------------------------
        # Validation accuracy
        # -----------------------------------------------------
        model.eval()
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        if val_total > 0:
            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1}: Val accuracy   = {val_acc:.3f}")
        else:
            print(f"Epoch {epoch+1}: No validation samples!")

    # ---------------------------------------------------------
    # Save model checkpoint for later inference
    # ---------------------------------------------------------
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_to_idx": label_to_idx,
        "classes": classes
    }, "animal_model_stream.pth")

    print("Saved streaming-trained model to animal_model_stream.pth")


# Run training when executing the script
if __name__ == "__main__":
    train()
