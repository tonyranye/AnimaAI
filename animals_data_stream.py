# animal_dataset_stream.py

import io
from PIL import Image
from torch.utils.data import Dataset

class GCSImageDataset(Dataset):
    def __init__(self, blobs, labels, label_to_idx, transform=None):
        self.blobs = blobs                # list of blob objects
        self.labels = labels              # list of class names
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, idx):
        blob = self.blobs[idx]
        label_str = self.labels[idx]

        # Download image bytes directly to memory (NO FILE)
        img_bytes = blob.download_as_bytes()

        # Convert bytes → PIL image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_idx = self.label_to_idx[label_str]
        return img, label_idx
