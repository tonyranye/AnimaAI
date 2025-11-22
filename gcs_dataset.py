import csv
import io
from typing import List, Tuple, Dict

from google.cloud import storage
from PIL import Image
from torch.utils.data import Dataset


# Global client cache (NOT stored on the Dataset instance, so it won't be pickled)
_gcs_client = None


def _get_client() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


class GCSImageDataset(Dataset):
    """
    Dataset that loads images from a PRIVATE Google Cloud Storage bucket
    using a CSV manifest of (gcs_path, label) rows.

    The CSV is expected to have a header row:
        gcs_path,label

    gcs_path examples:
        gs://animal-ai-images/poised-gateway-478017-a4/cats/001.jpg
    """

    def __init__(self, manifest_path: str, transform=None):
        self.manifest_path = manifest_path
        self.transform = transform

        self.entries: List[Tuple[str, str]] = []
        with open(self.manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gcs_path = row["gcs_path"]
                label = row["label"]
                self.entries.append((gcs_path, label))

        if not self.entries:
            raise ValueError(f"No entries found in manifest: {self.manifest_path}")

        labels = sorted({label for _, label in self.entries})
        self.label_to_idx: Dict[str, int] = {label: i for i, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.entries)

    def _download_bytes(self, gcs_path: str) -> bytes:
        """
        Convert gs://bucket/path/to/file.jpg into bucket + blob and
        download its bytes using an authenticated storage client.
        """
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"Expected GCS path starting with 'gs://', got: {gcs_path}")

        path = gcs_path[len("gs://") :]  # strip prefix
        bucket_name, blob_name = path.split("/", 1)

        client = _get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        return blob.download_as_bytes()

    def __getitem__(self, idx: int):
        gcs_path, label = self.entries[idx]

        img_bytes = self._download_bytes(gcs_path)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label_idx = self.label_to_idx[label]
        return img, label_idx
