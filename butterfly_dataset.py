# animal_dataset.py

from PIL import Image
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_idx = None, transform=None):
        """
        image_paths: list[str] - paths to image files on disk
        labels:      list[str] - class names, same length as image_paths
        label_to_idx: dict[str, int] - maps class name -> integer index
        transform: torchvision transforms to apply to each image
        """
        assert len(image_paths) == len(labels), "image_paths and labels must be same length"
        
        self.image_paths = image_paths
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_str = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
            
         # CASE 1 → labels are integers already
        if isinstance(label_str, int):
            return img, label_str

        # CASE 2 → labels are strings (use label_to_idx)
        if self.label_to_idx is None:
            raise ValueError(
                "String labels provided but label_to_idx is None. "
                "Pass label_to_idx or convert labels to integers first."
            )

        label_idx = self.label_to_idx[label_str]
        return img, label_idx
