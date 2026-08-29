"""PyTorch Dataset backed by the Rust indexing engine.

Filtering (which images to include, by class / object size) happens in
Rust via `rust_index.DatasetIndex`; this class only handles the
PyTorch-facing concerns of loading images and applying transforms.
"""

from pathlib import Path

import torch
from PIL import Image
from rust_index import DatasetIndex
from torch.utils.data import Dataset


class PerceptionDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations_path: str,
        class_name: str,
        min_instances: int = 1,
        transform=None,
    ):
        self.images_dir = Path(images_dir)
        self.index = DatasetIndex(annotations_path)
        self.file_names = self.index.query_by_class(class_name, min_count=min_instances)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int):
        file_name = self.file_names[idx]
        image = Image.open(self.images_dir / file_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Real project: return matching boxes/labels from the Rust index too.
        return image, file_name

    def class_distribution(self) -> dict:
        """Delegates to Rust for a fast per-class count, e.g. for logging
        dataset composition to MLflow before a training run starts."""
        return self.index.class_counts()
