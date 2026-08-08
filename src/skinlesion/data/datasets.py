"""PyTorch dataset and transforms for dermatoscopic images.

Augmentation is deliberately conservative. Diagnosis leans heavily on
pigmentation, colour pattern and border structure, so heavy colour jitter
would destroy the very signal the model needs -- the same reasoning that
keeps hue and saturation untouched in the nudibranch project. Horizontal
flips and mild geometric variation are safe; hue shifts are not.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from .metadata import LesionRecord

# ImageNet statistics, matching the pretrained backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.11)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            # Brightness and contrast only. Saturation and hue are left
            # alone because pigmentation colour is diagnostic.
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class LesionImageDataset(Dataset):
    """Images on disk, addressed by the metadata records.

    Unreadable files are skipped at construction time rather than raising
    mid-epoch, and the count of skipped files is exposed so a corrupted
    download is visible instead of silently shrinking the dataset.
    """

    def __init__(
        self,
        records: Sequence[LesionRecord],
        images_dir: Path,
        class_to_index: Mapping[str, int],
        transform: transforms.Compose,
    ) -> None:
        if not records:
            raise ValueError("cannot build a dataset from zero records")

        self._images_dir = Path(images_dir)
        self._class_to_index = dict(class_to_index)
        self._transform = transform

        usable: list[tuple[Path, int, LesionRecord]] = []
        missing = 0

        for record in records:
            if record.dx not in self._class_to_index:
                continue
            path = self._images_dir / record.filename
            if not path.is_file():
                missing += 1
                continue
            usable.append((path, self._class_to_index[record.dx], record))

        if not usable:
            raise ValueError(
                f"none of the {len(records)} records resolved to a file under {images_dir}. "
                "Run scripts/fetch_data.py to download images first."
            )

        self._items = tuple(usable)
        self.missing_files = missing

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label, _ = self._items[index]
        try:
            with Image.open(path) as image:
                converted = image.convert("RGB")
                tensor = self._transform(converted)
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"could not read image {path}: {exc}") from exc
        return tensor, label

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(label for _, label, _ in self._items)

    def class_counts(self) -> Mapping[int, int]:
        return dict(Counter(self.labels))

    def class_weights(self, num_classes: int) -> torch.Tensor:
        """Inverse-frequency weights, normalised to mean 1.

        The diagnosis distribution is long tailed -- melanocytic nevi
        outnumber vascular lesions by roughly 47 to 1 -- so without
        weighting the model can score well simply by favouring nevi.
        """
        counts = self.class_counts()
        weights = torch.ones(num_classes, dtype=torch.float32)
        for index in range(num_classes):
            count = counts.get(index, 0)
            weights[index] = 1.0 / count if count else 0.0

        positive = weights[weights > 0]
        if positive.numel():
            weights = weights / positive.mean()
        return weights
