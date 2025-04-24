from __future__ import annotations

"""Custom dataset & dataloader utilities for APTOS‑style CSVs.

`train.csv` must contain:
    * **id_code** – image file stem (without extension)
    * **diagnosis** – integer label 0 … 4

Images are searched in *img_dir* with common extensions (``.png``, ``.jpg``,
``.jpeg``).  All preprocessing is performed **once** inside ``__init__`` and the
processed tensors are cached in memory, so ``__getitem__`` is a constant‑time
lookup.
"""

from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm


class RetinopathyDataset(Dataset):
    """Dataset that eagerly loads & transforms all images in `__init__`."""

    def __init__(
        self,
        csv_file: str | Path,
        img_dir: str | Path,
        *,
        img_size: int = 224,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        df = pd.read_csv(csv_file)
        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []

        if df['labels'].dtype == 'O':
            classes = sorted(df['labels'].unique())
            self._map = {c: i for i, c in enumerate(classes)}
            df['labels'] = df['labels'].map(self._map)

        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = self._resolve_path(str(row["filepaths"]))
            img = Image.open(img_path).convert("RGB")
            self.images.append(self.transform(img))
            self.labels.append(int(row["labels"]))

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _resolve_path(self, value: str) -> Path:
        # absolute
        p = Path(value)
        if p.is_absolute() and p.exists():
            return p

        # relative sub‑folder path inside img_dir
        if "/" in value or "\\" in value:
            candidate = self.img_dir / value
            if candidate.exists():
                return candidate

        # bare stem → try extensions
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = self.img_dir / f"{value}{ext}"
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not locate image for '{value}' inside '{self.img_dir}'."
        )

    # ---------------------------------------------------------------------
    # PyTorch Dataset API
    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return self.images[idx], self.labels[idx]


def get_loaders(
    csv_path: str | Path,
    img_dir: str | Path,
    *,
    img_size: int = 224,
    batch_size: int = 32,
    split_ratio: float = 0.8,
    num_workers: int = 4,
):
    """Create train/val dataloaders backed by ``RetinopathyDataset``."""

    dataset = RetinopathyDataset(csv_path, img_dir, img_size=img_size)

    train_len = int(len(dataset) * split_ratio)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, list(range(100))  # 5 classes
