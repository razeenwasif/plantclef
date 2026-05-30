"""
Single-plant PC24 dataset for DINOv3 Phase-A full fine-tune.

Reads the stratified/cleaned training CSV (format ``image_name;species_id``),
maps species_id -> class_idx using a canonical sorted list (from the 7806 PC26
species lookup CSV), resolves ``{images_root}/{species_id}/{image_name}`` on
disk, and returns ``(image, class_idx)`` for a single-label CE loop.

Images that point to a species not in the canonical 7806 list are silently
skipped at init time — those are typically PC24-only species retired in PC26.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SinglePlantDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        images_root: str | Path,
        species_ids: list[str],
        transform=None,
        csv_sep: str = ";",
        image_col: str = "image_name",
        species_col: str = "species_id",
    ) -> None:
        self.images_root = Path(images_root)
        self.transform = transform

        # species_id (str) -> class_idx mapping; species_ids from the lookup CSV
        # are the canonical 7806-wide class ordering shared with Phase B / inference.
        self.species_ids: list[str] = [str(s) for s in species_ids]
        self.sid_to_idx: dict[str, int] = {sid: i for i, sid in enumerate(self.species_ids)}

        df = pd.read_csv(csv_path, sep=csv_sep, dtype={species_col: str, image_col: str})
        if image_col not in df.columns or species_col not in df.columns:
            raise ValueError(
                f"CSV missing required columns; saw {list(df.columns)}, "
                f"need {image_col!r} and {species_col!r}"
            )

        total = len(df)
        df = df[df[species_col].isin(self.sid_to_idx)]
        kept = len(df)
        dropped = total - kept
        if dropped:
            logger.info(
                f"[SinglePlantDataset] dropped {dropped}/{total} rows whose "
                f"species_id is not in the 7806-wide class list"
            )

        self.image_names: list[str] = df[image_col].tolist()
        self.targets: list[int] = [self.sid_to_idx[s] for s in df[species_col].tolist()]
        self.species_strs: list[str] = df[species_col].tolist()

        logger.info(
            f"[SinglePlantDataset] {len(self):,} samples over "
            f"{len(set(self.targets)):,} classes (of {len(self.species_ids)})"
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        sid = self.species_strs[idx]
        name = self.image_names[idx]
        path = self.images_root / sid / name
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            # Corrupt / missing images are rare post-cleaning, but fall back to
            # a blank tensor so a DataLoader worker never crashes a long run.
            logger.warning(f"[SinglePlantDataset] cannot read {path}: {exc}")
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[idx])


def compute_class_frequencies(
    dataset: SinglePlantDataset, n_classes: int
) -> torch.Tensor:
    """Return a (n_classes,) float tensor of per-class counts (for logit adj)."""
    counts = torch.zeros(n_classes, dtype=torch.float64)
    for t in dataset.targets:
        counts[t] += 1.0
    return counts


def split_indices(n: int, val_frac: float, seed: int = 42) -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(val_frac * n)))
    return perm[n_val:], perm[:n_val]
