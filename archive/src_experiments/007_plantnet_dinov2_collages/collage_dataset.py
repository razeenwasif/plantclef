"""
Pre-made labeled collage dataset for PlantCLEF 2026 multi-label training.

Data layout
-----------
  collages_root/
    collage_000000.jpg
    collage_000001.jpg
    ...

CSV format (synthetic_collages.csv):
  image_name;species_ids
  collage_000000.jpg;1392244,1550263,1668473,...

The CSV is semicolon-separated; `species_ids` is a comma-separated list of
species_id integers. Each image is a multi-species composite; the K-hot target
marks every listed species as positive.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class CollageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        images_root: str | Path,
        species_ids: list[str],
        transform: Optional[Callable] = None,
        verify_files: bool = False,
    ) -> None:
        super().__init__()
        csv_path = Path(csv_path)
        self.images_root = Path(images_root)
        if not csv_path.exists():
            raise FileNotFoundError(f"Collage CSV not found: {csv_path}")

        self.species_to_idx = {str(sid): i for i, sid in enumerate(species_ids)}
        self.n_classes = len(species_ids)
        self.transform = transform

        rows: list[tuple[str, list[int]]] = []
        n_unknown = 0
        n_empty = 0
        with open(csv_path) as f:
            header = f.readline().strip()
            if "image_name" not in header:
                raise ValueError(f"Unexpected header: {header}")
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split(";")
                if len(parts) < 2:
                    continue
                name = parts[0].strip()
                sid_str = parts[1].strip()
                if not sid_str:
                    n_empty += 1
                    continue
                idxs: list[int] = []
                for tok in sid_str.split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    j = self.species_to_idx.get(tok)
                    if j is None:
                        n_unknown += 1
                        continue
                    idxs.append(j)
                if not idxs:
                    n_empty += 1
                    continue
                rows.append((name, idxs))
        self.rows = rows
        logger.info(
            f"CollageDataset: {len(rows):,} rows loaded "
            f"({n_empty} empty, {n_unknown} unknown species_ids dropped)"
        )

        if verify_files:
            missing = [i for i, (n, _) in enumerate(rows) if not (self.images_root / n).exists()]
            logger.info(f"  verify_files: {len(missing)} missing out of {len(rows)}")
            self.rows = [r for i, r in enumerate(rows) if i not in set(missing)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        name, idxs = self.rows[i]
        path = self.images_root / name
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning(f"Collage read failed ({path}): {exc}; returning a black canvas.")
            img = Image.new("RGB", (336, 336))
        if self.transform is not None:
            img = self.transform(img)

        target = torch.zeros(self.n_classes, dtype=torch.float32)
        for j in idxs:
            target[j] = 1.0
        return img, target


def split_indices(n: int, val_frac: float, seed: int = 42) -> tuple[list[int], list[int]]:
    """Deterministic random split → (train_idx, val_idx)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]
