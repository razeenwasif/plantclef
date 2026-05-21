"""Phase cRT dataset utilities."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Reads (feature, label) pairs from a Phase 1 .pt cache file.

    Supports both the legacy {'bio':…, 'dino':…, 'conv':…, 'labels':…} dict
    and the newer {'features':…, 'labels':…} dict.
    """

    def __init__(self, cache_path: str) -> None:
        data = torch.load(cache_path, weights_only=False, mmap=True)
        if "bio" in data:
            self.features = torch.cat([data["bio"], data["dino"], data["conv"]], dim=1)
        else:
            self.features = data["features"]
        self.labels = data["labels"].long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]
