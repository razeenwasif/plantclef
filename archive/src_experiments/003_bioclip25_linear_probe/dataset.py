"""
Dataset utilities for the BioCLIP 2.5 linear probe experiment.

Image layout on disk (labeled data)
-------------------------------------
  {train_image_root}/{species_id}/{image_name}

e.g.
  /workspace/plantclef/raw/train/images_max_side_800/1396710/59feabe1...jpg

CSV parsing: semicolon-delimited with quoted fields.
  Required columns: image_name, species_id.

Label format: single integer class index (sorted species_id → 0..N-1).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_META_CSV = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "002_bioclip_tile_zero_shot_v2/data/"
    "PlantCLEF2024_single_plant_training_metadata.csv"
)
DEFAULT_TRAIN_IMAGE_ROOT = "/workspace/plantclef/raw/train/images_max_side_800"
DEFAULT_TEST_IMAGES_DIR = "/workspace/plantclef/kaggle_uploads/test/images"

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_train_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load the PlantCLEF single-plant training metadata CSV.

    The file is semicolon-delimited with quoted fields.
    Returns a DataFrame with stripped string columns.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training metadata CSV not found: {path}")
    logger.info(f"Loading training metadata from {path} ...")
    df = pd.read_csv(path, sep=";", dtype=str, low_memory=False)
    df.columns = [c.strip('"').strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip('"').str.strip()
    required = {"image_name", "species_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Training metadata missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    logger.info(
        f"  Loaded {len(df):,} rows, {df['species_id'].nunique():,} species"
    )
    return df


def resolve_image_paths(
    df: pd.DataFrame,
    image_root: str,
    verify: bool = False,
    log_every: int = 200_000,
) -> pd.DataFrame:
    """
    Add a ``resolved_path`` column: ``{image_root}/{species_id}/{image_name}``

    If verify=True checks each path exists (slow for 1.4M rows).
    """
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Train image root not found: {root}")
    paths: list[Optional[str]] = []
    n_missing = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if log_every and i % log_every == 0:
            logger.info(
                "  path resolution: %s/%s rows (%.1f%%)  missing=%s",
                f"{i:,}", f"{len(df):,}", 100.0 * i / len(df), f"{n_missing:,}",
            )
        candidate = root / str(row["species_id"]) / str(row["image_name"])
        if not verify or candidate.exists():
            paths.append(str(candidate))
        else:
            paths.append(None)
            n_missing += 1
    df = df.copy()
    df["resolved_path"] = paths
    if n_missing:
        logger.warning(
            f"{n_missing:,} images not found under {root}"
        )
    else:
        logger.info(f"  All {len(df):,} paths constructed (verify={verify})")
    return df


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

def build_class_mapping(df: pd.DataFrame) -> tuple[dict[str, int], list[str]]:
    """
    Build sorted species_id → class_index mapping.

    Returns (species_to_idx, idx_to_species).
    """
    unique_species = sorted(df["species_id"].unique())
    species_to_idx = {s: i for i, s in enumerate(unique_species)}
    logger.info(f"Class mapping: {len(unique_species):,} unique species")
    return species_to_idx, unique_species


def save_class_mapping(idx_to_species: list[str], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for s in idx_to_species:
            f.write(s + "\n")
    logger.info(f"Class mapping saved: {p}  ({len(idx_to_species):,} classes)")


def load_class_mapping(path: str) -> tuple[dict[str, int], list[str]]:
    with open(path) as f:
        idx_to_species = [line.strip() for line in f if line.strip()]
    species_to_idx = {s: i for i, s in enumerate(idx_to_species)}
    logger.info(f"Class mapping loaded: {path}  ({len(idx_to_species):,} classes)")
    return species_to_idx, idx_to_species


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def build_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified random 90/10 split by species.

    Species with fewer than 5 images are kept entirely in train.
    Returns (train_df, val_df).
    """
    rng = random.Random(seed)
    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []

    for _, group in df.groupby("species_id", sort=False):
        n = len(group)
        if n < 5:
            train_rows.append(group)
            continue
        n_val = max(1, int(n * val_fraction))
        indices = list(range(n))
        rng.shuffle(indices)
        val_idx = set(indices[:n_val])
        train_rows.append(group.iloc[[i for i in range(n) if i not in val_idx]])
        val_rows.append(group.iloc[sorted(val_idx)])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = (
        pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    )
    logger.info(
        f"Split → train {len(train_df):,} images "
        f"({train_df['species_id'].nunique():,} species), "
        f"val {len(val_df):,} images "
        f"({val_df['species_id'].nunique():,} species)"
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class PlantCLEFDataset(Dataset):
    """
    Labeled single-plant training dataset.
    Each item: (image_tensor, class_index).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        species_to_idx: dict[str, int],
        transform,
    ) -> None:
        valid = df[df["resolved_path"].notna()].reset_index(drop=True)
        if len(valid) < len(df):
            logger.info(
                f"PlantCLEFDataset: dropped {len(df) - len(valid):,} rows "
                f"with missing image paths"
            )
        self.df = valid
        self.species_to_idx = species_to_idx
        self.transform = transform
        logger.info(
            f"PlantCLEFDataset: {len(self.df):,} images, "
            f"{self.df['species_id'].nunique():,} species"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["resolved_path"]).convert("RGB")
        label = self.species_to_idx[row["species_id"]]
        return self.transform(image), label


class EmbeddingDataset(Dataset):
    """
    Training dataset over pre-computed backbone embeddings (from build_cache.py).

    Each item: (embedding_tensor, class_index).
    Avoids re-running the frozen backbone every step → orders-of-magnitude faster.
    """

    def __init__(self, embeddings_path: str, labels_path: str) -> None:
        import torch
        self.embeddings = torch.load(embeddings_path, weights_only=True)  # (N, D)
        self.labels = torch.load(labels_path, weights_only=True)          # (N,)
        assert len(self.embeddings) == len(self.labels), (
            f"Embedding/label count mismatch: "
            f"{len(self.embeddings)} vs {len(self.labels)}"
        )
        logger.info(
            f"EmbeddingDataset: {len(self.embeddings):,} embeddings, "
            f"dim={self.embeddings.shape[1]}"
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.labels[idx]
