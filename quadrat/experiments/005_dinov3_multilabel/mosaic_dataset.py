"""
Synthetic-mosaic dataset for Phase 2 multi-label fine-tuning.

Each sample composes K single-plant crops from K *distinct* species into one
canvas, producing a K-hot multi-label target. K is sampled from a fixed
distribution intended to match the typical species density of a quadrat.

Single-image decoding/augmentation runs on CPU via PIL + torchvision; mosaic
composition is a thin Python wrapper around per-image crops. For training
throughput, swap in DALI by feeding K decoded crops through the same
`compose_mosaic()` function on the GPU.

Targets are returned as float32 K-hot tensors of length n_classes, ready for
sigmoid + AsymmetricLoss.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# Default K distribution: P(K=1..5).
# Heavy on small K because real quadrats are mostly 1-3 species.
DEFAULT_K_DIST: tuple[float, ...] = (0.30, 0.30, 0.20, 0.12, 0.08)


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------

def compose_mosaic(
    crops: Sequence[Image.Image],
    canvas_size: int,
    rng: random.Random,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Place K crops onto a canvas using a simple grid + jitter strategy.

    For K==1 we just resize the single crop to the canvas to keep the simple
    single-plant signal intact (matches the linear-probe domain).

    For K>=2 we tile a roughly square grid (ceil(sqrt(K)) cells per side),
    resize each crop to the cell size, and paste in random cell order. This
    avoids having to do real collision-aware non-overlapping placement, which
    was overkill for a Phase-2 augmentation.
    """
    canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    K = len(crops)
    if K == 0:
        return canvas

    if K == 1:
        return crops[0].resize((canvas_size, canvas_size), Image.BILINEAR)

    # Grid of size g x g where g = ceil(sqrt(K))
    g = int(np.ceil(np.sqrt(K)))
    cell = canvas_size // g

    # Assign crops to randomly chosen cells
    cells = list(range(g * g))
    rng.shuffle(cells)
    for crop, cell_idx in zip(crops, cells[:K]):
        cy, cx = divmod(cell_idx, g)
        # Slight random jitter inside the cell so the model does not lock onto a grid prior.
        max_jitter = max(0, cell // 8)
        jx = rng.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
        jy = rng.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
        x0 = max(0, min(canvas_size - cell, cx * cell + jx))
        y0 = max(0, min(canvas_size - cell, cy * cell + jy))
        resized = crop.resize((cell, cell), Image.BILINEAR)
        canvas.paste(resized, (x0, y0))

    return canvas


def random_crop_pil(
    image: Image.Image,
    rng: random.Random,
    min_scale: float = 0.6,
    max_scale: float = 1.0,
) -> Image.Image:
    """Random square crop with scale in [min_scale, max_scale] of min side."""
    w, h = image.size
    short = min(w, h)
    s = rng.uniform(min_scale, max_scale)
    side = max(1, int(short * s))
    x0 = rng.randint(0, max(0, w - side))
    y0 = rng.randint(0, max(0, h - side))
    return image.crop((x0, y0, x0 + side, y0 + side))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MosaicDataset(Dataset):
    """
    Synthetic K-plant mosaic dataset.

    Parameters
    ----------
    metadata_df       : DataFrame with columns ['resolved_path', 'species_id'].
                        Rows with missing 'resolved_path' are dropped.
    species_ids       : ordered list of all competition species IDs (length n_classes).
                        Determines the position of each species in the K-hot label.
    canvas_size       : output canvas side length in pixels (must be a multiple of 16).
    k_dist            : tuple of probabilities for K=1..len(k_dist).
    samples_per_epoch : number of synthetic samples per epoch.
    transform         : torchvision-style transform applied to the composed PIL canvas.
                        Should produce an ImageNet-normalised float tensor.
    seed              : RNG seed (worker-aware: each worker uses seed + worker_id).
    augment           : if True, apply random crop + horizontal flip per sub-image.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        species_ids: Sequence[str],
        canvas_size: int = 384,
        k_dist: Sequence[float] = DEFAULT_K_DIST,
        samples_per_epoch: int = 1_000_000,
        transform: Optional[callable] = None,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        if canvas_size % 16 != 0:
            raise ValueError(f"canvas_size must be a multiple of 16, got {canvas_size}")
        if "resolved_path" not in metadata_df.columns or "species_id" not in metadata_df.columns:
            raise ValueError("metadata_df must contain 'resolved_path' and 'species_id'")

        df = metadata_df[metadata_df["resolved_path"].notna()].copy()
        df["species_id"] = df["species_id"].astype(str)

        self.species_ids: list[str] = list(species_ids)
        self.species_to_idx: dict[str, int] = {sid: i for i, sid in enumerate(self.species_ids)}

        # Drop training rows whose species is not in the official species list
        before = len(df)
        df = df[df["species_id"].isin(self.species_to_idx)]
        if len(df) < before:
            logger.warning(
                f"MosaicDataset: dropped {before - len(df):,} rows for species "
                "not in the species_ids list"
            )

        # Index rows by species for fast K-distinct sampling
        self._rows_by_species: dict[str, list[str]] = (
            df.groupby("species_id")["resolved_path"].apply(list).to_dict()
        )
        self._species_with_data: list[str] = list(self._rows_by_species.keys())
        if not self._species_with_data:
            raise ValueError("MosaicDataset: no species have any images after filtering")

        self.canvas_size = canvas_size
        self.k_dist = np.array(k_dist, dtype=np.float64)
        self.k_dist /= self.k_dist.sum()
        self.k_values = np.arange(1, len(k_dist) + 1)
        self.samples_per_epoch = int(samples_per_epoch)
        self.transform = transform
        self.seed = int(seed)
        self.augment = bool(augment)

        logger.info(
            f"MosaicDataset ready: {len(df):,} source images, "
            f"{len(self._species_with_data):,} species with data, "
            f"{self.samples_per_epoch:,} samples/epoch, "
            f"canvas={canvas_size}, K-dist={tuple(round(p, 3) for p in self.k_dist)}"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng_for(self, idx: int) -> random.Random:
        # Worker-aware: torch DataLoader sets a base seed per worker, but we
        # also want determinism per (seed, idx) so repeated runs match.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        return random.Random((self.seed * 1_000_003) ^ (worker_id * 9176) ^ idx)

    def _sample_k(self, rng: random.Random) -> int:
        # numpy choice is slow per-sample; do a manual roulette with stdlib random
        u = rng.random()
        acc = 0.0
        for k, p in zip(self.k_values, self.k_dist):
            acc += p
            if u <= acc:
                return int(k)
        return int(self.k_values[-1])

    def _sample_image(self, species_id: str, rng: random.Random) -> Optional[Image.Image]:
        paths = self._rows_by_species.get(species_id, [])
        if not paths:
            return None
        path = paths[rng.randrange(len(paths))]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, Image.UnidentifiedImageError) as exc:
            logger.warning(f"MosaicDataset: failed to open {path}: {exc}")
            return None
        if self.augment:
            img = random_crop_pil(img, rng)
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._rng_for(idx)
        K = self._sample_k(rng)

        # Sample K distinct species (with retry if a species draws no openable image)
        species_pool = self._species_with_data
        species_choices = rng.sample(species_pool, k=min(K, len(species_pool)))

        crops: list[Image.Image] = []
        chosen_species: list[str] = []
        for sid in species_choices:
            img = self._sample_image(sid, rng)
            if img is not None:
                crops.append(img)
                chosen_species.append(sid)
            if len(crops) >= K:
                break

        # Top up if image opens failed
        attempts = 0
        while len(crops) < K and attempts < 10:
            sid = species_pool[rng.randrange(len(species_pool))]
            if sid in chosen_species:
                attempts += 1
                continue
            img = self._sample_image(sid, rng)
            if img is not None:
                crops.append(img)
                chosen_species.append(sid)
            attempts += 1

        if not crops:
            # Fallback: blank canvas, all-zero label. Should be vanishingly rare.
            canvas = Image.new("RGB", (self.canvas_size, self.canvas_size))
            label = torch.zeros(len(self.species_ids), dtype=torch.float32)
        else:
            canvas = compose_mosaic(crops, self.canvas_size, rng)
            label = torch.zeros(len(self.species_ids), dtype=torch.float32)
            for sid in chosen_species:
                label[self.species_to_idx[sid]] = 1.0

        if self.transform is not None:
            tensor = self.transform(canvas)
        else:
            tensor = torch.from_numpy(np.asarray(canvas)).permute(2, 0, 1).float() / 255.0

        return tensor, label


# ---------------------------------------------------------------------------
# Single-label dataset (Phase 1 feature caching)
# ---------------------------------------------------------------------------

class SinglePlantDataset(Dataset):
    """
    Plain single-plant dataset for Phase 1 feature caching.

    Returns (image_tensor, class_idx, resolved_path) tuples.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        species_ids: Sequence[str],
        transform: Optional[callable] = None,
    ) -> None:
        if "resolved_path" not in metadata_df.columns or "species_id" not in metadata_df.columns:
            raise ValueError("metadata_df must contain 'resolved_path' and 'species_id'")

        df = metadata_df[metadata_df["resolved_path"].notna()].copy()
        df["species_id"] = df["species_id"].astype(str)

        self.species_ids = list(species_ids)
        self.species_to_idx = {sid: i for i, sid in enumerate(self.species_ids)}

        before = len(df)
        df = df[df["species_id"].isin(self.species_to_idx)].reset_index(drop=True)
        if len(df) < before:
            logger.warning(
                f"SinglePlantDataset: dropped {before - len(df):,} rows for species "
                "not in the species_ids list"
            )

        self._paths: list[str] = df["resolved_path"].tolist()
        self._labels: list[int] = [self.species_to_idx[s] for s in df["species_id"]]
        self.transform = transform

        logger.info(
            f"SinglePlantDataset ready: {len(self._paths):,} images, "
            f"{len(set(self._labels)):,} species with data"
        )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        path = self._paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            # Return a blank image with label -1 to signal "skip me" upstream.
            img = Image.new("RGB", (16, 16))
            tensor = self.transform(img) if self.transform else torch.zeros(3, 16, 16)
            return tensor, -1, path
        tensor = self.transform(img) if self.transform else torch.from_numpy(
            np.asarray(img)
        ).permute(2, 0, 1).float() / 255.0
        return tensor, self._labels[idx], path


# ---------------------------------------------------------------------------
# Visualisation utility (for verification step #2 of the plan)
# ---------------------------------------------------------------------------

def visualize_samples(dataset: MosaicDataset, n: int = 5, out_dir: str = "outputs/mosaic_preview") -> None:
    """
    Save the first N samples and their K-hot label summary to disk for sanity checks.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        tensor, label = dataset[i]
        # Undo ImageNet normalization for preview
        if tensor.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            preview = (tensor * std + mean).clamp(0, 1)
            arr = (preview.permute(1, 2, 0).numpy() * 255).astype("uint8")
            Image.fromarray(arr).save(out / f"sample_{i:02d}.png")
        active = (label > 0.5).nonzero(as_tuple=True)[0].tolist()
        species = [dataset.species_ids[j] for j in active]
        logger.info(f"sample {i}: K={len(active)} species={species}")
