"""
WebDataset-backed loader (TPU path)
====================================

CUDA hosts use the high-throughput NVIDIA DALI pipeline in dataloader.py.
TPU hosts don't have DALI, so we fall back to torch.utils.data + the
``webdataset`` package over the same ``.tar`` shards produced by
``src/data/shard_manager.py``.

This loader matches the DALI iterator's output shape — each ``next(loader)``
yields ``[{'data': Tensor[B,3,H,W], 'label': Tensor[B]}]`` — so the rest of
the training code (loops.py, trainer.py) stays mode-agnostic.

The .tar layout is the one written by the team's existing sharder:

    sample123.jpg     # raw JPEG bytes
    sample123.cls     # 4-byte little-endian int32 label (matches DALI's
                      # `fn.reinterpret(labels, dtype=types.INT32)` step)
"""

from __future__ import annotations

import glob
import io
import os
import struct
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

try:
    import webdataset as wds
    HAS_WDS = True
except ImportError:
    HAS_WDS = False

# Sentinel logger
try:
    import logging
    logger = logging.getLogger(__name__)
except Exception:  # pragma: no cover
    logger = None


# ── ImageNet normalisation ──────────────────────────────────────────────────
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_transforms(resolution: int, training: bool):
    """Match the DALI pipeline's pre-processing as closely as torchvision can.

    DALI does: decode → Resize(LANCZOS3) → optional flip → CropMirrorNormalize.
    We do:    decode → RandomResizedCrop or Resize+CenterCrop → flip → ToTensor → Normalize.
    """
    from torchvision import transforms  # imported lazily so a non-torchvision env still imports this module
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                resolution,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(
            int(resolution * 256 / 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _decode_cls_bytes(raw: bytes) -> int:
    """Reinterpret the 4-byte .cls block as a little-endian int32 — same
    semantics as DALI's `fn.reinterpret(labels, dtype=types.INT32)`."""
    return struct.unpack("<i", raw[:4])[0]


def _sample_to_pair(sample: dict, transform):
    """Pull ('jpg', 'cls') out of a webdataset sample and tensorise."""
    from PIL import Image
    img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
    img_t = transform(img)
    label = _decode_cls_bytes(sample["cls"])
    return img_t, label


# ── DALI-shape adapter ──────────────────────────────────────────────────────
class _DaliShapeAdapter:
    """Wrap a torch DataLoader so its iterator yields
    ``[{'data': ..., 'label': ...}]`` — the shape the rest of ORACLE expects
    from the DALI iterator.

    Also exposes ``reset()`` for parity with DALIGenericIterator (no-op here;
    webdataset auto-cycles per epoch).
    """

    def __init__(self, loader: DataLoader, length: Optional[int] = None):
        self._loader = loader
        self._length = length

    def __iter__(self):
        for images, labels in self._loader:
            yield [{"data": images, "label": labels}]

    def __len__(self):
        if self._length is not None:
            return self._length
        try:
            return len(self._loader)
        except TypeError:
            # webdataset pipelines are iterable but generally not Sized
            return 0

    def reset(self):  # noqa: D401 — parity with DALIGenericIterator
        """No-op: webdataset auto-restarts each epoch."""
        return None


# ── Public entry point ─────────────────────────────────────────────────────
def get_webdataset_loaders(
    batch_size: int,
    resolution: int,
    *,
    shard_path: str,
    num_shards: int = 1,
    shard_id: int = 0,
    seed: int = 42,
    num_workers: int = 8,
    train_prefix: str = "train_",
    val_prefix: str = "val_",
    min_shard_bytes: int = 1024 * 1024,
    shuffle_buffer: int = 1000,
) -> Tuple[_DaliShapeAdapter, _DaliShapeAdapter, Optional[int], None]:
    """Build train/val loaders that mimic the DALI iterator's surface.

    Returns ``(train, val, num_classes, _)`` where the last slot is reserved
    for API parity with ``get_dali_loaders`` (DALI returns an indexer there).

    ``num_classes`` is returned as ``None`` because WebDataset shards don't
    carry a class index — the caller is expected to know the taxonomy size
    (it's a fixed 7806 for PlantCLEF 2026 and stored in config).
    """
    if not HAS_WDS:
        raise ImportError(
            "The 'webdataset' package is required for the TPU data path. "
            "Install it via `pip install webdataset`. "
            "Alternatively, run with --mode cuda to use the DALI pipeline."
        )

    train_shards = sorted(glob.glob(os.path.join(shard_path, f"{train_prefix}*.tar")))
    val_shards   = sorted(glob.glob(os.path.join(shard_path, f"{val_prefix}*.tar")))

    # Filter sub-1 MB cruft (matches DALI loader's defensive check).
    train_shards = [f for f in train_shards if os.path.getsize(f) > min_shard_bytes]
    val_shards   = [f for f in val_shards   if os.path.getsize(f) > min_shard_bytes]

    if not train_shards:
        raise FileNotFoundError(
            f"No training shards matching '{train_prefix}*.tar' (>{min_shard_bytes//1024} KB) "
            f"found in {shard_path}. Build shards with src/data/shard_manager.py first."
        )

    # DDP / multi-host sharding (the inner shuffle still gives per-rank
    # randomisation, but cross-rank we want disjoint shard sets).
    if num_shards > 1:
        train_shards = [train_shards[i] for i in range(shard_id, len(train_shards), num_shards)]

    train_transform = _build_transforms(resolution, training=True)
    val_transform   = _build_transforms(resolution, training=False)

    def _make_pipeline(shards: List[str], transform, training: bool):
        ds = wds.WebDataset(
            shards,
            shardshuffle=training,
            empty_check=False,
            nodesplitter=wds.split_by_node,   # multi-host TPU pods
            handler=wds.handlers.warn_and_continue,
        )
        if training and shuffle_buffer > 0:
            ds = ds.shuffle(shuffle_buffer)
        # Decode + tensorise inside the worker so DataLoader can parallelise
        ds = ds.map(lambda s: _sample_to_pair(s, transform))
        return ds

    train_ds = _make_pipeline(train_shards, train_transform, training=True)
    val_ds   = _make_pipeline(val_shards,   val_transform,   training=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=False,  # TPU pinning is handled by xla_multiprocessing; safe default for CPU dev too
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=max(1, num_workers // 2),
        pin_memory=False,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    if logger:
        logger.info(
            f"[WDS Loader] {len(train_shards)} train shards, {len(val_shards)} val shards "
            f"(rank {shard_id}/{num_shards}, resolution={resolution})."
        )

    return _DaliShapeAdapter(train_loader), _DaliShapeAdapter(val_loader), None, None
