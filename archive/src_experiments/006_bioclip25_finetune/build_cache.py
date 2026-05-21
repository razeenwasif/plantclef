"""
Pre-compute and cache BioCLIP 2.5 backbone embeddings for all training images.

Why use a cache?
----------------
The backbone is fully frozen during linear-probe training.  Re-running the
ViT-H/14 forward pass on every training image every epoch wastes GPU compute.
Instead, run the backbone once, save all embeddings to disk, and train the
linear head on the cached tensors.  This typically yields a 10-50x speedup
per training epoch at the cost of ~7 GB disk space.

Outputs (all written to --cache-dir)
--------------------------------------
  train_embeddings.pt   float32 tensor (N_train, embed_dim)
  train_labels.pt       int64  tensor  (N_train,)
  val_embeddings.pt     float32 tensor (N_val,   embed_dim)
  val_labels.pt         int64  tensor  (N_val,)
  class_mapping.txt     one species_id per line (idx → species_id)
  cache_meta.json       config and stats

Usage
-----
  python build_cache.py \\
      --output-dir ./cache \\
      --batch-size 128

  # Then train with:
  python train.py --use-cache --cache-dir ./cache --epochs 30
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from dataset import (
    DEFAULT_TRAIN_META_CSV,
    DEFAULT_TRAIN_IMAGE_ROOT,
    load_train_metadata,
    resolve_image_paths,
    build_class_mapping,
    save_class_mapping,
    build_val_split,
    PlantCLEFDataset,
)
from model import BioCLIP25LinearProbe, BIOCLIP25_MODEL_NAME
from transforms import bioclip_val_transform
from utils import setup_logging, resolve_device, save_json

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "./cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute BioCLIP 2.5 embeddings for all training images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-meta-csv",   default=DEFAULT_TRAIN_META_CSV)
    p.add_argument("--train-image-root", default=DEFAULT_TRAIN_IMAGE_ROOT)
    p.add_argument("--model-name",       default=BIOCLIP25_MODEL_NAME)
    p.add_argument("--output-dir",       default=DEFAULT_CACHE_DIR)
    p.add_argument("--val-fraction",     type=float, default=0.1)
    p.add_argument("--val-seed",         type=int,   default=42)
    p.add_argument("--img-size",         type=int,   default=224)
    p.add_argument("--batch-size",       type=int,   default=128)
    p.add_argument("--num-workers",      type=int,   default=8)
    p.add_argument("--device",           default="auto")
    p.add_argument(
        "--max-samples-per-class", type=int, default=0,
        help="Cap samples per species (0 = no cap). Useful for quick smoke test.",
    )
    return p.parse_args()


@torch.no_grad()
def encode_dataset(
    model: BioCLIP25LinearProbe,
    loader: DataLoader,
    device: str,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward all images through the frozen backbone and collect embeddings.

    Returns
    -------
    embeddings : float32 tensor (N, embed_dim)
    labels     : int64  tensor (N,)
    """
    all_embeddings: list[torch.Tensor] = []
    all_labels:     list[torch.Tensor] = []
    t_start = time.perf_counter()

    model.eval()
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with autocast(enabled=device.startswith("cuda")):
            embs = model.encode(images).float()   # (B, D), no grad, no norm
        all_embeddings.append(embs.cpu())
        all_labels.append(labels)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            done    = (i + 1) * loader.batch_size
            total   = len(loader.dataset)
            logger.info(
                f"  [{split_name}] {done:>8,}/{total:,}  {elapsed:.0f}s elapsed  "
                f"~{elapsed / (i + 1) * (len(loader) - i - 1):.0f}s remaining"
            )

    embeddings = torch.cat(all_embeddings, dim=0)
    labels     = torch.cat(all_labels,     dim=0).long()
    elapsed    = time.perf_counter() - t_start
    logger.info(
        f"  [{split_name}] Done: {len(embeddings):,} embeddings  "
        f"shape={tuple(embeddings.shape)}  "
        f"time={elapsed:.0f}s"
    )
    return embeddings, labels


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir=str(out_dir))

    logger.info("=" * 60)
    logger.info("BioCLIP 2.5 Embedding Cache Builder")
    logger.info(f"  model_name  : {args.model_name}")
    logger.info(f"  output_dir  : {out_dir}")
    logger.info(f"  batch_size  : {args.batch_size}")
    logger.info(f"  device      : {device}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading training metadata ...")
    df = load_train_metadata(args.train_meta_csv)
    df = resolve_image_paths(df, args.train_image_root, verify=False)
    df = df[df["resolved_path"].notna()].reset_index(drop=True)

    species_to_idx, idx_to_species = build_class_mapping(df)
    num_classes = len(idx_to_species)
    save_class_mapping(idx_to_species, str(out_dir / "class_mapping.txt"))

    train_df, val_df = build_val_split(
        df, val_fraction=args.val_fraction, seed=args.val_seed
    )

    if args.max_samples_per_class > 0:
        train_df = (
            train_df
            .groupby("species_id", sort=False)
            .head(args.max_samples_per_class)
            .reset_index(drop=True)
        )
        logger.info(
            f"Capped at {args.max_samples_per_class}/class → "
            f"{len(train_df):,} training images"
        )

    transform = bioclip_val_transform(img_size=args.img_size)

    train_dataset = PlantCLEFDataset(train_df, species_to_idx, transform)
    val_dataset   = PlantCLEFDataset(val_df,   species_to_idx, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Load model (backbone only needed for encoding)
    # ------------------------------------------------------------------
    logger.info("Loading BioCLIP 2.5 backbone ...")
    model = BioCLIP25LinearProbe(
        num_classes=num_classes, model_name=args.model_name
    ).to(device)

    logger.info(f"  embed_dim = {model.embed_dim}")
    logger.info(
        f"  backbone params: "
        f"{sum(p.numel() for p in model.backbone.parameters()):,}"
    )

    # ------------------------------------------------------------------
    # Encode train split
    # ------------------------------------------------------------------
    logger.info(f"Encoding train split ({len(train_dataset):,} images) ...")
    t0 = time.perf_counter()
    train_embs, train_labels = encode_dataset(model, train_loader, device, "train")
    torch.save(train_embs,   out_dir / "train_embeddings.pt")
    torch.save(train_labels, out_dir / "train_labels.pt")
    logger.info(f"Train embeddings saved  ({time.perf_counter()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # Encode val split
    # ------------------------------------------------------------------
    logger.info(f"Encoding val split ({len(val_dataset):,} images) ...")
    t0 = time.perf_counter()
    val_embs, val_labels = encode_dataset(model, val_loader, device, "val")
    torch.save(val_embs,   out_dir / "val_embeddings.pt")
    torch.save(val_labels, out_dir / "val_labels.pt")
    logger.info(f"Val embeddings saved  ({time.perf_counter()-t0:.0f}s)")

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    cache_meta = {
        "model_name":   args.model_name,
        "embed_dim":    int(model.embed_dim),
        "num_classes":  num_classes,
        "img_size":     args.img_size,
        "n_train":      int(len(train_embs)),
        "n_val":        int(len(val_embs)),
        "val_fraction": args.val_fraction,
        "val_seed":     args.val_seed,
        "train_meta_csv":   args.train_meta_csv,
        "train_image_root": args.train_image_root,
        "max_samples_per_class": args.max_samples_per_class,
        "train_emb_bytes": int(train_embs.numel() * train_embs.element_size()),
        "val_emb_bytes":   int(val_embs.numel()   * val_embs.element_size()),
    }
    save_json(cache_meta, str(out_dir / "cache_meta.json"))

    total_gb = (
        train_embs.numel() * train_embs.element_size()
        + val_embs.numel() * val_embs.element_size()
    ) / 1e9
    logger.info(f"Cache complete: {out_dir}  ({total_gb:.2f} GB on disk)")
    logger.info(
        f"  → train with:  python train.py --use-cache --cache-dir {out_dir} ..."
    )


if __name__ == "__main__":
    main()
