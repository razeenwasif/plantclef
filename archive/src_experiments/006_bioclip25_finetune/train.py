"""
Training: BioCLIP 2.5 backbone + linear head, with optional partial fine-tuning.

Modes
-----
  Linear probe (default, --unfreeze-blocks 0):
      Backbone fully frozen; only the linear head trains.

  Partial fine-tuning (--unfreeze-blocks N):
      Last N transformer blocks + ln_post/proj unfrozen.
      Two-group optimizer: backbone at (lr × backbone-lr-scale), head at lr.

Training variants
-----------------
  Single GPU:
      python train.py [args]

  Multi-GPU via torchrun:
      torchrun --nproc_per_node=2 train.py [args]

  Cached embeddings (fastest - run build_cache.py first; linear probe only):
      python train.py --use-cache --cache-dir ./cache [args]

Checkpoint format (.pt file)
-----------------------------
  epoch, model_state_dict (DDP-unwrapped), optimizer_state_dict,
  scheduler_state_dict, scaler_state_dict, metrics, config, idx_to_species
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import (
    DEFAULT_TRAIN_META_CSV,
    DEFAULT_TRAIN_IMAGE_ROOT,
    load_train_metadata,
    resolve_image_paths,
    build_class_mapping,
    save_class_mapping,
    build_val_split,
    PlantCLEFDataset,
    EmbeddingDataset,
)
from model import BioCLIP25LinearProbe, BIOCLIP25_MODEL_NAME
from transforms import bioclip_train_transform, bioclip_val_transform
from utils import (
    setup_logging,
    resolve_device,
    save_checkpoint,
    load_checkpoint,
    compute_recall_at_k,
    topk_predictions,
    save_json,
    is_main_process,
    get_rank,
    get_world_size,
    all_reduce_mean,
    barrier,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "./outputs/train"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BioCLIP 2.5 training (linear probe or partial fine-tune).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--train-meta-csv", default=DEFAULT_TRAIN_META_CSV)
    p.add_argument("--train-image-root", default=DEFAULT_TRAIN_IMAGE_ROOT)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--val-seed", type=int, default=42)
    p.add_argument(
        "--max-samples-per-class", type=int, default=0,
        help="Cap training samples per species (0 = no cap). Useful for fast debug.",
    )
    p.add_argument("--img-size", type=int, default=224)

    # Embedding cache mode (skip backbone forward during training)
    p.add_argument(
        "--use-cache", action="store_true",
        help="Train head-only on pre-computed embeddings from build_cache.py. "
             "Incompatible with --unfreeze-blocks.",
    )
    p.add_argument("--cache-dir", default="./cache")

    # Model
    p.add_argument("--model-name", default=BIOCLIP25_MODEL_NAME)

    # Partial fine-tuning
    p.add_argument(
        "--unfreeze-blocks", type=int, default=0,
        metavar="N",
        help="Number of trailing ViT transformer blocks to unfreeze for "
             "partial fine-tuning (0 = fully frozen linear probe). "
             "BioCLIP 2.5 ViT-H/14 has 32 blocks total.",
    )
    p.add_argument(
        "--backbone-lr-scale", type=float, default=0.1,
        metavar="SCALE",
        help="LR multiplier for unfrozen backbone params relative to head LR. "
             "Effective backbone LR = lr × backbone-lr-scale.",
    )

    # Optimiser
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=1)

    # AMP
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")

    # DataLoader
    p.add_argument("--num-workers", type=int, default=8)

    # Checkpointing / output
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")

    # Validation
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-top-n", type=int, default=20)
    p.add_argument("--log-every", type=int, default=50)

    # Device (single-GPU only; multi-GPU uses LOCAL_RANK env var)
    p.add_argument("--device", default="auto")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation (rank-0 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: BioCLIP25LinearProbe,
    loader: DataLoader,
    idx_to_species: list[str],
    device: str,
    use_cache: bool = False,
    top_n: int = 20,
    amp: bool = True,
) -> dict:
    model.eval()
    results: list[dict] = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if use_cache:
            embeddings, labels = batch
            embeddings = embeddings.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model.head(embeddings)
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=amp):
                logits = model(images)

        loss = nn.functional.cross_entropy(logits.float(), labels)
        total_loss += loss.item()
        n_batches += 1

        for i in range(logits.size(0)):
            pred_species, _ = topk_predictions(
                logits[i].float().cpu(), idx_to_species, top_n=top_n
            )
            gt_species = idx_to_species[labels[i].item()]
            results.append({"gt_species_id": gt_species, "pred_ids": pred_species})

    metrics = compute_recall_at_k(results, k_values=(1, 5, 10, 20))
    metrics["val_loss"]      = round(total_loss / max(n_batches, 1), 4)
    metrics["top1_accuracy"] = metrics.get("recall_at_1", 0.0)
    metrics["n_val_images"]  = len(results)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    use_amp = not args.no_amp

    if args.use_cache and args.unfreeze_blocks > 0:
        sys.exit(
            "ERROR: --use-cache and --unfreeze-blocks are incompatible. "
            "Cache mode skips the backbone forward pass, so unfreezing blocks "
            "has no effect. Run without --use-cache to fine-tune the backbone."
        )

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    is_ddp = "RANK" in os.environ and "LOCAL_RANK" in os.environ
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
    else:
        device = resolve_device(args.device)
        rank = 0

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"

    setup_logging(output_dir=str(out_dir), rank=rank)

    if is_main_process():
        logger.info("=" * 60)
        logger.info("BioCLIP 2.5 Training")
        logger.info(f"  model_name       : {args.model_name}")
        logger.info(f"  epochs           : {args.epochs}")
        logger.info(f"  batch_size       : {args.batch_size} (per GPU)")
        logger.info(f"  lr               : {args.lr}")
        logger.info(f"  unfreeze_blocks  : {args.unfreeze_blocks}")
        logger.info(f"  backbone_lr_scale: {args.backbone_lr_scale}")
        logger.info(f"  use_cache        : {args.use_cache}")
        logger.info(f"  amp              : {use_amp}")
        logger.info(f"  ddp              : {is_ddp}  world_size={get_world_size()}")
        logger.info(f"  device           : {device}")
        logger.info(f"  output_dir       : {out_dir}")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.use_cache:
        cache_dir = Path(args.cache_dir)
        from dataset import load_class_mapping
        species_to_idx, idx_to_species = load_class_mapping(
            str(cache_dir / "class_mapping.txt")
        )
        num_classes = len(idx_to_species)
        train_dataset = EmbeddingDataset(
            str(cache_dir / "train_embeddings.pt"),
            str(cache_dir / "train_labels.pt"),
        )
        val_dataset = EmbeddingDataset(
            str(cache_dir / "val_embeddings.pt"),
            str(cache_dir / "val_labels.pt"),
        )
    else:
        if is_main_process():
            logger.info("Loading training metadata ...")
        df = load_train_metadata(args.train_meta_csv)
        df = resolve_image_paths(df, args.train_image_root, verify=False)
        df = df[df["resolved_path"].notna()].reset_index(drop=True)

        species_to_idx, idx_to_species = build_class_mapping(df)
        num_classes = len(idx_to_species)
        class_map_path = str(out_dir / "class_mapping.txt")

        if is_main_process():
            save_class_mapping(idx_to_species, class_map_path)
            logger.info(f"  {num_classes:,} classes")

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
            if is_main_process():
                logger.info(
                    f"Capped at {args.max_samples_per_class}/class → "
                    f"{len(train_df):,} training images"
                )

        train_transform = bioclip_train_transform(img_size=args.img_size)
        val_transform   = bioclip_val_transform(img_size=args.img_size)

        train_dataset = PlantCLEFDataset(
            df=train_df, species_to_idx=species_to_idx, transform=train_transform
        )
        val_dataset = PlantCLEFDataset(
            df=val_df, species_to_idx=species_to_idx, transform=val_transform
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    if is_ddp:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.startswith("cuda")),
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    if is_main_process():
        logger.info(
            f"Train: {len(train_dataset):,} samples  {len(train_loader):,} batches/epoch"
        )
        logger.info(f"Val  : {len(val_dataset):,} samples")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.use_cache:
        embed_dim = train_dataset[0][0].shape[0]
        if is_main_process():
            logger.info(f"Cache mode: embed_dim={embed_dim}, training head-only")

        class _HeadOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(embed_dim, num_classes)
            def forward(self, x):
                return self.head(x)

        model = _HeadOnly()
    else:
        model = BioCLIP25LinearProbe(
            num_classes=num_classes,
            model_name=args.model_name,
            unfreeze_blocks=args.unfreeze_blocks,
        )

    model = model.to(device)

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters=False works for ViT (all blocks always used).
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ------------------------------------------------------------------
    # Optimiser - two param groups when backbone blocks are unfrozen
    # ------------------------------------------------------------------
    raw_model = model.module if is_ddp else model
    head_params = list(raw_model.head.parameters())

    if not args.use_cache and args.unfreeze_blocks > 0:
        backbone_trainable = [
            p for p in raw_model.backbone.parameters() if p.requires_grad
        ]
        param_groups = [
            {
                "params":       backbone_trainable,
                "lr":           args.lr * args.backbone_lr_scale,
                "name":         "backbone",
            },
            {
                "params":       head_params,
                "lr":           args.lr,
                "name":         "head",
            },
        ]
        if is_main_process():
            n_bb = sum(p.numel() for p in backbone_trainable)
            n_hd = sum(p.numel() for p in head_params)
            logger.info(
                f"Optimizer: two groups - "
                f"backbone {n_bb:,} params @ lr={args.lr * args.backbone_lr_scale:.2e}  "
                f"head {n_hd:,} params @ lr={args.lr:.2e}"
            )
    else:
        param_groups = head_params
        if is_main_process():
            n_hd = sum(p.numel() for p in head_params)
            logger.info(f"Optimizer: head-only - {n_hd:,} params @ lr={args.lr:.2e}")

    optimizer = optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = GradScaler(enabled=use_amp and device.startswith("cuda"))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    history: list[dict] = []
    best_recall_at_5 = 0.0

    if args.resume:
        start_epoch, prev_metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device=device
        )
        best_recall_at_5 = prev_metrics.get("recall_at_5", 0.0)
        if is_main_process():
            logger.info(
                f"Resuming from epoch {start_epoch}  best_r@5={best_recall_at_5:.4f}"
            )

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    if is_main_process():
        config = {
            **vars(args),
            "device":     device,
            "num_classes": num_classes,
            "n_train":    len(train_dataset),
            "n_val":      len(val_dataset),
            "world_size": get_world_size(),
            "amp":        use_amp,
        }
        if not args.use_cache:
            config["class_mapping_path"] = class_map_path
        save_json(config, str(out_dir / "train_config.json"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    barrier()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    t_total = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_t = time.perf_counter()

        for step, batch in enumerate(train_loader):
            if args.use_cache:
                embeddings, labels = batch
                embeddings = embeddings.to(device, non_blocking=True)
                labels     = labels.to(device, non_blocking=True)
                with autocast(enabled=use_amp and device.startswith("cuda")):
                    logits = raw_model.head(embeddings)
                    loss   = criterion(logits, labels)
            else:
                images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=use_amp and device.startswith("cuda")):
                    logits = model(images)
                    loss   = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches  += 1

            if is_main_process() and (step + 1) % args.log_every == 0:
                avg = epoch_loss / n_batches
                elapsed = time.perf_counter() - epoch_t
                # Log LR for each param group
                lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
                lr_str = "/".join(lrs)
                logger.info(
                    f"[Epoch {epoch+1}/{args.epochs}]  "
                    f"step {step+1}/{len(train_loader)}  "
                    f"loss={avg:.4f}  lr={lr_str}  "
                    f"elapsed={elapsed:.0f}s"
                )

        scheduler.step()

        avg_train_loss_t = torch.tensor(
            epoch_loss / max(n_batches, 1), device=device
        )
        avg_train_loss = all_reduce_mean(avg_train_loss_t).item()

        epoch_secs = time.perf_counter() - epoch_t
        current_lr = optimizer.param_groups[-1]["lr"]  # head LR

        if is_main_process():
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"lr={current_lr:.2e}  "
                f"time={epoch_secs:.0f}s"
            )

        # ------------------------------------------------------------------
        # Validation (rank 0 only)
        # ------------------------------------------------------------------
        val_metrics: dict = {}
        if is_main_process() and (epoch + 1) % args.val_every == 0 and len(val_dataset) > 0:
            logger.info("Running validation ...")
            raw_model_for_val = model.module if is_ddp else model
            val_metrics = validate(
                model=raw_model_for_val,
                loader=val_loader,
                idx_to_species=idx_to_species,
                device=device,
                use_cache=args.use_cache,
                top_n=args.val_top_n,
                amp=use_amp,
            )
            logger.info(
                f"  val_loss={val_metrics['val_loss']:.4f}  "
                f"recall@1={val_metrics.get('recall_at_1', 0):.4f}  "
                f"recall@5={val_metrics.get('recall_at_5', 0):.4f}  "
                f"recall@20={val_metrics.get('recall_at_20', 0):.4f}  "
                f"n={val_metrics.get('n_val_images', 0):,}"
            )

        # ------------------------------------------------------------------
        # Checkpoint (rank 0 only)
        # ------------------------------------------------------------------
        if is_main_process():
            entry = {
                "epoch":      epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "lr":         round(current_lr, 8),
                "epoch_secs": round(epoch_secs, 1),
                **val_metrics,
            }
            history.append(entry)
            save_json({"history": history}, str(out_dir / "train_history.json"))

            is_best = val_metrics.get("recall_at_5", 0.0) > best_recall_at_5
            if is_best:
                best_recall_at_5 = val_metrics["recall_at_5"]
                logger.info(f"  New best recall@5: {best_recall_at_5:.4f}")

            raw_model_for_save = model.module if is_ddp else model
            if args.use_cache:
                ckpt_state = {
                    "epoch":                epoch,
                    "head_state_dict":      raw_model_for_save.head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict":    scaler.state_dict(),
                    "metrics":              entry,
                    "config":               config,
                    "idx_to_species":       idx_to_species,
                }
            else:
                ckpt_state = {
                    "epoch":                epoch,
                    "model_state_dict":     raw_model_for_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict":    scaler.state_dict(),
                    "metrics":              entry,
                    "config":               config,
                    "idx_to_species":       idx_to_species,
                }

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    state=ckpt_state,
                    path=str(ckpt_dir / f"epoch_{epoch+1:03d}.pt"),
                    is_best=is_best,
                    best_path=str(ckpt_dir / "best.pt"),
                )
            save_checkpoint(
                state=ckpt_state,
                path=str(ckpt_dir / "latest.pt"),
            )

        barrier()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    if is_main_process():
        total_secs = time.perf_counter() - t_total
        logger.info(
            f"Training complete: {args.epochs} epochs in {total_secs / 60:.1f} min"
        )
        logger.info(f"Best recall@5 on validation: {best_recall_at_5:.4f}")
        logger.info(f"Best checkpoint: {ckpt_dir}/best.pt")

    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
