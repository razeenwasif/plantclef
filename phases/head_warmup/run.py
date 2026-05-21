"""Phase 2a — PCA Head Warmup entry point.

Trains WarmupHead on the Phase 1 feature cache.  No backbone dependency.
Multi-GPU supported via DDP; single-GPU also works.

Usage:
    torchrun --nproc_per_node=N -m phases.head_warmup.run \
        --config configs/p2a_warmup.yaml
    python -m phases.head_warmup.run --config configs/p2a_warmup.yaml
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from .config  import load, P2aConfig
from .dataset import CachedFeatureDataset
from .model   import WarmupHead

from src.training.losses import AsymmetricLoss


def _train(cfg: P2aConfig) -> None:
    multi_gpu = cfg.world_size > 1
    if multi_gpu and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))

    local_rank = cfg.local_rank
    is_master  = (cfg.rank == 0)
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset — 90/10 train/val split over the cached features
    full_ds = CachedFeatureDataset(cfg.feature_cache)
    val_len  = max(1, int(len(full_ds) * 0.05))
    train_ds = torch.utils.data.Subset(full_ds, range(val_len, len(full_ds)))
    val_ds   = torch.utils.data.Subset(full_ds, range(val_len))

    sampler = DistributedSampler(train_ds, num_replicas=cfg.world_size,
                                 rank=cfg.rank) if multi_gpu else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Loss — use genus-hierarchy loss if genus_ids file is present
    if os.path.exists(cfg.genus_ids):
        genus_ids = torch.load(cfg.genus_ids, map_location=device,
                               weights_only=True).int()[:cfg.num_classes]
        criterion = AsymmetricLoss(genus_ids=genus_ids, use_fused=False).to(device)
    else:
        criterion = AsymmetricLoss(use_fused=False).to(device)

    # Model
    model = WarmupHead(
        in_features     = cfg.in_features,
        hidden_features = cfg.hidden_features,
        out_features    = cfg.num_classes,
        dropout         = cfg.dropout,
    ).to(device)

    if multi_gpu:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    os.makedirs(os.path.dirname(os.path.abspath(cfg.checkpoint)), exist_ok=True)

    if is_master:
        print(f"[P2a] Training WarmupHead: {len(train_ds)} train / {len(val_ds)} val, "
              f"{cfg.epochs} epochs, lr={cfg.lr}")

    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        # Train
        model.train()
        train_correct = train_total = 0
        for feats, labels in train_loader:
            feats  = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feats)
            loss   = criterion(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()

        # Val (master only for speed — all GPUs see the same cached data)
        if is_master:
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats  = feats.to(device)
                    labels = labels.to(device)
                    logits = model(feats)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total   += labels.size(0)

            train_acc = train_correct / train_total * 100 if train_total else 0.0
            val_acc   = val_correct   / val_total   * 100 if val_total   else 0.0
            print(f"[P2a] Epoch {epoch+1}/{cfg.epochs}  "
                  f"train={train_acc:.1f}%  val={val_acc:.1f}%  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                raw = model.module if hasattr(model, "module") else model
                from src.training.checkpoints import atomic_torch_save
                atomic_torch_save(
                    {"model_state": raw.state_dict(), "acc": best_val_acc},
                    cfg.checkpoint,
                )
                print(f"[P2a]   → best val_acc={best_val_acc:.2f}%  saved {cfg.checkpoint}")

    if is_master:
        print(f"[P2a] Done. Best val accuracy: {best_val_acc:.2f}%")

    if multi_gpu:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume",     type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = load(args.config)
    _train(cfg)


if __name__ == "__main__":
    main()
