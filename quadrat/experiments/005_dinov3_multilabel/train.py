"""
Two-phase training for the DINOv3 multi-label PlantCLEF model.

Phase 1 (linear probe on cached features)
-----------------------------------------
Loads the .pt file produced by extract_features.py and trains the classifier
head only. Loss = LogitAdjustmentLoss (cross-entropy + class-prior shift).
Fast (~30 min on a single GPU once features are cached).

Phase 2 (LoRA fine-tune on synthetic mosaics)
---------------------------------------------
Loads the trained head, applies LoRA to the DINOv3 backbone, switches to
synthetic K-plant mosaics with K-hot multi-label targets, and fine-tunes
end-to-end with AsymmetricLoss + class-frequency logit adjustment.

Phase 1 outputs:  models/dinov3_v1/phase1_head.pth
Phase 2 outputs:  models/dinov3_v1/phase2_lora.pth

Run:
    cd src_experiments/005_dinov3_multilabel
    python train.py --phase 1 --feature-cache ../../models/dinov3_v1/phase1_feature_cache.pt
    python train.py --phase 2 \
        --metadata-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
        --image-root   /workspace/plantclef/raw/train/images_max_side_800 \
        --species-csv  .../species_lookup_with_gbif_cleaned_names.csv \
        --p1-head      ../../models/dinov3_v1/phase1_head.pth
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Sibling imports
from model import DINOv3MultiLabel, build_default_transform, DEFAULT_HUB_NAME
from mosaic_dataset import MosaicDataset, DEFAULT_K_DIST

# Reuse 004 metadata utilities
_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
if str(_FOUR) not in sys.path:
    sys.path.insert(0, str(_FOUR))
from dataset import (  # type: ignore  # noqa: E402
    load_train_metadata,
    resolve_image_paths,
    load_species_ids,
)

# Reuse losses from src/
_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from training.losses import (  # type: ignore  # noqa: E402
    AsymmetricLoss,
    LogitAdjustmentLoss,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv3 two-phase trainer.")
    p.add_argument("--phase", type=int, choices=[1, 2], required=True)

    # Common
    p.add_argument("--backbone-source", choices=["torch_hub", "timm"], default="torch_hub")
    p.add_argument("--backbone", default=DEFAULT_HUB_NAME)
    p.add_argument("--ssl-checkpoint", default=None,
                   help="Path to SSL backbone .pt for Phase-2 backbone init.")
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="../../models/dinov3_v1")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)

    # Phase 1
    p.add_argument("--feature-cache",
                   help="Path to .pt feature cache (Phase 1).")
    p.add_argument("--p1-epochs", type=int, default=10)
    p.add_argument("--p1-batch-size", type=int, default=4096)
    p.add_argument("--p1-lr", type=float, default=1e-3)
    p.add_argument("--p1-val-frac", type=float, default=0.05)

    # Phase 2
    p.add_argument("--metadata-csv",
                   help="Cleaned + verified single-plant metadata CSV (Phase 2).")
    p.add_argument("--image-root",
                   help="Root dir of training images (Phase 2).")
    p.add_argument("--species-csv",
                   help="Species lookup CSV (defines class index ordering).")
    p.add_argument("--p1-head",
                   help="Path to phase1_head.pth (loaded into Phase 2 head).")
    p.add_argument("--p2-epochs", type=int, default=5)
    p.add_argument("--p2-samples-per-epoch", type=int, default=1_000_000)
    p.add_argument("--p2-batch-size", type=int, default=64)
    p.add_argument("--p2-accum", type=int, default=8)
    p.add_argument("--p2-lr-lora", type=float, default=1e-4)
    p.add_argument("--p2-lr-head", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true",
                   help="Run forward/backward in bf16 autocast (recommended).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Class-prior helpers
# ---------------------------------------------------------------------------

def class_counts_from_labels(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Return (n_classes,) tensor of integer class frequencies."""
    counts = torch.zeros(n_classes, dtype=torch.long)
    bincount = torch.bincount(labels[labels >= 0], minlength=n_classes)
    counts[: len(bincount)] = bincount
    return counts


def class_counts_from_metadata(df, species_ids: list[str]) -> torch.Tensor:
    species_to_idx = {sid: i for i, sid in enumerate(species_ids)}
    c = Counter()
    for sid in df["species_id"].astype(str):
        if sid in species_to_idx:
            c[species_to_idx[sid]] += 1
    counts = torch.zeros(len(species_ids), dtype=torch.long)
    for i, n in c.items():
        counts[i] = n
    return counts


def logit_adjustments_from_counts(counts: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Laplacian-smoothed log-prior shift. Matches LogitAdjustmentLoss."""
    counts = counts.float()
    priors = (counts + 1) / (counts.sum() + len(counts))
    return tau * torch.log(priors)


# ---------------------------------------------------------------------------
# Phase 1: cached-feature linear probe
# ---------------------------------------------------------------------------

def run_phase1(args: argparse.Namespace) -> None:
    if not args.feature_cache:
        raise SystemExit("--feature-cache is required for --phase 1")

    cache_path = Path(args.feature_cache)
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")

    logger.info(f"Loading feature cache: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    features: torch.Tensor = cache["features"]            # (N, 2D) fp16
    labels: torch.Tensor = cache["labels"]                # (N,) int64
    species_ids: list[str] = cache["species_ids"]
    n_classes = len(species_ids)
    feat_dim = features.shape[1]
    logger.info(f"  features: {tuple(features.shape)} {features.dtype}, classes: {n_classes}")

    # Train/val split
    n = features.shape[0]
    n_val = max(1, int(n * args.p1_val_frac))
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(features[train_idx], labels[train_idx])
    val_ds = TensorDataset(features[val_idx], labels[val_idx])

    train_loader = DataLoader(
        train_ds, batch_size=args.p1_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.p1_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Build a head-only model wrapper (we don't need the backbone in Phase 1)
    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, 1024),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(1024, n_classes),
    ).to(args.device)

    # Loss: cross-entropy + logit adjustment from class priors
    counts = class_counts_from_labels(labels, n_classes)
    criterion = LogitAdjustmentLoss(counts.tolist(), tau=1.0).to(args.device)

    # Optimizer + OneCycle
    optim = torch.optim.AdamW(head.parameters(), lr=args.p1_lr, weight_decay=0.05)
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=args.p1_lr,
        epochs=args.p1_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    best_top1 = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / "phase1_head.pth"

    for epoch in range(1, args.p1_epochs + 1):
        head.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(args.device, non_blocking=True).float()
            yb = yb.to(args.device, non_blocking=True)
            logits = head(xb)
            loss = criterion(logits, yb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optim.step()
            sched.step()

            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            if step % args.log_every == 0:
                logger.info(
                    f"  P1 ep{epoch} step{step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}"
                )

        train_loss = running / max(n_seen, 1)
        top1, top5 = evaluate_top_k(head, val_loader, args.device)
        dur = time.time() - t0
        logger.info(
            f"P1 epoch {epoch}/{args.p1_epochs} | train_loss={train_loss:.4f} "
            f"| val_top1={top1:.4f} val_top5={top5:.4f} | {dur/60:.1f}m"
        )

        if top1 > best_top1:
            best_top1 = top1
            torch.save(
                {
                    "head_state": head.state_dict(),
                    "feat_dim": feat_dim,
                    "n_classes": n_classes,
                    "species_ids": species_ids,
                    "backbone": cache.get("backbone", args.backbone),
                    "val_top1": top1,
                    "val_top5": top5,
                    "epoch": epoch,
                },
                head_path,
            )
            logger.info(f"  saved best head -> {head_path} (val_top1={top1:.4f})")

    logger.info(f"Phase 1 complete. best val_top1={best_top1:.4f}, head at {head_path}")


@torch.no_grad()
def evaluate_top_k(
    head: nn.Module,
    loader: DataLoader,
    device: str,
    k_values: tuple[int, ...] = (1, 5),
) -> tuple[float, float]:
    head.eval()
    correct = {k: 0 for k in k_values}
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True)
        logits = head(xb)
        topk = logits.topk(max(k_values), dim=1).indices  # (B, K)
        for k in k_values:
            correct[k] += (topk[:, :k] == yb.unsqueeze(1)).any(dim=1).sum().item()
        total += yb.size(0)
    return correct[1] / max(total, 1), correct[5] / max(total, 1)


# ---------------------------------------------------------------------------
# Phase 2: LoRA on mosaics
# ---------------------------------------------------------------------------

def run_phase2(args: argparse.Namespace) -> None:
    for required in ("metadata_csv", "image_root", "species_csv"):
        if getattr(args, required) is None:
            raise SystemExit(f"--{required.replace('_', '-')} is required for --phase 2")

    species_ids = load_species_ids(args.species_csv)
    n_classes = len(species_ids)
    logger.info(f"Phase 2: {n_classes} classes")

    df = load_train_metadata(args.metadata_csv)
    df = resolve_image_paths(df, args.image_root, verify=False)

    transform = build_default_transform(args.resolution)
    dataset = MosaicDataset(
        metadata_df=df,
        species_ids=species_ids,
        canvas_size=args.resolution,
        k_dist=DEFAULT_K_DIST,
        samples_per_epoch=args.p2_samples_per_epoch,
        transform=transform,
        seed=args.seed,
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.p2_batch_size,
        shuffle=False,                 # MosaicDataset does its own randomisation
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    model = DINOv3MultiLabel(
        n_classes=n_classes,
        backbone_source=args.backbone_source,
        backbone_name=args.backbone,
        pretrained=True,
        ssl_checkpoint=args.ssl_checkpoint,
    )

    # Load Phase-1 head if provided
    if args.p1_head:
        ckpt = torch.load(args.p1_head, map_location="cpu", weights_only=False)
        model.head.load_state_dict(ckpt["head_state"])
        logger.info(f"  loaded Phase-1 head from {args.p1_head}")

    # LoRA on backbone
    model.apply_lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model = model.to(args.device)

    # Logit adjustment is the softmax-CE long-tail trick (Menon 2021); applying
    # log(prior) ≈ -9 to a sigmoid head makes the loss-minimum collapse to
    # all-positive (verified by diagnose_predictions.py). ASL's gamma_neg=4 +
    # clip=0.05 already handle multi-label class imbalance — leave adj=None.
    criterion = AsymmetricLoss(
        gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
        logit_adjustments=None, use_fused=False,
    ).to(args.device)

    # Two LR groups
    lora_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters()) + [model.gem.p]
    optim = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.p2_lr_lora},
            {"params": head_params, "lr": args.p2_lr_head},
        ],
        weight_decay=0.05,
    )

    steps_per_epoch = max(1, len(loader) // args.p2_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=[args.p2_lr_lora, args.p2_lr_head],
        epochs=args.p2_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "phase2_lora.pth"

    use_amp = args.bf16 and args.device.startswith("cuda")
    scaler = None  # bf16 needs no grad scaler

    for epoch in range(1, args.p2_epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_micro = 0
        optim.zero_grad(set_to_none=True)

        for step, (imgs, targets) in enumerate(loader):
            imgs = imgs.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                logits = model(imgs)
                loss = criterion(logits, targets) / args.p2_accum

            loss.backward()
            running += loss.item() * args.p2_accum
            n_micro += 1

            if (step + 1) % args.p2_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                lrs = [g["lr"] for g in optim.param_groups]
                logger.info(
                    f"  P2 ep{epoch} step{step}/{len(loader)} "
                    f"loss={loss.item() * args.p2_accum:.4f} "
                    f"lr_lora={lrs[0]:.2e} lr_head={lrs[1]:.2e}"
                )

        avg_loss = running / max(n_micro, 1)
        dur = time.time() - t0
        logger.info(
            f"P2 epoch {epoch}/{args.p2_epochs} | avg_loss={avg_loss:.4f} | {dur/60:.1f}m"
        )

        # Save backbone-LoRA + head + gem each epoch
        torch.save(
            {
                "backbone_state": model.backbone.state_dict(),  # PEFT-wrapped
                "head_state": model.head.state_dict(),
                "gem_state": model.gem.state_dict(),
                "n_classes": n_classes,
                "species_ids": species_ids,
                "backbone_source": args.backbone_source,
                "backbone_name": args.backbone,
                "ssl_checkpoint": args.ssl_checkpoint,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "epoch": epoch,
            },
            ckpt_path,
        )
        logger.info(f"  saved -> {ckpt_path}")

    logger.info(f"Phase 2 complete. Final checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.phase == 1:
        run_phase1(args)
    else:
        run_phase2(args)


if __name__ == "__main__":
    main()
