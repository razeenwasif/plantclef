"""
Phase A — BioCLIP-2.5 ViT-H/14 full fine-tune on the 1.4M PC24 single-plant
corpus. PlantNet-style recipe, identical to ``008_dinov3_plantnet_finetune/
train_phase_a.py`` except the backbone is BioCLIP-2.5 instead of DINOv3-L.

Why this exists
---------------
008 hit a 0.305 ceiling standalone, and a rank-fusion ensemble with frozen
BioCLIP-2.5 (Arjun's 0.33) cracked 0.34642. The next ceiling lift requires a
*stronger* second leg in the ensemble — i.e. fine-tuning BioCLIP-2.5 on the
same PC24 corpus that 008 was trained on, then fusing two strong species
classifiers instead of one strong + one frozen prototype matcher.

Recipe (matches 008 deliberately)
---------------------------------
  * Warmup: freeze backbone, train LayerNorm+Linear head only for 1 epoch.
  * Main: unfreeze, two LR groups (backbone 5e-5, head 1e-4), OneCycleLR,
          AdamW(beta=(0.9,0.95), wd=0.05), 8 epochs, bf16.
  * Loss: CE + label_smoothing=0.1 + per-class logit adjustment.
  * Aug: RandomResizedCrop + HFlip + ColorJitter + RandomErasing.
  * Norm stats: BioCLIP's CLIP-style stats (NOT ImageNet — see ``bioclip_model``).
  * img_size must be multiple of 14 (patch-14). 224 is the default.

Cost
----
ViT-H/14 is ~2× the params of DINOv3-L. Expect ~25-40h on a single 5090,
~10-15h on 2× 5090 DDP. Use bf16 + grad checkpoint if VRAM is tight.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from bioclip_model import (
    BioCLIP25SinglePlantClassifier,
    build_default_transform,
    build_train_transform,
)

# Reuse 008's single-plant dataset and 004's species-id loader without
# duplicating code.
_HERE = Path(__file__).resolve().parent
_EIGHT = _HERE.parent / "008_dinov3_plantnet_finetune"
_FOUR = _HERE.parent / "004_bioclip_few_shot"
for _p in (_EIGHT, _FOUR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from single_plant_dataset import SinglePlantDataset, split_indices  # noqa: E402
from dataset import load_species_ids  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl",
            timeout=__import__("datetime").timedelta(minutes=30),
        )
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank, True
    return 0, 1, 0, False


def inner(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--species-csv", required=True)
    p.add_argument("--img-size", type=int, default=224,
                   help="Must be a multiple of 14 for BioCLIP-2.5 patch-14.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=24,
                   help="Per-GPU micro batch size (ViT-H/14 is heavy).")
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr-backbone", type=float, default=5e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Enable gradient checkpointing on the visual tower "
                        "to fit larger batch in VRAM (slows step ~25%%).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device: str, bf16: bool, is_dist: bool
             ) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    top1 = 0
    top5 = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16)
        if device.startswith("cuda")
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(imgs)
            loss = criterion(logits.float(), targets)
        loss_sum += float(loss.item())
        _, top5_idx = logits.topk(5, dim=1)
        top1 += int((top5_idx[:, 0] == targets).sum().item())
        top5 += int((top5_idx == targets.unsqueeze(1)).any(dim=1).sum().item())
        total += int(targets.size(0))

    if is_dist:
        buf = torch.tensor([loss_sum, top1, top5, total],
                           device=device, dtype=torch.float64)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        loss_sum, top1, top5, total = buf.tolist()

    total = max(int(total), 1)
    return loss_sum / total, top1 / total, top5 / total


# ---------------------------------------------------------------------------

def main() -> None:
    rank, world, local_rank, is_dist = setup_distributed()
    is_main = rank == 0

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARN,
        format=f"%(asctime)s [%(levelname)s] [r{rank}] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    torch.manual_seed(args.seed + rank)
    if args.img_size % 14 != 0:
        sys.exit("img-size must be a multiple of 14 for BioCLIP-2.5 patch-14.")

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    species_ids = load_species_ids(args.species_csv)
    n_classes = len(species_ids)
    if is_main:
        logger.info(
            f"World size: {world} | device: {device} | N classes: {n_classes}"
        )

    # Model ------------------------------------------------------------------
    model = BioCLIP25SinglePlantClassifier(n_classes=n_classes).to(device)

    if args.grad_checkpoint:
        # open_clip's VisionTransformer supports grad_checkpointing via attr.
        try:
            model.backbone.set_grad_checkpointing(True)
            if is_main:
                logger.info("Enabled grad checkpointing on visual tower")
        except AttributeError:
            if is_main:
                logger.warning(
                    "Could not enable grad checkpointing — open_clip API "
                    "may have changed; continuing without it."
                )

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        res = model.load_state_dict(ck["model_state"], strict=False)
        if is_main:
            logger.info(
                f"Resumed from {args.resume}: missing={len(res.missing_keys)}, "
                f"unexpected={len(res.unexpected_keys)}"
            )

    # Data -------------------------------------------------------------------
    train_tf = build_train_transform(args.img_size)
    val_tf = build_default_transform(args.img_size)
    full_train = SinglePlantDataset(
        args.train_csv, args.images_root, species_ids, transform=train_tf
    )
    full_val = SinglePlantDataset(
        args.train_csv, args.images_root, species_ids, transform=val_tf
    )
    n_all = len(full_train)
    if args.limit and args.limit < n_all:
        if is_main:
            logger.info(f"--limit active: capping to first {args.limit:,} samples")
        n_all = args.limit
    train_idx, val_idx = split_indices(n_all, args.val_frac, seed=args.seed)
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_val, val_idx)
    if is_main:
        logger.info(f"Train={len(train_ds):,}  Val={len(val_ds):,}")

    # Logit-adjust prior over the training subset.
    freqs = torch.zeros(n_classes, dtype=torch.float64)
    for i in train_idx:
        freqs[full_train.targets[i]] += 1.0
    total = float(freqs.sum().clamp_min(1.0))
    log_prior = torch.log(freqs.clamp_min(1.0) / total).to(
        device=device, dtype=torch.float32
    )
    if is_main:
        logger.info(
            f"Logit-adjust prior: min={float(log_prior.min()):.3f}, "
            f"max={float(log_prior.max()):.3f}, "
            f"n_zero_classes={int((freqs == 0).sum().item())}"
        )

    # Samplers / loaders -----------------------------------------------------
    if is_dist:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world, rank=rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world, rank=rank,
            shuffle=False, drop_last=False,
        )
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=train_shuffle,
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=max(2, args.num_workers // 2), pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "phase_a_last.pth"
    best_path = out_dir / "phase_a_best.pth"

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ------- Warmup: head-only ----------------------------------------------
    if args.warmup_epochs > 0 and args.resume is None:
        if is_main:
            logger.info(f"Warmup: head-only for {args.warmup_epochs} epoch(s)")
        inner(model).freeze_backbone()
        if is_dist:
            m = DDP(
                model if not isinstance(model, DDP) else model.module,
                device_ids=[local_rank], find_unused_parameters=True,
            )
        else:
            m = model

        head_params = list(inner(m).head_parameters())
        optim = torch.optim.AdamW(
            head_params, lr=args.lr_head, weight_decay=args.weight_decay
        )
        steps_per_epoch = max(1, len(train_loader) // args.accum)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=args.lr_head,
            epochs=args.warmup_epochs, steps_per_epoch=steps_per_epoch,
        )
        for epoch in range(args.warmup_epochs):
            m.train()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            t0 = time.time()
            running = 0.0
            n_micro = 0
            optim.zero_grad(set_to_none=True)
            for step, (imgs, targets) in enumerate(train_loader):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16
                ):
                    logits = m(imgs) + log_prior
                    loss = criterion(logits, targets) / args.accum
                loss.backward()
                running += float(loss.item()) * args.accum
                n_micro += 1
                if (step + 1) % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(head_params, 1.0)
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                if is_main and step % args.log_every == 0:
                    logger.info(
                        f"  warmup ep{epoch} step{step}/{len(train_loader)} "
                        f"loss={loss.item()*args.accum:.3f} "
                        f"lr={optim.param_groups[0]['lr']:.2e}"
                    )
            if is_main:
                logger.info(
                    f"warmup epoch {epoch} done in {(time.time()-t0)/60:.1f}m, "
                    f"avg loss={running/max(n_micro,1):.3f}"
                )
        model = inner(m)

    # ------- Main: unfreeze backbone, DDP-wrap ------------------------------
    inner(model).unfreeze_backbone()
    if is_dist:
        model = DDP(
            model if not isinstance(model, DDP) else model.module,
            device_ids=[local_rank], find_unused_parameters=False,
        )

    head_ids = {id(p) for p in inner(model).head_parameters()}
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_ids
    ]
    head_params = list(inner(model).head_parameters())

    if is_main:
        n_bb = sum(p.numel() for p in backbone_params)
        n_hd = sum(p.numel() for p in head_params)
        logger.info(
            f"Trainable: backbone {n_bb/1e6:.1f}M, head {n_hd/1e6:.1f}M  "
            f"(effective batch = {args.batch_size * args.accum * world})"
        )

    optim = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params,     "lr": args.lr_head / 10.0},
        ],
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    steps_per_epoch = max(1, len(train_loader) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=[args.lr_backbone, args.lr_head / 10.0],
        epochs=args.epochs, steps_per_epoch=steps_per_epoch,
    )

    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        t0 = time.time()
        running = 0.0
        n_micro = 0
        optim.zero_grad(set_to_none=True)
        for step, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16
            ):
                logits = model(imgs) + log_prior
                loss = criterion(logits, targets) / args.accum
            loss.backward()
            running += float(loss.item()) * args.accum
            n_micro += 1
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
            if is_main and step % args.log_every == 0:
                lrs = [g["lr"] for g in optim.param_groups]
                logger.info(
                    f"  ep{epoch} step{step}/{len(train_loader)} "
                    f"loss={loss.item()*args.accum:.3f} "
                    f"lr_bb={lrs[0]:.2e} lr_hd={lrs[1]:.2e}"
                )

        if is_dist:
            reduce_buf = torch.tensor(
                [running, n_micro], device=device, dtype=torch.float64
            )
            dist.all_reduce(reduce_buf, op=dist.ReduceOp.SUM)
            running, n_micro = reduce_buf.tolist()
        avg_loss = running / max(n_micro, 1)

        val_loss, val_top1, val_top5 = validate(
            model, val_loader, device, bf16=args.bf16, is_dist=is_dist
        )
        if is_main:
            logger.info(
                f"epoch {epoch}/{args.epochs} | train_loss={avg_loss:.3f} "
                f"val_loss={val_loss:.3f} top1={val_top1:.4f} "
                f"top5={val_top5:.4f} | {(time.time()-t0)/60:.1f}m"
            )

            ckpt = {
                "model_state": inner(model).state_dict(),
                "species_ids": species_ids,
                "img_size": args.img_size,
                "n_classes": n_classes,
                "backbone_name": inner(model).backbone_name,
                "log_prior": log_prior.detach().cpu(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top5": val_top5,
            }
            torch.save(ckpt, last_path)
            torch.save(ckpt, out_dir / f"phase_a_ep{epoch}.pth")
            if val_top1 > best_top1:
                best_top1 = val_top1
                torch.save(ckpt, best_path)
                logger.info(
                    f"  new best val_top1={val_top1:.4f} -> {best_path.name}"
                )

        if is_dist:
            bt = torch.tensor([best_top1], device=device, dtype=torch.float64)
            dist.broadcast(bt, src=0)
            best_top1 = float(bt.item())

    if is_main:
        logger.info(f"Phase A complete. best val_top1={best_top1:.4f}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
