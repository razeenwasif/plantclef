"""
Phase B — LoRA fine-tune on labeled collages with ASL.

Load the DINOv3-L backbone fine-tuned in Phase A (full FT on 1.4M PC24 single
plants), swap in 005's `DINOv3MultiLabel` head (CLS + GeM patches -> MLP ->
sigmoid), apply LoRA to the backbone, and train on the 50k labeled collages
with AsymmetricLoss — mirror of 007's train.py but with a stronger backbone.

Inputs
------
  --phase-a-ckpt   .../dinov3_plantnet_v1_ddp/phase_a_best.pth
  --collage-csv    synthetic_collages.csv
  --collages-root  /workspace/plantclef/processed/collages
  --species-csv    species_lookup_with_gbif_cleaned_names.csv

Output
------
  {output_dir}/phase_b_ep{N}.pth
  {output_dir}/phase_b_last.pth
  {output_dir}/phase_b_best.pth
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Local — re-exports DINOv3MultiLabel from 005 + timm helpers
from model import DINOv3MultiLabel, build_default_transform

# 007's CollageDataset is reused verbatim.
_SEVEN = Path(__file__).resolve().parent.parent / "007_plantnet_dinov2_collages"
_coll_spec = _ilu.spec_from_file_location(
    "collage_dataset_007", str(_SEVEN / "collage_dataset.py")
)
_coll_mod = _ilu.module_from_spec(_coll_spec)
_coll_spec.loader.exec_module(_coll_mod)
CollageDataset = _coll_mod.CollageDataset
split_indices = _coll_mod.split_indices

# 004's load_species_ids
_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
if str(_FOUR) not in sys.path:
    sys.path.insert(0, str(_FOUR))
from dataset import load_species_ids  # type: ignore  # noqa: E402

# ASL from src/training/losses.py (bypass the package init to avoid wandb/deepspeed)
_SRC_TRAIN = Path(__file__).resolve().parent.parent.parent / "src" / "training"
_asl_spec = _ilu.spec_from_file_location(
    "plantclef_losses", str(_SRC_TRAIN / "losses.py")
)
_asl_mod = _ilu.module_from_spec(_asl_spec)
_asl_spec.loader.exec_module(_asl_mod)
AsymmetricLoss = _asl_mod.AsymmetricLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--collage-csv", required=True)
    p.add_argument("--collages-root", required=True)
    p.add_argument("--species-csv", required=True)
    # Phase A backbone
    p.add_argument("--phase-a-ckpt", required=True,
                   help="Phase A best checkpoint (phase_a_best.pth).")
    p.add_argument("--img-size", type=int, default=448,
                   help="Must be a multiple of 16. 336/448/512 all valid.")
    # LoRA
    p.add_argument("--no-lora", action="store_true",
                   help="Disable LoRA; train full backbone + head.")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # Training
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", default=None, help="Resume from a phase_b checkpoint.")
    return p.parse_args()


# ---------------------------------------------------------------------------

def build_collage_train_transform(resolution: int):
    """Slightly lighter than single-plant aug — collages are already chaotic."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomResizedCrop(
            resolution,
            scale=(0.7, 1.0),
            ratio=(0.85, 1.18),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.25, 0.25, 0.25),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_phase_a_backbone_into(model: DINOv3MultiLabel, phase_a_ckpt: str) -> None:
    """Load the Phase A `backbone.*` weights into `model.backbone`.

    Phase A saved a `DINOv3SinglePlantClassifier` whose backbone is the same
    timm `vit_large_patch16_dinov3.lvd1689m` instance used here, so the
    `backbone.*` keys map one-to-one. `norm.*` and `head.*` are Phase A's own
    classifier readout and are not reused.
    """
    ckpt = torch.load(phase_a_ckpt, map_location="cpu", weights_only=False)
    full_sd = ckpt["model_state"] if "model_state" in ckpt else ckpt

    # Strip DDP 'module.' prefix if present.
    if any(k.startswith("module.") for k in full_sd.keys()):
        full_sd = {k[len("module."):]: v for k, v in full_sd.items()}

    bb_sd = {k[len("backbone."):]: v for k, v in full_sd.items()
             if k.startswith("backbone.")}
    if not bb_sd:
        raise ValueError(
            f"No 'backbone.*' keys found in {phase_a_ckpt} — "
            f"first 5 keys: {list(full_sd.keys())[:5]}"
        )

    result = model.backbone.load_state_dict(bb_sd, strict=False)
    logger.info(
        f"Loaded Phase A backbone from {phase_a_ckpt}: "
        f"{len(bb_sd) - len(result.unexpected_keys)}/{len(bb_sd)} keys matched, "
        f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}"
    )
    if result.missing_keys:
        logger.info(f"  first 5 missing: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        logger.info(f"  first 5 unexpected: {result.unexpected_keys[:5]}")


# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device, bf16: bool) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n_batch = 0
    f1s: list[float] = []
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16)
        if device.startswith("cuda") else torch.amp.autocast(device_type="cpu", enabled=False)
    )
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(imgs)
            loss = criterion(logits, targets)
        total_loss += float(loss.item())
        n_batch += 1

        probs = torch.sigmoid(logits.float())
        pred = (probs >= 0.5).float()
        tp = (pred * targets).sum(dim=1)
        fp = (pred * (1 - targets)).sum(dim=1)
        fn = ((1 - pred) * targets).sum(dim=1)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = torch.where(
            (prec + rec) > 0,
            2 * prec * rec / (prec + rec + 1e-9),
            torch.zeros_like(prec),
        )
        f1s.extend(f1.tolist())
    avg_loss = total_loss / max(n_batch, 1)
    macro_f1 = float(sum(f1s) / max(len(f1s), 1))
    return avg_loss, macro_f1


# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.img_size % 16 != 0:
        raise ValueError(f"--img-size must be a multiple of 16, got {args.img_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    species_ids = load_species_ids(args.species_csv)
    n_classes = len(species_ids)
    logger.info(f"N classes: {n_classes}")

    # Build model: DINOv3MultiLabel (timm source) with no pretrained weights — we
    # will overwrite them with the Phase A fine-tuned backbone below.
    model = DINOv3MultiLabel(
        n_classes=n_classes,
        backbone_source="timm",
        pretrained=False,
    )
    load_phase_a_backbone_into(model, args.phase_a_ckpt)

    if not args.no_lora:
        model.apply_lora(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        logger.info("LoRA disabled — training full backbone end-to-end.")
    model = model.to(device)

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        sd = ck["model_state"] if "model_state" in ck else ck
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        res = model.load_state_dict(sd, strict=False)
        logger.info(
            f"Resumed from {args.resume}: missing={len(res.missing_keys)}, "
            f"unexpected={len(res.unexpected_keys)}"
        )

    # Data -------------------------------------------------------------------
    train_tf = build_collage_train_transform(args.img_size)
    val_tf = build_default_transform(args.img_size)

    full_ds_train = CollageDataset(
        csv_path=args.collage_csv,
        images_root=args.collages_root,
        species_ids=species_ids,
        transform=train_tf,
    )
    full_ds_val = CollageDataset(
        csv_path=args.collage_csv,
        images_root=args.collages_root,
        species_ids=species_ids,
        transform=val_tf,
    )
    train_idx, val_idx = split_indices(len(full_ds_train), args.val_frac, seed=args.seed)
    train_ds = Subset(full_ds_train, train_idx)
    val_ds = Subset(full_ds_val, val_idx)
    logger.info(f"Train={len(train_ds):,}  Val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(2, args.num_workers // 2), pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Loss — ASL (no logit adjustment, matches 007)
    criterion = AsymmetricLoss(
        gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
        logit_adjustments=None, use_fused=False,
    ).to(device)

    # Optimizer
    head_params = list(model.head_parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    n_bb = sum(p.numel() for p in backbone_params)
    n_hd = sum(p.numel() for p in head_params)
    logger.info(
        f"Trainable params — backbone/LoRA: {n_bb/1e6:.2f}M, head: {n_hd/1e6:.2f}M  "
        f"(effective batch = {args.batch_size * args.accum})"
    )

    optim = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params,     "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader) // args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=[args.lr_backbone, args.lr_head],
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "phase_b_last.pth"
    best_path = out_dir / "phase_b_best.pth"

    use_amp = args.bf16 and device.startswith("cuda")
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_micro = 0
        optim.zero_grad(set_to_none=True)

        for step, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_amp
            ):
                logits = model(imgs)
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

            if step % args.log_every == 0:
                lrs = [g["lr"] for g in optim.param_groups]
                logger.info(
                    f"  ep{epoch} step{step}/{len(train_loader)} "
                    f"loss={loss.item() * args.accum:.4f} "
                    f"lr_bb={lrs[0]:.2e} lr_hd={lrs[1]:.2e}"
                )

        avg_loss = running / max(n_micro, 1)
        val_loss, val_f1 = validate(model, val_loader, criterion, device, bf16=use_amp)
        dur = time.time() - t0
        logger.info(
            f"epoch {epoch}/{args.epochs} | train_loss={avg_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1@0.5={val_f1:.4f} | {dur/60:.1f}m"
        )

        ckpt_obj = {
            "model_state": model.state_dict(),
            "species_ids": species_ids,
            "img_size": args.img_size,
            "n_classes": n_classes,
            "lora": not args.no_lora,
            "backbone_name": getattr(model, "backbone_name", None),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_f1": val_f1,
        }
        torch.save(ckpt_obj, last_path)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ckpt_obj, best_path)
            logger.info(f"  new best val_f1={val_f1:.4f} -> {best_path.name}")
        torch.save(ckpt_obj, out_dir / f"phase_b_ep{epoch}.pth")

    logger.info(f"Phase B complete. best val_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
