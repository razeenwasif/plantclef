"""
SimSiam-style SSL pre-training on unlabeled pseudo-quadrat images.

Uses BioCLIP 2.5 as the backbone with a 3-layer projector and 2-layer predictor.
Two independently augmented views of each image are produced and the symmetric
negative cosine similarity loss is minimised.

After training the script saves:
  {output_dir}/checkpoints/last.pt      — full checkpoint (model + opt + sched)
  {output_dir}/checkpoints/best.pt      — full checkpoint at lowest loss epoch
  {output_dir}/checkpoints/backbone.pt  — backbone state dict only (for warm-start)

The backbone.pt checkpoint can be loaded into BioCLIP25MultiTask via:
  python train.py --ssl-backbone-checkpoint outputs/ssl_bioclip25/checkpoints/backbone.pt

Usage
-----
  python train_ssl.py --image-dirs /workspace/plantclef/raw/pseudo_quadrats

Smoke test
----------
  python train_ssl.py --image-dirs ... --epochs 1 --batch-size 8 --limit 32 --num-workers 2
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from model import BioCLIP25SSL, BIOCLIP25_MODEL_NAME, simsiam_loss
from transforms import SSLTwoViewTransform
from utils import (
    setup_logging,
    resolve_device,
    amp_autocast,
    build_cosine_schedule,
    save_json,
    append_metrics_csv,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

DEFAULT_OUTPUT_DIR = "./outputs/ssl_bioclip25"


# ---------------------------------------------------------------------------
# Unlabeled dataset
# ---------------------------------------------------------------------------

class UnlabeledImageDataset(Dataset):
    """
    Recursively discovers images under one or more directories and returns
    two independently augmented views per image for SSL training.
    """

    def __init__(
        self,
        image_dirs: list[str],
        transform: SSLTwoViewTransform,
        limit: int = 0,
    ) -> None:
        paths: list[Path] = []
        for d in image_dirs:
            root = Path(d)
            if not root.exists():
                logger.warning(f"Image dir not found, skipping: {root}")
                continue
            found = sorted(p for p in root.rglob("*") if p.suffix in _IMAGE_EXTS)
            logger.info(f"  {root}: {len(found):,} images")
            paths.extend(found)

        paths = sorted(set(paths))
        if limit > 0:
            paths = paths[:limit]
            logger.info(f"Applying --limit {limit}: using {len(paths):,} images")

        if not paths:
            raise RuntimeError(
                "No images found. Check --image-dirs paths and file extensions."
            )

        self.paths     = paths
        self.transform = transform
        logger.info(f"UnlabeledImageDataset: {len(self.paths):,} images total")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)  # (view1, view2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SimSiam SSL pre-training on unlabeled images with BioCLIP 2.5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument(
        "--image-dirs",
        nargs="+",
        default=["/workspace/plantclef/raw/pseudo_quadrats"],
        metavar="DIR",
        help="One or more directories containing unlabeled images (searched recursively).",
    )
    p.add_argument("--img-size",    type=int, default=224)
    p.add_argument(
        "--limit",
        type=int, default=0,
        help="Use only the first N images (0 = all). Useful for smoke testing.",
    )

    # Model
    p.add_argument("--model-name",  default=BIOCLIP25_MODEL_NAME)
    p.add_argument("--proj-hidden", type=int, default=2048,
                   help="Hidden dim of the 3-layer SimSiam projector.")
    p.add_argument("--proj-out",    type=int, default=256,
                   help="Output dim of projector (= input dim of predictor).")
    p.add_argument("--pred-hidden", type=int, default=512,
                   help="Hidden dim of the 2-layer SimSiam predictor.")

    # Fine-tuning
    p.add_argument(
        "--unfreeze-last-n-blocks",
        type=int, default=4, metavar="N", dest="unfreeze_n",
        help="Unfreeze last N transformer blocks + ln_post/proj of BioCLIP backbone.",
    )

    # Learning rates
    p.add_argument("--backbone-lr",  type=float, default=1e-6)
    p.add_argument("--head-lr",      type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # Schedule
    p.add_argument("--epochs",        type=int, default=2)
    p.add_argument("--warmup-epochs", type=int, default=0)

    # Training
    p.add_argument("--batch-size",   type=int, default=128)
    p.add_argument("--num-workers",  type=int, default=16)
    p.add_argument("--precision",    default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--log-every",    type=int,   default=50)

    # Output
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume",     default=None,
                   help="Path to a previous SSL checkpoint to resume from.")

    # Device
    p.add_argument("--device", default="auto")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_ssl_checkpoint(
    state: dict,
    path: str,
    is_best: bool = False,
    best_path: str | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)
    logger.info(f"SSL checkpoint saved: {p}  (epoch {state.get('epoch', '?')})")
    if is_best and best_path:
        shutil.copyfile(p, best_path)
        logger.info(f"Best SSL checkpoint updated: {best_path}")


def _save_backbone_checkpoint(
    backbone_state: dict,
    epoch: int,
    ssl_loss: float,
    path: str,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone_state_dict": backbone_state,
        "epoch":    epoch,
        "ssl_loss": round(ssl_loss, 6),
    }, p)
    logger.info(f"Backbone checkpoint saved: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------
    amp_enabled = args.precision != "fp32"
    amp_dtype   = torch.float16 if args.precision == "fp16" else torch.bfloat16
    use_scaler  = amp_enabled and args.precision == "fp16"

    device = resolve_device(args.device)

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(out_dir), rank=0)

    logger.info("=" * 65)
    logger.info("BioCLIP 2.5 SSL Pre-Training (SimSiam)")
    logger.info(f"  image_dirs   : {args.image_dirs}")
    logger.info(f"  limit        : {args.limit or 'all'}")
    logger.info(f"  precision    : {args.precision}")
    logger.info(f"  epochs       : {args.epochs}")
    logger.info(f"  batch_size   : {args.batch_size}")
    logger.info(f"  backbone_lr  : {args.backbone_lr}  head_lr: {args.head_lr}")
    logger.info(f"  unfreeze_n   : {args.unfreeze_n} blocks")
    logger.info(f"  device       : {device}")
    logger.info(f"  output_dir   : {out_dir}")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    two_view = SSLTwoViewTransform(img_size=args.img_size)
    dataset  = UnlabeledImageDataset(args.image_dirs, two_view, limit=args.limit)

    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        drop_last   = True,
        num_workers = args.num_workers,
        pin_memory  = device.startswith("cuda"),
    )
    logger.info(f"DataLoader: {len(dataset):,} images  {len(loader):,} batches/epoch")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BioCLIP25SSL(
        model_name  = args.model_name,
        proj_hidden = args.proj_hidden,
        proj_out    = args.proj_out,
        pred_hidden = args.pred_hidden,
    )
    if args.unfreeze_n > 0:
        model.configure_backbone("last_n", n_blocks=args.unfreeze_n)
    else:
        model.configure_backbone("freeze")

    model = model.to(device)

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    param_groups = model.get_param_groups(
        backbone_lr  = args.backbone_lr,
        head_lr      = args.head_lr,
        weight_decay = args.weight_decay,
    )
    if not param_groups:
        logger.error("No trainable parameters. Check --unfreeze-last-n-blocks.")
        sys.exit(1)

    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = max(1, len(loader))
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = args.warmup_epochs * steps_per_epoch
    scheduler       = build_cosine_schedule(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_loss   = float("inf")
    history: list[dict] = []

    if args.resume:
        p = Path(args.resume)
        if not p.exists():
            logger.error(f"Resume checkpoint not found: {p}")
            sys.exit(1)
        logger.info(f"Resuming from {p}")
        ckpt = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss   = ckpt.get("best_loss", float("inf"))
        history     = ckpt.get("history", [])
        logger.info(f"Resumed: epoch={start_epoch}  best_loss={best_loss:.4f}")

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    save_json(
        {**vars(args), "device": device, "total_steps": total_steps},
        str(out_dir / "ssl_config.json"),
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    t_total = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss   = 0.0
        epoch_z_std  = 0.0
        n_batches    = 0
        epoch_t      = time.perf_counter()
        optimizer.zero_grad()

        for step, (v1, v2) in enumerate(loader):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            with amp_autocast(device, amp_enabled, amp_dtype):
                p1, p2, z1, z2 = model(v1, v2)
                loss = 0.5 * (simsiam_loss(p1, z2) + simsiam_loss(p2, z1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            n_batches  += 1

            # Monitor collapse: z std across batch (should be > 0)
            with torch.no_grad():
                z_std = z1.float().std(dim=0).mean().item()
            epoch_z_std += z_std

            if (step + 1) % args.log_every == 0:
                avg  = epoch_loss / n_batches
                z_s  = epoch_z_std / n_batches
                elapsed = time.perf_counter() - epoch_t
                lr_vals = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
                logger.info(
                    f"[{epoch+1}/{args.epochs}] step {step+1}/{len(loader)}  "
                    f"loss={avg:.4f}  z_std={z_s:.4f}  lr={'/'.join(lr_vals)}  "
                    f"t={elapsed:.0f}s"
                )

        avg_loss  = epoch_loss  / max(n_batches, 1)
        avg_z_std = epoch_z_std / max(n_batches, 1)
        epoch_secs = time.perf_counter() - epoch_t
        head_lr    = optimizer.param_groups[-1]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"ssl_loss={avg_loss:.4f}  z_std={avg_z_std:.4f}  "
            f"lr={head_lr:.2e}  time={epoch_secs:.0f}s"
        )
        if avg_z_std < 0.001:
            logger.warning(
                "z_std is very small — possible representation collapse. "
                "Consider reducing backbone_lr or using larger batches."
            )

        entry = {
            "epoch":      epoch + 1,
            "ssl_loss":   round(avg_loss,  6),
            "z_std":      round(avg_z_std, 6),
            "head_lr":    round(head_lr,   10),
            "epoch_secs": round(epoch_secs, 1),
        }
        history.append(entry)
        save_json({"history": history}, str(out_dir / "ssl_metrics.json"))
        append_metrics_csv(entry, str(out_dir / "ssl_metrics.csv"))

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            logger.info(f"  New best SSL loss: {best_loss:.4f}")

        ckpt_state = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_loss":           best_loss,
            "history":             history,
            "config":              vars(args),
        }

        _save_ssl_checkpoint(
            ckpt_state,
            path      = str(ckpt_dir / "last.pt"),
            is_best   = is_best,
            best_path = str(ckpt_dir / "best.pt"),
        )
        if is_best:
            # Update standalone backbone checkpoint from the best epoch
            _save_backbone_checkpoint(
                backbone_state = model.backbone.state_dict(),
                epoch          = epoch,
                ssl_loss       = best_loss,
                path           = str(ckpt_dir / "backbone.pt"),
            )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = (time.perf_counter() - t_total) / 60
    logger.info(f"SSL pre-training complete: {args.epochs} epochs in {elapsed:.1f} min")
    logger.info(f"Best SSL loss: {best_loss:.4f}")
    logger.info(f"Checkpoints: {ckpt_dir}")
    logger.info(
        "Use backbone.pt for supervised warm-start:\n"
        f"  python train.py --ssl-backbone-checkpoint {ckpt_dir}/backbone.pt ..."
    )


if __name__ == "__main__":
    main()
