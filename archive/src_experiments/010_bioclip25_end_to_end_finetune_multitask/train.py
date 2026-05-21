"""
BioCLIP 2.5 end-to-end fine-tuning with multi-task taxonomy learning.

Fine-tuning modes
-----------------
  --freeze-backbone        Frozen backbone, train head only (default)
  --unfreeze-last-n N      Unfreeze last N transformer blocks + projection
  --full-finetune          Unfreeze entire backbone

Precision
---------
  --precision fp16         FP16 AMP with GradScaler
  --precision bf16         BF16 AMP (no GradScaler; stable on Ampere+)
  --precision fp32         No AMP

Smoke test
----------
  --smoke-test             Run with ~200 samples for 1 epoch to verify end-to-end

Multi-GPU (torchrun)
--------------------
  torchrun --nproc_per_node=N train.py [args]
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import (
    DEFAULT_TRAIN_META_CSV,
    DEFAULT_TAXONOMY_CSV,
    DEFAULT_TRAIN_IMAGE_ROOT,
    load_metadata,
    build_label_encoders,
    resolve_image_paths,
    build_val_split,
    MultiTaskDataset,
)
from model import BioCLIP25MultiTask, BIOCLIP25_MODEL_NAME
from transforms import train_transform, val_transform
from utils import (
    setup_logging,
    resolve_device,
    amp_autocast,
    build_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
    compute_multitask_loss,
    topk_accuracy,
    taxonomy_accuracy,
    save_json,
    append_metrics_csv,
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
        description="BioCLIP 2.5 multi-task fine-tuning for PlantCLEF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--train-meta-csv",   default=DEFAULT_TRAIN_META_CSV)
    p.add_argument("--taxonomy-csv",     default=DEFAULT_TAXONOMY_CSV)
    p.add_argument("--train-image-root", default=DEFAULT_TRAIN_IMAGE_ROOT)
    p.add_argument("--val-fraction",     type=float, default=0.1)
    p.add_argument("--val-seed",         type=int,   default=42)
    p.add_argument("--max-samples",      type=int,   default=0,
                   help="Cap total training samples (0=no cap). Useful for debugging.")
    p.add_argument("--img-size",         type=int,   default=224)

    # Model
    p.add_argument("--model-name",    default=BIOCLIP25_MODEL_NAME)
    p.add_argument("--hidden-dim",    type=int,   default=1024)
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--use-taxonomy-heads", action="store_true", default=True)
    p.add_argument("--no-taxonomy-heads",  action="store_false", dest="use_taxonomy_heads")

    # Fine-tuning mode (mutually exclusive)
    ft = p.add_mutually_exclusive_group()
    ft.add_argument("--freeze-backbone",      action="store_true", default=True,
                    help="Freeze backbone entirely (default).")
    ft.add_argument("--unfreeze-last-n-blocks", type=int, default=0, metavar="N",
                    dest="unfreeze_n",
                    help="Unfreeze last N transformer blocks + projection.")
    ft.add_argument("--full-finetune",        action="store_true",
                    help="Unfreeze entire backbone.")

    # Learning rates
    p.add_argument("--backbone-lr",   type=float, default=1e-6)
    p.add_argument("--head-lr",       type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)

    # Schedule
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--warmup-epochs",  type=int,   default=1)

    # Training
    p.add_argument("--batch-size",        type=int, default=64)
    p.add_argument("--grad-accum-steps",  type=int, default=1)
    p.add_argument("--precision",         default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--num-workers",       type=int, default=8)
    p.add_argument("--grad-clip",         type=float, default=1.0)

    # Output / logging
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--val-every",  type=int, default=1)
    p.add_argument("--log-every",  type=int, default=50)
    p.add_argument("--resume",     default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--resume-weights-only", action="store_true")

    # Device
    p.add_argument("--device", default="auto")

    # Optional W&B
    p.add_argument("--wandb",     action="store_true")
    p.add_argument("--run-name",  default=None)

    # Smoke test
    p.add_argument("--smoke-test", action="store_true",
                   help="Use ~200 samples, 1 epoch. Verifies end-to-end pipeline.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: BioCLIP25MultiTask,
    loader: DataLoader,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches  = 0
    all_sp_pred, all_sp_lbls = [], []
    tax_correct: dict[str, list] = {
        "genus": [], "family": [], "order": [], "class": []
    }
    tax_names = ["genus", "family", "order", "class"]

    for batch in loader:
        images, sp_lbl, gen_lbl, fam_lbl, ord_lbl, cls_lbl = batch
        images  = images.to(device,  non_blocking=True)
        sp_lbl  = sp_lbl.to(device,  non_blocking=True)
        gen_lbl = gen_lbl.to(device, non_blocking=True)
        fam_lbl = fam_lbl.to(device, non_blocking=True)
        ord_lbl = ord_lbl.to(device, non_blocking=True)
        cls_lbl = cls_lbl.to(device, non_blocking=True)

        with amp_autocast(device, amp_enabled, amp_dtype):
            outputs = model(images)

        sp_log = outputs[0].float()
        loss   = criterion(sp_log, sp_lbl)
        total_loss += loss.item()
        n_batches  += 1

        all_sp_pred.append(sp_log.cpu())
        all_sp_lbls.append(sp_lbl.cpu())

        aux_logits = outputs[1:]
        aux_labels = [gen_lbl, fam_lbl, ord_lbl, cls_lbl]
        for name, logits, lbls in zip(tax_names, aux_logits, aux_labels):
            if logits is None:
                continue
            mask = lbls != -1
            if not mask.any():
                continue
            correct = logits[mask].float().argmax(1).eq(lbls[mask]).cpu().tolist()
            tax_correct[name].extend(correct)

    # Aggregate species metrics
    all_sp_pred = torch.cat(all_sp_pred, dim=0)  # (N, C)
    all_sp_lbls = torch.cat(all_sp_lbls, dim=0)  # (N,)

    top1 = topk_accuracy(all_sp_pred, all_sp_lbls, k=1)
    top5 = topk_accuracy(all_sp_pred, all_sp_lbls, k=5)

    metrics: dict = {
        "val_loss": round(total_loss / max(n_batches, 1), 4),
        "top1_acc": round(top1, 4),
        "top5_acc": round(top5, 4),
        "n_val":    len(all_sp_lbls),
    }

    for name, vals in tax_correct.items():
        if vals:
            metrics[f"{name}_acc"] = round(sum(vals) / len(vals), 4)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Smoke test overrides
    # ------------------------------------------------------------------
    if args.smoke_test:
        args.epochs       = min(args.epochs, 1)
        args.max_samples  = args.max_samples or 200
        args.log_every    = min(args.log_every, 5)
        args.num_workers  = min(args.num_workers, 2)

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------
    amp_enabled = args.precision != "fp32"
    amp_dtype   = torch.float16 if args.precision == "fp16" else torch.bfloat16
    use_scaler  = amp_enabled and args.precision == "fp16"

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
        rank   = dist.get_rank()
    else:
        device = resolve_device(args.device)
        rank   = 0

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"

    setup_logging(str(out_dir), rank=rank)

    if is_main_process():
        logger.info("=" * 65)
        logger.info("BioCLIP 2.5 Multi-Task Fine-Tuning")
        logger.info(f"  smoke_test    : {args.smoke_test}")
        logger.info(f"  precision     : {args.precision}")
        logger.info(f"  epochs        : {args.epochs}")
        logger.info(f"  batch_size    : {args.batch_size}  grad_accum: {args.grad_accum_steps}")
        logger.info(f"  backbone_lr   : {args.backbone_lr}  head_lr: {args.head_lr}")
        logger.info(f"  use_taxonomy  : {args.use_taxonomy_heads}")
        logger.info(f"  device        : {device}")
        logger.info(f"  output_dir    : {out_dir}")
        logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if is_main_process():
        logger.info("Loading and merging data ...")

    df = load_metadata(args.train_meta_csv, args.taxonomy_csv)
    df = resolve_image_paths(df, args.train_image_root)
    df = df[df["resolved_path"].notna()].reset_index(drop=True)

    if is_main_process():
        logger.info(f"Total images after path resolution: {len(df):,}")

    # Only rank 0 writes encoder JSON files; all ranks share the in-memory dict.
    encoders = build_label_encoders(
        df, output_dir=str(out_dir / "encoders") if is_main_process() else None
    )
    barrier()  # ensure files are on disk before any rank tries to read them
    idx_to_species = encoders.get("idx_to_species", [])
    num_species    = len(idx_to_species)
    num_genus      = len(encoders.get("idx_to_genus",  []))
    num_family     = len(encoders.get("idx_to_family", []))
    num_order      = len(encoders.get("idx_to_order",  []))
    num_class      = len(encoders.get("idx_to_class",  []))

    if is_main_process():
        logger.info(
            f"Classes — species={num_species:,}  genus={num_genus:,}  "
            f"family={num_family:,}  order={num_order:,}  class={num_class:,}"
        )

    train_df, val_df = build_val_split(df, val_fraction=args.val_fraction, seed=args.val_seed)

    if args.max_samples > 0 and len(train_df) > args.max_samples:
        train_df = train_df.sample(args.max_samples, random_state=42).reset_index(drop=True)
        if is_main_process():
            logger.info(f"Capped training set to {len(train_df):,} samples")

    # Proportional val cap for smoke test
    if args.smoke_test and len(val_df) > 50:
        val_df = val_df.sample(50, random_state=42).reset_index(drop=True)

    t_tfm = train_transform(img_size=args.img_size)
    v_tfm = val_transform(img_size=args.img_size)

    train_ds = MultiTaskDataset(train_df, encoders, t_tfm)
    val_ds   = MultiTaskDataset(val_df,   encoders, v_tfm)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    if is_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        train_loader  = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader  = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
        )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    if is_main_process():
        logger.info(
            f"Train: {len(train_ds):,} samples  {len(train_loader):,} batches/epoch"
        )
        logger.info(f"Val  : {len(val_ds):,} samples")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BioCLIP25MultiTask(
        num_species = num_species,
        num_genus   = num_genus  if args.use_taxonomy_heads else 0,
        num_family  = num_family if args.use_taxonomy_heads else 0,
        num_order   = num_order  if args.use_taxonomy_heads else 0,
        num_class   = num_class  if args.use_taxonomy_heads else 0,
        model_name  = args.model_name,
        hidden_dim  = args.hidden_dim,
        dropout     = args.dropout,
        use_taxonomy_heads = args.use_taxonomy_heads,
    )

    # Determine fine-tuning mode
    if args.full_finetune:
        model.configure_backbone("full")
    elif args.unfreeze_n and args.unfreeze_n > 0:
        model.configure_backbone("last_n", n_blocks=args.unfreeze_n)
    else:
        model.configure_backbone("freeze")

    model = model.to(device)

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters=True because aux heads may be unused for some samples
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    raw_model = model.module if is_ddp else model
    param_groups = raw_model.get_param_groups(
        backbone_lr  = args.backbone_lr,
        head_lr      = args.head_lr,
        weight_decay = args.weight_decay,
    )
    if not param_groups:
        logger.error("No trainable parameters found! Check fine-tuning mode.")
        sys.exit(1)

    optimizer = torch.optim.AdamW(param_groups)

    # ------------------------------------------------------------------
    # Scheduler (step-based cosine + warmup)
    # ------------------------------------------------------------------
    steps_per_epoch   = max(1, len(train_loader) // args.grad_accum_steps)
    total_steps       = args.epochs * steps_per_epoch
    warmup_steps      = args.warmup_epochs * steps_per_epoch

    scheduler = build_cosine_schedule(optimizer, warmup_steps, total_steps)

    # ------------------------------------------------------------------
    # Scaler (for fp16 only)
    # ------------------------------------------------------------------
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ------------------------------------------------------------------
    # Loss criterion
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch   = 0
    history: list[dict] = []
    best_top5     = -1.0  # -1 ensures best.pt is always saved on first epoch
    opt_step      = 0

    if args.resume:
        if args.resume_weights_only:
            start_epoch, prev_metrics = load_checkpoint(
                args.resume,
                model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=device,
            )
            start_epoch = 0
            opt_step = 0
            best_top5 = prev_metrics.get("top5_acc", 0.0)
            if is_main_process():
                logger.info("Loaded model weights only; optimizer/scheduler reset for new stage.")
        else:
            start_epoch, prev_metrics = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler, device=device
            )
            best_top5 = prev_metrics.get("top5_acc", 0.0)
            opt_step = prev_metrics.get("opt_step", 0)

            if is_main_process():
                logger.info(f"Resumed epoch={start_epoch}  best_top5={best_top5:.4f}")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    use_wandb = args.wandb and is_main_process()
    if use_wandb:
        import wandb
        wandb.init(
            project="plantclef2026",
            name=args.run_name or Path(args.output_dir).name,
            config=vars(args),
        )

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    if is_main_process():
        config = {
            **vars(args),
            "device":       device,
            "num_species":  num_species,
            "num_genus":    num_genus,
            "num_family":   num_family,
            "num_order":    num_order,
            "num_class":    num_class,
            "world_size":   get_world_size(),
            "total_steps":  total_steps,
            "warmup_steps": warmup_steps,
        }
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
        epoch_loss  = 0.0
        n_batches   = 0
        epoch_t     = time.perf_counter()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images, sp_lbl, gen_lbl, fam_lbl, ord_lbl, cls_lbl = batch

            images  = images.to(device,  non_blocking=True)
            sp_lbl  = sp_lbl.to(device,  non_blocking=True)
            gen_lbl = gen_lbl.to(device, non_blocking=True)
            fam_lbl = fam_lbl.to(device, non_blocking=True)
            ord_lbl = ord_lbl.to(device, non_blocking=True)
            cls_lbl = cls_lbl.to(device, non_blocking=True)

            with amp_autocast(device, amp_enabled, amp_dtype):
                outputs = model(images)
                loss, loss_parts = compute_multitask_loss(
                    outputs,
                    (sp_lbl, gen_lbl, fam_lbl, ord_lbl, cls_lbl),
                    criterion,
                )
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            is_update_step = (
                (step + 1) % args.grad_accum_steps == 0
                or (step + 1) == len(train_loader)
            )
            if is_update_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                opt_step += 1

            epoch_loss += loss_parts["total_loss"]
            n_batches  += 1

            if is_main_process() and (step + 1) % args.log_every == 0:
                avg = epoch_loss / n_batches
                elapsed = time.perf_counter() - epoch_t
                lr_vals = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
                logger.info(
                    f"[{epoch+1}/{args.epochs}] step {step+1}/{len(train_loader)}  "
                    f"loss={avg:.4f}  lr={'/'.join(lr_vals)}  "
                    f"t={elapsed:.0f}s"
                )

        # Average train loss across ranks
        avg_loss_t = torch.tensor(epoch_loss / max(n_batches, 1), device=device)
        avg_loss   = all_reduce_mean(avg_loss_t).item()
        epoch_secs = time.perf_counter() - epoch_t
        head_lr    = optimizer.param_groups[-1]["lr"]

        if is_main_process():
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"train_loss={avg_loss:.4f}  lr={head_lr:.2e}  "
                f"time={epoch_secs:.0f}s"
            )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        val_metrics: dict = {}
        if is_main_process() and (epoch + 1) % args.val_every == 0 and len(val_ds) > 0:
            logger.info("Validating ...")
            eval_model = model.module if is_ddp else model
            val_metrics = validate(eval_model, val_loader, device, amp_enabled, amp_dtype)
            logger.info(
                f"  val_loss={val_metrics['val_loss']:.4f}  "
                f"top1={val_metrics['top1_acc']:.4f}  "
                f"top5={val_metrics['top5_acc']:.4f}  "
                f"n={val_metrics['n_val']:,}"
            )
            # Log taxonomy accuracies
            for level in ["genus", "family", "order", "class"]:
                k = f"{level}_acc"
                if k in val_metrics:
                    logger.info(f"    {level}_acc={val_metrics[k]:.4f}")

        # --------------------------------------------------------------
        # Checkpoint + metrics
        # --------------------------------------------------------------
        if is_main_process():
            entry = {
                "epoch":      epoch + 1,
                "train_loss": round(avg_loss, 6),
                "head_lr":    round(head_lr, 10),
                "epoch_secs": round(epoch_secs, 1),
                "opt_step":   opt_step,
                **val_metrics,
            }
            history.append(entry)
            save_json({"history": history}, str(out_dir / "metrics.json"))
            append_metrics_csv(entry, str(out_dir / "metrics.csv"))

            is_best = val_metrics.get("top5_acc", 0.0) > best_top5
            if is_best:
                best_top5 = val_metrics["top5_acc"]
                logger.info(f"  New best top5_acc: {best_top5:.4f}")

            raw_save = model.module if is_ddp else model
            ckpt_state = {
                "epoch":               epoch,
                "model_state_dict":    raw_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "metrics":             entry,
                "config":              config,
                "idx_to_species":      idx_to_species,
                "encoders":            encoders,
            }

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    ckpt_state,
                    path=str(ckpt_dir / f"epoch_{epoch+1:03d}.pt"),
                    is_best=is_best,
                    best_path=str(ckpt_dir / "best.pt"),
                )
            save_checkpoint(ckpt_state, path=str(ckpt_dir / "last.pt"))

            if use_wandb:
                import wandb
                wandb.log({"train_loss": avg_loss, **val_metrics}, step=epoch + 1)

        barrier()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    if is_main_process():
        elapsed = (time.perf_counter() - t_total) / 60
        logger.info(f"Training complete: {args.epochs} epochs in {elapsed:.1f} min")
        logger.info(f"Best top5_acc: {best_top5:.4f}")
        logger.info(f"Checkpoints: {ckpt_dir}")
        if args.smoke_test:
            logger.info("SMOKE TEST PASSED ✓")

    if use_wandb:
        import wandb
        wandb.finish()

    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
