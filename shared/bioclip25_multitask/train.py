"""
BioCLIP 2.5 end-to-end fine-tuning with multi-task taxonomy learning.

New in i002
-----------
  --metadata-csv               Path to training metadata CSV (new default points
                               to metadata_filled_genus_family.csv).
  --taxonomy-csv               Optional; set to path for old-format CSVs that
                               need a separate taxonomy merge. Default: None.
  --max-images-per-species N   Cap each species to at most N training images.
                               <=0 means no cap.
  --max-train-rows N           Cap total training rows after per-species cap.
                               <=0 means no cap.
  --cap-seed S                 RNG seed for both capping steps (default 42).
  --use-sample-weights         Use WeightedRandomSampler built from the
                               sample_weight column in the metadata CSV.
                               Falls back to shuffle=True if the column is
                               absent or in DDP mode (with a warning).

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
from data.metadata_utils import (
    apply_max_images_per_species_cap,
    apply_max_train_rows_cap,
    print_species_distribution,
    build_weighted_sampler,
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
# Cached-features dataset (head-only fast path)
# ---------------------------------------------------------------------------

class CachedFeaturesDataset(torch.utils.data.Dataset):
    """
    Returns (feature, species_idx, genus_idx, family_idx) per index.

    Backed by in-memory tensors produced by
    `tools/cache_features_i002.py`. The trainer treats the feature
    tensor the same way it would treat an image tensor in shape
    `(B, embed_dim)`, but routes it through `model.forward_heads()`
    rather than `model.forward()`.
    """

    def __init__(self, features, sp, gen, fam):
        self.features = features
        self.sp = sp
        self.gen = gen
        self.fam = fam

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Cast to fp32 here so the AMP autocast in train.py controls dtype
        # rather than the cache file's storage dtype.
        return (
            self.features[idx].float(),
            int(self.sp[idx]),
            int(self.gen[idx]),
            int(self.fam[idx]),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BioCLIP 2.5 multi-task fine-tuning for PlantCLEF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument(
        "--metadata-csv", "--train-meta-csv",
        dest="metadata_csv",
        default=DEFAULT_TRAIN_META_CSV,
        help="Path to training metadata CSV.",
    )
    p.add_argument(
        "--taxonomy-csv",
        default=DEFAULT_TAXONOMY_CSV,
        help=(
            "Optional taxonomy CSV for old-format CSVs. "
            "None = taxonomy already embedded in metadata."
        ),
    )
    p.add_argument(
        "--train-image-root", default=DEFAULT_TRAIN_IMAGE_ROOT,
        help="Image root for fallback path resolution.",
    )
    p.add_argument("--val-fraction",  type=float, default=0.1)
    p.add_argument("--val-seed",      type=int,   default=42)
    p.add_argument("--img-size",      type=int,   default=224)

    # Capping
    p.add_argument(
        "--max-images-per-species", type=int, default=0, metavar="N",
        help="Cap each species to at most N training images (0 = no cap).",
    )
    p.add_argument(
        "--max-train-rows", type=int, default=0, metavar="N",
        help="Cap total training rows after per-species cap (0 = no cap).",
    )
    p.add_argument(
        "--cap-seed", type=int, default=42,
        help="RNG seed for reproducible capping.",
    )
    # Legacy alias kept for smoke-test backward compat
    p.add_argument(
        "--max-samples", type=int, default=0,
        help="Alias for --max-train-rows (legacy).",
    )

    # Sample weights
    p.add_argument(
        "--use-sample-weights", action="store_true", default=False,
        help="Use WeightedRandomSampler from sample_weight column.",
    )

    # Model
    p.add_argument("--model-name",    default=BIOCLIP25_MODEL_NAME)
    p.add_argument("--hidden-dim",    type=int,   default=1024)
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--use-taxonomy-heads", action="store_true", default=True)
    p.add_argument("--no-taxonomy-heads",  action="store_false",
                   dest="use_taxonomy_heads")

    # Fine-tuning mode (mutually exclusive)
    ft = p.add_mutually_exclusive_group()
    ft.add_argument("--freeze-backbone", action="store_true", default=True,
                    help="Freeze backbone entirely (default).")
    ft.add_argument("--unfreeze-last-n-blocks", type=int, default=0, metavar="N",
                    dest="unfreeze_n",
                    help="Unfreeze last N transformer blocks + projection.")
    ft.add_argument("--full-finetune", action="store_true",
                    help="Unfreeze entire backbone.")

    # Learning rates
    p.add_argument("--backbone-lr",     type=float, default=1e-6)
    p.add_argument("--head-lr",         type=float, default=1e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)

    # Schedule
    p.add_argument("--epochs",        type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=1)

    # Training
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--grad-accum-steps", type=int,   default=1)
    p.add_argument("--precision",        default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--num-workers",      type=int,   default=8)
    p.add_argument("--grad-clip",        type=float, default=1.0)

    # Output / logging
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--val-every",  type=int, default=1)
    p.add_argument("--log-every",  type=int, default=50)
    p.add_argument("--resume",     default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--resume-weights-only", action="store_true")
    p.add_argument("--no-epoch-snapshots", action="store_true",
                   help="Skip writing per-epoch epoch_NNN.pt files. best.pt is "
                        "still updated on val improvement and last.pt is still "
                        "written every epoch. Use for scout runs to save disk.")

    # Device
    p.add_argument("--device", default="auto")

    # Optional W&B
    p.add_argument("--wandb",    action="store_true")
    p.add_argument("--run-name", default=None)

    # Smoke test
    p.add_argument("--smoke-test", action="store_true",
                   help="Use ~200 samples, 1 epoch. Verifies end-to-end pipeline.")

    # Feature cache (head-only fast path)
    p.add_argument("--feature-cache", default=None,
                   help="Path to a cache.pt produced by tools/cache_features_i002.py. "
                        "When set, the trainer skips the frozen-backbone forward and "
                        "trains the head MLPs + classifiers against the cached "
                        "(N, embed_dim) tensors. Loads the encoders / val split from "
                        "the cache (--metadata-csv et al. are ignored).")

    # Performance knobs (default-off compile; default-on persistent workers)
    p.add_argument("--compile-head", action="store_true",
                   help="Wrap the model in torch.compile(mode='reduce-overhead'). "
                        "Adds ~30 s warm-up on first epoch; saves ~10-20% per "
                        "subsequent epoch on the head-MLP forward.")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="DataLoader prefetch_factor (non-cache path only).")
    p.add_argument("--no-persistent-workers", action="store_false",
                   dest="persistent_workers", default=True,
                   help="Disable persistent_workers in the non-cache DataLoaders.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: BioCLIP25MultiTask,
    loader: DataLoader,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    is_ddp: bool = False,
    rank: int = 0,
    from_features: bool = False,
) -> dict:
    """
    If from_features=True, the loader yields pre-computed backbone features
    (shape (B, embed_dim)) and we call `model.forward_heads()` to skip the
    frozen backbone. Otherwise the loader yields image tensors and we call
    `model()` as before.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    loss_sum     = 0.0
    n_samples    = 0
    top1_correct = 0
    top5_correct = 0
    gen_correct  = 0
    gen_total    = 0
    fam_correct  = 0
    fam_total    = 0

    logger.warning(f"[rank{rank}] validation loop start ({len(loader)} batches)")

    raw_model = model.module if is_ddp else model

    for batch in loader:
        inputs, sp_lbl, gen_lbl, fam_lbl = batch
        inputs  = inputs.to(device, non_blocking=True)
        sp_lbl  = sp_lbl.to(device, non_blocking=True)
        gen_lbl = gen_lbl.to(device, non_blocking=True)
        fam_lbl = fam_lbl.to(device, non_blocking=True)

        with amp_autocast(device, amp_enabled, amp_dtype):
            outputs = raw_model.forward_heads(inputs) if from_features else model(inputs)

        sp_log     = outputs[0].float()
        loss_sum  += criterion(sp_log, sp_lbl).item()
        n_samples += sp_lbl.size(0)

        k = min(5, sp_log.size(1))
        _, topk_idx  = sp_log.topk(k, dim=1)
        top1_correct += topk_idx[:, 0].eq(sp_lbl).sum().item()
        top5_correct += topk_idx.eq(sp_lbl.unsqueeze(1)).any(dim=1).sum().item()

        gen_log = outputs[1]
        fam_log = outputs[2] if len(outputs) > 2 else None

        if gen_log is not None:
            mask = gen_lbl != -1
            if mask.any():
                gen_correct += gen_log[mask].float().argmax(1).eq(gen_lbl[mask]).sum().item()
                gen_total   += mask.sum().item()

        if fam_log is not None:
            mask = fam_lbl != -1
            if mask.any():
                fam_correct += fam_log[mask].float().argmax(1).eq(fam_lbl[mask]).sum().item()
                fam_total   += mask.sum().item()

    logger.warning(f"[rank{rank}] validation loop done, entering metric all_reduce")

    stats = torch.tensor(
        [loss_sum, n_samples, top1_correct, top5_correct,
         gen_correct, gen_total, fam_correct, fam_total],
        dtype=torch.float64, device=device,
    )

    if is_ddp:
        import torch.distributed as dist
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    logger.warning(f"[rank{rank}] finished metric all_reduce")

    (loss_sum, n_samples, top1_correct, top5_correct,
     gen_correct, gen_total, fam_correct, fam_total) = stats.tolist()
    n = max(int(n_samples), 1)

    metrics: dict = {
        "val_loss": round(loss_sum / n, 4),
        "top1_acc": round(top1_correct / n, 4),
        "top5_acc": round(top5_correct / n, 4),
        "n_val":    int(n_samples),
    }
    if gen_total > 0:
        metrics["genus_acc"]  = round(gen_correct / gen_total, 4)
    if fam_total > 0:
        metrics["family_acc"] = round(fam_correct / fam_total, 4)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # GPU performance knobs (Ampere/Ada/Hopper safe; no-op on older HW)
    # ------------------------------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    # ------------------------------------------------------------------
    # Resolve legacy --max-samples alias
    # ------------------------------------------------------------------
    if args.max_samples > 0 and args.max_train_rows <= 0:
        args.max_train_rows = args.max_samples

    # ------------------------------------------------------------------
    # Smoke test overrides
    # ------------------------------------------------------------------
    if args.smoke_test:
        args.epochs         = min(args.epochs, 1)
        args.max_train_rows = args.max_train_rows or 200
        args.log_every      = min(args.log_every, 5)
        args.num_workers    = min(args.num_workers, 2)

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
        logger.info("BioCLIP 2.5 Multi-Task Fine-Tuning  [shared/bioclip25_multitask]")
        logger.info(f"  smoke_test             : {args.smoke_test}")
        logger.info(f"  precision              : {args.precision}")
        logger.info(f"  epochs                 : {args.epochs}")
        logger.info(f"  batch_size             : {args.batch_size}  "
                    f"grad_accum: {args.grad_accum_steps}")
        logger.info(f"  backbone_lr            : {args.backbone_lr}  "
                    f"head_lr: {args.head_lr}")
        logger.info(f"  use_taxonomy_heads     : {args.use_taxonomy_heads}")
        logger.info(f"  max_images_per_species : {args.max_images_per_species}")
        logger.info(f"  max_train_rows         : {args.max_train_rows}")
        logger.info(f"  cap_seed               : {args.cap_seed}")
        logger.info(f"  use_sample_weights     : {args.use_sample_weights}")
        logger.info(f"  device                 : {device}")
        logger.info(f"  output_dir             : {out_dir}")
        logger.info(f"  metadata_csv           : {args.metadata_csv}")
        logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    use_feature_cache = bool(args.feature_cache)

    if use_feature_cache:
        if is_main_process():
            logger.info(f"Feature cache: {args.feature_cache}")
            logger.info("Skipping backbone forward; training heads on cached features.")

        cache = torch.load(args.feature_cache, map_location="cpu",
                           weights_only=False)
        encoders = cache["encoders"]
        idx_to_species = encoders.get("idx_to_species", [])
        num_species    = len(idx_to_species)
        num_genus      = len(encoders.get("idx_to_genus",  []))
        num_family     = len(encoders.get("idx_to_family", []))

        if is_main_process():
            logger.info(
                f"Classes (from cache) — species={num_species:,}  "
                f"genus={num_genus:,}  family={num_family:,}"
            )
            # mirror the encoders the original path writes
            enc_dir = out_dir / "encoders"
            enc_dir.mkdir(parents=True, exist_ok=True)
            for name in ["species", "genus", "family"]:
                for key in [f"idx_to_{name}", f"{name}_to_idx"]:
                    val = encoders.get(key, {} if "to_idx" in key else [])
                    with open(enc_dir / f"{key}.json", "w") as f:
                        import json as _json
                        _json.dump(val, f, indent=2)

        train_ds = CachedFeaturesDataset(
            cache["train_features"], cache["train_sp"],
            cache["train_gen"], cache["train_fam"],
        )
        val_ds = CachedFeaturesDataset(
            cache["val_features"], cache["val_sp"],
            cache["val_gen"], cache["val_fam"],
        )
        if args.smoke_test and len(val_ds) > 50:
            # honour smoke-test val cap: sample 50 indices deterministically
            import random as _random
            _random.seed(42)
            idxs = sorted(_random.sample(range(len(val_ds)), 50))
            sub_f   = cache["val_features"][idxs]
            sub_sp  = cache["val_sp"][idxs]
            sub_gen = cache["val_gen"][idxs]
            sub_fam = cache["val_fam"][idxs]
            val_ds  = CachedFeaturesDataset(sub_f, sub_sp, sub_gen, sub_fam)
        if args.smoke_test:
            # honour smoke-test train cap
            n = min(len(train_ds), max(args.max_train_rows or 200, 200))
            train_ds = CachedFeaturesDataset(
                cache["train_features"][:n], cache["train_sp"][:n],
                cache["train_gen"][:n], cache["train_fam"][:n],
            )

        if is_main_process():
            logger.info(f"Train (cached): {len(train_ds):,} samples")
            logger.info(f"Val   (cached): {len(val_ds):,} samples")

        # Cached DataLoaders - bypass the disk path entirely.
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=0, pin_memory=device.startswith("cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=device.startswith("cuda"),
        )
        train_sampler = None  # no DDP sampler needed in cache mode (single GPU)
    else:
        if is_main_process():
            logger.info("Loading metadata ...")

        df = load_metadata(args.metadata_csv, args.taxonomy_csv)
        df = resolve_image_paths(df, args.train_image_root)
        df = df[df["resolved_path"].notna()].reset_index(drop=True)

        if is_main_process():
            logger.info(f"Total images after path resolution: {len(df):,}")

        # Rank 0 writes encoder JSON; all ranks share the in-memory dict.
        encoders = build_label_encoders(
            df,
            output_dir=str(out_dir / "encoders") if is_main_process() else None,
        )
        barrier()  # ensure files are on disk before other ranks read them

        idx_to_species = encoders.get("idx_to_species", [])
        num_species    = len(idx_to_species)
        num_genus      = len(encoders.get("idx_to_genus",  []))
        num_family     = len(encoders.get("idx_to_family", []))

        if is_main_process():
            logger.info(
                f"Classes — species={num_species:,}  genus={num_genus:,}  "
                f"family={num_family:,}"
            )

        train_df, val_df = build_val_split(
            df, val_fraction=args.val_fraction, seed=args.val_seed
        )

        # ------------------------------------------------------------------
        # Capping — training split only
        # ------------------------------------------------------------------
        if is_main_process():
            print_species_distribution(train_df, label="Train distribution BEFORE cap")

        if args.max_images_per_species > 0:
            n_before = len(train_df)
            train_df = apply_max_images_per_species_cap(
                train_df,
                max_per_species=args.max_images_per_species,
                seed=args.cap_seed,
            )
            if is_main_process():
                logger.info(
                    f"Per-species cap ({args.max_images_per_species}): "
                    f"{n_before:,} → {len(train_df):,} rows"
                )

        if args.max_train_rows > 0 and len(train_df) > args.max_train_rows:
            n_before = len(train_df)
            train_df = apply_max_train_rows_cap(
                train_df,
                max_rows=args.max_train_rows,
                seed=args.cap_seed,
            )
            if is_main_process():
                logger.info(
                    f"Total-rows cap ({args.max_train_rows}): "
                    f"{n_before:,} → {len(train_df):,} rows"
                )

        if is_main_process() and (
            args.max_images_per_species > 0 or args.max_train_rows > 0
        ):
            print_species_distribution(train_df, label="Train distribution AFTER cap")

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
        want_weighted = args.use_sample_weights
        has_weights   = "sample_weight" in train_df.columns
        use_weighted  = want_weighted and has_weights and not is_ddp

        if want_weighted and is_ddp:
            if is_main_process():
                logger.warning(
                    "WeightedRandomSampler is not supported in DDP mode; "
                    "falling back to DistributedSampler (shuffle=True)."
                )
        if want_weighted and not has_weights:
            if is_main_process():
                logger.warning(
                    "sample_weight column not found in metadata — "
                    "falling back to uniform shuffle."
                )

        train_sampler = None

        if is_ddp:
            train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
            train_loader  = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=train_sampler,
                num_workers=args.num_workers,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None, pin_memory=True,
            )
        elif use_weighted:
            if is_main_process():
                logger.info("Using WeightedRandomSampler for training.")
            w_sampler    = build_weighted_sampler(train_df)
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=w_sampler,
                num_workers=args.num_workers,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                pin_memory=device.startswith("cuda"),
                drop_last=True,
            )
        else:
            if is_main_process():
                logger.info("Using shuffle=True for training.")
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                num_workers=args.num_workers,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                pin_memory=device.startswith("cuda"),
            )

        val_sampler = None
        if is_ddp and len(val_ds) > 0:
            val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)

        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size * 2,
            sampler=val_sampler,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            pin_memory=device.startswith("cuda"),
        )

        if is_main_process():
            logger.info(
                f"Train: {len(train_ds):,} samples  "
                f"{len(train_loader):,} batches/epoch"
            )
            logger.info(f"Val  : {len(val_ds):,} samples")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BioCLIP25MultiTask(
        num_species        = num_species,
        num_genus          = num_genus  if args.use_taxonomy_heads else 0,
        num_family         = num_family if args.use_taxonomy_heads else 0,
        model_name         = args.model_name,
        hidden_dim         = args.hidden_dim,
        dropout            = args.dropout,
        use_taxonomy_heads = args.use_taxonomy_heads,
    )

    if args.full_finetune:
        model.configure_backbone("full")
    elif args.unfreeze_n and args.unfreeze_n > 0:
        model.configure_backbone("last_n", n_blocks=args.unfreeze_n)
    else:
        model.configure_backbone("freeze")

    model = model.to(device)

    if args.compile_head:
        if is_main_process():
            logger.info("Compiling model with torch.compile(mode='reduce-overhead')")
        # mode="reduce-overhead" wraps the forward in a CUDA graph after the
        # first ~30 s warmup pass; per-step launch overhead disappears.
        # Safe with DDP if applied before the DDP wrap.
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    raw_model    = model.module if is_ddp else model
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
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = args.warmup_epochs * steps_per_epoch

    scheduler = build_cosine_schedule(optimizer, warmup_steps, total_steps)

    # ------------------------------------------------------------------
    # Scaler (fp16 only)
    # ------------------------------------------------------------------
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch         = 0
    history: list[dict] = []
    best_top5           = -1.0
    opt_step            = 0

    if args.resume:
        if args.resume_weights_only:
            start_epoch, prev_metrics = load_checkpoint(
                args.resume, model,
                optimizer=None, scheduler=None, scaler=None,
                device=device,
            )
            start_epoch = 0
            opt_step    = 0
            best_top5   = prev_metrics.get("top5_acc", 0.0)
            if is_main_process():
                logger.info(
                    "Loaded model weights only; optimizer/scheduler reset for new stage."
                )
        else:
            start_epoch, prev_metrics = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler, device=device
            )
            best_top5 = prev_metrics.get("top5_acc", 0.0)
            opt_step  = prev_metrics.get("opt_step", 0)
            if is_main_process():
                logger.info(
                    f"Resumed epoch={start_epoch}  best_top5={best_top5:.4f}"
                )

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
        epoch_loss = 0.0
        n_batches  = 0
        epoch_t    = time.perf_counter()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            inputs, sp_lbl, gen_lbl, fam_lbl = batch

            inputs  = inputs.to(device, non_blocking=True)
            sp_lbl  = sp_lbl.to(device, non_blocking=True)
            gen_lbl = gen_lbl.to(device, non_blocking=True)
            fam_lbl = fam_lbl.to(device, non_blocking=True)

            with amp_autocast(device, amp_enabled, amp_dtype):
                if use_feature_cache:
                    raw = model.module if is_ddp else model
                    outputs = raw.forward_heads(inputs)
                else:
                    outputs = model(inputs)
                loss, loss_parts = compute_multitask_loss(
                    outputs,
                    (sp_lbl, gen_lbl, fam_lbl),
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
                avg     = epoch_loss / n_batches
                elapsed = time.perf_counter() - epoch_t
                lr_vals = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
                logger.info(
                    f"[{epoch+1}/{args.epochs}] step {step+1}/{len(train_loader)}  "
                    f"loss={avg:.4f}  lr={'/'.join(lr_vals)}  t={elapsed:.0f}s"
                )

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

        # ------------------------------------------------------------------
        # Validation  (all ranks participate to avoid NCCL desync)
        # ------------------------------------------------------------------
        val_metrics: dict = {}
        if (epoch + 1) % args.val_every == 0 and len(val_ds) > 0:
            logger.warning(f"[rank{rank}] entering validation epoch {epoch+1}")
            eval_model  = model.module if is_ddp else model
            val_metrics = validate(
                eval_model, val_loader, device, amp_enabled, amp_dtype,
                is_ddp=is_ddp, rank=rank,
                from_features=use_feature_cache,
            )
            logger.warning(f"[rank{rank}] finished validation epoch {epoch+1}")
            if is_main_process():
                logger.info(
                    f"  val_loss={val_metrics['val_loss']:.4f}  "
                    f"top1={val_metrics['top1_acc']:.4f}  "
                    f"top5={val_metrics['top5_acc']:.4f}  "
                    f"n={val_metrics['n_val']:,}"
                )
                for level in ["genus", "family"]:
                    k = f"{level}_acc"
                    if k in val_metrics:
                        logger.info(f"    {level}_acc={val_metrics[k]:.4f}")

        # ------------------------------------------------------------------
        # Checkpoint + metrics
        # ------------------------------------------------------------------
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

            raw_save   = model.module if is_ddp else model
            ckpt_state = {
                "epoch":                epoch,
                "model_state_dict":     raw_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "metrics":              entry,
                "config":               config,
                "idx_to_species":       idx_to_species,
                "encoders":             encoders,
            }

            if args.no_epoch_snapshots:
                # Scout policy: only best.pt (when val improves) + last.pt.
                # Net writes per run: 2 files instead of N epoch snapshots.
                if is_best:
                    save_checkpoint(
                        ckpt_state, path=str(ckpt_dir / "best.pt"),
                    )
            elif (epoch + 1) % args.save_every == 0:
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

        logger.warning(f"[rank{rank}] entering checkpoint barrier epoch {epoch+1}")
        barrier()
        logger.warning(f"[rank{rank}] finished checkpoint barrier epoch {epoch+1}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    if is_main_process():
        elapsed = (time.perf_counter() - t_total) / 60
        logger.info(
            f"Training complete: {args.epochs} epochs in {elapsed:.1f} min"
        )
        logger.info(f"Best top5_acc: {best_top5:.4f}")
        logger.info(f"Checkpoints: {ckpt_dir}")
        if args.smoke_test:
            logger.info("SMOKE TEST PASSED")

    if use_wandb:
        import wandb
        wandb.finish()

    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
