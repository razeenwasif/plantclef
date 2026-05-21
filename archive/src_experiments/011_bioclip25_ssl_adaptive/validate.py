"""
Standalone evaluation: load a checkpoint, compute metrics, save predictions CSV.

Usage
-----
  python validate.py --checkpoint outputs/train/checkpoints/best.pt \\
                     --train-meta-csv <path> --taxonomy-csv <path> \\
                     --train-image-root <path> \\
                     --output-dir outputs/eval
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
from model import load_checkpoint_model
from transforms import val_transform
from utils import (
    setup_logging,
    resolve_device,
    amp_autocast,
    topk_accuracy,
    save_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a BioCLIP 2.5 multi-task checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--train-meta-csv",   default=DEFAULT_TRAIN_META_CSV)
    p.add_argument("--taxonomy-csv",     default=DEFAULT_TAXONOMY_CSV)
    p.add_argument("--train-image-root", default=DEFAULT_TRAIN_IMAGE_ROOT)
    p.add_argument("--output-dir",       default="./outputs/eval")
    p.add_argument("--batch-size",       type=int, default=128)
    p.add_argument("--num-workers",      type=int, default=8)
    p.add_argument("--img-size",         type=int, default=224)
    p.add_argument("--val-fraction",     type=float, default=0.1)
    p.add_argument("--val-seed",         type=int,   default=42)
    p.add_argument("--top-k",            type=int,   default=5)
    p.add_argument("--device",           default="auto")
    p.add_argument("--precision",        default="fp16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--save-predictions", action="store_true", default=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(model, loader, idx_to_species, device, amp_enabled, amp_dtype, top_k=5):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    records  = []
    total_loss = 0.0
    n_batches  = 0

    top_k = min(top_k, len(idx_to_species))

    for batch in loader:
        images, sp_lbl, gen_lbl, fam_lbl, ord_lbl, cls_lbl = batch
        images  = images.to(device,  non_blocking=True)
        sp_lbl  = sp_lbl.to(device,  non_blocking=True)

        with amp_autocast(device, amp_enabled, amp_dtype):
            outputs = model(images)

        sp_log = outputs[0].float()
        loss   = criterion(sp_log, sp_lbl)
        total_loss += loss.item()
        n_batches  += 1

        probs = torch.softmax(sp_log, dim=1)
        _, topk_idx = probs.topk(top_k, dim=1)

        batch_df = loader.dataset.df.iloc[
            # reconstruct row indices – approximate using batch size
            # We iterate in order, so we can use a counter
            # Actually just use what we have
        ] if False else None  # skip – capture from batch metadata instead

        for i in range(sp_log.size(0)):
            true_idx  = sp_lbl[i].item()
            pred_idx  = topk_idx[i, 0].item()
            conf      = probs[i, pred_idx].item()
            top5_ids  = [idx_to_species[j.item()] for j in topk_idx[i]]
            records.append({
                "true_species": idx_to_species[true_idx] if true_idx < len(idx_to_species) else "?",
                "pred_species": idx_to_species[pred_idx] if pred_idx < len(idx_to_species) else "?",
                "top5_species": "|".join(top5_ids),
                "confidence":   round(conf, 5),
                "correct_top1": int(pred_idx == true_idx),
                "correct_top5": int(true_idx in topk_idx[i].tolist()),
            })

    n = len(records)
    top1 = sum(r["correct_top1"] for r in records) / max(n, 1)
    top5 = sum(r["correct_top5"] for r in records) / max(n, 1)

    metrics = {
        "val_loss": round(total_loss / max(n_batches, 1), 4),
        "top1_acc": round(top1, 4),
        "top5_acc": round(top5, 4),
        "n_val":    n,
    }
    return metrics, records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.output_dir)

    amp_enabled = args.precision != "fp32"
    amp_dtype   = torch.float16 if args.precision == "fp16" else torch.bfloat16
    device      = resolve_device(args.device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, encoders, config = load_checkpoint_model(args.checkpoint, device=device)

    idx_to_species = encoders.get("idx_to_species", [])
    if not idx_to_species:
        raise ValueError("Checkpoint missing idx_to_species in encoders.")

    # Build val dataset
    logger.info("Loading dataset ...")
    df = load_metadata(args.train_meta_csv, args.taxonomy_csv)
    df = resolve_image_paths(df, args.train_image_root)
    df = df[df["resolved_path"].notna()].reset_index(drop=True)

    # Use same encoders as training (from checkpoint)
    _, val_df = build_val_split(df, val_fraction=args.val_fraction, seed=args.val_seed)
    tfm = val_transform(img_size=args.img_size)
    val_ds = MultiTaskDataset(val_df, encoders, tfm)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )
    logger.info(f"Val set: {len(val_ds):,} images")

    # Run evaluation
    logger.info("Running evaluation ...")
    metrics, records = run_eval(
        model, val_loader, idx_to_species,
        device, amp_enabled, amp_dtype, top_k=args.top_k
    )

    logger.info(
        f"Results: val_loss={metrics['val_loss']:.4f}  "
        f"top1={metrics['top1_acc']:.4f}  "
        f"top5={metrics['top5_acc']:.4f}  "
        f"n={metrics['n_val']:,}"
    )

    # Save outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(metrics, str(out / "eval_metrics.json"))

    if args.save_predictions:
        pred_path = out / "predictions.csv"
        with open(pred_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else [])
            writer.writeheader()
            writer.writerows(records)
        logger.info(f"Predictions saved: {pred_path}")

    logger.info(f"Evaluation complete. Results in {out}")


if __name__ == "__main__":
    main()
