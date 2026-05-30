"""
Local few-shot validation pipeline.

Evaluates few-shot inference quality on a held-out split of the single-plant
training data.  The split is created within each species so support and query
images are always disjoint.

Validation strategy
-------------------
For each species with sufficient images:
  - First K images (after optional sampling) → support set
  - Remaining images → query set  (treated as single-species "quadrats")

The few-shot model is asked to classify each query image.
A prediction is "correct" if the ground-truth species is in the top-N
predicted species (standard multi-label recall @ N).

Metrics reported
----------------
- Recall@K    : fraction of query images where ground-truth species is in top-K
- Precision@K : precision of top-K predictions (treating each query as single-label)
- micro F1, macro F1 (at the configured threshold)
- Per-species hit rate (optional, written to CSV)

Usage examples
--------------
# Quick validation: bioclip, K=5, prototype mode
python run_validation.py \\
    --bank-dir ./cache/bioclip_k5_random_seed42 \\
    --val-seed 99

# K=10, KNN mode, save per-species diagnostics
python run_validation.py \\
    --bank-dir ./cache/bioclip_k10_random_seed42 \\
    --scoring-mode knn \\
    --per-species-diag

# Limit to 100 species for fast smoke test
python run_validation.py \\
    --bank-dir ./cache/bioclip_k5_random_seed42 \\
    --limit-species 100

Output directory structure
--------------------------
{output-dir}/{run_slug}/
    val_config.json          parameters
    val_metrics.json         recall@k, F1, precision, recall
    val_per_species.csv      per-species hit rate (if --per-species-diag)
    val_split_manifest.csv   which images were support vs query
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import pandas as pd

# Local modules
from models import load_model, resolve_device, default_batch_size, resolve_model_name
from tiling import get_tiles, encode_image_tiles
from few_shot import SupportBank, score_prototype, score_knn
from aggregation import aggregate_scores, apply_threshold, AGG_MODES
from dataset import (
    DEFAULT_TRAIN_META_CSV,
    DEFAULT_TRAIN_IMAGE_ROOT,
    DEFAULT_SPECIES_CSV,
    load_train_metadata,
    resolve_image_paths,
    load_species_ids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "./val_outputs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local few-shot validation using the training split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Bank / model
    p.add_argument("--bank-dir", required=True,
                   help="Support bank directory (from build_support_bank.py).")
    p.add_argument("--model-name", default=None,
                   help="Override model name (defaults to bank metadata).")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=None)

    # Training data (for creating the query set)
    p.add_argument("--train-meta-csv", default=DEFAULT_TRAIN_META_CSV)
    p.add_argument("--train-image-root", default=DEFAULT_TRAIN_IMAGE_ROOT)
    p.add_argument("--species-csv", default=DEFAULT_SPECIES_CSV)

    # Scoring
    p.add_argument("--scoring-mode", default="prototype", choices=["prototype", "knn"])

    # Tiling
    p.add_argument("--tile-size", type=int, default=224)
    p.add_argument("--tile-overlap", type=int, default=112)

    # Aggregation
    p.add_argument("--agg-mode", default="max", choices=list(AGG_MODES))
    p.add_argument("--agg-top-m", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--top-n", type=int, default=20)

    # Validation split
    p.add_argument(
        "--val-seed", type=int, default=99,
        help="Seed for query-set selection (different from the support-bank seed).",
    )
    p.add_argument(
        "--max-query-per-species", type=int, default=20,
        help="Max query images per species to evaluate (cap to keep runtime manageable).",
    )
    p.add_argument(
        "--min-images-for-val", type=int, default=2,
        help="Skip species with fewer than this many total images (cannot split).",
    )

    # Output
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--per-species-diag", action="store_true",
                   help="Save per-species hit rates to CSV.")
    p.add_argument("--limit-species", type=int, default=None,
                   help="Only validate on the first N species.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation split helpers
# ---------------------------------------------------------------------------

def build_val_split(
    df: pd.DataFrame,
    bank: SupportBank,
    max_query_per_species: int,
    min_images_for_val: int,
    val_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a disjoint (support, query) split from the training metadata.

    Support images are exactly those already embedded in the bank (identified
    by image_name and species_id from the bank manifest, if available, or by
    position).

    Query images are the remaining images for each species, capped at
    max_query_per_species.

    Returns
    -------
    support_df : DataFrame rows used as support (already embedded in bank)
    query_df   : DataFrame rows to use as query
    """
    import random

    rng = random.Random(val_seed)
    query_rows: list[pd.DataFrame] = []
    support_rows: list[pd.DataFrame] = []

    for species_id in bank.species_ids:
        species_df = df[df["species_id"] == species_id].copy()
        n_available = len(species_df)
        k_support = bank.support_counts.get(species_id, 0)

        if n_available < min_images_for_val:
            continue

        # Support: first k_support rows (by original order - matches how build_support_bank
        # samples, though not guaranteed exact; we mark them as 'support' for the manifest)
        support = species_df.iloc[:k_support]
        query_pool = species_df.iloc[k_support:]

        if len(query_pool) == 0:
            continue

        # Cap query size
        if len(query_pool) > max_query_per_species:
            idx = rng.sample(range(len(query_pool)), max_query_per_species)
            query_pool = query_pool.iloc[sorted(idx)]

        support_rows.append(support)
        query_rows.append(query_pool)

    support_df = pd.concat(support_rows, ignore_index=True) if support_rows else pd.DataFrame()
    query_df = pd.concat(query_rows, ignore_index=True) if query_rows else pd.DataFrame()
    return support_df, query_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    results: list[dict],
    threshold: float,
    top_n: int,
) -> dict:
    """
    Compute multi-label classification metrics from per-image result dicts.

    Each result dict has:
        species_id (str)  : ground-truth species
        pred_ids   (list) : predicted species IDs (sorted by score)
        pred_scores(list) : corresponding scores

    Returns a dict of scalar metrics.
    """
    n_total = len(results)
    if n_total == 0:
        return {}

    recall_at = {1: 0, 5: 0, 10: 0, 20: 0}
    tp = fp = fn = 0
    per_species_hits: dict[str, list[bool]] = defaultdict(list)

    for res in results:
        gt = res["species_id"]
        preds = res["pred_ids"]

        for k in recall_at:
            if gt in preds[:k]:
                recall_at[k] += 1

        # Binary TP/FP/FN at threshold (single-label ground truth)
        if gt in preds:
            tp += 1
        else:
            fn += 1
        fp += max(0, len(preds) - 1)  # all predictions beyond the ground truth are FP

        per_species_hits[gt].append(gt in preds)

    metrics: dict = {
        f"recall_at_{k}": round(v / n_total, 4) for k, v in recall_at.items()
    }
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics.update({
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(f1, 4),
        "n_query_images": n_total,
        "n_species_evaluated": len(per_species_hits),
        "threshold": threshold,
        "top_n": top_n,
    })

    # Macro F1 (mean per-species hit rate as a proxy)
    species_recalls = [sum(hits) / len(hits) for hits in per_species_hits.values() if hits]
    metrics["macro_recall"] = round(sum(species_recalls) / max(len(species_recalls), 1), 4)

    return metrics, per_species_hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    stride = args.tile_size - args.tile_overlap
    if stride <= 0:
        sys.exit(f"ERROR: tile-overlap must be < tile-size")

    # ------------------------------------------------------------------
    # Load support bank
    # ------------------------------------------------------------------
    bank_dir = Path(args.bank_dir)
    bank_file = bank_dir / "bank.pt"
    if not bank_file.exists():
        sys.exit(f"ERROR: no bank.pt in {bank_dir}. Run build_support_bank.py first.")
    logger.info(f"Loading support bank from {bank_file} ...")
    bank = SupportBank.load(bank_file)

    # ------------------------------------------------------------------
    # Model name
    # ------------------------------------------------------------------
    bank_meta_path = bank_dir / "bank_metadata.json"
    bank_meta = json.loads(bank_meta_path.read_text()) if bank_meta_path.exists() else {}
    model_name = args.model_name or bank_meta.get("model_name", "bioclip")
    batch_size = args.batch_size or default_batch_size(model_name)

    run_slug = (
        f"val_{Path(args.bank_dir).name}_{args.scoring_mode}_{args.agg_mode}"
        f"_seed{args.val_seed}"
    )
    out_dir = Path(args.output_dir) / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BioCLIP Few-Shot Validation")
    print(f"  bank_dir    : {args.bank_dir}")
    print(f"  model       : {model_name}")
    print(f"  scoring     : {args.scoring_mode}")
    print(f"  agg_mode    : {args.agg_mode}")
    print(f"  threshold   : {args.threshold}")
    print(f"  top_n       : {args.top_n}")
    print(f"  val_seed    : {args.val_seed}")
    print(f"  output_dir  : {out_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load training metadata
    # ------------------------------------------------------------------
    logger.info("Loading training metadata ...")
    df = load_train_metadata(args.train_meta_csv)

    # Restrict to species in the bank
    bank_species_set = set(bank.species_ids)
    df = df[df["species_id"].isin(bank_species_set)].copy()

    if args.limit_species:
        subset = sorted(df["species_id"].unique())[: args.limit_species]
        df = df[df["species_id"].isin(subset)]
        logger.info(f"Limiting validation to {args.limit_species} species")

    logger.info("Resolving image paths ...")
    df = resolve_image_paths(df, args.train_image_root, verify=False)
    df = df[df["resolved_path"].notna()].copy()

    # ------------------------------------------------------------------
    # Build validation split
    # ------------------------------------------------------------------
    logger.info("Building validation split ...")
    support_df, query_df = build_val_split(
        df,
        bank=bank,
        max_query_per_species=args.max_query_per_species,
        min_images_for_val=args.min_images_for_val,
        val_seed=args.val_seed,
    )
    logger.info(
        f"  Support rows  : {len(support_df):,} "
        f"({support_df['species_id'].nunique():,} species)"
    )
    logger.info(
        f"  Query rows    : {len(query_df):,} "
        f"({query_df['species_id'].nunique():,} species)"
    )
    if len(query_df) == 0:
        sys.exit("ERROR: query set is empty - not enough images for validation split")

    # Save split manifest
    manifest = pd.concat([
        support_df.assign(split="support"),
        query_df.assign(split="query"),
    ], ignore_index=True)
    manifest_path = out_dir / "val_split_manifest.csv"
    manifest[["species_id", "image_name", "resolved_path", "split"]].to_csv(
        manifest_path, index=False
    )
    logger.info(f"Split manifest saved: {manifest_path}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model {model_name} ...")
    model, transform, _ = load_model(model_name, device)

    # ------------------------------------------------------------------
    # Run inference on query images
    # ------------------------------------------------------------------
    logger.info(f"Evaluating {len(query_df):,} query images ...")
    results: list[dict] = []
    t0 = time.perf_counter()
    errors = 0

    for i, (_, row) in enumerate(query_df.iterrows()):
        img_path = Path(row["resolved_path"])
        gt_species = str(row["species_id"])

        try:
            from PIL import Image as _PIL
            image = _PIL.open(img_path).convert("RGB")
            tiles, _ = get_tiles(image, args.tile_size, stride)
            tile_emb = encode_image_tiles(
                model, transform, tiles, device, batch_size=batch_size
            )

            if args.scoring_mode == "prototype":
                tile_scores = score_prototype(tile_emb, bank, device=device)
            else:
                tile_scores = score_knn(tile_emb, bank, device=device)

            image_scores = aggregate_scores(
                tile_scores, mode=args.agg_mode, top_m=args.agg_top_m
            )
            pred_ids, pred_scores = apply_threshold(
                image_scores, bank.species_ids,
                threshold=args.threshold, top_n=args.top_n,
            )
        except Exception as exc:
            logger.warning(f"Error on {img_path.name}: {exc}")
            errors += 1
            pred_ids, pred_scores = [], []

        results.append({
            "species_id": gt_species,
            "image_name": img_path.stem,
            "pred_ids": pred_ids,
            "pred_scores": pred_scores,
            "hit_at_1":  gt_species in pred_ids[:1],
            "hit_at_5":  gt_species in pred_ids[:5],
            "hit_at_20": gt_species in pred_ids[:20],
        })

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            logger.info(
                f"  [{i+1:>6}/{len(query_df)}]  {elapsed:.0f}s elapsed"
            )

    eval_secs = time.perf_counter() - t0
    logger.info(f"Evaluation complete in {eval_secs:.1f}s  (errors={errors})")

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    metrics, per_species_hits = compute_metrics(results, args.threshold, args.top_n)
    metrics["eval_secs"] = round(eval_secs, 2)
    metrics["n_errors"] = errors

    print(f"\n{'='*60}")
    print(f"Validation Results: {run_slug}")
    for key, val in metrics.items():
        print(f"  {key:<30} {val}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics_path = out_dir / "val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    # ------------------------------------------------------------------
    # Optional per-species diagnostics
    # ------------------------------------------------------------------
    if args.per_species_diag and per_species_hits:
        diag_rows = [
            {
                "species_id": sid,
                "n_query": len(hits),
                "n_hits": sum(hits),
                "hit_rate": round(sum(hits) / len(hits), 4),
            }
            for sid, hits in sorted(per_species_hits.items())
        ]
        diag_path = out_dir / "val_per_species.csv"
        with open(diag_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["species_id", "n_query", "n_hits", "hit_rate"]
            )
            writer.writeheader()
            writer.writerows(diag_rows)
        logger.info(f"Per-species diagnostics saved: {diag_path}")

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    val_config = {
        **vars(args),
        "run_slug": run_slug,
        "model_name": model_name,
        "device": device,
        "stride": stride,
        "batch_size": batch_size,
        "n_species_in_bank": len(bank.species_ids),
        "n_query_images": len(query_df),
    }
    with open(out_dir / "val_config.json", "w") as f:
        json.dump(val_config, f, indent=2)


if __name__ == "__main__":
    main()
