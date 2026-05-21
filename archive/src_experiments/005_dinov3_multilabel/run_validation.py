"""
Local mosaic-based validation: macro-F1 per sample on a held-out species split.

Builds a synthetic mosaic validation set from a 5% species hold-out of the
training metadata, runs tiled multi-label inference, and reports macro-F1
per sample (averaged over mosaics) for each combination of:

  - aggregation mode  (max, mean, mean_top_m, noisy_or)
  - global threshold  (sweep)

This is the local proxy for the PlantCLEF leaderboard metric. Use it to pick
the best (agg_mode, threshold) without paying the Kaggle round-trip cost.

Usage:
    cd src_experiments/005_dinov3_multilabel
    python run_validation.py \
        --checkpoint   ../../models/dinov3_v1/phase2_lora.pth \
        --metadata-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
        --image-root   /workspace/plantclef/raw/train/images_max_side_800 \
        --species-csv  .../species_lookup_with_gbif_cleaned_names.csv \
        --n-mosaics    1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Sibling
from model import build_default_transform
from mosaic_dataset import MosaicDataset, DEFAULT_K_DIST
from run_inference import (
    load_model_from_checkpoint,
    encode_tiles_to_probs,
)

# 004 utilities
_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
if str(_FOUR) not in sys.path:
    sys.path.insert(0, str(_FOUR))
from tiling import get_tiles  # type: ignore  # noqa: E402
from aggregation import aggregate_scores, AGG_MODES  # type: ignore  # noqa: E402
from dataset import (  # type: ignore  # noqa: E402
    load_train_metadata,
    resolve_image_paths,
    load_species_ids,
    DEFAULT_SPECIES_CSV,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mosaic validation for DINOv3 multi-label model.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--species-csv", default=DEFAULT_SPECIES_CSV)
    p.add_argument("--n-mosaics", type=int, default=1000)
    p.add_argument("--canvas-size", type=int, default=384)

    # Tiling
    p.add_argument("--tile-size", type=int, default=384)
    p.add_argument("--tile-overlap", type=int, default=128)

    # Sweeps
    p.add_argument("--agg-modes", nargs="+", default=list(AGG_MODES))
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    p.add_argument("--top-n", type=int, default=20)

    # Splitting
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="Fraction of species held out for validation.")
    p.add_argument("--seed", type=int, default=42)

    # Compute
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--output-dir", default="./outputs/validation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Macro-F1 per sample
# ---------------------------------------------------------------------------

def f1_per_sample(true_set: set[str], pred_set: set[str]) -> float:
    """Per-sample F1: harmonic mean of precision & recall on the species sets."""
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.canvas_size % 16 != 0 or args.tile_size % 16 != 0:
        sys.exit("canvas_size and tile_size must be multiples of 16 for DINOv3.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model + species
    model, species_ids = load_model_from_checkpoint(args.checkpoint, args.device)
    species_to_idx = {sid: i for i, sid in enumerate(species_ids)}
    transform = build_default_transform(args.tile_size)

    # Load metadata, resolve paths
    df = load_train_metadata(args.metadata_csv)
    df = resolve_image_paths(df, args.image_root, verify=False)

    # Hold out a fraction of species for validation. This mirrors the test
    # distribution (the model has not optimised mosaics built from these species).
    rng = np.random.default_rng(args.seed)
    all_species = sorted(df["species_id"].astype(str).unique())
    n_val_species = max(1, int(len(all_species) * args.val_frac))
    val_species = set(rng.choice(all_species, size=n_val_species, replace=False).tolist())
    val_species &= set(species_ids)  # only score on species the model knows
    logger.info(f"Validation species held out: {len(val_species)}")

    val_df = df[df["species_id"].astype(str).isin(val_species)]
    if len(val_df) == 0:
        sys.exit("ERROR: no images in held-out species.")

    val_dataset = MosaicDataset(
        metadata_df=val_df,
        species_ids=species_ids,
        canvas_size=args.canvas_size,
        k_dist=DEFAULT_K_DIST,
        samples_per_epoch=args.n_mosaics,
        transform=None,             # we want PIL canvases, not tensors
        seed=args.seed,
        augment=False,              # deterministic for repeatable F1
    )

    # Build all mosaics first (PIL canvases + ground-truth label sets)
    logger.info(f"Generating {args.n_mosaics} validation mosaics ...")
    canvases: list[Image.Image] = []
    truths: list[set[str]] = []
    for i in range(args.n_mosaics):
        # MosaicDataset.transform=None => returns tensor from numpy; reach into compose
        # logic by re-using the dataset internals manually for PIL output.
        rng_i = val_dataset._rng_for(i)
        K = val_dataset._sample_k(rng_i)
        species_choices = rng_i.sample(
            val_dataset._species_with_data,
            k=min(K, len(val_dataset._species_with_data)),
        )
        crops = []
        chosen = []
        for sid in species_choices:
            img = val_dataset._sample_image(sid, rng_i)
            if img is not None:
                crops.append(img)
                chosen.append(sid)
            if len(crops) >= K:
                break
        if not crops:
            continue
        from mosaic_dataset import compose_mosaic
        canvas = compose_mosaic(crops, args.canvas_size, rng_i)
        canvases.append(canvas)
        truths.append(set(chosen))
    logger.info(f"Built {len(canvases)} mosaic canvases")

    # Score each mosaic once: cache per-tile probabilities so we can sweep
    # aggregation modes + thresholds without re-running the backbone.
    stride = max(1, args.tile_size - args.tile_overlap)

    logger.info("Running tile inference on all mosaics ...")
    t0 = time.time()
    all_tile_probs: list[torch.Tensor] = []
    for i, canvas in enumerate(canvases):
        tiles, _ = get_tiles(canvas, args.tile_size, stride)
        if not tiles:
            all_tile_probs.append(torch.empty(0, len(species_ids)))
            continue
        probs = encode_tiles_to_probs(
            model, transform, tiles, args.device, args.batch_size, args.bf16
        )
        all_tile_probs.append(probs)
        if (i + 1) % 50 == 0 or i + 1 == len(canvases):
            logger.info(f"  scored {i+1}/{len(canvases)} mosaics "
                        f"({(time.time()-t0):.1f}s)")
    logger.info(f"Tile scoring done in {(time.time()-t0):.1f}s")

    # Sweep
    results: list[dict] = []
    for agg_mode in args.agg_modes:
        # Convert probs -> logits for noisy_or only (matches run_inference.py)
        for thr in args.thresholds:
            scores_per_mosaic: list[float] = []
            for tile_probs, truth in zip(all_tile_probs, truths):
                if tile_probs.numel() == 0:
                    pred_set: set[str] = set()
                else:
                    if agg_mode == "noisy_or":
                        eps = 1e-6
                        clamped = tile_probs.clamp(eps, 1 - eps)
                        scores_in = torch.log(clamped / (1 - clamped))
                    else:
                        scores_in = tile_probs
                    image_scores = aggregate_scores(scores_in, mode=agg_mode, top_m=3)
                    # Top-N cap + threshold
                    sorted_vals, sorted_idx = image_scores.sort(descending=True)
                    sorted_vals = sorted_vals[: args.top_n]
                    sorted_idx = sorted_idx[: args.top_n]
                    pred_set = {
                        species_ids[idx]
                        for val, idx in zip(sorted_vals.tolist(), sorted_idx.tolist())
                        if val >= thr
                    }
                scores_per_mosaic.append(f1_per_sample(truth, pred_set))

            macro_f1 = float(np.mean(scores_per_mosaic)) if scores_per_mosaic else 0.0
            results.append({
                "agg_mode": agg_mode,
                "threshold": thr,
                "macro_f1_per_sample": macro_f1,
                "n_mosaics": len(scores_per_mosaic),
            })
            logger.info(
                f"  {agg_mode:<11s} thr={thr:.2f}  macro_F1={macro_f1:.4f}"
            )

    # Sort and report best
    results.sort(key=lambda r: r["macro_f1_per_sample"], reverse=True)
    out_path = out_dir / f"validation_{Path(args.checkpoint).stem}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"results": results, "config": vars(args)}, f, indent=2,
        )
    logger.info(f"Saved validation report -> {out_path}")

    print("\nTop 5 (agg_mode, threshold) by macro-F1 per sample:")
    for r in results[:5]:
        print(f"  {r['agg_mode']:<11s} thr={r['threshold']:.2f}  "
              f"F1={r['macro_f1_per_sample']:.4f}")


if __name__ == "__main__":
    main()
