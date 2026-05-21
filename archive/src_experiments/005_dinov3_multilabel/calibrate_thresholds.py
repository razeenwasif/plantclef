"""
Per-species threshold calibration for the DINOv3 Phase-2 multi-label head.

Builds a validation set of synthetic multi-tile mosaics, runs tiled inference,
aggregates per-mosaic probabilities, and for each species sweeps a threshold
grid to find the value that maximises the species' binary F1 across the
validation set. Species that appear too rarely in val fall back to the global
best threshold.

Output JSON: {species_id (str): threshold (float)}.
Also reports the expected macro-F1-per-sample on the val set under two regimes:
  - global best threshold
  - calibrated per-species thresholds
so we can see whether the calibration actually helps before spending a Kaggle
submission slot.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import build_default_transform
from mosaic_dataset import MosaicDataset, DEFAULT_K_DIST, compose_mosaic
from run_inference import load_model_from_checkpoint, encode_tiles_to_probs

_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
if str(_FOUR) not in sys.path:
    sys.path.insert(0, str(_FOUR))
from tiling import get_tiles  # type: ignore  # noqa: E402
from dataset import load_train_metadata, resolve_image_paths  # type: ignore  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--n-mosaics", type=int, default=1500)
    p.add_argument("--canvas-size", type=int, default=896)
    p.add_argument("--tile-size", type=int, default=384)
    p.add_argument("--tile-overlap", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--min-positives", type=int, default=3,
                   help="Species needs this many val positives to get a tuned threshold.")
    p.add_argument("--threshold-grid", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
                   help="Comma-separated thresholds to sweep.")
    p.add_argument("--top-n-eval", type=int, default=10,
                   help="Top-N cap when evaluating macro-F1-per-sample (matches submission format).")
    return p.parse_args()


def build_val_mosaics(
    args: argparse.Namespace,
    species_ids: list[str],
) -> tuple[list[Image.Image], list[set[str]]]:
    df = load_train_metadata(args.metadata_csv)
    df = resolve_image_paths(df, args.image_root, verify=False)
    df = df[df["species_id"].astype(str).isin(set(species_ids))]
    if len(df) == 0:
        sys.exit("ERROR: no training images match species_ids from checkpoint.")

    ds = MosaicDataset(
        metadata_df=df,
        species_ids=species_ids,
        canvas_size=args.canvas_size,
        k_dist=DEFAULT_K_DIST,
        samples_per_epoch=args.n_mosaics,
        transform=None,
        seed=args.seed,
        augment=False,
    )

    canvases: list[Image.Image] = []
    truths: list[set[str]] = []
    for i in range(args.n_mosaics):
        rng_i = ds._rng_for(i)
        K = ds._sample_k(rng_i)
        chosen_species = rng_i.sample(
            ds._species_with_data, k=min(K, len(ds._species_with_data))
        )
        crops, picked = [], []
        for sid in chosen_species:
            img = ds._sample_image(sid, rng_i)
            if img is not None:
                crops.append(img)
                picked.append(sid)
            if len(crops) >= K:
                break
        if not crops:
            continue
        canvas = compose_mosaic(crops, args.canvas_size, rng_i)
        canvases.append(canvas)
        truths.append(set(picked))
    logger.info(f"Built {len(canvases)} val mosaics")
    return canvases, truths


@torch.no_grad()
def score_mosaics(
    canvases: list[Image.Image],
    model,
    transform,
    args: argparse.Namespace,
) -> np.ndarray:
    stride = max(1, args.tile_size - args.tile_overlap)
    n_classes = model.n_classes
    out = np.zeros((len(canvases), n_classes), dtype=np.float32)
    for i, canvas in enumerate(canvases):
        tiles, _ = get_tiles(canvas, args.tile_size, stride)
        if not tiles:
            continue
        tile_probs = encode_tiles_to_probs(
            model, transform, tiles, args.device, args.batch_size, args.bf16
        )
        out[i] = tile_probs.max(dim=0).values.numpy()
        if (i + 1) % 100 == 0 or i + 1 == len(canvases):
            logger.info(f"  scored {i+1}/{len(canvases)}")
    return out


def macro_f1_per_sample(
    probs: np.ndarray,
    truths: list[set[str]],
    species_ids: list[str],
    thresholds: np.ndarray,
    top_n: int,
) -> float:
    """Average F1 over samples, predicting top_n species that pass per-class threshold."""
    f1s = []
    for i, true in enumerate(truths):
        score = probs[i]
        passes = score >= thresholds
        idx = np.where(passes)[0]
        if idx.size > top_n:
            # Keep the top_n by score
            order = np.argsort(-score[idx])
            idx = idx[order[:top_n]]
        pred = {species_ids[j] for j in idx}
        if not true and not pred:
            f1s.append(1.0); continue
        if not true or not pred:
            f1s.append(0.0); continue
        tp = len(true & pred)
        if tp == 0:
            f1s.append(0.0); continue
        p = tp / len(pred); r = tp / len(true)
        f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


def main() -> None:
    args = parse_args()
    grid = np.array([float(t) for t in args.threshold_grid.split(",")], dtype=np.float32)

    model, species_ids = load_model_from_checkpoint(args.checkpoint, args.device)
    transform = build_default_transform(args.tile_size)
    species_to_idx = {sid: i for i, sid in enumerate(species_ids)}

    canvases, truths = build_val_mosaics(args, species_ids)
    logger.info(f"Scoring {len(canvases)} mosaics (canvas={args.canvas_size}) ...")
    probs = score_mosaics(canvases, model, transform, args)

    # Build binary label matrix (N, C)
    N, C = probs.shape
    y = np.zeros((N, C), dtype=np.uint8)
    for i, true in enumerate(truths):
        for sid in true:
            j = species_to_idx.get(sid)
            if j is not None:
                y[i, j] = 1
    pos_counts = y.sum(axis=0)
    logger.info(f"Species with ≥{args.min_positives} positives: {(pos_counts >= args.min_positives).sum()} / {C}")

    # Global best threshold (baseline)
    logger.info("Sweeping global threshold ...")
    best_global_f1, best_global_thr = -1.0, float(grid[0])
    for tau in grid:
        f1 = macro_f1_per_sample(
            probs, truths, species_ids,
            thresholds=np.full(C, tau, dtype=np.float32),
            top_n=args.top_n_eval,
        )
        if f1 > best_global_f1:
            best_global_f1, best_global_thr = f1, float(tau)
        logger.info(f"  global thr={tau:.2f}  macro_F1={f1:.4f}")
    logger.info(f"Best global: thr={best_global_thr:.2f}  F1={best_global_f1:.4f}")

    # Per-species: for each species with enough positives, choose threshold that
    # maximises that species' binary F1 across the val set.
    logger.info("Calibrating per-species thresholds (binary-F1 proxy) ...")
    per_thr = np.full(C, best_global_thr, dtype=np.float32)
    n_tuned = 0
    for j in range(C):
        if pos_counts[j] < args.min_positives:
            continue
        yj = y[:, j]
        sj = probs[:, j]
        best_f1, best_tau = -1.0, best_global_thr
        for tau in grid:
            pred = (sj >= tau).astype(np.uint8)
            tp = int(((pred == 1) & (yj == 1)).sum())
            fp = int(((pred == 1) & (yj == 0)).sum())
            fn = int(((pred == 0) & (yj == 1)).sum())
            if tp == 0:
                f1 = 0.0
            else:
                p = tp / (tp + fp); r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1, best_tau = f1, float(tau)
        per_thr[j] = best_tau
        n_tuned += 1
    logger.info(f"Tuned {n_tuned} / {C} species (others keep global thr={best_global_thr:.2f})")

    # Evaluate calibrated macro-F1-per-sample
    cal_f1 = macro_f1_per_sample(
        probs, truths, species_ids, thresholds=per_thr, top_n=args.top_n_eval
    )
    logger.info(f"Calibrated macro-F1-per-sample: {cal_f1:.4f}  (global baseline: {best_global_f1:.4f})")

    # Save threshold JSON (sid -> float)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {species_ids[j]: float(per_thr[j]) for j in range(C)}
    report = {
        "best_global_threshold": best_global_thr,
        "best_global_macro_f1": best_global_f1,
        "calibrated_macro_f1": cal_f1,
        "n_species_tuned": int(n_tuned),
        "n_species_fallback": int(C - n_tuned),
        "n_val_mosaics": int(N),
        "min_positives": int(args.min_positives),
        "top_n_eval": int(args.top_n_eval),
        "canvas_size": int(args.canvas_size),
        "grid": [float(x) for x in grid],
    }
    with open(out_path, "w") as f:
        json.dump({"thresholds": out, "report": report}, f, indent=2)
    logger.info(f"Wrote {out_path}")
    logger.info(f"Report: {json.dumps(report, indent=2)}")


if __name__ == "__main__":
    main()
