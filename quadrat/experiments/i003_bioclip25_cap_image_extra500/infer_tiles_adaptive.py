"""
Tile-based inference with a trained BioCLIP 2.5 multi-task checkpoint.

Slides a configurable tiling pattern over each full-resolution quadrat image,
runs the species classification head on every tile, then aggregates per-tile
logits to a single image-level score.

Tiling modes
------------
  whole         single tile = the full image
  grid_2x2      2×2 grid  (4 tiles)
  grid_3x3      3×3 grid  (9 tiles)
  grid_4x4      4×4 grid  (16 tiles)
  five_crop     centre + 4 corners (5 tiles)
  sliding       sliding window; stride = tile_size × (1 - overlap)
  multiscale    whole + grid_2x2 + grid_4x4  (21 tiles)

Aggregation modes
-----------------
  max           element-wise max over tile logits
  mean          simple mean of tile logits
  softmax_mean  mean over softmax probabilities (in log-space for stability)

Selection modes (adaptive variable-length prediction)
------------------------------------------------------
  fixed_topk          : classic top-k (uses --top-ks values)
  prob_threshold      : keep all species with prob >= threshold (clamped to [min-k, max-k])
  relative_threshold  : keep species within threshold fraction of top-1 prob
  gap                 : cut at the first large relative drop in sorted probabilities

Outputs
-------
  {output_dir}/
    {agg}_{selection_key}/
      submission.csv           PlantCLEF format: quadrat_id, species_ids
      predictions_scored.csv   quadrat_id, rank, species_id, score
      run_config.json
      summary.json
    logits/                    (only when --save-logits)
      {agg}_logits.pt
    tile_preview_{stem}.png    (if --save-tile-preview, first image only)

Usage examples
--------------
  # Whole-image inference (fastest)
  python infer_tiles.py --checkpoint outputs/train/checkpoints/best.pt \\
                        --image-dir /data/test/images

  # Multi-agg + adaptive selection sweep
  python infer_tiles.py --checkpoint ... --tile-mode grid_4x4 \\
                        --agg-modes softmax_mean \\
                        --selection-modes fixed_topk relative_threshold prob_threshold gap \\
                        --top-ks 2 3 4 --min-k 2 --max-k 5 \\
                        --relative-thresholds 0.15 0.20 0.25 0.30 \\
                        --prob-thresholds 0.02 0.03 0.05 \\
                        --gap-ratios 0.40 0.50 0.60 \\
                        --save-logits
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from model import load_checkpoint_model, BIOCLIP25_MODEL_NAME
from transforms import val_transform
from utils import resolve_device, amp_autocast, save_json

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

TILING_MODES = (
    "whole", "grid_2x2", "grid_3x3", "grid_4x4",
    "five_crop", "sliding", "multiscale",
)
AGG_MODES = ("max", "mean", "softmax_mean")
SELECTION_MODES = ("fixed_topk", "prob_threshold", "relative_threshold", "gap")


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

class TileInfo(NamedTuple):
    left:   int
    top:    int
    right:  int
    bottom: int
    mode:   str
    index:  int


def _clamp(l: float, t: float, r: float, b: float,
           w: int, h: int) -> tuple[int, int, int, int]:
    cl = max(0, int(round(l)))
    ct = max(0, int(round(t)))
    cr = min(w, int(round(r)))
    cb = min(h, int(round(b)))
    cr = max(cl + 1, cr)
    cb = max(ct + 1, cb)
    return cl, ct, cr, cb


def _crop_boxes(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    mode: str,
) -> list[tuple[TileInfo, Image.Image]]:
    w, h = image.size
    out = []
    for idx, (l, t, r, b) in enumerate(boxes):
        cl, ct, cr, cb = _clamp(l, t, r, b, w, h)
        info = TileInfo(cl, ct, cr, cb, mode, idx)
        out.append((info, image.crop((cl, ct, cr, cb))))
    return out


def _tiles_whole(img: Image.Image) -> list[tuple[TileInfo, Image.Image]]:
    w, h = img.size
    return _crop_boxes(img, [(0, 0, w, h)], "whole")


def _tiles_grid(
    img: Image.Image,
    n: int,
    overlap: float = 0.0,
) -> list[tuple[TileInfo, Image.Image]]:
    w, h = img.size
    cw, ch = w / n, h / n
    boxes = []
    for row in range(n):
        for col in range(n):
            x1 = col * cw - (cw * overlap if overlap > 0 else 0)
            y1 = row * ch - (ch * overlap if overlap > 0 else 0)
            x2 = (col + 1) * cw + (cw * overlap if overlap > 0 else 0)
            y2 = (row + 1) * ch + (ch * overlap if overlap > 0 else 0)
            boxes.append((x1, y1, x2, y2))
    return _crop_boxes(img, boxes, f"grid_{n}x{n}")


def _tiles_five_crop(
    img: Image.Image, tile_size: int
) -> list[tuple[TileInfo, Image.Image]]:
    w, h = img.size
    s = min(w, h, tile_size)
    cx, cy, hs = w // 2, h // 2, s // 2
    boxes = [
        (cx - hs, cy - hs, cx - hs + s, cy - hs + s),  # centre
        (0, 0, s, s),                                    # top-left
        (w - s, 0, w, s),                               # top-right
        (0, h - s, s, h),                               # bottom-left
        (w - s, h - s, w, h),                           # bottom-right
    ]
    return _crop_boxes(img, boxes, "five_crop")


def _tiles_sliding(
    img: Image.Image, tile_size: int, overlap: float
) -> list[tuple[TileInfo, Image.Image]]:
    w, h = img.size
    if w <= tile_size and h <= tile_size:
        return _tiles_whole(img)

    stride = max(1, int(tile_size * (1.0 - overlap)))

    def _positions(dim: int) -> list[int]:
        pos = list(range(0, max(dim - tile_size + 1, 1), stride))
        if not pos or pos[-1] + tile_size < dim:
            pos.append(max(0, dim - tile_size))
        return sorted(set(pos))

    boxes: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for y in _positions(h):
        for x in _positions(w):
            box = (x, y,
                   min(x + tile_size, w),
                   min(y + tile_size, h))
            if box not in seen:
                seen.add(box)
                boxes.append(box)

    return _crop_boxes(img, boxes, "sliding")


def extract_tiles(
    image: Image.Image,
    mode: str,
    tile_size: int = 448,
    overlap: float = 0.5,
    max_tiles: int | None = None,
) -> list[tuple[TileInfo, Image.Image]]:
    """
    Extract tiles from a PIL image according to the specified mode.

    Parameters
    ----------
    image     : input PIL image (RGB)
    mode      : one of TILING_MODES
    tile_size : pixel side-length for sliding / five_crop modes
    overlap   : overlap fraction for grid / sliding modes
    max_tiles : hard cap on number of tiles (deterministic truncation)

    Returns
    -------
    List of (TileInfo, PIL.Image) pairs.
    """
    if mode == "whole":
        tiles = _tiles_whole(image)
    elif mode == "grid_2x2":
        tiles = _tiles_grid(image, 2, overlap)
    elif mode == "grid_3x3":
        tiles = _tiles_grid(image, 3, overlap)
    elif mode == "grid_4x4":
        tiles = _tiles_grid(image, 4, overlap)
    elif mode == "five_crop":
        tiles = _tiles_five_crop(image, tile_size)
    elif mode == "sliding":
        tiles = _tiles_sliding(image, tile_size, overlap)
    elif mode == "multiscale":
        tiles = _tiles_whole(image)
        tiles += _tiles_grid(image, 2, overlap)
        tiles += _tiles_grid(image, 4, overlap)
    else:
        raise ValueError(f"Unknown tiling mode {mode!r}. Choose from {TILING_MODES}")

    if max_tiles and len(tiles) > max_tiles:
        tiles = tiles[:max_tiles]

    return tiles


# ---------------------------------------------------------------------------
# Tile preview (debug)
# ---------------------------------------------------------------------------

def save_tile_preview(
    image: Image.Image,
    tiles: list[tuple[TileInfo, Image.Image]],
    output_path: str,
    max_preview: int = 64,
) -> None:
    from PIL import ImageDraw
    W, H = image.size
    scale = min(1.0, 1024 / max(W, H))
    pw, ph = max(1, int(W * scale)), max(1, int(H * scale))
    preview = image.resize((pw, ph), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(preview, "RGBA")
    colors = [
        (255, 80, 80, 140), (80, 200, 80, 140),
        (80, 120, 255, 140), (255, 200, 0, 140), (200, 80, 255, 140),
    ]
    for i, (info, _) in enumerate(tiles[:max_preview]):
        c = colors[i % len(colors)]
        draw.rectangle(
            [int(info.left * scale), int(info.top * scale),
             int(info.right * scale) - 1, int(info.bottom * scale) - 1],
            outline=c[:3], width=2,
        )
        draw.text((int(info.left * scale) + 3, int(info.top * scale) + 2),
                  str(info.index), fill=(255, 255, 255, 230))
    preview.save(output_path, format="PNG")
    logger.info(f"Tile preview saved: {output_path}  ({len(tiles)} tiles)")


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_image(
    image: Image.Image,
    model,
    preprocess,
    device: str,
    tile_mode: str,
    tile_size: int,
    overlap: float,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_tiles: int | None,
) -> tuple[list[tuple[TileInfo, Image.Image]], torch.Tensor]:
    """
    Extract tiles, preprocess, and run one model forward pass.

    Returns
    -------
    tiles       : list of (TileInfo, PIL.Image)
    tile_logits : float32 CPU tensor of shape (num_tiles, num_species)
    """
    tiles = extract_tiles(image, tile_mode, tile_size=tile_size,
                          overlap=overlap, max_tiles=max_tiles)
    if not tiles:
        return tiles, torch.empty(0)

    tile_tensors = torch.stack([preprocess(t) for _, t in tiles])  # (N, 3, H, W)

    all_logits: list[torch.Tensor] = []
    for i in range(0, len(tile_tensors), batch_size):
        batch = tile_tensors[i : i + batch_size].to(device)
        with amp_autocast(device, amp_enabled, amp_dtype):
            sp_logits, *_ = model(batch)
        all_logits.append(sp_logits.float().cpu())

    tile_logits = torch.cat(all_logits, dim=0)  # (num_tiles, num_species)
    return tiles, tile_logits


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_logits(tile_logits: torch.Tensor, agg_mode: str) -> torch.Tensor:
    if agg_mode == "max":
        return tile_logits.max(dim=0).values
    elif agg_mode == "mean":
        return tile_logits.mean(dim=0)
    elif agg_mode == "softmax_mean":
        probs = F.softmax(tile_logits, dim=1).mean(dim=0)
        return torch.log(probs.clamp_min(1e-12))
    else:
        raise ValueError(f"Unknown agg_mode {agg_mode!r}")


# ---------------------------------------------------------------------------
# Adaptive selection
# ---------------------------------------------------------------------------

@dataclass
class SelectionSpec:
    """Fully describes one selection strategy for per-image species prediction."""
    mode:   str    # fixed_topk | prob_threshold | relative_threshold | gap
    key:    str    # unique string used as part of the output subdirectory name
    params: dict   # mode-specific parameters
    min_k:  int = 1
    max_k:  int = 10


def build_selection_specs(args: argparse.Namespace) -> list[SelectionSpec]:
    """Construct the full list of SelectionSpec objects from CLI args."""
    specs: list[SelectionSpec] = []
    enabled = set(args.selection_modes)

    if "fixed_topk" in enabled:
        for k in args.top_ks:
            specs.append(SelectionSpec(
                mode="fixed_topk",
                key=f"top{k}",
                params={"top_k": k},
                min_k=args.min_k,
                max_k=args.max_k,
            ))

    if "prob_threshold" in enabled:
        for t in args.prob_thresholds:
            tag = f"{t:.4g}".rstrip("0").rstrip(".")
            specs.append(SelectionSpec(
                mode="prob_threshold",
                key=f"probT{tag}",
                params={"threshold": t},
                min_k=args.min_k,
                max_k=args.max_k,
            ))

    if "relative_threshold" in enabled:
        for t in args.relative_thresholds:
            tag = f"{t:.4g}".rstrip("0").rstrip(".")
            specs.append(SelectionSpec(
                mode="relative_threshold",
                key=f"relT{tag}",
                params={"threshold": t},
                min_k=args.min_k,
                max_k=args.max_k,
            ))

    if "gap" in enabled:
        for r in args.gap_ratios:
            tag = f"{r:.4g}".rstrip("0").rstrip(".")
            specs.append(SelectionSpec(
                mode="gap",
                key=f"gap{tag}",
                params={"gap_ratio": r},
                min_k=args.min_k,
                max_k=args.max_k,
            ))

    return specs


def adaptive_select(
    img_logits: torch.Tensor,
    spec: SelectionSpec,
    idx_to_species: list[str],
) -> tuple[list[str], list[float]]:
    """
    Select species from image-level aggregated logits according to spec.

    Returns (pred_species, pred_scores) with spec.min_k <= len <= spec.max_k.
    """
    n_classes = img_logits.shape[0]
    probs = torch.softmax(img_logits, dim=0)
    sorted_probs, sorted_indices = probs.sort(descending=True)

    mode = spec.mode
    if mode == "fixed_topk":
        k = spec.params["top_k"]

    elif mode == "prob_threshold":
        threshold = spec.params["threshold"]
        k = int((sorted_probs >= threshold).sum().item())

    elif mode == "relative_threshold":
        threshold = spec.params["threshold"]
        top_prob  = sorted_probs[0].item()
        cutoff    = top_prob * (1.0 - threshold)
        k = int((sorted_probs >= cutoff).sum().item())

    elif mode == "gap":
        gap_ratio = spec.params["gap_ratio"]
        # Find the first relative drop >= gap_ratio within [1, max_k]
        k = spec.max_k  # fallback: use max_k if no qualifying gap is found
        limit = min(len(sorted_probs), spec.max_k + 1)
        for i in range(1, limit):
            prev = sorted_probs[i - 1].item()
            curr = sorted_probs[i].item()
            if prev > 1e-8 and (prev - curr) / prev >= gap_ratio:
                k = i
                break

    else:
        raise ValueError(f"Unknown selection mode: {mode!r}")

    k = max(spec.min_k, min(spec.max_k, k, n_classes))
    selected_indices = sorted_indices[:k]

    species = [idx_to_species[i.item()] for i in selected_indices]
    scores  = [round(probs[i].item(), 6) for i in selected_indices]
    return species, scores


# ---------------------------------------------------------------------------
# Classic top-k helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def topk_from_logits(
    img_logits: torch.Tensor,
    top_k: int,
    idx_to_species: list[str],
) -> tuple[list[str], list[float]]:
    k = min(top_k, img_logits.shape[0])
    scores, indices = img_logits.topk(k)
    probs = torch.softmax(img_logits, dim=0)

    pred_species = [idx_to_species[i.item()] for i in indices]
    pred_scores  = [round(probs[i].item(), 6) for i in indices]

    return pred_species, pred_scores


# ---------------------------------------------------------------------------
# Logit adjustment (long-tail calibration)
# ---------------------------------------------------------------------------

def build_logit_adjustment(
    metadata_csv: str,
    idx_to_species: list[str],
    tau: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-class logit adjustment vector from training class frequencies.

    At inference: adjusted_logit[c] = raw_logit[c] - tau * log(prior[c])
    Subtracting log(prior) boosts rare classes (whose log(prior) is large-negative)
    and suppresses common ones, correcting the calibration bias of a CE-trained model
    on a long-tail distribution.

    Uses Laplace-smoothed priors matching LogitAdjustmentLoss in src/training/losses.py.

    Returns: (num_species,) float32 tensor — subtract from logits at inference.
    """
    df = pd.read_csv(metadata_csv, usecols=["species_id"])
    df["species_id"] = df["species_id"].astype(str).str.split(".").str[0].str.strip()
    counts = df["species_id"].value_counts()

    n = len(idx_to_species)
    raw_counts = torch.tensor(
        [counts.get(sid, 0) for sid in idx_to_species], dtype=torch.float32
    )
    priors = (raw_counts + 1) / (raw_counts.sum() + n)
    return tau * torch.log(priors + eps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Tile-based inference for BioCLIP 2.5 multi-task model → PlantCLEF submission.\n"
            "Supports adaptive variable-length prediction selection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .pt checkpoint from train.py.")
    p.add_argument("--image-dir",  required=True,
                   help="Directory containing test quadrat images.")

    # Tiling
    p.add_argument("--tile-mode",  default="multiscale", choices=list(TILING_MODES))
    p.add_argument("--tile-size",  type=int,   default=448)
    p.add_argument("--overlap",    type=float, default=0.25)
    p.add_argument("--max-tiles",  type=int,   default=None)

    # Aggregation — multi-value
    p.add_argument("--agg-modes", nargs="+", default=["max"], choices=list(AGG_MODES))

    # ---- Selection modes ----
    p.add_argument(
        "--selection-modes",
        nargs="+",
        default=["fixed_topk"],
        choices=list(SELECTION_MODES),
        help=(
            "Which selection strategies to run. "
            "'fixed_topk' uses --top-ks; others use their respective threshold/ratio args."
        ),
    )

    # fixed_topk parameters
    p.add_argument("--top-ks", nargs="+", type=int, default=[5],
                   help="Top-k values used by the fixed_topk selection mode.")

    # Shared min/max bounds for adaptive modes
    p.add_argument("--min-k", type=int, default=1,
                   help="Minimum number of species to predict (adaptive modes).")
    p.add_argument("--max-k", type=int, default=10,
                   help="Maximum number of species to predict (adaptive modes).")

    # prob_threshold parameters
    p.add_argument("--prob-thresholds", nargs="+", type=float, default=[0.03],
                   help="Probability thresholds for prob_threshold mode.")

    # relative_threshold parameters
    p.add_argument("--relative-thresholds", nargs="+", type=float, default=[0.20],
                   help="Relative drop thresholds for relative_threshold mode.")

    # gap parameters
    p.add_argument("--gap-ratios", nargs="+", type=float, default=[0.50],
                   help="Relative gap ratios for gap selection mode.")

    # Logits saving
    p.add_argument("--save-logits", action="store_true",
                   help="Save aggregated logits per image/agg for later ensembling.")

    # Output
    p.add_argument("--output-dir", default="./outputs/tile_inference")
    p.add_argument("--save-tile-preview", action="store_true")

    # Compute
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--img-size",    type=int, default=224)
    p.add_argument("--precision",   default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device",      default="auto")
    p.add_argument("--num-workers", type=int, default=0)

    # Logit adjustment (long-tail correction)
    p.add_argument(
        "--logit-adj-tau", type=float, default=0.0, metavar="TAU",
        help=(
            "Logit adjustment strength. 0 = disabled. "
            "Subtracts tau*log(prior[c]) from each class logit before selection, "
            "boosting rare-species scores to correct long-tail calibration bias. "
            "Requires --metadata-csv. Good range to sweep: 0.25, 0.5, 0.75, 1.0."
        ),
    )
    p.add_argument(
        "--metadata-csv", default=None,
        help=(
            "Path to training metadata CSV (needs 'species_id' column). "
            "Required when --logit-adj-tau > 0."
        ),
    )

    # Misc
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N images (0 = all). Useful for smoke tests.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_combo_outputs(
    out_dir: Path,
    subdir_name: str,
    submission_rows: list[dict],
    scored_rows: list[dict],
    run_config: dict,
    total_secs: float,
    n_images: int,
    errors: int,
    idx_to_species: list[str],
    tile_mode: str,
) -> None:
    combo_dir = out_dir / subdir_name
    combo_dir.mkdir(parents=True, exist_ok=True)

    sub_path = combo_dir / "submission.csv"
    with open(sub_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quadrat_id", "species_ids"],
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(submission_rows)
    logger.info(f"Submission [{subdir_name}]: {sub_path}  ({len(submission_rows):,} rows)")

    if scored_rows:
        scored_path = combo_dir / "predictions_scored.csv"
        with open(scored_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["quadrat_id", "rank", "species_id", "score"],
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            writer.writerows(scored_rows)

    save_json(run_config, str(combo_dir / "run_config.json"))
    save_json({
        "n_images":       n_images,
        "n_errors":       errors,
        "total_secs":     round(total_secs, 1),
        "images_per_sec": round(n_images / max(total_secs, 1e-6), 2),
        "tile_mode":      tile_mode,
        "subdir":         subdir_name,
    }, str(combo_dir / "summary.json"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    device      = resolve_device(args.device)
    amp_enabled = args.precision != "fp32"
    amp_dtype   = torch.float16 if args.precision == "fp16" else torch.bfloat16

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BioCLIP 2.5 Multi-Task — Tile Inference")
    logger.info(f"  checkpoint      : {args.checkpoint}")
    logger.info(f"  image_dir       : {args.image_dir}")
    logger.info(f"  tile_mode       : {args.tile_mode}")
    logger.info(f"  tile_size       : {args.tile_size}  overlap: {args.overlap}")
    logger.info(f"  agg_modes       : {args.agg_modes}")
    logger.info(f"  selection_modes : {args.selection_modes}")
    logger.info(f"  top_ks          : {args.top_ks}")
    logger.info(f"  min_k/max_k     : {args.min_k}/{args.max_k}")
    logger.info(f"  prob_thresholds : {args.prob_thresholds}")
    logger.info(f"  rel_thresholds  : {args.relative_thresholds}")
    logger.info(f"  gap_ratios      : {args.gap_ratios}")
    logger.info(f"  save_logits     : {args.save_logits}")
    logger.info(f"  logit_adj_tau   : {args.logit_adj_tau}")
    logger.info(f"  metadata_csv    : {args.metadata_csv}")
    logger.info(f"  precision       : {args.precision}")
    logger.info(f"  device          : {device}")
    logger.info(f"  output_dir      : {out_dir}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model, encoders, config = load_checkpoint_model(args.checkpoint, device=device)
    model.eval()

    idx_to_species = encoders.get("idx_to_species", [])
    if not idx_to_species:
        raise ValueError("Checkpoint missing idx_to_species in encoders.")
    logger.info(f"Model loaded: {len(idx_to_species):,} species classes")

    preprocess = val_transform(img_size=args.img_size)

    # ------------------------------------------------------------------
    # Logit adjustment vector (optional)
    # ------------------------------------------------------------------
    logit_adjustment: torch.Tensor | None = None
    if args.logit_adj_tau > 0.0:
        if not args.metadata_csv:
            raise ValueError("--metadata-csv is required when --logit-adj-tau > 0")
        logit_adjustment = build_logit_adjustment(
            args.metadata_csv, idx_to_species, tau=args.logit_adj_tau
        ).to(device)
        logger.info(
            f"Logit adjustment loaded: tau={args.logit_adj_tau}  "
            f"min={logit_adjustment.min():.3f}  max={logit_adjustment.max():.3f}  "
            f"mean={logit_adjustment.mean():.3f}"
        )

    # ------------------------------------------------------------------
    # Build selection specs
    # ------------------------------------------------------------------
    specs = build_selection_specs(args)
    if not specs:
        raise RuntimeError(
            "No selection specs generated. Check --selection-modes and associated args."
        )
    logger.info(f"Selection specs ({len(specs)}):")
    for s in specs:
        logger.info(f"  [{s.mode}] key={s.key}  params={s.params}  k=[{s.min_k},{s.max_k}]")

    # ------------------------------------------------------------------
    # Discover images
    # ------------------------------------------------------------------
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    all_images = sorted(p for p in image_dir.rglob("*") if p.suffix in _IMAGE_EXTS)
    if not all_images:
        raise RuntimeError(f"No images found under {image_dir}")

    if args.limit > 0:
        all_images = all_images[: args.limit]
        logger.info(f"Limiting to first {args.limit} images")

    logger.info(f"Found {len(all_images):,} images to process")

    # ------------------------------------------------------------------
    # Accumulators: keyed by (agg_mode, spec.key)
    # ------------------------------------------------------------------
    results: dict[tuple[str, str], dict] = {
        (agg, spec.key): {"submission_rows": [], "scored_rows": []}
        for agg  in args.agg_modes
        for spec in specs
    }

    saved_logits: dict[str, list[torch.Tensor]] = {agg: [] for agg in args.agg_modes}
    logit_quadrat_ids: list[str] = []

    errors = 0
    t_start = time.perf_counter()
    tile_preview_saved = False
    first_agg = args.agg_modes[0]

    # ------------------------------------------------------------------
    # Inference loop — model forward only once per image
    # ------------------------------------------------------------------
    for i, img_path in enumerate(all_images):
        quadrat_id = img_path.stem
        success = False
        tiles: list = []
        per_agg_logits: dict[str, torch.Tensor] = {}

        try:
            image = Image.open(img_path).convert("RGB")

            tiles, tile_logits = infer_image(
                image       = image,
                model       = model,
                preprocess  = preprocess,
                device      = device,
                tile_mode   = args.tile_mode,
                tile_size   = args.tile_size,
                overlap     = args.overlap,
                batch_size  = args.batch_size,
                amp_enabled = amp_enabled,
                amp_dtype   = amp_dtype,
                max_tiles   = args.max_tiles,
            )

            for agg in args.agg_modes:
                agg_logit = aggregate_logits(tile_logits, agg)
                if logit_adjustment is not None:
                    agg_logit = agg_logit.to(device) - logit_adjustment
                per_agg_logits[agg] = agg_logit.cpu()

            if i == 0:
                first_logits = per_agg_logits[first_agg]
                top1_sp = idx_to_species[first_logits.argmax().item()] if tiles else "?"
                logger.info(
                    f"First image {img_path.name}: size={image.size}  "
                    f"n_tiles={len(tiles)}  top1={top1_sp}"
                )
                for ti, (info, _) in enumerate(tiles[:5]):
                    logger.info(
                        f"  tile[{info.index}] ({info.left},{info.top})-"
                        f"({info.right},{info.bottom}) "
                        f"[{info.right-info.left}×{info.bottom-info.top}px] {info.mode}"
                    )
                if len(tiles) > 5:
                    logger.info(f"  ... ({len(tiles)-5} more tiles)")

                if args.save_tile_preview and not tile_preview_saved:
                    preview_path = str(out_dir / f"tile_preview_{img_path.stem}.png")
                    save_tile_preview(image, tiles, preview_path)
                    tile_preview_saved = True

            success = True

        except Exception as exc:
            logger.warning(f"Error on {img_path.name}: {exc}")
            errors += 1

        # Append rows for every (agg, spec) combination
        for agg in args.agg_modes:
            for spec in specs:
                key = (agg, spec.key)
                if success:
                    img_logits = per_agg_logits[agg]
                    pred_species, pred_scores = adaptive_select(img_logits, spec, idx_to_species)
                else:
                    pred_species, pred_scores = [], []

                species_ids_str = (
                    "[" + ", ".join(
                        s for s in pred_species
                        if s.strip().lstrip("-").isdigit()
                    ) + "]"
                )
                results[key]["submission_rows"].append(
                    {"quadrat_id": quadrat_id, "species_ids": species_ids_str}
                )
                for rank, (sid, score) in enumerate(zip(pred_species, pred_scores), start=1):
                    results[key]["scored_rows"].append({
                        "quadrat_id": quadrat_id,
                        "rank":       rank,
                        "species_id": sid,
                        "score":      score,
                    })

        if success and args.save_logits:
            logit_quadrat_ids.append(quadrat_id)
            for agg in args.agg_modes:
                saved_logits[agg].append(per_agg_logits[agg].cpu())

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.perf_counter() - t_start
            rate    = (i + 1) / max(elapsed, 1e-6)
            eta     = (len(all_images) - i - 1) / max(rate, 1e-6)
            logger.info(
                f"  [{i+1:>5}/{len(all_images)}]  "
                f"{elapsed:.0f}s elapsed  {rate:.1f} img/s  "
                f"ETA {eta:.0f}s  errors={errors}"
            )

    total_secs = time.perf_counter() - t_start
    logger.info(
        f"Inference complete: {len(all_images):,} images in "
        f"{total_secs:.0f}s  ({len(all_images)/max(total_secs,1e-6):.1f} img/s)  "
        f"errors={errors}"
    )

    # ------------------------------------------------------------------
    # Save per-(agg, spec) outputs
    # ------------------------------------------------------------------
    base_config = {
        **vars(args),
        "device":     str(device),
        "n_species":  len(idx_to_species),
    }

    for (agg, spec_key), data in results.items():
        spec = next(s for s in specs if s.key == spec_key)
        run_cfg = {
            **base_config,
            "agg_mode":       agg,
            "selection_mode": spec.mode,
            "selection_key":  spec.key,
            "selection_params": spec.params,
            "min_k":          spec.min_k,
            "max_k":          spec.max_k,
        }
        _save_combo_outputs(
            out_dir         = out_dir,
            subdir_name     = f"{agg}_{spec_key}",
            submission_rows = data["submission_rows"],
            scored_rows     = data["scored_rows"],
            run_config      = run_cfg,
            total_secs      = total_secs,
            n_images        = len(all_images),
            errors          = errors,
            idx_to_species  = idx_to_species,
            tile_mode       = args.tile_mode,
        )

    # ------------------------------------------------------------------
    # Save aggregated logits for ensembling
    # ------------------------------------------------------------------
    if args.save_logits and logit_quadrat_ids:
        logits_dir = out_dir / "logits"
        logits_dir.mkdir(parents=True, exist_ok=True)
        for agg in args.agg_modes:
            logit_tensor = torch.stack(saved_logits[agg], dim=0)  # (N, num_species)
            save_path = logits_dir / f"{agg}_logits.pt"
            torch.save({
                "quadrat_ids":    logit_quadrat_ids,
                "logits":         logit_tensor,
                "agg_mode":       agg,
                "idx_to_species": idx_to_species,
                "checkpoint":     args.checkpoint,
                "tile_mode":      args.tile_mode,
                "tile_size":      args.tile_size,
                "overlap":        args.overlap,
                "max_tiles":      args.max_tiles,
            }, save_path)
            logger.info(
                f"Logits [{agg}]: {save_path}  "
                f"shape={tuple(logit_tensor.shape)}"
            )

    logger.info(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
