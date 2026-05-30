"""
inf_script_phen.py - PlantCLEF 2026 seasonal-phenology pivot
(paper pivot 3, 0.41346 public Macro F1).

This script departs from the anchor recipe (`inf_script.py`) in four
places, all documented in the report's Appendix C (Seasonal Phenology
Prior: Full Method):

  1. **Multi-scale tiling.** Each quadrat is partitioned into a 4x4 grid
     at the native scale (16 tiles) and re-partitioned at scale 0.8 (a
     centred inner crop covering 80% of the quadrat, again 4x4 = 16
     tiles), yielding 32 tiles per resolution.
  2. **ExG vegetation filter.** Per tile we compute the Excess Green
     index ExG = 2G - R - B and the fraction of pixels above an ExG
     threshold; tiles with fewer than 15% green pixels are dropped
     before the encoder forward pass.
  3. **Entropy-weighted Bayesian aggregation.** Tiles that survive the
     filter are pooled with the per-tile weight w_t proportional to
     exp(-H_t) * ExG_t, where H_t is the predictive entropy of tile t
     and ExG_t is its vegetation fraction. Replaces the anchor's
     unweighted softmax-mean.
  4. **Circular Gaussian day-of-year prior.** The per-species GBIF
     month-of-observation counts are smoothed into a 365-day pdf via
     circular Gaussian smoothing (sigma = 18 d) centred on each month's
     mid-point, with a small uniform mass epsilon = 0.05 to keep the
     log finite. The quadrat date d is parsed from the filename
     (YYYYMMDD); the prior log P(s | d) is added in log space at
     beta = 1.0 before logit adjustment and adaptive threshold
     selection.

Phenology CSV format (`--phenology-csv`):
  Required columns: species_id, m_01, m_02, ..., m_12   (raw GBIF
  observation counts; rows can be missing - defaults to uniform).

Usage:

  python inf_script_phen.py \\
      --checkpoint     /path/to/i002/outputs/last_blocks/checkpoints/best.pt \\
      --image-dir      /path/to/plantclef/test \\
      --metadata-csv   /path/to/training_manifest.csv \\
      --phenology-csv  /path/to/gbif_phenology.csv \\
      --output         submission_phen.csv

The model loader, preprocessing transform, and tile-extraction code
live in `../shared/bioclip25_multitask/` relative to this script.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Pull in the trained model + transforms + tiling helpers from the
# shared BioCLIP-2.5 multitask package. Resolved relative to this file
# so the script remains portable across hosts.
EXP_DIR = (Path(__file__).resolve().parent.parent
           / "shared" / "bioclip25_multitask")
if not EXP_DIR.is_dir():
    raise SystemExit(
        f"Expected shared dir not found: {EXP_DIR}\n"
        f"Edit EXP_DIR in this file if the BioCLIP 2.5 multitask sources live elsewhere."
    )
sys.path.insert(0, str(EXP_DIR))

from model import load_checkpoint_model               # noqa: E402
from transforms import val_transform                  # noqa: E402
from utils import resolve_device, amp_autocast        # noqa: E402


# --- Recipe constants (anchor + phenology pivot) -----------------------------
TILE_GRID       = 4         # 4x4 = 16 tiles per scale
TILE_SCALES     = (1.0, 0.8)
RESOLUTIONS     = (224, 336)
LOGIT_ADJ_TAU   = 0.25
PROB_THRESHOLD  = 0.03
K_MIN           = 2
K_MAX           = 10

# Phenology
PHEN_BETA       = 1.0       # mixing weight; paper used 1.0
PHEN_SIGMA_DAYS = 18.0      # circular Gaussian smoothing
PHEN_UNIFORM_EPS = 0.05     # uniform mass in the DOY pdf

# Vegetation filter
EXG_THRESHOLD   = 20.0      # per-pixel ExG threshold (matches experiments/008)
MIN_VEG_FRAC    = 0.15      # tile dropped below this vegetation fraction


# --- Positional-embedding resampling (open_clip ships fixed-size pos_embed) --

def _resample_pos_embed(model, res: int) -> None:
    """
    Resample the ViT-H/14 positional embedding from its native 224 px grid
    (16x16 patches + CLS) to the patch grid implied by `res`. Idempotent
    at the trained resolution. Required to run the same model at 336 px.
    """
    visual     = model.backbone.visual
    patch_size = visual.conv1.kernel_size[0]
    new_grid   = res // patch_size

    pe       = visual.positional_embedding
    cls_pe   = pe[:1]
    grid_pe  = pe[1:]
    old_grid = int(math.sqrt(grid_pe.shape[0]))
    if old_grid == new_grid:
        return

    embed_dim = grid_pe.shape[-1]
    grid_pe = grid_pe.reshape(1, old_grid, old_grid, embed_dim).permute(0, 3, 1, 2)
    grid_pe = F.interpolate(grid_pe, size=(new_grid, new_grid),
                            mode='bicubic', align_corners=False)
    grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(new_grid * new_grid, embed_dim)

    visual.positional_embedding = nn.Parameter(torch.cat([cls_pe, grid_pe], dim=0))


# --- Tiling: 4x4 grid at scales 1.0 and 0.8 ----------------------------------

def grid4x4_tiles(image: Image.Image, scale: float) -> list[Image.Image]:
    """
    Partition `image` into a 4x4 grid taken from a centred sub-rectangle
    of side `scale * min_side` (scale = 1.0 reproduces the anchor tiling;
    scale = 0.8 yields a centred inner crop, matching the dashed inset in
    Fig. C of the report).
    """
    W, H = image.size
    sw = int(round(W * scale))
    sh = int(round(H * scale))
    ox = (W - sw) // 2
    oy = (H - sh) // 2
    cw = sw / TILE_GRID
    ch = sh / TILE_GRID
    tiles: list[Image.Image] = []
    for row in range(TILE_GRID):
        for col in range(TILE_GRID):
            x1 = int(round(ox + col * cw))
            y1 = int(round(oy + row * ch))
            x2 = int(round(ox + (col + 1) * cw))
            y2 = int(round(oy + (row + 1) * ch))
            x2 = max(x1 + 1, min(W, x2))
            y2 = max(y1 + 1, min(H, y2))
            tiles.append(image.crop((x1, y1, x2, y2)))
    return tiles


# --- Excess Green vegetation filter ------------------------------------------

def _exg_vegetation_fraction(tile: Image.Image, thresh: float = EXG_THRESHOLD) -> float:
    """
    Fraction of tile pixels classified as vegetation by Excess Green.
    ExG = 2G - R - B, with an extra G > R AND G > B guard that rejects
    bright non-green patches (concrete, sky, skin).
    """
    arr = np.asarray(tile.convert("RGB"), dtype=np.int16)
    r = arr[..., 0]; g = arr[..., 1]; b = arr[..., 2]
    exg  = 2 * g - r - b
    mask = (exg > thresh) & (g > r) & (g > b)
    return float(mask.mean())


# --- Day-of-year extraction and circular Gaussian phenology pdf --------------

_DATE_RE = re.compile(r"(\d{8})")


def doy_from_stem(stem: str) -> int | None:
    """
    Pull the day-of-year (1..365) out of a PlantCLEF quadrat filename
    stem of the form `...-YYYYMMDD` (e.g. `CBN-Pla-A1-20130808`).
    Day 366 in leap years is clamped to 365 to keep the pdf domain fixed.
    Returns None if no valid YYYYMMDD substring is found.
    """
    for hit in _DATE_RE.findall(stem):
        try:
            y = int(hit[0:4]); m = int(hit[4:6]); d = int(hit[6:8])
            if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                from datetime import date
                doy = date(y, m, d).timetuple().tm_yday
                return min(doy, 365)
        except (ValueError, OverflowError):
            continue
    return None


# Month mid-points (day-of-year of the 15th of each month, non-leap)
_MONTH_MID_DOY = np.array(
    [15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350], dtype=np.float32
)


def _circular_gaussian_kernel(sigma: float = PHEN_SIGMA_DAYS) -> np.ndarray:
    """
    Pre-compute the (365, 12) matrix K[d, m] = N_circ(d | mu_m, sigma),
    each column normalised so it sums to 1 over d = 1..365. Circular
    distance is min(|d - mu_m|, 365 - |d - mu_m|).
    """
    doy = np.arange(1, 366, dtype=np.float32)[:, None]    # (365, 1)
    mu  = _MONTH_MID_DOY[None, :]                          # (1, 12)
    raw_diff = np.abs(doy - mu)
    circ_diff = np.minimum(raw_diff, 365.0 - raw_diff)
    k = np.exp(-0.5 * (circ_diff / sigma) ** 2)
    k /= k.sum(axis=0, keepdims=True)                      # normalise per month
    return k


def build_phenology_pdf(
    phenology_csv: Path,
    idx_to_species: list[str],
    sigma_days: float = PHEN_SIGMA_DAYS,
    uniform_eps: float = PHEN_UNIFORM_EPS,
) -> torch.Tensor:
    """
    Load per-species GBIF month-of-observation counts and convert them
    into a (num_species, 365) day-of-year pdf via the circular Gaussian
    smoothing of report Appendix C:

        P(s | d) = (1 - eps) * (1 / Z_s) * sum_m c_{s,m} * N_circ(d | mu_m, sigma)
                 + eps / 365

    where mu_m is the mid-point day-of-year of month m, sigma = 18 d by
    default, and eps = 0.05 prevents hard zeros in tail months.

    Species absent from the CSV (or whose counts sum to zero) fall back
    to a uniform pdf, so they contribute no phenological signal.
    """
    if not phenology_csv.is_file():
        raise SystemExit(f"--phenology-csv not found: {phenology_csv}")

    df = pd.read_csv(phenology_csv)
    df["species_id"] = df["species_id"].astype(str)
    by_id = df.set_index("species_id")
    month_cols = [f"m_{m:02d}" for m in range(1, 13)]
    missing = [c for c in month_cols if c not in by_id.columns]
    if missing:
        raise SystemExit(
            f"--phenology-csv missing expected columns: {missing}. "
            f"Need species_id + m_01..m_12."
        )

    kernel = _circular_gaussian_kernel(sigma_days)         # (365, 12)
    uniform = np.full(365, 1.0 / 365.0, dtype=np.float32)

    rows: list[np.ndarray] = []
    n_missing = 0
    for s in idx_to_species:
        if s in by_id.index:
            counts = by_id.loc[s, month_cols].to_numpy(dtype=np.float32)
            tot = float(counts.sum())
            if tot > 0:
                # Smoothed component: weighted sum of per-month kernels,
                # already normalised per month.
                smoothed = kernel @ (counts / tot)         # (365,)
                pdf = (1.0 - uniform_eps) * smoothed + uniform_eps * uniform
                rows.append(pdf.astype(np.float32))
                continue
        rows.append(uniform.copy())
        n_missing += 1

    if n_missing:
        print(f"[inf_phen] phenology: {n_missing:,} / {len(idx_to_species):,} "
              f"species had no GBIF rows -> uniform DOY fallback")

    return torch.tensor(np.stack(rows))                    # (K, 365)


# --- Training-prior for logit adjustment -------------------------------------

def build_log_prior(metadata_csv: Path, idx_to_species: list[str]) -> torch.Tensor:
    """Laplace-smoothed training-set prior log pi_tilde_s, aligned with idx_to_species."""
    if not metadata_csv.is_file():
        raise SystemExit(f"--metadata-csv not found: {metadata_csv}")
    df = pd.read_csv(metadata_csv, usecols=["species_id"])
    counts_by_id = df["species_id"].astype(str).value_counts().to_dict()
    counts = torch.tensor(
        [counts_by_id.get(str(s), 0) for s in idx_to_species],
        dtype=torch.float32,
    )
    smoothed = counts + 1.0
    return torch.log(smoothed / smoothed.sum())


# --- Per-image inference -----------------------------------------------------

@torch.no_grad()
def encode_tiles(
    model,
    tile_pils: list[Image.Image],
    transform,
    device: str,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Preprocess each PIL tile through `transform` (which sets the
    resolution via its CenterCrop), forward through the species head,
    return per-tile softmax probabilities at the given resolution.

    Returns: (N_tiles, num_species) float32 tensor on `device`.
    """
    chunks: list[torch.Tensor] = []
    for i in range(0, len(tile_pils), batch_size):
        batch = tile_pils[i : i + batch_size]
        x = torch.stack([transform(im) for im in batch]).to(device)
        with amp_autocast(device, amp_enabled, amp_dtype):
            sp_logits, _, _ = model(x)
        chunks.append(F.softmax(sp_logits.float(), dim=-1))
    return torch.cat(chunks, dim=0)


def aggregate_image_probs(
    tile_probs: torch.Tensor,
    exg_fracs: torch.Tensor,
) -> torch.Tensor | None:
    """
    Bayesian entropy-weighted aggregation of per-tile distributions.

    w_t proportional to exp(-H_t) * ExG_t, where H_t is the per-tile
    predictive entropy (natural log) and ExG_t is the tile's vegetation
    fraction. Returns the weighted mean over tiles, or None if every
    weight collapsed to zero (e.g. all tiles dropped by the ExG filter
    at this resolution).
    """
    if tile_probs.numel() == 0:
        return None

    # Predictive entropy per tile (natural log).
    log_probs = torch.log(tile_probs.clamp_min(1e-12))
    H = -(tile_probs * log_probs).sum(dim=-1)              # (N_tiles,)
    w = torch.exp(-H) * exg_fracs.to(tile_probs.device)    # (N_tiles,)
    Z = float(w.sum().item())
    if not math.isfinite(Z) or Z <= 0.0:
        return None
    w = w / Z
    return (w[:, None] * tile_probs).sum(dim=0)            # (K,)


def select_species(
    image_probs: torch.Tensor,
    log_prior: torch.Tensor,
    phen_pdf: torch.Tensor,
    doy: int | None,
    beta: float,
    idx_to_species: list[str],
) -> list[str]:
    """
    Apply (1) the circular-Gaussian phenology log-prior, (2) the
    training-prior logit adjustment, then (3) the adaptive probability
    threshold. Returns species-id strings in descending post-prior order.
    """
    log_probs = torch.log(image_probs.clamp_min(1e-12))

    # Phenology log-prior (paper Eq. for log p_final).
    if doy is not None and 1 <= doy <= 365:
        log_phen = torch.log(phen_pdf[:, doy - 1].to(log_probs.device).clamp_min(1e-12))
        log_probs = log_probs + beta * log_phen

    # Class-prior logit adjustment.
    log_adj = log_probs - LOGIT_ADJ_TAU * log_prior

    adj_probs = F.softmax(log_adj, dim=-1)

    sorted_probs, sorted_idx = adj_probs.sort(descending=True)
    n_above = int((sorted_probs >= PROB_THRESHOLD).sum().item())
    k = max(K_MIN, min(K_MAX, n_above))

    return [idx_to_species[i.item()] for i in sorted_idx[:k]]


# --- Driver ------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix in _IMAGE_EXTS)


def format_species_ids(species: list[str]) -> str:
    clean = [s for s in species if s.strip().lstrip("-").isdigit()]
    return "[" + ", ".join(clean) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Seasonal-phenology pivot inference (paper pivot 3, 0.41346 public).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint",    required=True, type=Path,
                    help="Path to i002 last-blocks best.pt")
    ap.add_argument("--image-dir",     required=True, type=Path,
                    help="Directory of PlantCLEF test quadrat images")
    ap.add_argument("--metadata-csv",  required=True, type=Path,
                    help="Training manifest CSV (needs species_id column) "
                         "used for the Laplace-smoothed prior pi_tilde_s")
    ap.add_argument("--phenology-csv", required=True, type=Path,
                    help="GBIF month-of-observation histogram CSV - "
                         "columns: species_id, m_01, m_02, ..., m_12")
    ap.add_argument("--output",        default=Path("submission_phen.csv"), type=Path,
                    help="Where to write the PlantCLEF submission CSV")
    ap.add_argument("--beta",          type=float, default=PHEN_BETA,
                    help=f"Phenology strength beta (default {PHEN_BETA}, paper value)")
    ap.add_argument("--min-veg-frac",  type=float, default=MIN_VEG_FRAC,
                    help="ExG vegetation-fraction floor (paper: 0.15)")
    ap.add_argument("--batch-size",    type=int, default=32)
    ap.add_argument("--precision",     choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device",        default="auto",
                    help="auto | cuda | cuda:N | cpu")
    ap.add_argument("--limit",         type=int, default=0,
                    help="Process only the first N images (0 = all)")
    args = ap.parse_args()

    device = resolve_device(args.device)
    amp_enabled = args.precision != "fp32"
    amp_dtype   = torch.float16 if args.precision == "fp16" else torch.bfloat16

    print(f"[inf_phen] device={device}  precision={args.precision}")
    print(f"[inf_phen] loading checkpoint @ {len(RESOLUTIONS)} resolutions  {args.checkpoint}")

    # One model instance per resolution: open_clip's pos_embed is fixed
    # at load time and the same instance cannot be reused at 336 px.
    models_by_size: dict[int, object] = {}
    encoders = None
    for sz in RESOLUTIONS:
        m, enc, _ = load_checkpoint_model(str(args.checkpoint), device=device)
        _resample_pos_embed(m, sz)
        m.eval()
        models_by_size[sz] = m
        if encoders is None:
            encoders = enc
    idx_to_species = encoders["idx_to_species"]
    print(f"[inf_phen] {len(idx_to_species):,} species classes")

    print(f"[inf_phen] building Laplace-smoothed training prior from {args.metadata_csv}")
    log_prior = build_log_prior(args.metadata_csv, idx_to_species).to(device)

    print(f"[inf_phen] building circular-Gaussian DOY pdf from {args.phenology_csv}  "
          f"(sigma={PHEN_SIGMA_DAYS} d, eps={PHEN_UNIFORM_EPS})")
    phen_pdf = build_phenology_pdf(args.phenology_csv, idx_to_species).to(device)

    transforms_by_size = {sz: val_transform(sz) for sz in RESOLUTIONS}

    images = find_images(args.image_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")
    print(f"[inf_phen] {len(images):,} images  -  4x4 grid at scales {list(TILE_SCALES)}, "
          f"resolutions {list(RESOLUTIONS)}")
    print(f"[inf_phen] ExG filter min_frac={args.min_veg_frac} (thresh={EXG_THRESHOLD})  "
          f"entropy-weighted aggregation  beta={args.beta}  "
          f"LA tau={LOGIT_ADJ_TAU}  T={PROB_THRESHOLD}  k in [{K_MIN},{K_MAX}]")
    print()

    rows: list[dict[str, str]] = []
    t0 = time.time()
    n_errors = 0
    n_no_doy = 0
    n_all_filtered = 0

    for i, img_path in enumerate(images, 1):
        quadrat_id = img_path.stem
        doy = doy_from_stem(quadrat_id)
        if doy is None:
            n_no_doy += 1

        try:
            with Image.open(img_path) as im:
                image = im.convert("RGB")
        except Exception as e:
            print(f"  ! {quadrat_id}: failed to read ({e}) - emitting empty row")
            rows.append({"quadrat_id": quadrat_id, "species_ids": "[]"})
            n_errors += 1
            continue

        # 1. Multi-scale tiling: 4x4 at native scale + 4x4 at scale 0.8.
        all_tiles: list[Image.Image] = []
        for scale in TILE_SCALES:
            all_tiles.extend(grid4x4_tiles(image, scale))

        # 2. ExG vegetation filter (drop tiles below threshold).
        exg_fracs_full = [_exg_vegetation_fraction(t) for t in all_tiles]
        keep_mask = [f >= args.min_veg_frac for f in exg_fracs_full]
        kept_tiles  = [t for t, k in zip(all_tiles, keep_mask) if k]
        kept_fracs  = [f for f, k in zip(exg_fracs_full, keep_mask) if k]

        if not kept_tiles:
            # Fall back to the top quarter of tiles by ExG so we still
            # score the least-bad ones rather than emit an empty set.
            order = sorted(range(len(all_tiles)),
                           key=lambda j: exg_fracs_full[j], reverse=True)
            keep = max(1, len(all_tiles) // 4)
            kept_tiles = [all_tiles[j] for j in order[:keep]]
            kept_fracs = [exg_fracs_full[j] for j in order[:keep]]
            n_all_filtered += 1

        exg_t = torch.tensor(kept_fracs, dtype=torch.float32)

        # 3. Dual-resolution forward + entropy-weighted Bayesian aggregation.
        per_res: list[torch.Tensor] = []
        for sz in RESOLUTIONS:
            tile_probs = encode_tiles(
                models_by_size[sz], kept_tiles, transforms_by_size[sz], device,
                args.batch_size, amp_enabled, amp_dtype,
            )
            agg = aggregate_image_probs(tile_probs, exg_t)
            if agg is not None:
                per_res.append(agg)

        if not per_res:
            rows.append({"quadrat_id": quadrat_id, "species_ids": "[]"})
            continue

        image_probs = torch.stack(per_res, dim=0).mean(dim=0)

        # 4. Phenology log-prior + LA + adaptive threshold selection.
        species = select_species(
            image_probs, log_prior, phen_pdf, doy, args.beta, idx_to_species,
        )
        rows.append({
            "quadrat_id": quadrat_id,
            "species_ids": format_species_ids(species),
        })

        if i % 50 == 0 or i == len(images):
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  {i:5d}/{len(images)}  ·  {rate:5.1f} img/s  ·  last={quadrat_id}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["quadrat_id", "species_ids"], quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n[inf_phen] wrote {len(rows):,} rows to {args.output}")
    if n_errors:
        print(f"[inf_phen]   {n_errors} read errors -> empty rows")
    if n_no_doy:
        print(f"[inf_phen]   {n_no_doy} images had no parseable date -> "
              f"phenology skipped (anchor recipe used)")
    if n_all_filtered:
        print(f"[inf_phen]   {n_all_filtered} images had no tile above min_veg_frac -> "
              f"fell back to top-quarter by ExG")
    print(f"[inf_phen] {elapsed:.1f}s total · {len(images)/max(elapsed, 1e-6):.1f} img/s")


if __name__ == "__main__":
    main()
