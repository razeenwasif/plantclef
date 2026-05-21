"""
inf_script_phen.py — PlantCLEF 2026 submission with the seasonal-phenology
pivot stacked on top of the anchor recipe (paper pivot 3,
0.41346 public Macro F1).

This is the SAME architecture and SAME tile-ensemble + logit adjustment
pipeline as `inf_script.py`. The only addition is a per-image
multiplicative phenology prior that boosts species observed in the
image's month-of-collection and suppresses out-of-season species,
derived from per-species GBIF month-of-observation histograms.

Pipeline (additions vs. anchor in **bold**):

  Checkpoint   : i002 BioCLIP 2.5 ViT-H/14, last-4-blocks fine-tune
  Tiling       : 4×4 grid, 448-px tiles, no overlap (16 tiles / image)
  Inference    : forward each tile at BOTH 224 and 336 pixels
  Aggregation  : softmax-mean across tiles, then mean across resolutions
  LA           : τ = 0.25 logit adjustment vs. Laplace-smoothed training prior
  **Phenology**: multiply per-species probability by
                 (h_s(m_image) + ε)^β, then re-normalise.
                 h_s(m) is the row-normalised GBIF month histogram for
                 species s — i.e. the prior probability of observing s
                 in month m. β controls strength (paper used β = 1.0).
  Selection    : adaptive prob threshold T = 0.03, clamped to [k=2..10]

Image-month extraction:
  PlantCLEF quadrat filenames carry an 8-digit YYYYMMDD substring
  (e.g. `CBN-Pla-A1-20130808.jpg`). We pull the month from there. For
  any image where the regex doesn't match, the phenology stage is
  skipped for that image (falls back to the anchor recipe's post-LA
  probabilities) so partial coverage doesn't tank the submission.

Phenology CSV format (`--phenology-csv`):
  Required columns: species_id, m_01, m_02, ..., m_12   (raw GBIF
  observation counts; rows can be missing — defaults to uniform).

Usage:

  python inf_script_phen.py \\
      --checkpoint     /path/to/i002/outputs/last_blocks/checkpoints/best.pt \\
      --image-dir      /path/to/plantclef/test \\
      --metadata-csv   /path/to/training_manifest.csv \\
      --phenology-csv  /path/to/gbif_phenology.csv \\
      --output         submission_phen.csv

The model loader, preprocessing transform, and tile-extraction code
live at the absolute path

    /workspace/scratch_space_arjun/PlantCLEF2026/src_experiments/i002_bioclip25_cap_image

— this script imports them by path so there's a single source of truth
and no duplication.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# Pull in the trained model + transforms + tiling helpers from the i002
# experiment directory. Absolute path so this script runs from anywhere
# on the workstation.
EXP_DIR = Path("/workspace/scratch_space_arjun/PlantCLEF2026/src_experiments/i002_bioclip25_cap_image")
if not EXP_DIR.is_dir():
    raise SystemExit(
        f"Expected experiment dir not found: {EXP_DIR}\n"
        f"Edit EXP_DIR in this file if the i002 sources live elsewhere on this host."
    )
sys.path.insert(0, str(EXP_DIR))

from model import load_checkpoint_model               # noqa: E402
from transforms import val_transform                  # noqa: E402
from infer_tiles_adaptive import extract_tiles        # noqa: E402
from utils import resolve_device, amp_autocast        # noqa: E402


# ─── Recipe constants (anchor + phenology pivot) ──────────────────────────
TILE_MODE       = "grid_4x4"
TILE_SIZE       = 448
TILE_OVERLAP    = 0.0
RESOLUTIONS     = (224, 336)
LOGIT_ADJ_TAU   = 0.25
PROB_THRESHOLD  = 0.03
K_MIN           = 2
K_MAX           = 10
PHEN_BETA       = 1.0     # phenology strength; paper used 1.0
PHEN_EPS        = 1e-3    # smoothing on h_s(m) so β·log doesn't explode


# ─── Prior + phenology table ──────────────────────────────────────────────

def build_log_prior(metadata_csv: Path, idx_to_species: list[str]) -> torch.Tensor:
    """Laplace-smoothed training-set prior log π̃_s. Same as inf_script.py."""
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


def build_phenology_table(
    phenology_csv: Path,
    idx_to_species: list[str],
) -> torch.Tensor:
    """
    Load per-species GBIF month-of-observation histograms and convert to
    a row-stochastic phenology table aligned with `idx_to_species`.

    Returns: (num_species, 12) float32 tensor; row s is the probability
    distribution h_s(m) over months 1..12 (sum to 1).

    Species missing from the CSV fall back to a uniform 1/12 row, so the
    phenology stage degrades to a no-op for them rather than zeroing out
    their probability.
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

    rows: list[np.ndarray] = []
    uniform = np.full(12, 1.0 / 12.0, dtype=np.float32)
    n_missing = 0
    for s in idx_to_species:
        if s in by_id.index:
            row = by_id.loc[s, month_cols].to_numpy(dtype=np.float32)
            tot = float(row.sum())
            if tot > 0:
                rows.append(row / tot)
                continue
        rows.append(uniform.copy())
        n_missing += 1
    if n_missing:
        print(f"[inf_phen] phenology: {n_missing:,} / {len(idx_to_species):,} "
              f"species had no GBIF rows → uniform fallback")
    return torch.tensor(np.stack(rows))


# ─── Image-month extraction ───────────────────────────────────────────────

# 8 consecutive digits → YYYYMMDD. PlantCLEF quadrat stems carry this
# at the end (CBN-Pla-A1-20130808.jpg, RNNB-1-1-20230512.jpg, etc.).
_DATE_RE = re.compile(r"(\d{8})")


def month_from_stem(stem: str) -> int | None:
    """
    Pull the month (1..12) out of a PlantCLEF quadrat filename stem.
    Returns None if no valid YYYYMMDD substring is found.
    """
    for hit in _DATE_RE.findall(stem):
        try:
            y = int(hit[0:4])
            m = int(hit[4:6])
            d = int(hit[6:8])
            if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                return m
        except ValueError:
            continue
    return None


# ─── Per-image inference ──────────────────────────────────────────────────

@torch.no_grad()
def forward_tiles_at_resolution(
    model,
    tile_pils: list[Image.Image],
    transform,
    device: str,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Same as inf_script.py — see there for full docstring."""
    chunks = []
    for i in range(0, len(tile_pils), batch_size):
        batch = tile_pils[i : i + batch_size]
        x = torch.stack([transform(im) for im in batch]).to(device)
        with amp_autocast(amp_enabled, amp_dtype):
            sp_logits, _, _ = model(x)
        chunks.append(F.softmax(sp_logits.float(), dim=-1))
    return torch.cat(chunks, dim=0)


def select_species(
    image_probs: torch.Tensor,
    log_prior: torch.Tensor,
    phen_table: torch.Tensor,
    month_idx: int | None,
    beta: float,
    idx_to_species: list[str],
) -> list[str]:
    """
    Apply LA, then (if month is known) the phenology prior, then the
    adaptive probability threshold. Returns species-id strings in
    descending post-prior score order.

    All re-weightings done in log-space then re-softmaxed to keep the
    output a proper distribution for the threshold check.
    """
    log_probs = torch.log(image_probs.clamp_min(1e-12))

    # Logit adjustment vs. the training prior.
    log_adj = log_probs - LOGIT_ADJ_TAU * log_prior

    # Seasonal phenology prior — only if we successfully parsed a month.
    if month_idx is not None:
        h_s = phen_table[:, month_idx - 1].to(log_adj.device)        # (K,)
        log_adj = log_adj + beta * torch.log(h_s + PHEN_EPS)

    adj_probs = F.softmax(log_adj, dim=-1)

    sorted_probs, sorted_idx = adj_probs.sort(descending=True)
    n_above = int((sorted_probs >= PROB_THRESHOLD).sum().item())
    k = max(K_MIN, min(K_MAX, n_above))

    return [idx_to_species[i.item()] for i in sorted_idx[:k]]


# ─── Driver ───────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix in _IMAGE_EXTS)


def format_species_ids(species: list[str]) -> str:
    clean = [s for s in species if s.strip().lstrip("-").isdigit()]
    return "[" + ", ".join(clean) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Oracle phenology-pivot inference (paper pivot 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint",    required=True, type=Path,
                    help="Path to i002 last-blocks best.pt")
    ap.add_argument("--image-dir",     required=True, type=Path,
                    help="Directory of PlantCLEF test quadrat images")
    ap.add_argument("--metadata-csv",  required=True, type=Path,
                    help="Training manifest CSV (needs species_id column) — "
                         "used for the Laplace-smoothed prior π̃_s")
    ap.add_argument("--phenology-csv", required=True, type=Path,
                    help="GBIF month-of-observation histogram CSV — "
                         "columns: species_id, m_01, m_02, …, m_12")
    ap.add_argument("--output",        default=Path("submission_phen.csv"), type=Path,
                    help="Where to write the PlantCLEF submission CSV")
    ap.add_argument("--beta",          type=float, default=PHEN_BETA,
                    help=f"Phenology strength β (default {PHEN_BETA}, paper value)")
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
    print(f"[inf_phen] loading checkpoint  {args.checkpoint}")
    model, encoders, _ = load_checkpoint_model(str(args.checkpoint), device=device)
    model.eval()
    idx_to_species = encoders["idx_to_species"]
    print(f"[inf_phen] {len(idx_to_species):,} species classes")

    print(f"[inf_phen] building Laplace-smoothed prior from {args.metadata_csv}")
    log_prior = build_log_prior(args.metadata_csv, idx_to_species).to(device)

    print(f"[inf_phen] building phenology table from {args.phenology_csv}")
    phen_table = build_phenology_table(args.phenology_csv, idx_to_species).to(device)

    transforms_by_size = {sz: val_transform(sz) for sz in RESOLUTIONS}

    images = find_images(args.image_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")
    print(f"[inf_phen] {len(images):,} images · "
          f"{TILE_MODE}@{TILE_SIZE}px · {len(RESOLUTIONS)}-res ensemble {list(RESOLUTIONS)}")
    print(f"[inf_phen] LA τ={LOGIT_ADJ_TAU}  phenology β={args.beta}  "
          f"threshold T={PROB_THRESHOLD}  k∈[{K_MIN},{K_MAX}]")
    print()

    rows: list[dict[str, str]] = []
    t0 = time.time()
    n_errors = 0
    n_no_month = 0

    for i, img_path in enumerate(images, 1):
        quadrat_id = img_path.stem
        month_idx = month_from_stem(quadrat_id)
        if month_idx is None:
            n_no_month += 1

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                tiles_with_info = extract_tiles(
                    im, TILE_MODE,
                    tile_size=TILE_SIZE, overlap=TILE_OVERLAP, max_tiles=None,
                )
        except Exception as e:
            print(f"  ! {quadrat_id}: failed to read ({e}) — emitting empty row")
            rows.append({"quadrat_id": quadrat_id, "species_ids": "[]"})
            n_errors += 1
            continue

        tile_pils = [tile for (_, tile) in tiles_with_info]

        # Dual-resolution tile ensemble → per-image probability vector
        per_res = []
        for sz in RESOLUTIONS:
            tile_probs = forward_tiles_at_resolution(
                model, tile_pils, transforms_by_size[sz], device,
                args.batch_size, amp_enabled, amp_dtype,
            )
            per_res.append(tile_probs.mean(dim=0))
        image_probs = torch.stack(per_res, dim=0).mean(dim=0)

        species = select_species(
            image_probs, log_prior, phen_table, month_idx, args.beta, idx_to_species,
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
        print(f"[inf_phen]   {n_errors} read errors → empty rows")
    if n_no_month:
        print(f"[inf_phen]   {n_no_month} images had no parseable month → "
              f"phenology skipped (anchor recipe used)")
    print(f"[inf_phen] {elapsed:.1f}s total · {len(images)/max(elapsed, 1e-6):.1f} img/s")


if __name__ == "__main__":
    main()
