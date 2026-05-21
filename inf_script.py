"""
inf_script.py — clean entry point for the PlantCLEF 2026 headline
submission recipe (0.418265 public Macro F1).

Recipe (fixed; this script is single-purpose, no sweeps):

  Checkpoint   : i002 BioCLIP 2.5 ViT-H/14, last-4-blocks fine-tune
  Tiling       : 4×4 grid, 448-px tiles, no overlap (16 tiles / image)
  Inference    : forward each tile at BOTH 224 and 336 pixels
  Aggregation  : softmax-mean across the 16 tiles per resolution,
                 then mean across the 2 resolutions
  LA           : logit adjustment with τ = 0.25, Laplace-smoothed
                 training prior (matches LogitAdjustmentLoss / the
                 paper's `\\tilde\\pi_s`)
  Selection    : adaptive probability threshold T = 0.03, clamped to
                 [k_min = 2, k_max = 10] species per quadrat

Usage:

  python inf_script.py \\
      --checkpoint /path/to/i002/outputs/last_blocks/checkpoints/best.pt \\
      --image-dir  /path/to/plantclef/test \\
      --metadata-csv /path/to/training_manifest.csv \\
      --output     submission.csv

The model loader, preprocessing transform, and tile-extraction code
live at the absolute path

    /workspace/scratch_space_arjun/PlantCLEF2026/src_experiments/i002_bioclip25_cap_image

— this script imports them by path so there's a single source of truth
and no duplication.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

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


# ─── Recipe constants (the winning submission's hyperparameters) ──────────
TILE_MODE      = "grid_4x4"
TILE_SIZE      = 448
TILE_OVERLAP   = 0.0
RESOLUTIONS    = (224, 336)
LOGIT_ADJ_TAU  = 0.25
PROB_THRESHOLD = 0.03
K_MIN          = 2
K_MAX          = 10


# ─── Prior ────────────────────────────────────────────────────────────────

def build_log_prior(metadata_csv: Path, idx_to_species: list[str]) -> torch.Tensor:
    """
    Laplace-smoothed training-set prior log π̃_s, aligned with
    `idx_to_species` ordering.

    Returns a (num_species,) float32 tensor of log-probabilities, suitable
    for the standard logit-adjustment subtraction `logit -= τ * log_prior`.
    Matches the `build_logit_adjustment` smoothing in
    archive/.../infer_tiles_adaptive.py.
    """
    if not metadata_csv.is_file():
        raise SystemExit(f"--metadata-csv not found: {metadata_csv}")
    df = pd.read_csv(metadata_csv, usecols=["species_id"])
    counts_by_id = df["species_id"].astype(str).value_counts().to_dict()
    counts = torch.tensor(
        [counts_by_id.get(str(s), 0) for s in idx_to_species],
        dtype=torch.float32,
    )
    smoothed = counts + 1.0
    prior = smoothed / smoothed.sum()
    return torch.log(prior)


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
    """
    Preprocess each PIL tile through `transform` (which sets the resolution
    via its CenterCrop), forward through the species head, return the
    softmax probabilities at the given resolution.

    Returns: (N_tiles, num_species) float32 tensor on `device`.
    """
    chunks = []
    for i in range(0, len(tile_pils), batch_size):
        batch = tile_pils[i : i + batch_size]
        x = torch.stack([transform(im) for im in batch]).to(device)
        with amp_autocast(amp_enabled, amp_dtype):
            sp_logits, _, _ = model(x)
        # Cast to fp32 before softmax — log/exp on bf16 can be ugly with
        # 7,806 classes.
        chunks.append(F.softmax(sp_logits.float(), dim=-1))
    return torch.cat(chunks, dim=0)


def select_species(
    image_probs: torch.Tensor,
    log_prior: torch.Tensor,
    idx_to_species: list[str],
) -> list[str]:
    """
    Apply logit adjustment then the adaptive probability threshold to a
    single image-level probability vector. Returns the species-id strings
    in descending post-LA score order.

    LA in log-space: log p_s − τ log π̃_s, then re-softmax to renormalise.
    Threshold: keep species with post-LA prob ≥ T, clamp count to
    [K_MIN, K_MAX].
    """
    log_probs = torch.log(image_probs.clamp_min(1e-12))
    adj_logits = log_probs - LOGIT_ADJ_TAU * log_prior
    adj_probs  = F.softmax(adj_logits, dim=-1)

    sorted_probs, sorted_idx = adj_probs.sort(descending=True)
    n_above = int((sorted_probs >= PROB_THRESHOLD).sum().item())
    k = max(K_MIN, min(K_MAX, n_above))

    return [idx_to_species[i.item()] for i in sorted_idx[:k]]


# ─── Driver ───────────────────────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix in _IMAGE_EXTS)


def format_species_ids(species: list[str]) -> str:
    """PlantCLEF expects `"[123, 456, 789]"` — Python list literal of
    integer species IDs, joined by `, `, wrapped in CSV quoting."""
    clean = [s for s in species if s.strip().lstrip("-").isdigit()]
    return "[" + ", ".join(clean) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Oracle headline inference — 0.418265 Macro F1 recipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Path to i002 last-blocks best.pt")
    ap.add_argument("--image-dir", required=True, type=Path,
                    help="Directory of PlantCLEF test quadrat images")
    ap.add_argument("--metadata-csv", required=True, type=Path,
                    help="Training manifest CSV (needs a `species_id` column) "
                         "used to compute the Laplace-smoothed prior π̃_s")
    ap.add_argument("--output", default=Path("submission.csv"), type=Path,
                    help="Where to write the PlantCLEF submission CSV")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Tile batch size per forward pass")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--device", default="auto",
                    help="auto | cuda | cuda:N | cpu")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N images (0 = all). For smoke testing.")
    args = ap.parse_args()

    device = resolve_device(args.device)
    amp_enabled = args.precision != "fp32"
    amp_dtype   = torch.float16 if args.precision == "fp16" else torch.bfloat16

    print(f"[inf_script] device={device}  precision={args.precision}")
    print(f"[inf_script] loading checkpoint  {args.checkpoint}")
    model, encoders, _ = load_checkpoint_model(str(args.checkpoint), device=device)
    model.eval()
    idx_to_species = encoders["idx_to_species"]
    print(f"[inf_script] {len(idx_to_species):,} species classes")

    print(f"[inf_script] building prior from {args.metadata_csv}")
    log_prior = build_log_prior(args.metadata_csv, idx_to_species).to(device)

    # One transform per resolution — val_transform handles the resize +
    # center crop chain that matches OpenCLIP's preprocessing.
    transforms_by_size = {sz: val_transform(sz) for sz in RESOLUTIONS}

    images = find_images(args.image_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.image_dir}")
    print(f"[inf_script] {len(images):,} images · "
          f"{TILE_MODE}@{TILE_SIZE}px · {len(RESOLUTIONS)}-res ensemble {list(RESOLUTIONS)}")
    print(f"[inf_script] LA τ={LOGIT_ADJ_TAU}  threshold T={PROB_THRESHOLD}  "
          f"k∈[{K_MIN},{K_MAX}]")
    print()

    rows: list[dict[str, str]] = []
    t0 = time.time()
    n_errors = 0

    for i, img_path in enumerate(images, 1):
        quadrat_id = img_path.stem

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

        # Dual-resolution ensemble: per-tile softmax probs averaged across
        # tiles at each resolution, then averaged across the two resolutions
        # to give the per-image probability vector.
        per_res_image_probs = []
        for sz in RESOLUTIONS:
            tile_probs = forward_tiles_at_resolution(
                model, tile_pils, transforms_by_size[sz], device,
                args.batch_size, amp_enabled, amp_dtype,
            )                                      # (N_tiles, K)
            per_res_image_probs.append(tile_probs.mean(dim=0))   # (K,)
        image_probs = torch.stack(per_res_image_probs, dim=0).mean(dim=0)

        species = select_species(image_probs, log_prior, idx_to_species)
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
    print(f"\n[inf_script] wrote {len(rows):,} rows to {args.output}"
          + (f"  ({n_errors} read errors → empty rows)" if n_errors else ""))
    print(f"[inf_script] {elapsed:.1f}s total · {len(images)/max(elapsed, 1e-6):.1f} img/s")


if __name__ == "__main__":
    main()
