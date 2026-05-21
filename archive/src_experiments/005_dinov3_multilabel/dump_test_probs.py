"""
Dump full per-quadrat probability vectors from the DINOv3 Phase-2 checkpoint.

Purpose: pre-compute the expensive artifact (7806-dim probs for each of 2105 test
quadrats) so a future ensemble run against the 006_bioclip25_finetune probs
is a cheap averaging step, not another 25-min GPU pass.

Output: single .npz file containing
    quadrat_ids     (N,)       list of quadrat stems
    species_ids     (C,)       list of species_id strings
    probs_max       (N, C)     fp16 — per-quadrat max-over-tiles sigmoid prob
    probs_mean      (N, C)     fp16 — per-quadrat mean-over-tiles sigmoid prob
    probs_noisy_or  (N, C)     fp16 — 1 - prod(1 - p_tile)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import build_default_transform
from run_inference import load_model_from_checkpoint, encode_tiles_to_probs

_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
if str(_FOUR) not in sys.path:
    sys.path.insert(0, str(_FOUR))
from tiling import get_tiles  # type: ignore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--output", required=True,
                   help="Destination .npz path.")
    p.add_argument("--tile-size", type=int, default=384)
    p.add_argument("--tile-overlap", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def aggregate_modes(tile_probs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (max, mean, noisy_or) per-class probs as fp16 numpy arrays."""
    if tile_probs.numel() == 0:
        C = tile_probs.shape[-1] if tile_probs.ndim == 2 else 0
        zeros = np.zeros(C, dtype=np.float16)
        return zeros, zeros, zeros
    p_max = tile_probs.max(dim=0).values
    p_mean = tile_probs.mean(dim=0)
    eps = 1e-6
    p_noisy_or = 1.0 - (1.0 - tile_probs.clamp(eps, 1 - eps)).prod(dim=0)
    return (
        p_max.half().numpy(),
        p_mean.half().numpy(),
        p_noisy_or.half().numpy(),
    )


def main() -> None:
    args = parse_args()
    if args.tile_size % 16 != 0:
        sys.exit("tile_size must be a multiple of 16 for DINOv3.")

    model, species_ids = load_model_from_checkpoint(args.checkpoint, args.device)
    transform = build_default_transform(args.tile_size)
    stride = max(1, args.tile_size - args.tile_overlap)

    root = Path(args.images_root)
    image_paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        image_paths = image_paths[: args.limit]
    logger.info(f"Found {len(image_paths)} images under {root}")

    N = len(image_paths)
    C = len(species_ids)
    out_max = np.zeros((N, C), dtype=np.float16)
    out_mean = np.zeros((N, C), dtype=np.float16)
    out_noisyor = np.zeros((N, C), dtype=np.float16)
    quadrat_ids: list[str] = []

    t0 = time.time()
    for i, path in enumerate(image_paths):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning(f"Skip {path.name}: {exc}")
            quadrat_ids.append(path.stem)
            continue
        tiles, _ = get_tiles(image, args.tile_size, stride)
        if not tiles:
            quadrat_ids.append(path.stem)
            continue
        tile_probs = encode_tiles_to_probs(
            model, transform, tiles, args.device, args.batch_size, args.bf16
        )
        pm, pa, pn = aggregate_modes(tile_probs)
        out_max[i] = pm
        out_mean[i] = pa
        out_noisyor[i] = pn
        quadrat_ids.append(path.stem)
        if (i + 1) % 50 == 0 or i + 1 == N:
            rate = (i + 1) / (time.time() - t0)
            eta = (N - (i + 1)) / max(rate, 1e-9)
            logger.info(f"  {i+1}/{N}  ({rate:.2f} img/s, ETA {eta:.0f}s)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        quadrat_ids=np.array(quadrat_ids, dtype=object),
        species_ids=np.array(species_ids, dtype=object),
        probs_max=out_max,
        probs_mean=out_mean,
        probs_noisy_or=out_noisyor,
    )
    logger.info(f"Wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
