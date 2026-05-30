"""
Dump per-quadrat probability vectors from a trained DINOv3 Phase-B collage model.

Output .npz schema matches 005/007's dump_test_probs.py so the existing
ensemble_with_other.py / apply_thresholds_to_npz.py / make_submission.py tooling
works unchanged:

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

from model import DINOv3MultiLabel, build_default_transform
from vegetation_filter import filter_tiles_by_vegetation

_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
if str(_FOUR) not in sys.path:
    sys.path.insert(0, str(_FOUR))
from tiling import get_tiles  # type: ignore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Phase B checkpoint (phase_b_best.pth).")
    p.add_argument("--images-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tile-sizes", type=int, nargs="+", default=[448],
                   help="One or more tile sizes (each a multiple of 16). "
                        "Multi-scale fuses per-tile probs across scales before "
                        "aggregation — strictly better than single-scale on "
                        "PlantCLEF-style quadrats where target plants span a "
                        "wide size range. Typical: 336 448 560.")
    p.add_argument("--tile-overlap", type=int, default=112,
                   help="Overlap applied to each scale. Stride = tile_size - overlap.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--hflip-tta", action="store_true",
                   help="Also forward the horizontal flip of each tile and average.")
    p.add_argument("--min-vegetation-frac", type=float, default=0.0,
                   help="Drop tiles with <X fraction of Excess-Green-indexed "
                        "vegetation pixels. 0 disables. 0.15 is a good "
                        "starting point (drops pure-soil tiles without "
                        "losing flower-heavy ones).")
    p.add_argument("--exg-thresh", type=float, default=20.0,
                   help="Excess Green threshold (2G-R-B). Higher = stricter.")
    return p.parse_args()


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    device: str,
) -> tuple[DINOv3MultiLabel, list[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    species_ids: list[str] = ckpt["species_ids"]
    use_lora = bool(ckpt.get("lora", True))

    model = DINOv3MultiLabel(
        n_classes=len(species_ids),
        backbone_source="timm",
        pretrained=False,
    )
    if use_lora:
        r = int(ckpt.get("lora_r", 32))
        alpha = int(ckpt.get("lora_alpha", 64))
        model.apply_lora(r=r, alpha=alpha, dropout=0.0)

    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    result = model.load_state_dict(state, strict=False)
    logger.info(
        f"Loaded {ckpt_path}: missing={len(result.missing_keys)}, "
        f"unexpected={len(result.unexpected_keys)}"
    )
    model.eval().to(device)
    return model, species_ids


@torch.no_grad()
def encode_tiles_to_probs(
    model: DINOv3MultiLabel,
    transform,
    tiles: list[Image.Image],
    device: str,
    batch_size: int,
    bf16: bool,
    hflip_tta: bool = False,
) -> torch.Tensor:
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16)
        if device.startswith("cuda") else torch.amp.autocast(device_type="cpu", enabled=False)
    )
    out: list[torch.Tensor] = []
    for i in range(0, len(tiles), batch_size):
        batch = torch.stack([transform(t) for t in tiles[i:i + batch_size]]).to(device)
        with autocast_ctx:
            logits = model(batch)
            probs = torch.sigmoid(logits.float())
            if hflip_tta:
                logits_f = model(torch.flip(batch, dims=[-1]))
                probs = 0.5 * (probs + torch.sigmoid(logits_f.float()))
        out.append(probs.cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def aggregate_modes(tile_probs: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    tile_sizes: list[int] = list(args.tile_sizes)
    for ts in tile_sizes:
        if ts % 16 != 0:
            sys.exit(f"tile size {ts} must be a multiple of 16 for DINOv3 patch-16.")

    model, species_ids = load_model_from_checkpoint(args.checkpoint, args.device)
    # One transform per scale so patch counts line up with that scale's forward.
    scale_cfg = [
        (ts, build_default_transform(ts), max(1, ts - args.tile_overlap))
        for ts in tile_sizes
    ]
    logger.info(
        f"Tile scales: {tile_sizes}  (overlap={args.tile_overlap}, "
        f"hflip_tta={args.hflip_tta})"
    )

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
        scale_probs: list[torch.Tensor] = []
        for ts, transform, stride in scale_cfg:
            tiles, _ = get_tiles(image, ts, stride)
            if not tiles:
                continue
            if args.min_vegetation_frac > 0.0:
                tiles, _ = filter_tiles_by_vegetation(
                    tiles,
                    min_frac=args.min_vegetation_frac,
                    exg_thresh=args.exg_thresh,
                )
                if not tiles:
                    continue
            tp = encode_tiles_to_probs(
                model, transform, tiles, args.device, args.batch_size, args.bf16,
                hflip_tta=args.hflip_tta,
            )
            scale_probs.append(tp)
        if not scale_probs:
            quadrat_ids.append(path.stem)
            continue
        # Multi-scale fusion: concatenate tiles from all scales along dim 0.
        # Max/mean/noisy-or all operate tile-wise so concatenation is the
        # correct fusion — a tile from scale 2 is just another evidence row.
        tile_probs = torch.cat(scale_probs, dim=0)
        pm, pa, pn = aggregate_modes(tile_probs)
        out_max[i] = pm
        out_mean[i] = pa
        out_noisyor[i] = pn
        quadrat_ids.append(path.stem)
        if (i + 1) % 50 == 0 or i + 1 == N:
            rate = (i + 1) / (time.time() - t0)
            eta = (N - (i + 1)) / max(rate, 1e-9)
            logger.info(f"  {i+1}/{N} ({rate:.2f} img/s, ETA {eta:.0f}s)")

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
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
