"""
Dump per-quadrat probability vectors from a trained PlantNet-DINOv2 collage model.

Output .npz format matches 005's dump_test_probs.py so the existing
ensemble_with_other.py / apply_thresholds_to_npz.py tooling works unchanged:

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

from model import PlantNetDINOv2MultiLabel, build_default_transform

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


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tile-size", type=int, default=336,
                   help="Multiple of 14. 224/336/518.")
    p.add_argument("--tile-overlap", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    ckpt_path: str | Path,
    device: str,
    img_size_override: int | None = None,
) -> tuple[PlantNetDINOv2MultiLabel, list[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    species_ids: list[str] = ckpt["species_ids"]
    img_size = img_size_override or int(ckpt.get("img_size", 336))
    backbone_name = ckpt.get("backbone_name", "vit_base_patch14_reg4_dinov2.lvd142m")
    use_lora = bool(ckpt.get("lora", True))

    model = PlantNetDINOv2MultiLabel(
        n_classes=len(species_ids),
        pc24_checkpoint=None,   # weights come from the trained ckpt
        pc24_class_file=None,
        target_species_ids=None,
        img_size=img_size,
        backbone_name=backbone_name,
    )
    if use_lora:
        # Rebuild LoRA wrappers so load_state_dict finds the expected keys.
        r = int(ckpt.get("lora_r", 32))
        alpha = int(ckpt.get("lora_alpha", 64))
        model.apply_lora(r=r, alpha=alpha, dropout=0.0)

    result = model.load_state_dict(ckpt["model_state"], strict=False)
    logger.info(
        f"Loaded {ckpt_path}: missing={len(result.missing_keys)}, "
        f"unexpected={len(result.unexpected_keys)}"
    )
    model.eval().to(device)
    return model, species_ids


@torch.no_grad()
def encode_tiles_to_probs(
    model: PlantNetDINOv2MultiLabel,
    transform,
    tiles: list[Image.Image],
    device: str,
    batch_size: int,
    bf16: bool,
) -> torch.Tensor:
    """Return (n_tiles, n_classes) sigmoid probs (float32, on CPU)."""
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16)
        if device.startswith("cuda") else torch.amp.autocast(device_type="cpu", enabled=False)
    )
    out: list[torch.Tensor] = []
    for i in range(0, len(tiles), batch_size):
        batch = torch.stack([transform(t) for t in tiles[i:i + batch_size]]).to(device)
        with autocast_ctx:
            logits = model(batch)
        out.append(torch.sigmoid(logits.float()).cpu())
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


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.tile_size % 14 != 0:
        sys.exit("tile_size must be a multiple of 14 for DINOv2 patch-14.")

    model, species_ids = load_model_from_checkpoint(
        args.checkpoint, args.device, img_size_override=args.tile_size
    )
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
