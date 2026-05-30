"""
009 BioCLIP-2.5 grid_NxN tile-inference dump — matches Arjun's 010 winning
recipe (grid_4x4, overlap=0.0, img_size=224, softmax_mean, top-3 = 0.38333).

Per-tile pipeline:
  1. Partition the source image into N×N non-overlapping cells via PIL crop
     (cells take their natural size from the source resolution; we do NOT
     resize the source first).
  2. Resize each cell to ``img_size`` via the BioCLIP center-crop transform.
  3. Forward → softmax(logits) → tile_probs.
  4. Aggregate across the 16 tiles via {max, mean, noisy_or}; the team
     baseline uses ``probs_mean`` (= softmax_mean) + top-3.

Emits the same npz schema as 008/009/010 dumps so existing fusion /
submission tooling works unchanged.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bioclip_model import BioCLIP25SinglePlantClassifier, build_default_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--grid-n", type=int, default=4,
                   help="Partition into N×N tiles. Default 4 → 16 tiles.")
    p.add_argument("--img-size", type=int, default=224,
                   help="Per-tile model input size (multiple of 14).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--hflip-tta", action="store_true",
                   help="Average softmax over original + horizontally-flipped tiles.")
    p.add_argument("--include-whole", action="store_true",
                   help="Also include the resized whole image as an extra tile.")
    return p.parse_args()


def grid_tiles(image: Image.Image, n: int) -> list[Image.Image]:
    w, h = image.size
    tiles: list[Image.Image] = []
    for row in range(n):
        for col in range(n):
            x1 = int(round(col * w / n))
            y1 = int(round(row * h / n))
            x2 = int(round((col + 1) * w / n))
            y2 = int(round((row + 1) * h / n))
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            tiles.append(image.crop((x1, y1, x2, y2)))
    return tiles


def load_bioclip25(ckpt_path: str | Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    species_ids: list[str] = ckpt["species_ids"]
    n_classes = int(ckpt.get("n_classes", len(species_ids)))
    model = BioCLIP25SinglePlantClassifier(n_classes=n_classes, pretrained=True)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    logger.info(
        f"Loaded {ckpt_path}: missing={len(result.missing_keys)}, "
        f"unexpected={len(result.unexpected_keys)}"
    )
    if result.missing_keys:
        logger.warning(f"  first missing: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        logger.warning(f"  first unexpected: {result.unexpected_keys[:5]}")
    model.eval().to(device)
    return model, species_ids


@torch.no_grad()
def encode_tiles_to_probs(
    model: BioCLIP25SinglePlantClassifier,
    transform,
    tiles: list[Image.Image],
    device: str,
    batch_size: int,
    bf16: bool,
    hflip_tta: bool,
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
            probs = torch.softmax(logits.float(), dim=-1)
            if hflip_tta:
                logits_f = model(torch.flip(batch, dims=[-1]))
                probs = 0.5 * (probs + torch.softmax(logits_f.float(), dim=-1))
        out.append(probs.cpu())
    return torch.cat(out, dim=0)


def aggregate_modes(tile_probs: torch.Tensor):
    if tile_probs.numel() == 0:
        return None
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
    if args.img_size % 14 != 0:
        raise SystemExit(f"img_size {args.img_size} must be a multiple of 14 for BioCLIP-2.5 ViT-H/14")

    model, species_ids = load_bioclip25(args.checkpoint, args.device)
    transform = build_default_transform(args.img_size)
    logger.info(
        f"Tiling: grid_{args.grid_n}x{args.grid_n} (N={args.grid_n**2} tiles per image), "
        f"img_size={args.img_size}, batch={args.batch_size}, bf16={args.bf16}, "
        f"hflip_tta={args.hflip_tta}, include_whole={args.include_whole}"
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
        tiles = grid_tiles(image, args.grid_n)
        if args.include_whole:
            tiles.append(image)
        tile_probs = encode_tiles_to_probs(
            model, transform, tiles, args.device, args.batch_size,
            args.bf16, args.hflip_tta,
        )
        agg = aggregate_modes(tile_probs)
        if agg is not None:
            pm, pa, pn = agg
            out_max[i] = pm
            out_mean[i] = pa
            out_noisyor[i] = pn
        quadrat_ids.append(path.stem)
        if (i + 1) % 25 == 0 or i + 1 == N:
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
