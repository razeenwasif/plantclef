"""
Phase A direct tile-inference smoke test.

Phase A is single-plant CE (softmax over 7806 species). We tile the test
quadrat, forward each tile through the Phase-A backbone+head, softmax, and
aggregate per-species across tiles via max / mean / noisy_or.

Output schema matches dump_test_probs.py so make_topk_submission.py and
apply_thresholds_to_npz.py work unchanged.
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

from model import DINOv3SinglePlantClassifier, build_default_transform

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
    p.add_argument("--checkpoint", required=True, help="phase_a_best.pth")
    p.add_argument("--images-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tile-sizes", type=int, nargs="+", default=[448])
    p.add_argument("--tile-overlap", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--hflip-tta", action="store_true")
    p.add_argument("--whole-image", action="store_true",
                   help="In addition to tiles, also forward the whole image "
                        "resized to the smallest tile size — gives a global "
                        "softmax that captures dominant species at scene level.")
    return p.parse_args()


def load_phase_a(ckpt_path: str | Path, device: str) -> tuple[DINOv3SinglePlantClassifier, list[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    species_ids: list[str] = ckpt["species_ids"]
    n_classes = int(ckpt.get("n_classes", len(species_ids)))
    model = DINOv3SinglePlantClassifier(n_classes=n_classes, pretrained=False)
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
    model: DINOv3SinglePlantClassifier,
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
            probs = torch.softmax(logits.float(), dim=-1)
            if hflip_tta:
                logits_f = model(torch.flip(batch, dims=[-1]))
                probs = 0.5 * (probs + torch.softmax(logits_f.float(), dim=-1))
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

    model, species_ids = load_phase_a(args.checkpoint, args.device)
    scale_cfg = [
        (ts, build_default_transform(ts), max(1, ts - args.tile_overlap))
        for ts in tile_sizes
    ]
    logger.info(
        f"Tile scales: {tile_sizes}  (overlap={args.tile_overlap}, "
        f"hflip_tta={args.hflip_tta}, whole_image={args.whole_image})"
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

    smallest_ts = min(tile_sizes)
    whole_transform = build_default_transform(smallest_ts) if args.whole_image else None

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
            tp = encode_tiles_to_probs(
                model, transform, tiles, args.device, args.batch_size, args.bf16,
                hflip_tta=args.hflip_tta,
            )
            scale_probs.append(tp)
        if args.whole_image and whole_transform is not None:
            wp = encode_tiles_to_probs(
                model, whole_transform, [image], args.device, args.batch_size,
                args.bf16, hflip_tta=args.hflip_tta,
            )
            scale_probs.append(wp)
        if not scale_probs:
            quadrat_ids.append(path.stem)
            continue
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
