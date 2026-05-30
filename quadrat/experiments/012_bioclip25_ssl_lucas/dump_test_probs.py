"""
012 BioCLIP-2.5 (SSL-warm-started + 006 linear probe) tile-inference dump.

Mirrors 010's `dump_test_probs.py` so the resulting npz is drop-in compatible
with `011_bioclip25_aggregation_sweep/alpha_sweep.py` (which converts the npz
-> submission CSV with alpha=1.0 -> pure probs_mean = team-best softmax_mean recipe).

The only difference from 010 is the model class: 012 was trained with
`006_bioclip25_finetune/train.py` which uses `BioCLIP25LinearProbe`
(plain `Linear(embed_dim -> num_classes)` head, no SharedMLP, no aux heads).
We reuse 010's tiling helpers (`infer_tiles.extract_tiles`) and 006's val
preprocess to keep the geometry identical to the team-best 0.38333 run.

Run from repo root:

    python src_experiments/012_bioclip25_ssl_lucas/dump_test_probs.py \\
        --checkpoint src_experiments/012_bioclip25_ssl_lucas/outputs/finetune/checkpoints/best.pt \\
        --images-root /workspace/plantclef/kaggle_uploads/test/images \\
        --output     src_experiments/012_bioclip25_ssl_lucas/outputs/test_probs_012_grid4x4.npz \\
        --tile-mode  grid_4x4 \\
        --tile-size  448 --overlap 0.0 --img-size 224 \\
        --batch-size 64 --precision bf16
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

# Step 1: 010 path first so `from model import ...` inside infer_tiles.py resolves
# against 010's model.py (it needs load_checkpoint_model which only 010 has).
sys.path.insert(0, str(REPO_ROOT / "src_experiments" / "010_bioclip25_end_to_end_finetune_multitask"))
from infer_tiles import extract_tiles  # noqa: E402  (010)
from utils import resolve_device, amp_autocast  # noqa: E402  (010)

# Step 2: drop the 010-bound `model`/`transforms` modules so 006's same-named
# modules can be imported cleanly without name collision.
for _m in ("model", "transforms"):
    sys.modules.pop(_m, None)

sys.path.insert(0, str(REPO_ROOT / "src_experiments" / "006_bioclip25_finetune"))
from model import BioCLIP25LinearProbe, BIOCLIP25_MODEL_NAME  # noqa: E402  (006)
from transforms import bioclip_val_transform as build_val_transform  # noqa: E402  (006)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="012 best.pt (006-format).")
    p.add_argument("--images-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tile-mode", default="grid_4x4")
    p.add_argument("--tile-size", type=int, default=448)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--max-tiles", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--hflip-tta", action="store_true")
    p.add_argument("--unfreeze-blocks", type=int, default=4,
                   help="Match training value so module structure aligns when loading.")
    return p.parse_args()


def load_006_checkpoint(checkpoint_path: str, device: str, unfreeze_blocks: int):
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(p)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    idx_to_species = ckpt["idx_to_species"]
    model_name = config.get("model_name", BIOCLIP25_MODEL_NAME)
    model = BioCLIP25LinearProbe(
        num_classes=len(idx_to_species),
        model_name=model_name,
        unfreeze_blocks=unfreeze_blocks,
    )
    state_dict = ckpt["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            logger.warning(f"  first 5 missing: {missing[:5]}")
        if unexpected:
            logger.warning(f"  first 5 unexpected: {unexpected[:5]}")
    model = model.to(device)
    model.train(False)
    return model, idx_to_species, config


@torch.no_grad()
def encode_tiles_to_probs(model, preprocess, tiles, device, batch_size,
                          amp_enabled, amp_dtype, hflip_tta):
    if not tiles:
        return torch.empty(0)
    tensors = torch.stack([preprocess(t) for _, t in tiles])
    out = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size].to(device, non_blocking=True)
        with amp_autocast(device, amp_enabled, amp_dtype):
            logits = model(batch)
            probs = F.softmax(logits.float(), dim=-1)
            if hflip_tta:
                logits_f = model(torch.flip(batch, dims=[-1]))
                probs = 0.5 * (probs + F.softmax(logits_f.float(), dim=-1))
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
    device = resolve_device(args.device)
    amp_enabled = args.precision != "fp32"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16

    model, idx_to_species, _config = load_006_checkpoint(
        args.checkpoint, device, args.unfreeze_blocks,
    )
    logger.info(f"Loaded 012 ckpt: {len(idx_to_species)} species")

    preprocess = build_val_transform(img_size=args.img_size)

    root = Path(args.images_root)
    image_paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        image_paths = image_paths[: args.limit]
    logger.info(f"Found {len(image_paths)} images under {root}")
    logger.info(
        f"Tiling: mode={args.tile_mode} tile_size={args.tile_size} overlap={args.overlap} "
        f"img_size={args.img_size} batch={args.batch_size} hflip_tta={args.hflip_tta} "
        f"precision={args.precision}"
    )

    N = len(image_paths)
    C = len(idx_to_species)
    out_max = np.zeros((N, C), dtype=np.float16)
    out_mean = np.zeros((N, C), dtype=np.float16)
    out_noisyor = np.zeros((N, C), dtype=np.float16)
    quadrat_ids: list[str] = []

    max_tiles = args.max_tiles if args.max_tiles > 0 else None

    t0 = time.time()
    for i, path in enumerate(image_paths):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning(f"Skip {path.name}: {exc}")
            quadrat_ids.append(path.stem)
            continue
        tiles = extract_tiles(
            image, args.tile_mode,
            tile_size=args.tile_size, overlap=args.overlap, max_tiles=max_tiles,
        )
        tile_probs = encode_tiles_to_probs(
            model, preprocess, tiles, device, args.batch_size,
            amp_enabled, amp_dtype, args.hflip_tta,
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
        species_ids=np.array(idx_to_species, dtype=object),
        probs_max=out_max,
        probs_mean=out_mean,
        probs_noisy_or=out_noisyor,
    )
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
