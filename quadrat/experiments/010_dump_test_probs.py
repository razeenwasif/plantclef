"""
010 BioCLIP-2.5 multitask tile-inference dump → PhaseA-compatible npz.

Emits a (N_quadrats, 7806) probability matrix for each of probs_max,
probs_mean, probs_noisy_or, mirroring the schema 008/009 already use, so
``008/fuse_phase_a_bioclip.py`` works unchanged.

Per-tile pipeline:
  1. Extract tiles via 010's multiscale recipe (whole + grid_2x2 + grid_4x4).
  2. Forward through model.backbone → sp_logits (multitask head, take [0]).
  3. softmax(sp_logits, dim=-1) → tile_probs.
  4. Aggregate tile_probs across tiles via {max, mean, noisy_or}.

Aggregating *probs* (not logits) matches 008's npz semantics — the fuse
script does an alpha-mix of two PhaseA probability matrices.

Run from inside ``src_experiments/010_bioclip25_end_to_end_finetune_multitask``
so that ``model``, ``transforms``, ``utils`` import.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import BioCLIP25MultiTask, BIOCLIP25_MODEL_NAME  # type: ignore  # noqa: E402
from transforms import val_transform  # type: ignore  # noqa: E402
from infer_tiles import extract_tiles  # type: ignore  # noqa: E402
from utils import resolve_device, amp_autocast  # type: ignore  # noqa: E402


def load_checkpoint_model_safe(checkpoint_path, device):
    """Local replacement for 010's load_checkpoint_model (which has a Path/_Path bug)."""
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    encoders = ckpt.get("encoders", {})
    model = BioCLIP25MultiTask(
        num_species=len(encoders.get("idx_to_species", [])),
        num_genus=len(encoders.get("idx_to_genus", [])),
        num_family=len(encoders.get("idx_to_family", [])),
        num_order=len(encoders.get("idx_to_order", [])),
        num_class=len(encoders.get("idx_to_class", [])),
        model_name=config.get("model_name", BIOCLIP25_MODEL_NAME),
        hidden_dim=config.get("hidden_dim", 1024),
        dropout=config.get("dropout", 0.2),
        use_taxonomy_heads=config.get("use_taxonomy_heads", True),
    )
    state_dict = ckpt["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, encoders, config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="010 best.pt")
    p.add_argument("--images-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tile-mode", default="multiscale",
                   help="010 tiling mode (whole|grid_NxN|five_crop|sliding|multiscale)")
    p.add_argument("--tile-size", type=int, default=448,
                   help="Used by sliding/five_crop modes only.")
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--max-tiles", type=int, default=0,
                   help="Hard cap on tiles per image (0 = unlimited).")
    p.add_argument("--img-size", type=int, default=224,
                   help="Preprocess size fed to model.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--hflip-tta", action="store_true")
    return p.parse_args()


@torch.no_grad()
def encode_tiles_to_probs(
    model,
    preprocess,
    tiles,
    device: str,
    batch_size: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    hflip_tta: bool,
) -> torch.Tensor:
    if not tiles:
        return torch.empty(0)
    tensors = torch.stack([preprocess(t) for _, t in tiles])
    out: list[torch.Tensor] = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size].to(device, non_blocking=True)
        with amp_autocast(device, amp_enabled, amp_dtype):
            sp_logits, *_ = model(batch)
            probs = F.softmax(sp_logits.float(), dim=-1)
            if hflip_tta:
                sp_logits_f, *_ = model(torch.flip(batch, dims=[-1]))
                probs = 0.5 * (probs + F.softmax(sp_logits_f.float(), dim=-1))
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

    model, encoders, _config = load_checkpoint_model_safe(args.checkpoint, device=device)
    model.eval()
    species_ids = encoders.get("idx_to_species", [])
    if not species_ids:
        raise ValueError("Checkpoint missing encoders.idx_to_species")
    logger.info(f"Loaded 010 checkpoint: {len(species_ids)} species classes")

    preprocess = val_transform(img_size=args.img_size)

    root = Path(args.images_root)
    image_paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        image_paths = image_paths[:args.limit]
    logger.info(f"Found {len(image_paths)} images under {root}")
    logger.info(
        f"Tiling: mode={args.tile_mode} tile_size={args.tile_size} overlap={args.overlap} "
        f"img_size={args.img_size} batch={args.batch_size} hflip_tta={args.hflip_tta} "
        f"precision={args.precision}"
    )

    N = len(image_paths)
    C = len(species_ids)
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
        species_ids=np.array(species_ids, dtype=object),
        probs_max=out_max,
        probs_mean=out_mean,
        probs_noisy_or=out_noisyor,
    )
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
