"""
Pseudo-label LUCAS pseudo-quadrats with the Phase A teacher.

For each image under --images-root:
  - Tile at --tile-size with stride = tile_size - tile_overlap.
  - Forward each tile through Phase A, apply softmax, max-over-tiles per class.
  - Threshold: keep species with prob >= --tau-conf, cap at --top-k per image.
  - Drop images with fewer than --min-species-per-image positives.

Outputs:
  --csv-out      semicolon-separated, header `image_name;species_ids`
                 (compatible with 007's CollageDataset / train_phase_b.py)
  --probs-npz    float16 (N, C) max-over-tiles probs + image_paths + species_ids
                 (lets us re-threshold later without re-running inference).
"""
from __future__ import annotations

import argparse
import csv
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
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--csv-out", required=True)
    p.add_argument("--probs-npz", required=True)
    p.add_argument("--tile-size", type=int, default=224)
    p.add_argument("--tile-overlap", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--whole-image", action="store_true")
    p.add_argument("--hflip-tta", action="store_true")
    p.add_argument("--tau-conf", type=float, default=0.5,
                   help="Per-species probability threshold for inclusion.")
    p.add_argument("--top-k", type=int, default=5,
                   help="Cap species per image after thresholding.")
    p.add_argument("--min-species-per-image", type=int, default=1,
                   help="Drop image from CSV if fewer than this many species "
                        "pass tau-conf. 0 keeps all (background-only rows "
                        "become useless to ASL anyway, so default 1).")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resize-max-side", type=int, default=0,
                   help="Resize each image so its longest side is this many "
                        "pixels (preserving aspect). 0 = no resize. Set to 800 "
                        "to match the test pipeline (5x speedup on raw LUCAS).")
    p.add_argument("--checkpoint-every", type=int, default=2000,
                   help="Flush CSV + npz every N images so a crash doesn't "
                        "lose all progress.")
    return p.parse_args()


def load_phase_a(ckpt_path, device):
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
def encode_tiles_to_probs(model, transform, tiles, device, batch_size, bf16, hflip_tta=False):
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


def threshold_and_format(probs_max: np.ndarray, species_ids: list[str],
                         tau: float, k: int) -> tuple[list[str], np.ndarray]:
    """Return (species_id_strs, prob_values) for the kept species, sorted by prob desc."""
    order = np.argsort(-probs_max)
    keep_ids: list[str] = []
    keep_probs: list[float] = []
    for j in order:
        if probs_max[j] < tau:
            break
        keep_ids.append(str(species_ids[j]))
        keep_probs.append(float(probs_max[j]))
        if len(keep_ids) >= k:
            break
    return keep_ids, np.asarray(keep_probs, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.tile_size % 16 != 0:
        sys.exit(f"tile-size {args.tile_size} must be a multiple of 16.")

    model, species_ids = load_phase_a(args.checkpoint, args.device)
    transform = build_default_transform(args.tile_size)
    stride = max(1, args.tile_size - args.tile_overlap)
    logger.info(
        f"Tile {args.tile_size} stride={stride} (overlap={args.tile_overlap}), "
        f"hflip_tta={args.hflip_tta}, whole_image={args.whole_image}, "
        f"tau={args.tau_conf}, top_k={args.top_k}"
    )

    root = Path(args.images_root)
    logger.info(f"Walking {root} for image files (this may take ~30s)…")
    image_paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        image_paths = image_paths[: args.limit]
    logger.info(f"Found {len(image_paths)} images.")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = Path(args.probs_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    N = len(image_paths)
    C = len(species_ids)
    out_max = np.zeros((N, C), dtype=np.float16)
    rel_paths: list[str] = []
    csv_rows: list[tuple[str, str]] = []
    n_kept = 0
    n_dropped = 0
    n_failed = 0
    species_per_image_hist = np.zeros(args.top_k + 2, dtype=np.int64)

    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f, delimiter=";")
    csv_w.writerow(["image_name", "species_ids"])

    t0 = time.time()
    for i, path in enumerate(image_paths):
        rel = str(path.relative_to(root))
        rel_paths.append(rel)
        try:
            image = Image.open(path).convert("RGB")
            if args.resize_max_side > 0:
                w, h = image.size
                m = max(w, h)
                if m > args.resize_max_side:
                    s = args.resize_max_side / m
                    image = image.resize((max(1, int(w * s)), max(1, int(h * s))),
                                         Image.BILINEAR)
        except Exception as exc:
            logger.warning(f"Skip {rel}: {exc}")
            n_failed += 1
            continue
        all_tile_probs: list[torch.Tensor] = []
        tiles, _ = get_tiles(image, args.tile_size, stride)
        if tiles:
            tp = encode_tiles_to_probs(
                model, transform, tiles, args.device, args.batch_size, args.bf16,
                hflip_tta=args.hflip_tta,
            )
            all_tile_probs.append(tp)
        if args.whole_image:
            wp = encode_tiles_to_probs(
                model, transform, [image], args.device, args.batch_size, args.bf16,
                hflip_tta=args.hflip_tta,
            )
            all_tile_probs.append(wp)
        if not all_tile_probs:
            n_failed += 1
            continue
        tile_probs = torch.cat(all_tile_probs, dim=0)
        p_max = tile_probs.max(dim=0).values.numpy()
        out_max[i] = p_max.astype(np.float16)

        keep_ids, _ = threshold_and_format(p_max, species_ids, args.tau_conf, args.top_k)
        species_per_image_hist[min(len(keep_ids), args.top_k + 1)] += 1
        if len(keep_ids) >= args.min_species_per_image:
            csv_w.writerow([rel, ",".join(keep_ids)])
            n_kept += 1
        else:
            n_dropped += 1

        if (i + 1) % 50 == 0 or i + 1 == N:
            rate = (i + 1) / (time.time() - t0)
            eta = (N - (i + 1)) / max(rate, 1e-9)
            logger.info(
                f"  {i+1}/{N} ({rate:.2f} img/s, ETA {eta/60:.1f} min) "
                f"kept={n_kept} dropped={n_dropped} failed={n_failed}"
            )
        if (i + 1) % args.checkpoint_every == 0:
            csv_f.flush()
            np.savez_compressed(
                npz_path,
                image_paths=np.array(rel_paths + [""] * (N - len(rel_paths)), dtype=object),
                species_ids=np.array(species_ids, dtype=object),
                probs_max=out_max,
                tau_conf=np.float32(args.tau_conf),
                top_k=np.int32(args.top_k),
            )
            logger.info(f"  [checkpoint] flushed CSV + npz at {i+1}")

    csv_f.close()
    np.savez_compressed(
        npz_path,
        image_paths=np.array(rel_paths, dtype=object),
        species_ids=np.array(species_ids, dtype=object),
        probs_max=out_max,
        tau_conf=np.float32(args.tau_conf),
        top_k=np.int32(args.top_k),
    )
    logger.info(
        f"Done. CSV: {csv_path} ({n_kept} kept, {n_dropped} dropped, {n_failed} failed)\n"
        f"  species/image hist (0..{args.top_k}+): {species_per_image_hist.tolist()}\n"
        f"  npz: {npz_path} ({npz_path.stat().st_size/1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
