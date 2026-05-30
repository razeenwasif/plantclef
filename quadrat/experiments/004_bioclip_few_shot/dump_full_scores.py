"""
Dump FULL per-quadrat per-species score matrix from the BioCLIP few-shot pipeline.

Mirrors run_inference.py end-to-end (same tiling, same prototype scoring, same
aggregation) but saves the dense (n_quadrats, n_species) matrix instead of
truncating to top-N. Output schema matches the 008 Phase A npz so they can be
fused directly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image as _PIL

from models import load_model, resolve_device, default_batch_size
from tiling import get_tiles, encode_image_tiles
from few_shot import SupportBank, score_prototype, score_knn
from aggregation import aggregate_scores, AGG_MODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--images-root",
                   default="/workspace/plantclef/kaggle_uploads/test/images")
    p.add_argument("--output", required=True,
                   help="Path to write dense scores npz")
    p.add_argument("--scoring-mode", default="prototype",
                   choices=["prototype", "knn"])
    p.add_argument("--agg-modes", nargs="+", default=["max"],
                   choices=list(AGG_MODES),
                   help="Aggregation modes to dump (one matrix per mode)")
    p.add_argument("--agg-top-m", type=int, default=3)
    p.add_argument("--tile-size", type=int, default=224)
    p.add_argument("--tile-overlap", type=int, default=112)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    stride = args.tile_size - args.tile_overlap

    bank = SupportBank.load(Path(args.bank_dir) / "bank.pt")
    n_species = len(bank.species_ids)
    logger.info(f"Bank: {n_species} species")

    bank_meta_path = Path(args.bank_dir) / "bank_metadata.json"
    import json
    with open(bank_meta_path) as f:
        bank_meta = json.load(f)
    model_name = bank_meta["model_name"]
    batch_size = args.batch_size or default_batch_size(model_name)
    logger.info(f"Loading {model_name} ...")
    model, transform, _ = load_model(model_name, device)

    images_root = Path(args.images_root)
    image_paths = sorted(set(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for p in images_root.glob(ext)
    ))
    if args.limit:
        image_paths = image_paths[: args.limit]
    n = len(image_paths)
    logger.info(f"Found {n} images. Allocating ({n}, {n_species}) per agg mode.")

    # Allocate one (n, n_species) matrix per agg mode, fp16 to keep RAM small.
    score_mats = {m: np.zeros((n, n_species), dtype=np.float16) for m in args.agg_modes}
    quadrat_ids: list[str] = []

    t0 = time.perf_counter()
    for i, img_path in enumerate(image_paths):
        qid = img_path.stem
        quadrat_ids.append(qid)
        try:
            img = _PIL.open(img_path).convert("RGB")
            tiles, _ = get_tiles(img, args.tile_size, stride)
            tile_emb = encode_image_tiles(
                model, transform, tiles, device, batch_size=batch_size
            )
            if args.scoring_mode == "prototype":
                tile_scores = score_prototype(tile_emb, bank, device=device)
            else:
                tile_scores = score_knn(tile_emb, bank, device=device)
            for mode in args.agg_modes:
                img_scores = aggregate_scores(
                    tile_scores, mode=mode, top_m=args.agg_top_m
                )
                score_mats[mode][i] = img_scores.numpy().astype(np.float16)
        except Exception as exc:
            logger.error(f"Error on {qid}: {exc}")

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  [{i+1:>5}/{n}] {qid}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "quadrat_ids": np.array(quadrat_ids, dtype=object),
        "species_ids": np.array(bank.species_ids, dtype=object),
    }
    for mode, mat in score_mats.items():
        save_kwargs[f"scores_{mode}"] = mat
    np.savez_compressed(out_path, **save_kwargs)
    logger.info(f"Saved {out_path}  ({n} quadrats × {n_species} species, "
                f"modes={args.agg_modes})")


if __name__ == "__main__":
    main()
