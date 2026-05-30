"""
Quick recall@K probe for the Phase-2 model on the held-out mosaic split.

Outputs:
  - average truth size
  - recall@K for K in {1, 3, 5, 10, 20, 50}
  - macro-F1 per sample for top-K (no threshold) for the same K values

Decides whether the model has any useful signal that we can submit, even if
sigmoid is saturated.
"""
from __future__ import annotations

import sys
import torch
import numpy as np
from pathlib import Path

from model import build_default_transform
from mosaic_dataset import MosaicDataset, DEFAULT_K_DIST, compose_mosaic
from run_inference import load_model_from_checkpoint, encode_tiles_to_probs

_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
sys.path.insert(0, str(_FOUR))
from tiling import get_tiles  # noqa: E402
from aggregation import aggregate_scores  # noqa: E402
from dataset import load_train_metadata, resolve_image_paths  # noqa: E402


CKPT = "/workspace/working/PlantCLEF2026/models/dinov3_v1/phase2_lora.pth"
META = "/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv"
ROOT = "/workspace/plantclef/raw/train/images_max_side_800"

DEVICE = "cuda"
TILE = 384
CANVAS = 384
N_MOS = 200
KS = [1, 2, 3, 5, 10, 20, 50]


def f1_per_sample(true_set: set[str], pred_set: set[str]) -> float:
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return 2 * p * r / (p + r)


def main() -> None:
    model, species_ids = load_model_from_checkpoint(CKPT, DEVICE)
    transform = build_default_transform(TILE)

    df = load_train_metadata(META)
    df = resolve_image_paths(df, ROOT, verify=False)
    rng = np.random.default_rng(42)
    all_species = sorted(df["species_id"].astype(str).unique())
    n_val = max(1, int(len(all_species) * 0.05))
    val_species = set(rng.choice(all_species, size=n_val, replace=False).tolist())
    val_species &= set(species_ids)
    val_df = df[df["species_id"].astype(str).isin(val_species)]

    val_dataset = MosaicDataset(
        metadata_df=val_df,
        species_ids=species_ids,
        canvas_size=CANVAS,
        k_dist=DEFAULT_K_DIST,
        samples_per_epoch=N_MOS,
        transform=None,
        seed=42,
        augment=False,
    )

    canvases, truths = [], []
    for i in range(N_MOS):
        rng_i = val_dataset._rng_for(i)
        K = val_dataset._sample_k(rng_i)
        sps = rng_i.sample(
            val_dataset._species_with_data,
            k=min(K, len(val_dataset._species_with_data)),
        )
        crops, chosen = [], []
        for sid in sps:
            img = val_dataset._sample_image(sid, rng_i)
            if img is not None:
                crops.append(img); chosen.append(sid)
            if len(crops) >= K:
                break
        if not crops:
            continue
        canvas = compose_mosaic(crops, CANVAS, rng_i)
        canvases.append(canvas)
        truths.append(set(chosen))

    print(f"[data] N={len(canvases)} mosaics, avg_truth_size="
          f"{np.mean([len(t) for t in truths]):.2f}")

    stride = max(1, TILE - 128)
    image_scores_list = []
    for canvas in canvases:
        tiles, _ = get_tiles(canvas, TILE, stride)
        probs = encode_tiles_to_probs(model, transform, tiles, DEVICE, 16, True)
        # For 1-tile mosaics, all agg modes equivalent; use mean (== probs)
        s = aggregate_scores(probs, mode="mean", top_m=3)
        image_scores_list.append(s)

    species_to_idx = {sid: i for i, sid in enumerate(species_ids)}
    truth_idx_sets = [
        {species_to_idx[s] for s in t if s in species_to_idx}
        for t in truths
    ]

    print("\nTop-K analysis (top-K predictions, no threshold):")
    print(f"{'K':>4s}  {'recall':>8s}  {'precision':>10s}  {'macro_F1':>10s}")
    for K in KS:
        rec, f1s = [], []
        for s, truth_idx in zip(image_scores_list, truth_idx_sets):
            topk = set(s.argsort(descending=True)[:K].tolist())
            tp = len(topk & truth_idx)
            r = tp / max(1, len(truth_idx))
            rec.append(r)
            pred_set = {species_ids[i] for i in topk}
            true_set = {species_ids[i] for i in truth_idx}
            f1s.append(f1_per_sample(true_set, pred_set))
        print(f"{K:>4d}  {np.mean(rec):>8.4f}  {tp/K:>10.4f}  {np.mean(f1s):>10.4f}")

    # Empty prediction => 0 F1 (since truth nonempty). Just confirm.
    print("\nBest-K macro-F1 if we tuned K only (no threshold).")


if __name__ == "__main__":
    main()
