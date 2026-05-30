"""
Diagnostic: figure out why mosaic validation produces a constant macro-F1.

Steps:
  1. Load Phase-2 checkpoint.
  2. Forward 5 distinct mosaic canvases through the tiled-inference pipeline.
  3. Print:
       - sigmoid prob distribution (min/median/max/mean) per mosaic
       - top-20 species indices per agg_mode (max/mean/noisy_or) per mosaic
       - whether top-20 sets are identical across mosaics
       - whether top-20 sets are identical across agg modes
  4. Cross-check checkpoint species_ids vs CSV species_ids and confirm the
     classifier head's per-class output bias / norm.
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
N_MOS = 5


def main() -> None:
    print("=" * 70)
    print("DINOv3 multi-label model diagnostic")
    print("=" * 70)

    model, species_ids = load_model_from_checkpoint(CKPT, DEVICE)
    transform = build_default_transform(TILE)

    # Quick sanity: classifier-head output bias norm (collapsed model would
    # show a few classes with very large bias)
    head = model.head
    last_linear = None
    for m in head.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    if last_linear is not None and last_linear.bias is not None:
        bias = last_linear.bias.detach().float().cpu()
        weight = last_linear.weight.detach().float().cpu()
        print(f"[head] last Linear out_features={last_linear.out_features} "
              f"bias min/median/max = "
              f"{bias.min():.3f}/{bias.median():.3f}/{bias.max():.3f}")
        print(f"[head]   weight row-norm min/median/max = "
              f"{weight.norm(dim=1).min():.3f}/{weight.norm(dim=1).median():.3f}/"
              f"{weight.norm(dim=1).max():.3f}")
        # Classes most strongly preferred by the bias alone
        topb = bias.argsort(descending=True)[:10].tolist()
        print(f"[head]   top-10 by bias: {topb}")

    # Build mosaics
    df = load_train_metadata(META)
    df = resolve_image_paths(df, ROOT, verify=False)
    rng = np.random.default_rng(42)
    all_species = sorted(df["species_id"].astype(str).unique())
    n_val = max(1, int(len(all_species) * 0.05))
    val_species = set(rng.choice(all_species, size=n_val, replace=False).tolist())
    val_species &= set(species_ids)
    val_df = df[df["species_id"].astype(str).isin(val_species)]
    print(f"[data] held-out species count: {len(val_species)}")

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

    canvases = []
    truths = []
    for i in range(N_MOS):
        rng_i = val_dataset._rng_for(i)
        K = val_dataset._sample_k(rng_i)
        species_choices = rng_i.sample(
            val_dataset._species_with_data,
            k=min(K, len(val_dataset._species_with_data)),
        )
        crops, chosen = [], []
        for sid in species_choices:
            img = val_dataset._sample_image(sid, rng_i)
            if img is not None:
                crops.append(img)
                chosen.append(sid)
            if len(crops) >= K:
                break
        if not crops:
            continue
        canvas = compose_mosaic(crops, CANVAS, rng_i)
        canvases.append(canvas)
        truths.append(set(chosen))
    print(f"[data] built {len(canvases)} canvases")

    stride = max(1, TILE - 128)
    top20_by_mosaic = {"max": [], "mean": [], "noisy_or": []}

    for i, canvas in enumerate(canvases):
        tiles, _ = get_tiles(canvas, TILE, stride)
        print(f"\n[mosaic {i}] truth={sorted(truths[i])}, n_tiles={len(tiles)}")
        probs = encode_tiles_to_probs(model, transform, tiles, DEVICE, 8, True)
        # probs: (n_tiles, n_classes)
        flat = probs.flatten()
        print(f"  per-tile sigmoid prob: min={flat.min():.4f} median={flat.median():.4f} "
              f"max={flat.max():.4f} mean={flat.mean():.4f}")
        # Across-tile correlation: how much variation is there?
        per_tile_mean = probs.mean(dim=1)
        print(f"  per-tile mean prob (across classes): "
              f"{per_tile_mean.min():.4f}..{per_tile_mean.max():.4f}")
        # For each class, std across tiles
        per_class_std = probs.std(dim=0)
        print(f"  per-class std across tiles: "
              f"min={per_class_std.min():.4f} median={per_class_std.median():.4f} "
              f"max={per_class_std.max():.4f}")

        for agg in ["max", "mean", "noisy_or"]:
            if agg == "noisy_or":
                eps = 1e-6
                clamped = probs.clamp(eps, 1 - eps)
                scores_in = torch.log(clamped / (1 - clamped))
            else:
                scores_in = probs
            image_scores = aggregate_scores(scores_in, mode=agg, top_m=3)
            top20_idx = image_scores.argsort(descending=True)[:20].tolist()
            top20_vals = [image_scores[k].item() for k in top20_idx]
            top5_species = [species_ids[k] for k in top20_idx[:5]]
            top20_by_mosaic[agg].append(top20_idx)
            print(f"  agg={agg}: top5_score={top20_vals[:5]} top5_sp={top5_species}")
            # Truth-in-top20?
            truth_idx_set = {species_ids.index(s) for s in truths[i] if s in species_ids}
            hits = truth_idx_set & set(top20_idx)
            print(f"     truth_in_top20: {len(hits)}/{len(truth_idx_set)}  hit_idx={hits}")

    # Cross-mosaic constancy
    print("\n=== Top-20 set constancy across mosaics (per agg) ===")
    for agg, lists in top20_by_mosaic.items():
        sets = [frozenset(L) for L in lists]
        unique = set(sets)
        print(f"  agg={agg}: {len(unique)} distinct top-20 sets across {len(sets)} mosaics")
        if len(unique) > 1:
            # Show jaccard between mosaic 0 and others
            base = sets[0]
            for j in range(1, len(sets)):
                inter = len(base & sets[j])
                print(f"    mosaic0 vs mosaic{j}: |inter|={inter}/20")

    # Cross-agg constancy on mosaic 0
    print("\n=== Top-20 set agreement across agg_modes (mosaic 0) ===")
    mx = set(top20_by_mosaic["max"][0])
    mn = set(top20_by_mosaic["mean"][0])
    nz = set(top20_by_mosaic["noisy_or"][0])
    print(f"  max ∩ mean = {len(mx & mn)}/20")
    print(f"  max ∩ noisy_or = {len(mx & nz)}/20")
    print(f"  mean ∩ noisy_or = {len(mn & nz)}/20")


if __name__ == "__main__":
    main()
