#!/usr/bin/env python3
"""
Pre-compute and cache BioCLIP 2.5 ViT-H/14 features for an i002-style
manifest, so the head-only training stage can iterate over a tensor
in memory instead of re-running the frozen backbone on JPEGs every
epoch.

What it produces
----------------
A single `.pt` file with:

    {
      "train_features": fp16 tensor (N_train, 1024),
      "train_sp":       int64 tensor (N_train,)   species idx
      "train_gen":      int64 tensor (N_train,)   genus idx   (-1 = missing)
      "train_fam":      int64 tensor (N_train,)   family idx  (-1 = missing)
      "val_features":   fp16 tensor (N_val, 1024),
      "val_sp":         int64 tensor (N_val,)
      "val_gen":        int64 tensor (N_val,)
      "val_fam":        int64 tensor (N_val,)
      "encoders":       dict (species_to_idx, idx_to_species, ...)
      "config":         {dataset, val_fraction, val_seed, model, etc.}
    }

Plus, alongside it, the standard `encoders/idx_to_*.json` files the
i002 trainer also writes, so the cached run is drop-in compatible with
a non-cached run.

Why this exists
---------------
On a 4090 the head-only stage of the i002 schedule spends ~30 min/epoch
on JPEG decode + backbone forward, and the backbone is frozen.
Caching collapses that to a one-time ~10-15 min build, after which
each "epoch" is sub-second (head MLPs + classifiers on a (B, 1024)
tensor). Trade-off: no image augmentation — the cached features were
computed once at val_transform resolution.

Usage
-----

    python tools/cache_features_i002.py \\
        --metadata-csv      data/plantnet300k_manifest.csv \\
        --train-image-root  /mnt/d/PlantNet-300k/images/images/train \\
        --output-dir        outputs/feature_cache_plantnet300k \\
        --batch-size        128 \\
        --num-workers       8 \\
        --precision         bf16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EXP_DIR = (Path(__file__).resolve().parent.parent
           / "bioclip25_multitask")
if not EXP_DIR.is_dir():
    sys.exit(f"bioclip25_multitask sources not found at {EXP_DIR}")
sys.path.insert(0, str(EXP_DIR))

from dataset import (                                       # noqa: E402
    load_metadata, build_label_encoders,
    resolve_image_paths, build_val_split, MultiTaskDataset,
)
from model import BioCLIP25MultiTask, BIOCLIP25_MODEL_NAME  # noqa: E402
from transforms import val_transform                        # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cache BioCLIP 2.5 features for i002-compatible manifests.",
    )
    p.add_argument("--metadata-csv",     required=True,
                   help="i002-compatible manifest CSV (species_id, image_path, "
                        "optional genus/family).")
    p.add_argument("--train-image-root", default=None,
                   help="Image root for fallback path resolution; ignored if "
                        "the manifest has absolute image_path values.")
    p.add_argument("--output-dir", default="outputs/feature_cache",
                   help="Where to write cache.pt + encoders/.")
    p.add_argument("--model-name",  default=BIOCLIP25_MODEL_NAME)
    p.add_argument("--img-size",    type=int,   default=224)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--val-seed",    type=int,   default=42)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--num-workers", type=int,   default=8)
    p.add_argument("--precision",   default="bf16",
                   choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GPU performance knobs (Ampere/Ada/Hopper safe; no-op on older HW)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if args.precision == "bf16"
                 else torch.float16 if args.precision == "fp16"
                 else torch.float32)

    # ------------------------------------------------------------------
    # 1. Load + split manifest, build encoders (mirrors train.py exactly)
    # ------------------------------------------------------------------
    print(f"[cache] loading metadata: {args.metadata_csv}")
    df = load_metadata(args.metadata_csv, None)
    df = resolve_image_paths(df, args.train_image_root)
    df = df[df["resolved_path"].notna()].reset_index(drop=True)
    print(f"[cache] {len(df):,} rows with resolved paths")

    encoders = build_label_encoders(df, output_dir=str(out_dir / "encoders"))
    num_species = len(encoders["idx_to_species"])
    num_genus   = len(encoders["idx_to_genus"])
    num_family  = len(encoders["idx_to_family"])
    print(f"[cache] encoders: species={num_species:,}  "
          f"genus={num_genus:,}  family={num_family:,}")

    train_df, val_df = build_val_split(df, args.val_fraction, args.val_seed)
    print(f"[cache] split: train={len(train_df):,}  val={len(val_df):,}")

    # ------------------------------------------------------------------
    # 2. Build model in frozen-backbone mode
    # ------------------------------------------------------------------
    print(f"[cache] loading {args.model_name} (one-time download if uncached)")
    model = BioCLIP25MultiTask(
        num_species=num_species,
        num_genus=num_genus,
        num_family=num_family,
        model_name=args.model_name,
        hidden_dim=1024,
        dropout=0.2,
        use_taxonomy_heads=True,
    )
    model.configure_backbone("freeze")
    model = model.to(device).eval()

    v_tfm = val_transform(args.img_size)

    # ------------------------------------------------------------------
    # 3. Extract features for each split
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(ds: MultiTaskDataset, label: str):
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
        n = len(ds)
        feats   = torch.empty((n, 1024), dtype=torch.float16)
        sp_arr  = torch.empty((n,), dtype=torch.int64)
        gen_arr = torch.empty((n,), dtype=torch.int64)
        fam_arr = torch.empty((n,), dtype=torch.int64)

        i = 0
        t0 = time.perf_counter()
        for batch_idx, batch in enumerate(loader):
            images, sp, gen, fam = batch
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=device.type == "cuda",
                                    dtype=amp_dtype):
                feat = model._encode_raw(images)
            feats[i : i + feat.shape[0]] = feat.detach().to(torch.float16).cpu()
            sp_arr[i : i + feat.shape[0]]  = sp
            gen_arr[i : i + feat.shape[0]] = gen
            fam_arr[i : i + feat.shape[0]] = fam
            i += feat.shape[0]
            if (batch_idx + 1) % 50 == 0 or i == n:
                elapsed = time.perf_counter() - t0
                rate    = i / max(elapsed, 1e-6)
                eta     = (n - i) / max(rate, 1e-6)
                print(f"  [{label}] {i:>7,}/{n:>7,}   {rate:5.1f} img/s   "
                      f"ETA {eta:5.0f}s")
        return feats, sp_arr, gen_arr, fam_arr

    train_ds = MultiTaskDataset(train_df, encoders, v_tfm)
    val_ds   = MultiTaskDataset(val_df,   encoders, v_tfm)

    print(f"[cache] encoding train split ({len(train_ds):,} images)")
    train_feats, train_sp, train_gen, train_fam = encode(train_ds, "train")
    print(f"[cache] encoding val split ({len(val_ds):,} images)")
    val_feats, val_sp, val_gen, val_fam = encode(val_ds, "val")

    # ------------------------------------------------------------------
    # 4. Save the cache
    # ------------------------------------------------------------------
    cache_path = out_dir / "cache.pt"
    print(f"[cache] writing {cache_path}")
    torch.save({
        "train_features": train_feats,
        "train_sp":       train_sp,
        "train_gen":      train_gen,
        "train_fam":      train_fam,
        "val_features":   val_feats,
        "val_sp":         val_sp,
        "val_gen":        val_gen,
        "val_fam":        val_fam,
        "encoders":       encoders,
        "config": {
            "metadata_csv":     args.metadata_csv,
            "train_image_root": args.train_image_root,
            "model_name":       args.model_name,
            "img_size":         args.img_size,
            "val_fraction":     args.val_fraction,
            "val_seed":         args.val_seed,
            "precision":        args.precision,
        },
    }, cache_path)

    size_mb = cache_path.stat().st_size / 1e6
    print(f"[cache] done: {size_mb:.1f} MB at {cache_path}")
    print(f"[cache] train features shape: {tuple(train_feats.shape)}  "
          f"val features shape: {tuple(val_feats.shape)}")
    print(f"[cache] feed this to train.py via --feature-cache {cache_path}")


if __name__ == "__main__":
    main()
