"""
Build and cache a few-shot support bank from the PlantCLEF training images.

What this script does
---------------------
1. Loads the single-plant training metadata CSV.
2. Resolves image paths on disk  ({image_root}/{species_id}/{image_name}).
3. Samples up to K images per species using a configurable sampling strategy.
4. Encodes each image with the specified BioCLIP model.
5. Computes per-species prototype embeddings (mean-pooled, L2-normalised).
6. Saves the following artifacts to the cache directory:

   {cache_dir}/{run_slug}/
       bank_metadata.json   model name, K, seed, build timestamp, stats
       manifest.csv         one row per support image: species_id, image_name, path
       embeddings.pt        dict: species_id (str) -> (k_i, dim) tensor
       bank.pt              SupportBank (prototypes + embeddings), loadable by few_shot.py

Usage examples
--------------
# Minimal (uses all defaults: bioclip-1, K=5, random sampling)
python build_support_bank.py

# K=10 with BioCLIP 2
python build_support_bank.py --model-name bioclip-2 --k 10

# K=20 with top_n_per_species sampling and a custom cache location
python build_support_bank.py \\
    --k 20 \\
    --sampling-mode top_n_per_species \\
    --cache-dir /tmp/fewshot_cache

# Smoke-test: only build support for first 50 species
python build_support_bank.py --limit-species 50 --k 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

# Local modules
from models import load_model, resolve_device, default_batch_size, resolve_model_name
from dataset import (
    DEFAULT_TRAIN_META_CSV,
    DEFAULT_TRAIN_IMAGE_ROOT,
    DEFAULT_SPECIES_CSV,
    load_train_metadata,
    resolve_image_paths,
    sample_support_images,
    load_species_ids,
)
from tiling import encode_image_tiles
from few_shot import SupportBank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a BioCLIP few-shot support bank from training images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        "--model-name", default="bioclip",
        help="BioCLIP model. Shorthands: bioclip, bioclip-2, bioclip-2.5. "
             "Full HF-hub paths also accepted.",
    )
    p.add_argument("--device", default="auto",
                   help="'auto' | 'cuda' | 'cpu' | 'cuda:0' etc.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size for image encoding. Defaults to model-specific value.")

    # Data paths
    p.add_argument("--train-meta-csv", default=DEFAULT_TRAIN_META_CSV,
                   help="Training metadata CSV (semicolon-delimited).")
    p.add_argument("--train-image-root", default=DEFAULT_TRAIN_IMAGE_ROOT,
                   help="Root dir for training images ({root}/{species_id}/{image_name}).")
    p.add_argument("--species-csv", default=DEFAULT_SPECIES_CSV,
                   help="Enriched species CSV for filtering to competition species only.")

    # Support sampling
    p.add_argument("--k", type=int, default=5,
                   help="Max support images per species.")
    p.add_argument(
        "--sampling-mode", default="random",
        choices=["random", "capped_all", "top_n_per_species"],
        help=(
            "random: random K images per species. "
            "capped_all: use all images if <K available, else random K. "
            "top_n_per_species: prefer diverse organ types, fall back to random."
        ),
    )
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible sampling.")

    # Output
    p.add_argument("--cache-dir", default="./cache",
                   help="Root directory for cached support bank artifacts.")

    # Filtering / debug
    p.add_argument("--limit-species", type=int, default=None,
                   help="Only process the first N species (for smoke tests).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing cache even if it already exists.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    batch_size = args.batch_size or default_batch_size(args.model_name)
    full_model_name = resolve_model_name(args.model_name)
    model_slug = full_model_name.split("/")[-1].replace(".", "-")

    run_slug = f"{model_slug}_k{args.k}_{args.sampling_mode}_seed{args.seed}"
    out_dir = Path(args.cache_dir) / run_slug
    bank_file = out_dir / "bank.pt"

    print(f"\n{'='*60}")
    print(f"BioCLIP Few-Shot Support Bank Builder")
    print(f"  run_slug    : {run_slug}")
    print(f"  model       : {full_model_name}")
    print(f"  K           : {args.k}")
    print(f"  sampling    : {args.sampling_mode}")
    print(f"  seed        : {args.seed}")
    print(f"  device      : {device}")
    print(f"  batch_size  : {batch_size}")
    print(f"  cache dir   : {out_dir}")
    print(f"{'='*60}\n")

    if bank_file.exists() and not args.overwrite:
        print(f"Cache already exists: {bank_file}")
        print("Pass --overwrite to rebuild. Exiting.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------
    logger.info("Loading training metadata ...")
    df = load_train_metadata(args.train_meta_csv)

    # Optionally restrict to competition species only
    if Path(args.species_csv).exists():
        competition_ids = set(load_species_ids(args.species_csv))
        before = len(df["species_id"].unique())
        df = df[df["species_id"].isin(competition_ids)]
        after = len(df["species_id"].unique())
        logger.info(
            f"Filtered to competition species: {after:,} / {before:,} species retained "
            f"({len(df):,} image rows)"
        )
    else:
        logger.warning(f"Species CSV not found: {args.species_csv} - using all species in metadata")

    # Optional species limit (for smoke tests)
    if args.limit_species:
        species_subset = sorted(df["species_id"].unique())[: args.limit_species]
        df = df[df["species_id"].isin(species_subset)]
        logger.info(f"Limiting to {args.limit_species} species ({len(df):,} rows)")

    # ------------------------------------------------------------------
    # 2. Resolve image paths
    # ------------------------------------------------------------------
    logger.info("Resolving image paths ...")
    df = resolve_image_paths(df, args.train_image_root, verify=False)

    n_before = len(df)
    df_valid = df[df["resolved_path"].notna()]
    logger.info(
        f"Valid paths: {len(df_valid):,} / {n_before:,} rows "
        f"({n_before - len(df_valid):,} skipped)"
    )

    # ------------------------------------------------------------------
    # 3. Sample support images
    # ------------------------------------------------------------------
    logger.info(f"Sampling K={args.k} support images per species ...")
    support_df = sample_support_images(
        df_valid, k=args.k, mode=args.sampling_mode, seed=args.seed
    )

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model {full_model_name} ...")
    model, transform, _ = load_model(args.model_name, device)
    logger.info(f"  Model loaded on {device}")

    # ------------------------------------------------------------------
    # 5. Encode support images
    # ------------------------------------------------------------------
    logger.info(f"Encoding {len(support_df):,} support images ...")
    t0 = time.perf_counter()

    embeddings_by_species: dict[str, torch.Tensor] = {}
    species_list = support_df["species_id"].unique().tolist()
    n_species = len(species_list)
    errors = 0

    for si, species_id in enumerate(species_list):
        species_rows = support_df[support_df["species_id"] == species_id]
        image_paths = species_rows["resolved_path"].tolist()

        # Load images for this species
        loaded: list[Image.Image] = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                loaded.append(img)
            except Exception as exc:
                logger.warning(f"Could not load {img_path}: {exc}")
                errors += 1

        if not loaded:
            logger.warning(f"Species {species_id}: no images loaded, skipping")
            continue

        # Encode
        try:
            emb = encode_image_tiles(model, transform, loaded, device, batch_size=batch_size)
            embeddings_by_species[species_id] = emb  # (k_i, dim)
        except Exception as exc:
            logger.error(f"Species {species_id}: encoding failed: {exc}")
            errors += 1
            continue

        if (si + 1) % 500 == 0 or (si + 1) == n_species:
            elapsed = time.perf_counter() - t0
            rate = (si + 1) / elapsed
            eta = (n_species - si - 1) / rate if rate > 0 else 0
            logger.info(
                f"  [{si+1:>5}/{n_species}]  "
                f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s  "
                f"(errors={errors})"
            )

    encode_secs = time.perf_counter() - t0
    n_encoded = len(embeddings_by_species)
    logger.info(
        f"Encoding complete: {n_encoded:,} species in {encode_secs:.1f}s  "
        f"({errors} errors)"
    )

    if n_encoded == 0:
        sys.exit("ERROR: no species were successfully encoded - aborting")

    # ------------------------------------------------------------------
    # 6. Build SupportBank and save
    # ------------------------------------------------------------------
    logger.info("Building SupportBank ...")
    bank_species_ids = [sid for sid in species_list if sid in embeddings_by_species]
    bank = SupportBank(
        species_ids=bank_species_ids,
        embeddings_by_species=embeddings_by_species,
    )
    bank.save(bank_file)

    # ------------------------------------------------------------------
    # 7. Save manifest CSV
    # ------------------------------------------------------------------
    manifest_path = out_dir / "manifest.csv"
    manifest_rows = support_df[support_df["species_id"].isin(embeddings_by_species)][
        ["species_id", "image_name", "resolved_path", "organ"]
        if "organ" in support_df.columns
        else ["species_id", "image_name", "resolved_path"]
    ]
    manifest_rows.to_csv(manifest_path, index=False)
    logger.info(f"Manifest saved: {manifest_path}  ({len(manifest_rows):,} rows)")

    # ------------------------------------------------------------------
    # 8. Save metadata JSON
    # ------------------------------------------------------------------
    support_counts = {sid: bank.support_counts[sid] for sid in bank_species_ids}
    metadata = {
        "run_slug": run_slug,
        "model_name": full_model_name,
        "k": args.k,
        "sampling_mode": args.sampling_mode,
        "seed": args.seed,
        "device": device,
        "batch_size": batch_size,
        "train_meta_csv": args.train_meta_csv,
        "train_image_root": args.train_image_root,
        "n_species": n_encoded,
        "total_support_images": sum(support_counts.values()),
        "min_support": min(support_counts.values()),
        "max_support": max(support_counts.values()),
        "encode_secs": round(encode_secs, 2),
        "n_encode_errors": errors,
        "embed_dim": int(bank.prototypes.shape[1]),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_dir / "bank_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {meta_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Support bank ready: {out_dir}/")
    print(f"  Species encoded : {n_encoded:,}")
    print(f"  Support images  : {sum(support_counts.values()):,}")
    print(f"  Embed dim       : {int(bank.prototypes.shape[1])}")
    print(f"  Encode time     : {encode_secs:.1f}s")
    if errors:
        print(f"  Errors          : {errors}")
    print(f"  Bank file       : {bank_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
