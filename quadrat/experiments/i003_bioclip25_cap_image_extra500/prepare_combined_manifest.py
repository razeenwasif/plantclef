"""
Build the combined (old + extra_under100) training manifest for i003.

Pipeline (order matters):
  1. Load old + new manifests via load_metadata_csv (normalises species_id).
  2. Fill genus/family on the new frame from a species_id lookup built from old.
  3. Reduce both frames to a fixed KEEP_COLS column set.
  4. Concatenate.
  5. Deduplicate by image_path BEFORE the cap.
  6. Apply max-N-per-species cap (deterministic seed).
  7. Save: capped manifest CSV + per-species counts CSV + summary JSON.

Run:
    cd /root/workspace/PlantCLEF2026/src_experiments/i003_bioclip25_cap_image_extra500
    python prepare_combined_manifest.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

# Catch truncated images at validation time — do NOT enable LOAD_TRUNCATED_IMAGES
# here, otherwise PIL would silently accept short-data files.
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 300_000_000


def _is_valid_image_file(path: str) -> bool:
    """
    Strict image validation: full PIL decode.

    Catches:
      - Missing files (OSError on open)
      - Wrong-magic-bytes (e.g., JSON saved as .jpg) → UnidentifiedImageError
      - Truncated files where data ends mid-stream → OSError on .load()
      - Decompression-bomb files exceeding MAX_IMAGE_PIXELS

    Slower than a header-only check (we actually decode pixels), but the
    extra ~1 ms per file is worth eliminating epoch-time crashes.
    """
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError):
        return False

from data.metadata_utils import (
    apply_max_images_per_species_cap,
    load_metadata_csv,
    print_species_distribution,
)

logger = logging.getLogger(__name__)

I003_ROOT = Path(__file__).resolve().parent

DEFAULT_OLD_MANIFEST = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "i001_data_download/data/training_usage/metadata_filled_genus_family.csv"
)
DEFAULT_NEW_MANIFEST = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "i001_data_download/data/extra_under100/extra_under100_train_manifest.csv"
)
DEFAULT_OUTPUT_CSV   = str(I003_ROOT / "data" / "combined_old_extra_max500_train_manifest.csv")
DEFAULT_SUMMARY_JSON = str(I003_ROOT / "data" / "combined_old_extra_max500_summary.json")
DEFAULT_COUNTS_CSV   = str(I003_ROOT / "data" / "species_counts_before_after.csv")

KEEP_COLS = [
    "image_path",
    "image_name",
    "species_id",
    "scientific_name",
    "genus",
    "family",
    "source",
    "url",
    "gbif_species_id",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build i003 combined+capped training manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--old-manifest",    default=DEFAULT_OLD_MANIFEST)
    p.add_argument("--new-manifest",    default=DEFAULT_NEW_MANIFEST)
    p.add_argument("--output-csv",      default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--summary-json",    default=DEFAULT_SUMMARY_JSON)
    p.add_argument("--counts-csv",      default=DEFAULT_COUNTS_CSV)
    p.add_argument("--max-per-species", type=int, default=500)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument(
        "--validate-new-images", action="store_true", default=True,
        help=(
            "Magic-byte check on every new-manifest image_path; drop rows whose "
            "file is missing or not a real image. Old manifest is trusted (used "
            "in production by i002)."
        ),
    )
    p.add_argument(
        "--no-validate-new-images", action="store_false",
        dest="validate_new_images",
    )
    p.add_argument(
        "--validate-workers", type=int, default=64,
        help="Threads for parallel image validation.",
    )
    return p.parse_args()


def _align_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return df subset to `columns` in the given order, adding missing as NaN."""
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    return out[columns]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    logger.info("=" * 70)
    logger.info("  i003 prepare_combined_manifest.py")
    logger.info("=" * 70)
    logger.info(f"  Old manifest        : {args.old_manifest}")
    logger.info(f"  New manifest        : {args.new_manifest}")
    logger.info(f"  Output CSV          : {args.output_csv}")
    logger.info(f"  Summary JSON        : {args.summary_json}")
    logger.info(f"  Counts CSV          : {args.counts_csv}")
    logger.info(f"  Max images/species  : {args.max_per_species}")
    logger.info(f"  Seed                : {args.seed}")
    logger.info("=" * 70)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # 1. Load both manifests (load_metadata_csv normalises species_id)
    logger.info("Loading old manifest ...")
    df_old = load_metadata_csv(args.old_manifest)
    logger.info("Loading new manifest ...")
    df_new = load_metadata_csv(args.new_manifest)

    n_old, n_new = len(df_old), len(df_new)
    n_sp_old = df_old["species_id"].nunique()
    n_sp_new = df_new["species_id"].nunique()
    logger.info(f"  old: {n_old:,} rows, {n_sp_old:,} species")
    logger.info(f"  new: {n_new:,} rows, {n_sp_new:,} species")

    # Validate new manifest images (~9% are corrupt JSON download responses)
    n_new_invalid = 0
    if args.validate_new_images:
        logger.info(
            f"Validating {len(df_new):,} new-manifest image files "
            f"(full PIL decode, {args.validate_workers} threads — slow but catches truncation) ..."
        )
        paths = df_new["image_path"].tolist()
        with ThreadPoolExecutor(max_workers=args.validate_workers) as ex:
            valid = list(ex.map(_is_valid_image_file, paths))
        n_valid = int(sum(valid))
        n_new_invalid = len(df_new) - n_valid
        logger.info(
            f"  valid : {n_valid:,}/{len(df_new):,} "
            f"({100.0*n_valid/max(len(df_new),1):.1f}%)"
        )
        logger.info(f"  dropped (corrupt/missing) : {n_new_invalid:,}")
        df_new = df_new[pd.Series(valid, index=df_new.index)].reset_index(drop=True)
        n_new = len(df_new)
        n_sp_new = df_new["species_id"].nunique()
        logger.info(f"  new (post-validate): {n_new:,} rows, {n_sp_new:,} species")

    # 2. Fill genus/family on new from species_id lookup built from old
    logger.info("Filling genus/family on new manifest from old-manifest lookup ...")
    tax_lookup = (
        df_old.dropna(subset=["species_id"])
        .drop_duplicates("species_id")
        .set_index("species_id")[["genus", "family"]]
    )
    if "genus" not in df_new.columns:
        df_new["genus"] = pd.NA
    if "family" not in df_new.columns:
        df_new["family"] = pd.NA
    new_filled_genus = df_new["species_id"].map(tax_lookup["genus"])
    new_filled_family = df_new["species_id"].map(tax_lookup["family"])
    df_new["genus"]  = df_new["genus"].where(df_new["genus"].notna(),  new_filled_genus)
    df_new["family"] = df_new["family"].where(df_new["family"].notna(), new_filled_family)
    n_new_with_genus  = int(df_new["genus"].notna().sum())
    n_new_with_family = int(df_new["family"].notna().sum())
    logger.info(
        f"  new rows with genus  : {n_new_with_genus:,}/{n_new:,} "
        f"({100.0*n_new_with_genus/max(n_new,1):.1f}%)"
    )
    logger.info(
        f"  new rows with family : {n_new_with_family:,}/{n_new:,} "
        f"({100.0*n_new_with_family/max(n_new,1):.1f}%)"
    )

    # Track which species in new are not present in old (would have NaN taxonomy)
    sp_old_set = set(df_old["species_id"].dropna().unique())
    sp_new_set = set(df_new["species_id"].dropna().unique())
    sp_new_only = sorted(sp_new_set - sp_old_set)
    if sp_new_only:
        logger.info(
            f"  {len(sp_new_only):,} species are only in new manifest "
            f"(genus/family stay NaN; masked as -1 in loss)"
        )
        logger.info(f"    examples: {sp_new_only[:10]}")
    else:
        logger.info("  All new-manifest species are also in old manifest — full taxonomy coverage.")

    # 3. Align columns
    df_old_a = _align_columns(df_old, KEEP_COLS)
    df_new_a = _align_columns(df_new, KEEP_COLS)

    # 4. Concatenate
    df_concat = pd.concat([df_old_a, df_new_a], ignore_index=True)
    n_concat = len(df_concat)
    logger.info(f"After concat: {n_concat:,} rows")

    # 5. Dedup BEFORE cap
    n_before_dedup = len(df_concat)
    df_concat = df_concat.drop_duplicates(subset=["image_path"], keep="first").reset_index(drop=True)
    n_after_dedup  = len(df_concat)
    n_duplicates_removed = n_before_dedup - n_after_dedup
    logger.info(
        f"Dedup by image_path: removed {n_duplicates_removed:,} duplicate rows "
        f"({n_after_dedup:,} remain)"
    )

    print_species_distribution(df_concat, "After concat+dedup, BEFORE cap")
    counts_before = df_concat.groupby("species_id").size()

    n_species_combined = int(counts_before.shape[0])
    min_before    = int(counts_before.min())
    median_before = float(counts_before.median())
    max_before    = int(counts_before.max())
    n_species_under_100_before = int((counts_before < 100).sum())

    # 6. Apply 500-per-species cap
    logger.info(f"Applying cap: max {args.max_per_species} images per species (seed={args.seed}) ...")
    df_capped = apply_max_images_per_species_cap(
        df_concat,
        max_per_species=args.max_per_species,
        seed=args.seed,
    )
    n_after_cap = len(df_capped)
    logger.info(f"After cap: {n_after_cap:,} rows")

    print_species_distribution(df_capped, f"After {args.max_per_species}-cap")
    counts_after = df_capped.groupby("species_id").size()

    min_after    = int(counts_after.min())
    median_after = float(counts_after.median())
    max_after    = int(counts_after.max())
    n_species_at_cap = int((counts_after == args.max_per_species).sum())
    n_species_under_100_after = int((counts_after < 100).sum())

    sample_at_cap = sorted(counts_after[counts_after == args.max_per_species].index.tolist())[:20]
    sample_under_100 = sorted(counts_after[counts_after < 100].index.tolist())[:20]

    logger.info(f"  Species hitting the {args.max_per_species} cap                : {n_species_at_cap:,}")
    logger.info(f"  Species still under 100 after combining (post-cap)   : {n_species_under_100_after:,}")
    logger.info(f"  Species under 100 in concat+dedup (pre-cap, sanity)  : {n_species_under_100_before:,}")

    # 7. Save outputs
    logger.info(f"Writing combined manifest CSV: {args.output_csv}")
    df_capped.to_csv(args.output_csv, index=False)

    logger.info(f"Writing per-species counts CSV: {args.counts_csv}")
    sci_lookup = (
        pd.concat([df_old[["species_id", "scientific_name"]],
                   df_new[["species_id", "scientific_name"]]], ignore_index=True)
        .dropna(subset=["species_id"])
        .drop_duplicates("species_id")
        .set_index("species_id")["scientific_name"]
    )
    counts_df = pd.DataFrame({
        "species_id":       counts_before.index,
        "count_before_cap": counts_before.values,
    })
    counts_df["count_after_cap"] = counts_df["species_id"].map(counts_after).fillna(0).astype(int)
    counts_df["scientific_name"] = counts_df["species_id"].map(sci_lookup)
    counts_df = counts_df.sort_values("count_before_cap", ascending=False).reset_index(drop=True)
    counts_df.to_csv(args.counts_csv, index=False)

    summary = {
        "old_manifest":               args.old_manifest,
        "new_manifest":               args.new_manifest,
        "output_csv":                 args.output_csv,
        "max_per_species":            args.max_per_species,
        "seed":                       args.seed,
        "n_old":                      n_old,
        "n_new":                      n_new,
        "n_new_invalid_dropped":      int(n_new_invalid),
        "validate_new_images":        bool(args.validate_new_images),
        "n_species_old":              int(n_sp_old),
        "n_species_new":              int(n_sp_new),
        "n_new_with_genus":           n_new_with_genus,
        "n_new_with_family":          n_new_with_family,
        "n_species_only_in_new":      len(sp_new_only),
        "n_concat":                   n_concat,
        "n_duplicates_removed":       int(n_duplicates_removed),
        "n_after_dedup":              n_after_dedup,
        "n_after_cap":                n_after_cap,
        "n_species_combined":         n_species_combined,
        "min_before":                 min_before,
        "median_before":              median_before,
        "max_before":                 max_before,
        "min_after":                  min_after,
        "median_after":               median_after,
        "max_after":                  max_after,
        "n_species_at_cap":           n_species_at_cap,
        "n_species_under_100_before": n_species_under_100_before,
        "n_species_under_100_after":  n_species_under_100_after,
        "sample_species_at_cap":      sample_at_cap,
        "sample_species_under_100":   sample_under_100,
    }
    logger.info(f"Writing summary JSON: {args.summary_json}")
    with open(args.summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(
        f"DONE: wrote {args.output_csv} with {len(df_capped):,} rows, "
        f"{df_capped['species_id'].nunique():,} species."
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
