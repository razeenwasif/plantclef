#!/usr/bin/env python3
"""
Dataset verification script for i002_bioclip25_cap_image.

Loads the metadata CSV, applies the same capping logic as training,
prints before/after distribution statistics, verifies a sample of
image_path files actually exist, and checks that species_id / genus /
family columns are usable.

Usage
-----
  python scripts/verify_dataset_cap.py \\
      --metadata-csv /path/to/metadata_filled_genus_family.csv \\
      --max-images-per-species 500

  python scripts/verify_dataset_cap.py \\
      --metadata-csv /path/to/metadata_filled_genus_family.csv \\
      --max-images-per-species 500 \\
      --max-train-rows 200000 \\
      --cap-seed 42 \\
      --sample-check 200
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the experiment root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.metadata_utils import (
    load_metadata_csv,
    apply_max_images_per_species_cap,
    apply_max_train_rows_cap,
    print_species_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify metadata CSV and capping logic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--metadata-csv",
        default=(
            "/root/workspace/PlantCLEF2026/src_experiments/"
            "i001_data_download/data/training_usage/"
            "metadata_filled_genus_family.csv"
        ),
        help="Path to training metadata CSV.",
    )
    p.add_argument(
        "--max-images-per-species", type=int, default=0, metavar="N",
        help="Cap each species to N images (0 = no cap).",
    )
    p.add_argument(
        "--max-train-rows", type=int, default=0, metavar="N",
        help="Cap total rows (0 = no cap).",
    )
    p.add_argument(
        "--cap-seed", type=int, default=42,
        help="RNG seed for capping.",
    )
    p.add_argument(
        "--sample-check", type=int, default=100, metavar="N",
        help="Number of random rows to check for file existence.",
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction used for val split (informational only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Dataset Verification — i002_bioclip25_cap_image")
    logger.info(f"  CSV                  : {args.metadata_csv}")
    logger.info(f"  max_images_per_species: {args.max_images_per_species}")
    logger.info(f"  max_train_rows        : {args.max_train_rows}")
    logger.info(f"  cap_seed              : {args.cap_seed}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    df = load_metadata_csv(args.metadata_csv)

    # ------------------------------------------------------------------
    # Column checks
    # ------------------------------------------------------------------
    logger.info("\n--- Column checks ---")
    required = ["species_id"]
    for col in required:
        if col not in df.columns:
            logger.error(f"  MISSING required column: {col}")
            sys.exit(1)
        else:
            logger.info(f"  OK  {col}: {df[col].nunique():,} unique values")

    optional = ["genus", "family", "order", "class", "image_path", "image_name", "sample_weight"]
    for col in optional:
        if col in df.columns:
            n_valid = (
                df[col].notna() & (df[col].astype(str).str.strip() != "")
            ).sum()
            pct = 100.0 * n_valid / max(len(df), 1)
            logger.info(f"  OK  {col}: {n_valid:,}/{len(df):,} non-empty ({pct:.1f}%)")
        else:
            logger.info(f"  --  {col}: not present")

    # ------------------------------------------------------------------
    # Before-cap distribution
    # ------------------------------------------------------------------
    logger.info("")
    print_species_distribution(df, label="Full dataset BEFORE cap")

    # ------------------------------------------------------------------
    # Apply val split estimate (informational)
    # ------------------------------------------------------------------
    n_est_val   = int(len(df) * args.val_fraction)
    n_est_train = len(df) - n_est_val
    logger.info(
        f"\nEstimated split (val_fraction={args.val_fraction}): "
        f"~{n_est_train:,} train / ~{n_est_val:,} val"
    )

    # ------------------------------------------------------------------
    # Apply capping (mirrors training logic on full dataset for demo)
    # ------------------------------------------------------------------
    df_capped = df.copy()
    if args.max_images_per_species > 0:
        n_before  = len(df_capped)
        df_capped = apply_max_images_per_species_cap(
            df_capped,
            max_per_species=args.max_images_per_species,
            seed=args.cap_seed,
        )
        logger.info(
            f"\nPer-species cap ({args.max_images_per_species}): "
            f"{n_before:,} → {len(df_capped):,} rows"
        )

    if args.max_train_rows > 0 and len(df_capped) > args.max_train_rows:
        n_before  = len(df_capped)
        df_capped = apply_max_train_rows_cap(
            df_capped,
            max_rows=args.max_train_rows,
            seed=args.cap_seed,
        )
        logger.info(
            f"Total-rows cap ({args.max_train_rows}): "
            f"{n_before:,} → {len(df_capped):,} rows"
        )

    if args.max_images_per_species > 0 or args.max_train_rows > 0:
        logger.info("")
        print_species_distribution(df_capped, label="Dataset AFTER cap")

    # ------------------------------------------------------------------
    # File existence check
    # ------------------------------------------------------------------
    n_check = min(args.sample_check, len(df_capped))
    if n_check > 0 and "image_path" in df_capped.columns:
        logger.info(f"\n--- File existence check (n={n_check}) ---")
        sample = df_capped.sample(n=n_check, random_state=args.cap_seed)
        n_found   = 0
        n_missing = 0
        missing_examples: list[str] = []
        for _, row in sample.iterrows():
            p = str(row["image_path"]).strip()
            if not p:
                n_missing += 1
                missing_examples.append("(empty path)")
                continue
            if Path(p).exists():
                n_found += 1
            else:
                n_missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(p)

        pct_found = 100.0 * n_found / max(n_check, 1)
        logger.info(f"  Found   : {n_found:,}/{n_check:,} ({pct_found:.1f}%)")
        logger.info(f"  Missing : {n_missing:,}/{n_check:,}")
        if missing_examples:
            logger.info("  Missing examples (up to 5):")
            for ex in missing_examples:
                logger.info(f"    {ex}")
        if n_missing > 0:
            logger.warning(
                f"  {n_missing} missing files in sample — "
                "verify image_path values and disk mount."
            )
        else:
            logger.info("  All sampled files found on disk.")
    elif n_check > 0:
        logger.info(
            "\n--- File existence check skipped (no image_path column) ---"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n--- Summary ---")
    logger.info(f"  Total rows (original) : {len(df):,}")
    logger.info(f"  Total rows (after cap): {len(df_capped):,}")
    logger.info(f"  Species (original)    : {df['species_id'].nunique():,}")
    logger.info(f"  Species (after cap)   : {df_capped['species_id'].nunique():,}")
    logger.info(f"  Has genus column      : {'genus' in df_capped.columns}")
    logger.info(f"  Has family column     : {'family' in df_capped.columns}")
    logger.info(f"  Has sample_weight     : {'sample_weight' in df_capped.columns}")
    logger.info("Verification complete.")


if __name__ == "__main__":
    main()
