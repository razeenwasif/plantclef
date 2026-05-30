#!/usr/bin/env python3
"""
Build an i002-compatible training manifest from the PlantNet-300K v2 release.

The PlantNet-300K layout on disk is

    <root>/images/images/<split>/<species_id:04d>/<PN_hash>.jpg

with two metadata CSVs:

* `plantnet300K_metadata.csv` - one row per image, columns:
  species_id, PN_observation_id, organ, author, license, split, PN_hash.
* `species_metadata.csv` - one row per species, columns:
  species_id, full_species, species, genus, family, epithet, author,
  unmatched_terms, iucn_status.

The shared training script (`src/train.py` -> `shared/bioclip25_multitask/`)
expects a single CSV with at minimum

* `species_id` (string, normalised)
* `image_path` (absolute path to a JPEG)

and optionally `genus` / `family` for the auxiliary taxonomy heads. This
script joins the two CSVs on `species_id`, filters to a single split
(default `train`), constructs the absolute path, and writes the result
to a single file ready for `--metadata-csv`.

Usage
-----

    python tools/build_plantnet300k_manifest.py \\
        --root        /mnt/d/PlantNet-300k \\
        --split       train \\
        --output      data/plantnet300k_manifest.csv

Pass `--all-splits` to write three manifests (train / val / test) under
the same output stem.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _build_one_split(
    root: Path,
    image_meta: pd.DataFrame,
    species_meta: pd.DataFrame,
    split: str,
    output: Path,
) -> int:
    """Write a single-split manifest. Returns row count."""
    df = image_meta[image_meta["split"] == split].copy()
    if df.empty:
        print(f"  ! split={split!r}: no rows in image metadata, skipping")
        return 0

    df["species_id"] = df["species_id"].astype(int)
    species_meta = species_meta.copy()
    species_meta["species_id"] = species_meta["species_id"].astype(int)

    merged = df.merge(
        species_meta[["species_id", "species", "genus", "family"]],
        on="species_id",
        how="left",
    )

    missing_tax = (
        merged["genus"].isna() | merged["family"].isna()
    ).sum()
    if missing_tax:
        print(
            f"  ! split={split!r}: {missing_tax:,} rows missing genus/family"
            f" after merge -> will be encoded as -1 during training"
        )

    image_root = root / "images" / "images" / split
    merged["species_dir"] = merged["species_id"].apply(lambda i: f"{int(i):04d}")
    merged["image_path"] = (
        image_root.as_posix()
        + "/"
        + merged["species_dir"]
        + "/"
        + merged["PN_hash"].astype(str)
        + ".jpg"
    )

    out = merged[["species_id", "image_path", "genus", "family", "species"]].copy()
    out.rename(columns={"species": "species_name"}, inplace=True)

    out["species_id"] = out["species_id"].astype(str)

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print(
        f"  wrote split={split!r}: {len(out):,} rows, "
        f"{out['species_id'].nunique():,} species -> {output}"
    )
    return len(out)


def _spot_check(output: Path, n: int) -> None:
    """Verify that the first N image_paths actually resolve on disk."""
    df = pd.read_csv(output, usecols=["image_path"], nrows=n)
    missing = [p for p in df["image_path"] if not Path(p).is_file()]
    print(
        f"  spot-check: first {len(df)} paths -> "
        f"{len(df) - len(missing):,} present, {len(missing):,} missing"
    )
    for p in missing[:3]:
        print(f"     missing: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Adapter: PlantNet-300K v2 -> i002 training manifest.",
    )
    ap.add_argument(
        "--root", type=Path, default=Path("/mnt/d/PlantNet-300k"),
        help="PlantNet-300K v2 unpacked root (contains images/, plantnet300K_metadata.csv).",
    )
    ap.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Which split to write. Ignored when --all-splits is set.",
    )
    ap.add_argument(
        "--all-splits", action="store_true",
        help="Write three manifests (train/val/test) under the same output stem.",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("data/plantnet300k_manifest.csv"),
        help="Output CSV path (or stem when --all-splits).",
    )
    ap.add_argument(
        "--spot-check-n", type=int, default=20,
        help="Verify the first N image_path entries actually exist on disk.",
    )
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"--root not a directory: {args.root}")

    image_csv = args.root / "plantnet300K_metadata.csv"
    species_csv = args.root / "species_metadata.csv"
    for p in (image_csv, species_csv):
        if not p.is_file():
            sys.exit(f"required CSV not found: {p}")

    print(f"[*] loading {image_csv.name}")
    image_meta = pd.read_csv(image_csv)
    print(f"[*] loading {species_csv.name}")
    species_meta = pd.read_csv(species_csv)

    print(
        f"[*] image metadata: {len(image_meta):,} rows, "
        f"{image_meta['species_id'].nunique():,} species, "
        f"splits = {sorted(image_meta['split'].unique())}"
    )
    print(
        f"[*] species metadata: {len(species_meta):,} rows, "
        f"{species_meta['family'].nunique():,} families, "
        f"{species_meta['genus'].nunique():,} genera"
    )

    splits = ["train", "val", "test"] if args.all_splits else [args.split]
    output_stem = args.output

    for split in splits:
        if args.all_splits:
            out = output_stem.with_name(
                output_stem.stem + f"_{split}" + output_stem.suffix
            )
        else:
            out = output_stem
        _build_one_split(args.root, image_meta, species_meta, split, out)
        if args.spot_check_n > 0:
            _spot_check(out, args.spot_check_n)

    print("[+] done")


if __name__ == "__main__":
    main()
