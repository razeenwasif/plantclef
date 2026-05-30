"""
015 — Build a combined PC24 + iNat training manifest in 010's expected format.

010's dataset.py expects:
    - semicolon-separated CSV
    - required columns: image_name, species_id
    - optional: genus, family (used during taxonomy merge)
    - resolves images via {train_image_root}/{species_id}/{image_name}

We extend that contract with one new optional column:
    - image_path (absolute) — if present and non-empty, dataset.py uses it
      verbatim instead of the {root}/{sp}/{name} pattern.

PC24 rows:  copied from the existing PlantCLEF2024 metadata CSV; image_path empty.
iNat rows:  synthesized from /workspace/plantclef/processed/inat_research_grade_manifest.csv,
            with image_path set to the absolute path on disk.

Output: a single semicolon-separated CSV that 010/train.py can consume directly
via --train-meta-csv. Mix ratio is the natural concat (~53% PC24, ~47% iNat).
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd


logger = logging.getLogger("build_combined_manifest")


PC24_DEFAULT = "/workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/PlantCLEF2024_single_plant_training_metadata.csv"
INAT_DEFAULT = "/workspace/plantclef/processed/inat_research_grade_manifest.csv"
OUT_DEFAULT  = "/workspace/plantclef/processed/pc24_inat_combined_manifest.csv"
PC24_ROOT_DEFAULT = "/workspace/plantclef/raw/train/images_max_side_800"
INAT_ROOT_DEFAULT = "/workspace/plantclef/raw/inat_research_grade"


def load_pc24(path: str) -> pd.DataFrame:
    logger.info(f"Loading PC24 metadata: {path}")
    df = pd.read_csv(path, sep=";", dtype=str, low_memory=False)
    df.columns = [c.strip('"').strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip('"').str.strip()
    df["species_id"] = df["species_id"].astype(str).str.strip()
    df["source"] = "pc24"
    logger.info(f"  PC24: {len(df):,} rows, {df['species_id'].nunique():,} species")
    return df


def load_inat(path: str, pc24_root: str, inat_root: str) -> pd.DataFrame:
    """
    Encode iNat rows so that 010's untouched path resolver
    f"{pc24_root}/{species_id}/{image_name}" resolves to the actual iNat file.

    Computes the relative path from {pc24_root}/<sp>/ down to <inat_root>/<sp>/<file>,
    via their common ancestor. e.g. with
        pc24_root = /workspace/plantclef/raw/train/images_max_side_800
        inat_root = /workspace/plantclef/raw/inat_research_grade
    the relative encoding is `../../../inat_research_grade/<sp>/<file>`.

    All vectorized — no per-row Path() calls.
    """
    import os.path

    logger.info(f"Loading iNat manifest: {path}")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df["species_id"] = df["species_id"].astype(str).str.strip()

    common = Path(os.path.commonpath([pc24_root, inat_root]))
    pc24_extra = Path(pc24_root).relative_to(common).parts
    inat_extra = Path(inat_root).relative_to(common).parts
    # anchor is pc24_root/<sp>, so up from anchor to common = len(pc24_extra) + 1
    ups = "../" * (len(pc24_extra) + 1)
    down_prefix = ("/".join(inat_extra) + "/") if inat_extra else ""
    logger.info(f"  Relative encoding prefix: {ups}{down_prefix}<sp>/<file>")

    basenames = df["image_path"].str.rsplit("/", n=1).str[-1]
    df["image_name"] = ups + down_prefix + df["species_id"] + "/" + basenames
    df["source"] = "inat"
    keep = ["image_name", "species_id", "source"]
    if "gbif_species_id" in df.columns:
        keep.append("gbif_species_id")
    if "license" in df.columns:
        keep.append("license")
    if "scientific_name" in df.columns:
        df["species"] = df["scientific_name"]
        keep.append("species")
    df = df[keep]
    logger.info(f"  iNat: {len(df):,} rows, {df['species_id'].nunique():,} species")
    sample = df["image_name"].iloc[0] if len(df) else ""
    logger.info(f"  iNat first image_name (relative-to-pc24-sp-dir): {sample}")
    return df


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc24", default=PC24_DEFAULT)
    ap.add_argument("--inat", default=INAT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--pc24-root", default=PC24_ROOT_DEFAULT)
    ap.add_argument("--inat-root", default=INAT_ROOT_DEFAULT)
    args = ap.parse_args()

    pc24 = load_pc24(args.pc24)
    inat = load_inat(args.inat, args.pc24_root, args.inat_root)

    common = sorted(set(pc24.columns) & set(inat.columns))
    logger.info(f"Shared columns: {common}")

    union = sorted(set(pc24.columns) | set(inat.columns))
    for col in union:
        if col not in pc24.columns:
            pc24[col] = ""
        if col not in inat.columns:
            inat[col] = ""

    combined = pd.concat([pc24[union], inat[union]], ignore_index=True)
    logger.info(
        f"Combined: {len(combined):,} rows  ("
        f"pc24={len(pc24):,} + inat={len(inat):,})  "
        f"species={combined['species_id'].nunique():,}"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing: {out_path}")
    combined.to_csv(out_path, sep=";", index=False, quoting=csv.QUOTE_ALL)

    by_source = combined["source"].value_counts().to_dict()
    sp_overlap = (
        combined.groupby("source")["species_id"].nunique().to_dict()
    )
    logger.info(f"By source: {by_source}")
    logger.info(f"Species per source: {sp_overlap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
