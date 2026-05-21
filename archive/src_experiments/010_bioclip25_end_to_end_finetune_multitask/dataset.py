"""
Dataset + taxonomy merge for BioCLIP 2.5 multi-task fine-tuning.

Loads training metadata, merges with taxonomy lookup, builds label encoders
for species / genus / family / order / class.  Missing taxonomy labels are
encoded as -1 and masked during loss computation (never crash on missing data).

Image layout: {train_image_root}/{species_id}/{image_name}
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_META_CSV = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "002_bioclip_tile_zero_shot_v2/data/"
    "PlantCLEF2024_single_plant_training_metadata.csv"
)
DEFAULT_TAXONOMY_CSV = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "002_bioclip_tile_zero_shot_v2/data/"
    "species_lookup_with_gbif_cleaned_names.csv"
)
DEFAULT_TRAIN_IMAGE_ROOT = "/workspace/plantclef/raw/train/images_max_side_800"

# ---------------------------------------------------------------------------
# Column name candidates (checked in order; first match wins)
# ---------------------------------------------------------------------------

_SPECIES_ID_CANDIDATES = ["species_id", "taxon_id", "gbif_species_id"]
_GENUS_CANDIDATES      = ["genus", "genus_clean", "gbif_genus"]
_FAMILY_CANDIDATES     = ["family", "family_clean", "gbif_family"]
_ORDER_CANDIDATES      = ["gbif_order", "order_clean", "order"]
_CLASS_CANDIDATES      = ["gbif_class", "class_clean", "class"]


def _pick_col(df: pd.DataFrame, candidates: list[str], label: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    logger.warning(
        f"No {label} column found. Tried: {candidates}. "
        f"Available (first 15): {list(df.columns[:15])}"
    )
    return None


# ---------------------------------------------------------------------------
# Metadata loading + taxonomy merge
# ---------------------------------------------------------------------------

def load_metadata(meta_csv: str, taxonomy_csv: str) -> pd.DataFrame:
    """
    Load training metadata and merge with taxonomy lookup.

    Returns a DataFrame with at minimum:
        image_name, species_id, genus, family, order, class
    Missing taxonomy values are NaN.
    """
    # -- Main metadata CSV (semicolon-delimited, quoted fields) --
    logger.info(f"Loading metadata: {meta_csv}")
    meta = pd.read_csv(meta_csv, sep=";", dtype=str, low_memory=False)
    meta.columns = [c.strip('"').strip() for c in meta.columns]
    for col in meta.columns:
        if meta[col].dtype == object:
            meta[col] = meta[col].str.strip('"').str.strip()

    meta_sid = _pick_col(meta, _SPECIES_ID_CANDIDATES, "species_id (meta)")
    if meta_sid is None:
        raise ValueError(f"No species_id column in metadata. Columns: {list(meta.columns)}")
    if meta_sid != "species_id":
        meta = meta.rename(columns={meta_sid: "species_id"})
    meta["species_id"] = meta["species_id"].astype(str).str.strip()

    logger.info(
        f"  Main CSV: {len(meta):,} rows, "
        f"{meta['species_id'].nunique():,} unique species"
    )

    # -- Taxonomy lookup CSV (comma-delimited) --
    logger.info(f"Loading taxonomy: {taxonomy_csv}")
    tax = pd.read_csv(taxonomy_csv, dtype=str, low_memory=False)
    tax.columns = [c.strip() for c in tax.columns]

    tax_sid = _pick_col(tax, _SPECIES_ID_CANDIDATES, "species_id (taxonomy)")
    if tax_sid is None:
        raise ValueError(f"No species_id column in taxonomy CSV. Columns: {list(tax.columns)}")
    if tax_sid != "species_id":
        tax = tax.rename(columns={tax_sid: "species_id"})
    # Handle "1548094.0" → "1548094"
    tax["species_id"] = (
        tax["species_id"].astype(str).str.strip().str.split(".").str[0]
    )
    logger.info(
        f"  Taxonomy CSV: {len(tax):,} rows, "
        f"{tax['species_id'].nunique():,} unique species"
    )

    # -- Detect taxonomy columns in taxonomy CSV --
    genus_tax  = _pick_col(tax, _GENUS_CANDIDATES,  "genus (taxonomy)")
    family_tax = _pick_col(tax, _FAMILY_CANDIDATES, "family (taxonomy)")
    order_tax  = _pick_col(tax, _ORDER_CANDIDATES,  "order (taxonomy)")
    class_tax  = _pick_col(tax, _CLASS_CANDIDATES,  "class (taxonomy)")

    # Build slim taxonomy table with non-clashing column names
    tax_cols = ["species_id"]
    rename_map: dict[str, str] = {}
    for src_col, dst_name in [
        (genus_tax,  "tax_genus"),
        (family_tax, "tax_family"),
        (order_tax,  "tax_order"),
        (class_tax,  "tax_class"),
    ]:
        if src_col is not None and src_col not in tax_cols:
            tax_cols.append(src_col)
            rename_map[src_col] = dst_name

    tax_slim = (
        tax[tax_cols]
        .rename(columns=rename_map)
        .drop_duplicates("species_id")
    )

    # -- Detect taxonomy columns already in main CSV --
    genus_meta  = _pick_col(meta, _GENUS_CANDIDATES,  "genus (meta)")
    family_meta = _pick_col(meta, _FAMILY_CANDIDATES, "family (meta)")

    # -- Left-merge (keep all main CSV rows) --
    n_before = len(meta)
    df = meta.merge(tax_slim, on="species_id", how="left")
    assert len(df) == n_before, (
        f"Merge changed row count: {n_before} → {len(df)}. "
        "Possible duplicate species_id in taxonomy CSV."
    )
    logger.info(f"  Rows before merge: {n_before:,}  after: {len(df):,}")

    # -- Resolve final taxonomy columns --
    def _resolve(meta_col: Optional[str], tax_col: str, out_col: str) -> None:
        if meta_col and meta_col in df.columns and meta_col != out_col:
            # Prefer meta, fallback to taxonomy
            valid_meta = df[meta_col].notna() & (df[meta_col].astype(str).str.strip() != "")
            df[out_col] = df[meta_col].where(valid_meta, df.get(tax_col))
        elif meta_col and meta_col == out_col:
            # Column is already named correctly; fill blanks from taxonomy
            valid_meta = df[out_col].notna() & (df[out_col].astype(str).str.strip() != "")
            if tax_col in df.columns:
                df[out_col] = df[out_col].where(valid_meta, df[tax_col])
        elif tax_col in df.columns:
            df[out_col] = df[tax_col]
        else:
            df[out_col] = None

    _resolve(genus_meta,  "tax_genus",  "genus")
    _resolve(family_meta, "tax_family", "family")
    _resolve(None,        "tax_order",  "order")
    _resolve(None,        "tax_class",  "class")

    # -- Print coverage stats --
    logger.info("Taxonomy coverage:")
    for level in ["genus", "family", "order", "class"]:
        col = df.get(level)
        if col is None:
            logger.info(f"  {level:8s}: column not present")
            continue
        n_valid  = col.notna().sum()
        n_nonblk = (col.notna() & (col.astype(str).str.strip() != "")).sum()
        pct      = 100 * n_nonblk / max(len(df), 1)
        n_unique = col.nunique()
        logger.info(
            f"  {level:8s}: {n_nonblk:,}/{len(df):,} ({pct:.1f}%)  unique={n_unique:,}"
        )

    return df


# ---------------------------------------------------------------------------
# Label encoders
# ---------------------------------------------------------------------------

def build_label_encoders(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Build integer label encoders for species + all taxonomy levels.

    Returns a flat dict with keys:
        species_to_idx, idx_to_species,
        genus_to_idx,   idx_to_genus,
        family_to_idx,  idx_to_family,
        order_to_idx,   idx_to_order,
        class_to_idx,   idx_to_class
    """
    col_map = {
        "species": "species_id",
        "genus":   "genus",
        "family":  "family",
        "order":   "order",
        "class":   "class",
    }
    encoders: dict = {}
    for name, col in col_map.items():
        if col not in df.columns:
            logger.warning(f"Column {col!r} not found — skipping {name} encoder")
            encoders[f"{name}_to_idx"] = {}
            encoders[f"idx_to_{name}"] = []
            continue
        unique = sorted(
            str(v).strip()
            for v in df[col].dropna().unique()
            if str(v).strip()
        )
        to_idx = {v: i for i, v in enumerate(unique)}
        encoders[f"{name}_to_idx"] = to_idx
        encoders[f"idx_to_{name}"] = unique
        logger.info(f"  Encoder [{name:8s}]: {len(unique):,} classes")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in ["species", "genus", "family", "order", "class"]:
            key = f"idx_to_{name}"
            path = out / f"{key}.json"
            with open(path, "w") as f:
                json.dump(encoders.get(key, []), f, indent=2)
        # Also save to_idx mappings
        for name in ["species", "genus", "family", "order", "class"]:
            key = f"{name}_to_idx"
            path = out / f"{key}.json"
            with open(path, "w") as f:
                json.dump(encoders.get(key, {}), f, indent=2)
        logger.info(f"  Encoders saved to {out}")

    return encoders


# ---------------------------------------------------------------------------
# Path resolution + val split
# ---------------------------------------------------------------------------

def resolve_image_paths(df: pd.DataFrame, image_root: str) -> pd.DataFrame:
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    df = df.copy()
    df["resolved_path"] = df.apply(
        lambda r: str(root / str(r["species_id"]) / str(r["image_name"])),
        axis=1,
    )
    return df


def build_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split by species_id. Species with <5 images stay in train."""
    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for _, group in df.groupby("species_id", sort=False):
        n = len(group)
        if n < 5:
            train_rows.append(group)
            continue
        n_val = max(1, int(n * val_fraction))
        indices = list(range(n))
        rng.shuffle(indices)
        val_idx = set(indices[:n_val])
        train_rows.append(group.iloc[[i for i in range(n) if i not in val_idx]])
        val_rows.append(group.iloc[sorted(val_idx)])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    logger.info(
        f"Split: train={len(train_df):,} "
        f"({train_df['species_id'].nunique():,} sp)  "
        f"val={len(val_df):,} "
        f"({val_df['species_id'].nunique():,} sp)"
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MultiTaskDataset(Dataset):
    """
    Returns (image, species_idx, genus_idx, family_idx, order_idx, class_idx).
    Taxonomy indices are -1 when the label is missing (masked during loss).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        encoders: dict,
        transform,
    ) -> None:
        self.df        = df.reset_index(drop=True)
        self.transform = transform

        self._sp  = self._encode("species_id", encoders.get("species_to_idx", {}))
        self._gen = self._encode("genus",       encoders.get("genus_to_idx",   {}))
        self._fam = self._encode("family",      encoders.get("family_to_idx",  {}))
        self._ord = self._encode("order",       encoders.get("order_to_idx",   {}))
        self._cls = self._encode("class",       encoders.get("class_to_idx",   {}))

        logger.info(
            f"MultiTaskDataset: {len(self.df):,} images, "
            f"{self.df['species_id'].nunique():,} species"
        )

    def _encode(self, col: str, mapping: dict) -> list[int]:
        if col not in self.df.columns or not mapping:
            return [-1] * len(self.df)
        return [
            mapping.get(str(v).strip(), -1) if pd.notna(v) and str(v).strip() else -1
            for v in self.df[col]
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["resolved_path"]).convert("RGB")
        img = self.transform(img)
        return (
            img,
            self._sp[idx],
            self._gen[idx],
            self._fam[idx],
            self._ord[idx],
            self._cls[idx],
        )
