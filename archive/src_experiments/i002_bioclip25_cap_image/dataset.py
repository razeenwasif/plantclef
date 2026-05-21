"""
Dataset + taxonomy handling for BioCLIP 2.5 multi-task fine-tuning.

Supports two CSV formats
------------------------
New (default): comma-delimited, contains image_path, genus, family already.
  Example: metadata_filled_genus_family.csv
  No separate taxonomy CSV needed.

Old (backward-compat): semicolon-delimited, no image_path, requires a
  separate taxonomy CSV to be merged in.

Image path resolution
---------------------
1. If the DataFrame has an ``image_path`` column with a non-empty value,
   use it directly as ``resolved_path``.
2. Otherwise build ``{train_image_root}/{species_id}/{image_name}``.

Taxonomy
--------
genus and family come from the CSV itself (new format) or from the merged
taxonomy CSV (old format).  order and class are optional and may be absent;
they are encoded as -1 and masked during loss computation.
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

from data.metadata_utils import (
    load_metadata_csv,
    _pick_col,
    _sniff_delimiter,
    _GENUS_CANDIDATES,
    _FAMILY_CANDIDATES,
    _ORDER_CANDIDATES,
    _CLASS_CANDIDATES,
    _SPECIES_ID_CANDIDATES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_META_CSV = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "i001_data_download/data/training_usage/"
    "metadata_filled_genus_family.csv"
)
# taxonomy_csv is optional for the new format; kept for backward compat
DEFAULT_TAXONOMY_CSV: Optional[str] = None
DEFAULT_TRAIN_IMAGE_ROOT = "/workspace/plantclef/raw/train/images_max_side_800"


# ---------------------------------------------------------------------------
# Metadata loading + (optional) taxonomy merge
# ---------------------------------------------------------------------------

def load_metadata(
    meta_csv: str,
    taxonomy_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load training metadata.  Optionally merge with a separate taxonomy CSV.

    When taxonomy_csv is None (default for the new format) the function
    uses taxonomy columns already present in meta_csv (genus, family, etc.).

    When taxonomy_csv is provided (old format), it merges that CSV to
    fill in genus / family / order / class.

    Returns a DataFrame guaranteed to have:
        species_id (str), genus, family, order, class
    Missing taxonomy values are NaN — they are encoded as -1 downstream.
    """
    df = load_metadata_csv(meta_csv)

    if taxonomy_csv is not None:
        df = _merge_taxonomy(df, taxonomy_csv)
    else:
        _resolve_taxonomy_in_place(df)

    # Print coverage
    logger.info("Taxonomy coverage after load:")
    for level in ["genus", "family", "order", "class"]:
        col = df.get(level)
        if col is None:
            logger.info(f"  {level:8s}: not present")
            continue
        valid = (col.notna() & (col.astype(str).str.strip() != "")).sum()
        pct   = 100.0 * valid / max(len(df), 1)
        logger.info(
            f"  {level:8s}: {valid:,}/{len(df):,} ({pct:.1f}%)  "
            f"unique={col.nunique():,}"
        )

    return df


def _resolve_taxonomy_in_place(df: pd.DataFrame) -> None:
    """
    Normalise taxonomy column names in df to: genus, family, order, class.
    Only renames if the canonical name is absent but a candidate exists.
    Adds a None column if no candidate found.
    """
    for candidates, canonical in [
        (_GENUS_CANDIDATES,  "genus"),
        (_FAMILY_CANDIDATES, "family"),
        (_ORDER_CANDIDATES,  "order"),
        (_CLASS_CANDIDATES,  "class"),
    ]:
        if canonical in df.columns:
            continue
        alt = _pick_col(df, candidates)
        if alt:
            df.rename(columns={alt: canonical}, inplace=True)
        else:
            df[canonical] = None


def _merge_taxonomy(df: pd.DataFrame, taxonomy_csv: str) -> pd.DataFrame:
    """Merge a separate taxonomy lookup CSV (old format backward compat)."""
    logger.info(f"Loading taxonomy CSV: {taxonomy_csv}")

    delim = _sniff_delimiter(taxonomy_csv)
    tax   = pd.read_csv(taxonomy_csv, sep=delim, dtype=str, low_memory=False)
    tax.columns = [c.strip('"').strip() for c in tax.columns]

    sid_col = _pick_col(tax, _SPECIES_ID_CANDIDATES)
    if sid_col is None:
        raise ValueError(
            f"No species_id column in taxonomy CSV. Columns: {list(tax.columns)}"
        )
    if sid_col != "species_id":
        tax = tax.rename(columns={sid_col: "species_id"})
    tax["species_id"] = (
        tax["species_id"].astype(str).str.strip().str.split(".").str[0]
    )
    logger.info(
        f"  Taxonomy: {len(tax):,} rows, {tax['species_id'].nunique():,} species"
    )

    # Build slim taxonomy table with non-clashing column names
    rename_map: dict[str, str] = {}
    tax_cols = ["species_id"]
    for candidates, dst in [
        (_GENUS_CANDIDATES,  "tax_genus"),
        (_FAMILY_CANDIDATES, "tax_family"),
        (_ORDER_CANDIDATES,  "tax_order"),
        (_CLASS_CANDIDATES,  "tax_class"),
    ]:
        src = _pick_col(tax, candidates)
        if src and src not in tax_cols:
            tax_cols.append(src)
            rename_map[src] = dst

    tax_slim = (
        tax[tax_cols].rename(columns=rename_map).drop_duplicates("species_id")
    )

    n_before = len(df)
    df = df.merge(tax_slim, on="species_id", how="left")
    assert len(df) == n_before, (
        f"Merge changed row count {n_before} → {len(df)}"
    )

    # Resolve final canonical columns, preferring meta then taxonomy
    def _resolve(meta_col: Optional[str], tax_col: str, out_col: str) -> None:
        if meta_col and meta_col in df.columns and meta_col != out_col:
            valid = df[meta_col].notna() & (df[meta_col].astype(str).str.strip() != "")
            df[out_col] = df[meta_col].where(valid, df.get(tax_col))
        elif meta_col and meta_col == out_col:
            valid = df[out_col].notna() & (df[out_col].astype(str).str.strip() != "")
            if tax_col in df.columns:
                df[out_col] = df[out_col].where(valid, df[tax_col])
        elif tax_col in df.columns:
            df[out_col] = df[tax_col]
        else:
            df[out_col] = None

    _resolve(_pick_col(df, _GENUS_CANDIDATES),  "tax_genus",  "genus")
    _resolve(_pick_col(df, _FAMILY_CANDIDATES), "tax_family", "family")
    _resolve(None,                              "tax_order",  "order")
    _resolve(None,                              "tax_class",  "class")

    return df


# ---------------------------------------------------------------------------
# Label encoders
# ---------------------------------------------------------------------------

def build_label_encoders(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Build integer label encoders for species, genus, and family.

    Returns a flat dict with keys:
        species_to_idx, idx_to_species,
        genus_to_idx,   idx_to_genus,
        family_to_idx,  idx_to_family
    """
    col_map = {
        "species": "species_id",
        "genus":   "genus",
        "family":  "family",
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
        for name in ["species", "genus", "family"]:
            for key in [f"idx_to_{name}", f"{name}_to_idx"]:
                path = out / f"{key}.json"
                val  = encoders.get(key, {} if "to_idx" in key else [])
                with open(path, "w") as f:
                    json.dump(val, f, indent=2)
        logger.info(f"  Encoders saved to {out}")

    return encoders


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_image_paths(
    df: pd.DataFrame,
    image_root: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add a ``resolved_path`` column to df.

    Resolution order:
    1. Use ``image_path`` column directly if present and non-empty.
    2. Fall back to ``{image_root}/{species_id}/{image_name}``.

    Rows that cannot be resolved keep ``resolved_path = None``.
    """
    df = df.copy()

    has_image_path = "image_path" in df.columns
    has_fallback   = (
        image_root is not None
        and "image_name" in df.columns
        and "species_id" in df.columns
    )

    if has_image_path:
        nonempty = df["image_path"].notna() & (
            df["image_path"].astype(str).str.strip() != ""
        )
        df["resolved_path"] = df["image_path"].where(nonempty, None)

        if has_fallback:
            root    = Path(image_root)
            missing = df["resolved_path"].isna()
            if missing.any():
                df.loc[missing, "resolved_path"] = df[missing].apply(
                    lambda r: str(root / str(r["species_id"]) / str(r["image_name"])),
                    axis=1,
                )

        n_resolved = df["resolved_path"].notna().sum()
        logger.info(
            f"  Resolved {n_resolved:,}/{len(df):,} paths "
            f"(primary: image_path column)"
        )

    elif has_fallback:
        root = Path(image_root)
        if not root.exists():
            raise FileNotFoundError(f"Image root not found: {root}")
        df["resolved_path"] = df.apply(
            lambda r: str(root / str(r["species_id"]) / str(r["image_name"])),
            axis=1,
        )
        logger.info(
            f"  Resolved {len(df):,} paths "
            f"(fallback: image_root/{'{'}species_id{'}'}/{'{'}image_name{'}'})"
        )

    else:
        logger.warning(
            "Cannot resolve image paths: no 'image_path' column and no "
            "image_root + image_name available."
        )
        df["resolved_path"] = None

    return df


# ---------------------------------------------------------------------------
# Val split
# ---------------------------------------------------------------------------

def build_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split by species_id.  Species with <5 images stay in train."""
    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for _, group in df.groupby("species_id", sort=False):
        n = len(group)
        if n < 5:
            train_rows.append(group)
            continue
        n_val   = max(1, int(n * val_fraction))
        indices = list(range(n))
        rng.shuffle(indices)
        val_idx = set(indices[:n_val])
        train_rows.append(group.iloc[[i for i in range(n) if i not in val_idx]])
        val_rows.append(group.iloc[sorted(val_idx)])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df   = (
        pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    )
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
    Returns (image, species_idx, genus_idx, family_idx).
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

        logger.info(
            f"MultiTaskDataset: {len(self.df):,} images, "
            f"{self.df['species_id'].nunique():,} species"
        )

    def _encode(self, col: str, mapping: dict) -> list[int]:
        if col not in self.df.columns or not mapping:
            return [-1] * len(self.df)
        return [
            mapping.get(str(v).strip(), -1)
            if pd.notna(v) and str(v).strip()
            else -1
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
        )
