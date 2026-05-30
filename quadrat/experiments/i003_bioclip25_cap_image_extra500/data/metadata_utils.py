"""
Shared helpers for loading, normalising, and rebalancing the PlantCLEF
training metadata used by the i002 (uncapped) and i003 (per-species cap)
experiments.

The public surface is small and deliberately stable, because
`dataset.py`, `train.py`, and the verification scripts in both
experiments import from it directly:

    Column-candidate constants
        _SPECIES_ID_CANDIDATES
        _GENUS_CANDIDATES
        _FAMILY_CANDIDATES
        _ORDER_CANDIDATES
        _CLASS_CANDIDATES

    CSV helpers
        _sniff_delimiter(path)
        _pick_col(df, candidates)
        load_metadata_csv(path)

    Capping + balancing helpers
        apply_max_images_per_species_cap(df, max_per_species, seed)
        apply_max_train_rows_cap(df, max_rows, seed)
        print_species_distribution(df, label)
        build_weighted_sampler(df)

Both manifest formats are supported (see dataset.py for the full
description): the new comma-delimited format that already carries
`image_path`, `genus`, and `family`, and the older semicolon-delimited
PlantCLEF 2024 export that needs a separate taxonomy merge.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name candidates (first match wins). Lower-case is intentional - the
# match in _pick_col is case-insensitive so it tolerates the few "Genus" /
# "Family" capitalisations that slip in via GBIF exports.
# ---------------------------------------------------------------------------

_SPECIES_ID_CANDIDATES: tuple[str, ...] = (
    "species_id",
    "speciesid",
    "species",
    "species_key",
    "speciesKey",
    "gbif_species_id",
)
_GENUS_CANDIDATES: tuple[str, ...] = (
    "genus",
    "Genus",
    "genus_name",
    "genus_id",
)
_FAMILY_CANDIDATES: tuple[str, ...] = (
    "family",
    "Family",
    "family_name",
    "family_id",
)
_ORDER_CANDIDATES: tuple[str, ...] = (
    "order",
    "Order",
    "order_name",
    "order_id",
)
_CLASS_CANDIDATES: tuple[str, ...] = (
    "class",
    "Class",
    "class_name",
    "class_id",
)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _sniff_delimiter(path: str) -> str:
    """
    Detect whether `path` uses comma or semicolon delimiters.

    The new manifest emitted by i001 is comma-delimited; the original
    PlantCLEF 2024 metadata.csv is semicolon-delimited. We try Python's
    csv.Sniffer first, then fall back to counting separators on the
    header line.
    """
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        if dialect.delimiter in (",", ";", "\t"):
            return dialect.delimiter
    except csv.Error:
        pass
    header = sample.splitlines()[0] if sample else ""
    return ";" if header.count(";") > header.count(",") else ","


def _pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """
    Return the first column from `candidates` present in `df` (case
    insensitive). Returns None if none match.
    """
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        hit = lookup.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _normalise_species_id_column(df: pd.DataFrame) -> None:
    """
    Coerce species_id to a clean string in place.

    The PlantCLEF metadata sometimes parses species_id as float (e.g.
    `1738410.0`) when pandas guesses dtype. Strip whitespace, drop a
    trailing `.0`, and leave the result as a Python string so downstream
    string-keyed lookups (encoders, value_counts) behave deterministically.
    """
    if "species_id" not in df.columns:
        return
    df["species_id"] = (
        df["species_id"].astype(str).str.strip().str.split(".").str[0]
    )


def load_metadata_csv(path: str) -> pd.DataFrame:
    """
    Read a training-metadata CSV from `path`, auto-detecting the
    delimiter and normalising the species_id column.

    Renames the first matching species_id candidate column to the
    canonical `species_id` if necessary, so downstream code can rely on
    that exact column name.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"metadata CSV not found: {p}")

    delim = _sniff_delimiter(str(p))
    df = pd.read_csv(p, sep=delim, dtype=str, low_memory=False)
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]

    sid_col = _pick_col(df, _SPECIES_ID_CANDIDATES)
    if sid_col is None:
        raise ValueError(
            f"metadata CSV at {p} has no species_id column; "
            f"saw columns {list(df.columns)}"
        )
    if sid_col != "species_id":
        df = df.rename(columns={sid_col: "species_id"})

    _normalise_species_id_column(df)
    logger.info(
        f"Loaded metadata CSV {p.name}: {len(df):,} rows, "
        f"{df['species_id'].nunique():,} species, delimiter={delim!r}"
    )
    return df


# ---------------------------------------------------------------------------
# Capping
# ---------------------------------------------------------------------------

def apply_max_images_per_species_cap(
    df: pd.DataFrame,
    max_per_species: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Trim each species to at most `max_per_species` rows by deterministic
    random subsampling, then return a new DataFrame with the original
    index reset.

    Species already below the cap are left untouched. The shuffle is
    seeded so the same input + seed produces the same output across
    machines.
    """
    if max_per_species <= 0:
        return df.reset_index(drop=True)

    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).index
    df = df.iloc[rng].reset_index(drop=True)

    capped = (
        df.groupby("species_id", sort=False, group_keys=False)
          .head(max_per_species)
          .reset_index(drop=True)
    )
    return capped


def apply_max_train_rows_cap(
    df: pd.DataFrame,
    max_rows: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Reduce the manifest to at most `max_rows` rows by uniform random
    sampling without replacement. Returns the original DataFrame if it
    is already at or below the cap.
    """
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Distribution logging
# ---------------------------------------------------------------------------

# Bin edges chosen to match the histogram quoted in
# docs/exp_reports/i002_bioclip25_cap_image_report.md.
_BIN_EDGES: tuple[tuple[int, int, str], ...] = (
    (1,     1,    "       1"),
    (2,     2,    "       2"),
    (3,     5,    "     3-5"),
    (6,     10,   "    6-10"),
    (11,    20,   "   11-20"),
    (21,    50,   "   21-50"),
    (51,    100,  "  51-100"),
    (101,   250,  " 101-250"),
    (251,   500,  " 251-500"),
    (501,   1000, "501-1000"),
    (1001,  None, "   >1000"),
)


def print_species_distribution(df: pd.DataFrame, label: str = "Distribution") -> None:
    """
    Log per-species image-count summary statistics (median, mean, max)
    and a bin histogram matching the format used in the project
    experiment reports. Always goes through the module logger at INFO.
    """
    counts = df.groupby("species_id").size()
    if counts.empty:
        logger.info(f"{label}: empty manifest")
        return

    logger.info(f"{label}: {len(df):,} rows  /  {counts.size:,} species")
    logger.info(f"  Median per species : {int(counts.median())}")
    logger.info(f"  Mean per species   : {int(round(counts.mean()))}")
    logger.info(f"  Max  per species   : {int(counts.max())}")
    logger.info("  Bin counts (species per image-count bin):")
    for lo, hi, label_str in _BIN_EDGES:
        if hi is None:
            n = int((counts >= lo).sum())
        else:
            n = int(((counts >= lo) & (counts <= hi)).sum())
        logger.info(f"    [{label_str}]: {n:>5,} species")


# ---------------------------------------------------------------------------
# Weighted sampler for long-tail rebalancing
# ---------------------------------------------------------------------------

def build_weighted_sampler(df: pd.DataFrame):
    """
    Build a `torch.utils.data.WeightedRandomSampler` from the
    `sample_weight` column. The sampler draws one row per call and is
    sized to one full pass over `df`, so an epoch sees ~len(df) draws.

    Raises a ValueError if the column is missing; callers should check
    `'sample_weight' in df.columns` before calling.
    """
    from torch.utils.data import WeightedRandomSampler

    if "sample_weight" not in df.columns:
        raise ValueError(
            "build_weighted_sampler: 'sample_weight' column missing from "
            "metadata. Either supply per-row weights or fall back to "
            "shuffle=True."
        )
    weights = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(0.0)
    weights = weights.clip(lower=0.0).tolist()
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(df),
        replacement=True,
    )
