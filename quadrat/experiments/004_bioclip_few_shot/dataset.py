"""
Training metadata loading and image path resolution.

Image layout on disk
--------------------
  {train_image_root}/{species_id}/{image_name}

e.g.
  /workspace/plantclef/raw/train/images_max_side_800/1396710/59feabe1...jpg

This matches the actual on-disk layout confirmed during development.
The CSV column `image_name` contains only the filename (no directory prefix).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (all overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_META_CSV = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "002_bioclip_tile_zero_shot_v2/data/"
    "PlantCLEF2024_single_plant_training_metadata.csv"
)
DEFAULT_TRAIN_IMAGE_ROOT = "/workspace/plantclef/raw/train/images_max_side_800"
DEFAULT_SPECIES_CSV = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "002_bioclip_tile_zero_shot_v2/data/"
    "species_lookup_with_gbif_cleaned_names.csv"
)


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_train_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load the PlantCLEF single-plant training metadata CSV.

    The file is semicolon-delimited with quoted fields.

    Expected columns (among others):
        image_name, organ, species_id, obs_id, species, genus, family

    Returns
    -------
    DataFrame with string-typed columns, stripped of surrounding quotes.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training metadata CSV not found: {path}")

    logger.info(f"Loading training metadata from {path} ...")
    df = pd.read_csv(path, sep=";", dtype=str, low_memory=False)

    # Strip any residual quote wrapping from column names and values
    df.columns = [c.strip('"').strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip('"').str.strip()

    required = {"image_name", "species_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Training metadata is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    logger.info(f"  Loaded {len(df):,} rows, {df['species_id'].nunique():,} species")
    return df


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def build_image_path(image_root: str, species_id: str, image_name: str) -> Path:
    """Construct the canonical image path: root / species_id / image_name."""
    return Path(image_root) / str(species_id) / str(image_name)


def resolve_image_paths(
    df: pd.DataFrame,
    image_root: str,
    verify: bool = False,
    log_every: int = 50000,
) -> pd.DataFrame:
    """
    Add a ``resolved_path`` column to the DataFrame.

    Primary strategy: ``{image_root}/{species_id}/{image_name}``

    If a file is not found at the primary path, a warning is logged and
    the entry is marked None (will be skipped during bank building).

    Parameters
    ----------
    df         : training metadata DataFrame (must have image_name, species_id)
    image_root : root directory of training images
    verify     : if True, check that each resolved path actually exists

    Returns
    -------
    DataFrame with a new 'resolved_path' column (str or None).
    """
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"Train image root not found: {root}")

    paths: list[Optional[str]] = []
    n_missing = 0

    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        if i % log_every == 0:
            logger.info(
                "Processed %s/%s rows (%.2f%%) | missing so far: %s",
                f"{i:,}",
                f"{len(df):,}",
                100.0 * i / len(df),
                f"{n_missing:,}",
            )

        candidate = root / str(row["species_id"]) / str(row["image_name"])
        if not verify or candidate.exists():
            paths.append(str(candidate))
        else:
            paths.append(None)
            n_missing += 1

    df = df.copy()
    df["resolved_path"] = paths

    n_found = len(df) - n_missing
    logger.info(
        f"Path resolution: {n_found:,}/{len(df):,} images found "
        f"({n_missing:,} missing)"
    )
    if n_missing > 0:
        logger.warning(
            f"{n_missing:,} images not found under {root}. "
            "These rows will be skipped."
        )
    return df


# ---------------------------------------------------------------------------
# Support sampling
# ---------------------------------------------------------------------------

def sample_support_images(
    df: pd.DataFrame,
    k: int,
    mode: str = "random",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample up to K support images per species from the training metadata.

    Sampling modes
    --------------
    random        : random sample (reproducible via seed)
    capped_all    : use all images if species has fewer than K; else random K
    top_n_per_species : prefer images with quality metadata if available
                       (currently falls back to random - extend as needed)

    Parameters
    ----------
    df   : training metadata with a 'resolved_path' column (None rows skipped)
    k    : max support images per species
    mode : "random" | "capped_all" | "top_n_per_species"
    seed : random seed for reproducibility

    Returns
    -------
    DataFrame of sampled rows (one row per selected support image).
    """
    # Drop rows with no resolved image path
    valid = df[df["resolved_path"].notna()].copy()
    if len(valid) < len(df):
        logger.info(f"  Dropped {len(df)-len(valid):,} rows with missing images")

    rng = random.Random(seed)
    selected_rows: list[pd.DataFrame] = []

    for species_id, group in valid.groupby("species_id", sort=False):
        n_available = len(group)
        if mode == "top_n_per_species":
            # Placeholder: use 'organ' diversity as a simple quality proxy,
            # preferring images with diverse organs before falling back to random.
            if "organ" in group.columns:
                organ_order = ["flower", "fruit", "leaf", "stem", "bark", "root", "habit", "other"]
                organ_map = {o: i for i, o in enumerate(organ_order)}
                group = group.copy()
                group["_organ_rank"] = group["organ"].map(organ_map).fillna(len(organ_order))
                group = group.sort_values("_organ_rank").drop(columns=["_organ_rank"])
            samples = group.head(k)
        else:
            # random / capped_all: same behaviour (both cap at K)
            if n_available <= k:
                samples = group
            else:
                indices = rng.sample(range(n_available), k)
                samples = group.iloc[sorted(indices)]

        selected_rows.append(samples)

    result = pd.concat(selected_rows, ignore_index=True)
    logger.info(
        f"Support sampling: {len(result):,} images selected "
        f"across {result['species_id'].nunique():,} species "
        f"(K={k}, mode={mode}, seed={seed})"
    )
    return result


# ---------------------------------------------------------------------------
# Species lookup
# ---------------------------------------------------------------------------

def load_species_ids(species_csv: str) -> list[str]:
    """
    Load the list of competition species IDs from the enriched species CSV.

    Returns a sorted list of string species IDs.
    """
    path = Path(species_csv)
    if not path.exists():
        raise FileNotFoundError(f"Species CSV not found: {path}")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "species_id" not in df.columns:
        raise ValueError(
            f"Species CSV does not have a 'species_id' column. "
            f"Found: {list(df.columns)}"
        )
    ids = df["species_id"].str.strip().dropna().unique().tolist()
    logger.info(f"Loaded {len(ids):,} species IDs from {path.name}")
    return sorted(ids)


def load_species_name_map(species_csv: str) -> dict[str, str]:
    """
    Return a dict mapping species_id -> canonical scientific name.

    Uses the cleaned name columns if present, otherwise falls back to 'species'.
    """
    path = Path(species_csv)
    if not path.exists():
        raise FileNotFoundError(f"Species CSV not found: {path}")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Pick best available name column
    name_col = None
    for col in ["species_name_clean", "gbif_canonical_name", "species"]:
        if col in df.columns:
            name_col = col
            break
    if name_col is None:
        raise ValueError(f"No usable name column found in {path.name}")

    return {
        str(row["species_id"]).strip(): str(row[name_col]).strip()
        for _, row in df.iterrows()
        if pd.notna(row.get("species_id"))
    }
