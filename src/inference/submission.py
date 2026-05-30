"""Competition submission CSV generation.

Produces a CSV in the PlantCLEF 2026 format::

    "quadrat_id","species_ids"
    "CBN-Pla-B1-20130724","[1395806]"
    "CBN-PdlC-A1-20130807","[1351284, 1494911, 1381367]"

The ``species_ids`` column contains a comma-and-space-separated list of
*actual species IDs* (not class indices) enclosed in square brackets.

Class-index → species-ID mapping
---------------------------------
The model outputs class indices 0 … C-1.  These must be mapped to the
competition's numeric species IDs using the ``species_ids.csv`` file
(one species ID per row, no header, in the same order as the model output).

If no mapping file is available, class indices are used as species IDs.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .types import ImagePrediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Species-ID mapping
# ---------------------------------------------------------------------------

def load_species_mapping(species_csv: Optional[Path]) -> dict[int, int]:
    """Load the class-index → species-ID mapping from a CSV file.

    The CSV may be:
    - A single column with no header (one species ID per line).
    - A single column with a ``species_id`` header.
    - A two-column CSV with ``class_index`` and ``species_id`` columns.

    Parameters
    ----------
    species_csv : Path | None
        Path to the species mapping CSV, or ``None``.

    Returns
    -------
    dict[int, int]
        Dict mapping class index (int) to species ID (int).  If *species_csv*
        is ``None`` or the file does not exist, returns an empty dict (the
        caller should fall back to using indices directly).
    """
    if species_csv is None or not Path(species_csv).exists():
        if species_csv is not None:
            logger.warning(
                "Species CSV not found at %s - using class indices as species IDs.",
                species_csv,
            )
        return {}

    try:
        # plantclef: Ultimate Crash-Proof Loading
        # Directly read lines and extract the first integer found (handles headers and noisy lines)
        sids = []
        with open(species_csv, 'r') as f:
            for line in f:
                parts = line.replace(';', ' ').replace(',', ' ').split()
                if not parts: continue
                # Skip the header if present
                if "species_id" in parts[0].lower(): continue
                try:
                    sids.append(int(parts[0]))
                except ValueError:
                    continue
        
        if not sids:
            logger.error("No valid species IDs found in %s", species_csv)
            return {}
            
        unique_sids = sorted(list(set(sids)))
        mapping = {i: sid for i, sid in enumerate(unique_sids)}
    except Exception as exc:
        logger.error("Could not read species CSV at %s: %s", species_csv, exc)
        return {}

    logger.info("Loaded species mapping for %d classes from %s.", len(mapping), species_csv)
    return mapping


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_species_ids(
    class_indices: list[int],
    mapping: dict[int, int],
) -> str:
    """Convert predicted class indices to the competition ``species_ids`` string.

    Parameters
    ----------
    class_indices : list[int]
        Predicted class indices (from postprocessing).
    mapping : dict[int, int]
        Class-index → species-ID lookup.  If empty, indices are used.

    Returns
    -------
    str
        String like ``[1395806]`` or ``[1351284, 1494911, 1381367]``.
    """
    if mapping:
        species_ids = [mapping.get(i, i) for i in class_indices]
    else:
        species_ids = list(class_indices)

    if not species_ids:
        # Produce an empty bracket rather than crashing.  The competition
        # rules require at least one prediction; ensure min_predictions ≥ 1.
        return "[]"

    inner = ", ".join(str(sid) for sid in species_ids)
    return f"[{inner}]"


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_submission(
    predictions: list[ImagePrediction],
    output_path: Path,
    mapping: dict[int, int],
    append: bool = False,
) -> None:
    """Write or append the Kaggle submission CSV.

    Parameters
    ----------
    predictions : list[ImagePrediction]
        Final predictions for test images.
    output_path : Path
        Destination file path.
    mapping : dict[int, int]
        Class-index → species-ID lookup.
    append : bool, optional
        If True, append to existing file without writing header. Defaults to False.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for pred in predictions:
        species_str = format_species_ids(pred.predicted_class_indices, mapping)
        rows.append({
            "quadrat_id": pred.image_id,
            "species_ids": species_str,
        })

    df = pd.DataFrame(rows, columns=["quadrat_id", "species_ids"])
    
    # Mode: 'a' for append, 'w' for write. header=False if appending.
    mode = 'a' if append and output_path.exists() else 'w'
    header = not (append and output_path.exists())
    
    df.to_csv(output_path, mode=mode, index=False, header=header, quoting=csv.QUOTE_ALL)

    if not append:
        logger.info("Submission written to %s (%d rows).", output_path, len(df))
    _log_submission_stats(predictions)


def _load_existing_ids(output_path: Path) -> set[str]:
    """Read the output CSV and return the set of already-processed quadrat_ids.

    Parameters
    ----------
    output_path : Path
        Path to the output CSV.

    Returns
    -------
    set[str]
        Set of quadrat IDs that have already been processed.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        return set()
    try:
        df = pd.read_csv(output_path)
        if "quadrat_id" in df.columns:
            return set(df["quadrat_id"].unique())
    except Exception as exc:
        logger.warning("Could not read existing results from %s: %s", output_path, exc)
    return set()


def _log_submission_stats(predictions: list[ImagePrediction]) -> None:
    """Log basic statistics about the submission.

    Parameters
    ----------
    predictions : list[ImagePrediction]
        Final predictions for test images.
    """
    if not predictions:
        return
    counts = [p.num_predictions for p in predictions]
    arr = np.array(counts)
    logger.info(
        "Predictions per image - mean: %.1f, median: %.1f, min: %d, max: %d.",
        arr.mean(),
        float(np.median(arr)),
        int(arr.min()),
        int(arr.max()),
    )
