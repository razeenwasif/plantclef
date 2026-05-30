"""
Build per-species text prompts from the enriched GBIF species CSV.

Prompt modes
------------
scientific          Family A only: scientific-name templates.
scientific_common   Families A + B + D: adds common-name and cross prompts.
scientific_family   Families A + C: adds family-tagged prompts.
all                 Families A + B + C + D: everything.

Prompt families
---------------
A  scientific:       "a photo of {sci}", "a close-up photo of {sci}",
                     "a wild plant of species {sci}"
B  common:           "a photo of {common}", "a close-up photo of {common}",
                     "a wild plant called {common}"
C  sci + family:     "a photo of {sci}, a plant in the family {family}",
                     "a close-up photo of {sci}, a plant in the family {family}"
D  sci + common:     "a photo of {sci} ({common})",
                     "a close-up photo of {sci} ({common})"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

PromptMode = Literal["scientific", "scientific_common", "scientific_family", "all"]
PROMPT_MODES: tuple[str, ...] = ("scientific", "scientific_common", "scientific_family", "all")

REQUIRED_COLUMNS = ["species_id", "species"]

# Preferred columns in priority order (first non-empty value is used)
_SCIENTIFIC_COLS = ["species_name_clean", "gbif_scientific_name_clean", "gbif_canonical_name", "species"]
_GBIF_CANONICAL_COLS = ["gbif_canonical_name", "gbif_scientific_name_clean"]
_COMMON_COLS = ["primary_common_name_en"]
_COMMON_EXTRA_COLS = ["gbif_common_names_en_clean"]
_SYNONYM_COLS = ["gbif_synonyms_clean"]
_FAMILY_COLS = ["family", "gbif_family"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SpeciesLabel:
    species_id: str
    canonical_scientific: str       # primary scientific name for prompts
    scientific_aliases: list[str]   # extra scientific names (deduplicated, may be empty)
    primary_common: str             # primary English common name (may be "")
    extra_common: list[str]         # additional common names
    family: str                     # family name (may be "")

    @property
    def all_scientific(self) -> list[str]:
        return _dedup([self.canonical_scientific] + self.scientific_aliases)

    @property
    def all_common(self) -> list[str]:
        names = ([self.primary_common] if self.primary_common else []) + self.extra_common
        return _dedup(names)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(s) -> str:
    if s is None or (isinstance(s, float) and s != s):  # NaN check
        return ""
    return str(s).strip()


def _parse_pipe(s) -> list[str]:
    """Split a pipe-separated cell into a deduplicated list of non-empty strings."""
    raw = _clean(s)
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("|")]
    return _dedup([p for p in parts if p])


def _dedup(names: list[str]) -> list[str]:
    """Case-insensitive deduplication preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for n in names:
        if n and n.lower() not in seen:
            seen.add(n.lower())
            result.append(n)
    return result


def _pick(row: pd.Series, *cols: str) -> str:
    """Return the first non-empty value found across the listed columns."""
    for col in cols:
        if col in row.index:
            v = _clean(row[col])
            if v:
                return v
    return ""


# ---------------------------------------------------------------------------
# Label building
# ---------------------------------------------------------------------------

def build_species_labels(
    df: pd.DataFrame,
    max_common_names: int = 3,
    max_synonyms: int = 2,
) -> list[SpeciesLabel]:
    """Convert a species DataFrame into a list of SpeciesLabel objects."""
    labels: list[SpeciesLabel] = []

    for _, row in df.iterrows():
        species_id = str(row["species_id"])

        # Canonical scientific name (highest-quality available)
        canonical = _pick(row, *_SCIENTIFIC_COLS)

        # Additional scientific aliases: GBIF canonical + synonyms
        alt_sci: list[str] = []
        gbif_canonical = _pick(row, *_GBIF_CANONICAL_COLS)
        if gbif_canonical and gbif_canonical.lower() != canonical.lower():
            alt_sci.append(gbif_canonical)

        synonyms = _parse_pipe(_clean(row.get("gbif_synonyms_clean", "")))
        for syn in synonyms[:max_synonyms]:
            if syn.lower() not in {canonical.lower()} | {a.lower() for a in alt_sci}:
                alt_sci.append(syn)

        # Common names
        primary_common = _pick(row, *_COMMON_COLS)
        extra_raw = _parse_pipe(_clean(row.get("gbif_common_names_en_clean", "")))
        extra_common = [
            c for c in extra_raw
            if c.lower() != primary_common.lower()
        ][:max_common_names]

        # Family
        family = _pick(row, *_FAMILY_COLS)

        labels.append(SpeciesLabel(
            species_id=species_id,
            canonical_scientific=canonical,
            scientific_aliases=_dedup(alt_sci),
            primary_common=primary_common,
            extra_common=extra_common,
            family=family,
        ))

    return labels


def load_species_labels(
    csv_path: str | Path,
    max_common_names: int = 3,
    max_synonyms: int = 2,
) -> list[SpeciesLabel]:
    """Load species labels from the enriched GBIF CSV."""
    df = pd.read_csv(csv_path, dtype=str)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Species CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Warn about preferred columns that are absent
    all_preferred = _SCIENTIFIC_COLS + _COMMON_COLS + _FAMILY_COLS + ["gbif_synonyms_clean", "gbif_common_names_en_clean"]
    for col in all_preferred:
        if col not in df.columns:
            print(f"  [warn] preferred column '{col}' not found in CSV, will use fallback")

    return build_species_labels(df, max_common_names=max_common_names, max_synonyms=max_synonyms)


# ---------------------------------------------------------------------------
# Prompt template families
# ---------------------------------------------------------------------------

def _prompts_sci(name: str) -> list[str]:
    return [
        f"a photo of {name}",
        f"a close-up photo of {name}",
        f"a wild plant of species {name}",
    ]


def _prompts_common(common: str) -> list[str]:
    return [
        f"a photo of {common}",
        f"a close-up photo of {common}",
        f"a wild plant called {common}",
    ]


def _prompts_sci_family(sci: str, family: str) -> list[str]:
    return [
        f"a photo of {sci}, a plant in the family {family}",
        f"a close-up photo of {sci}, a plant in the family {family}",
    ]


def _prompts_sci_common(sci: str, common: str) -> list[str]:
    return [
        f"a photo of {sci} ({common})",
        f"a close-up photo of {sci} ({common})",
    ]


# ---------------------------------------------------------------------------
# Public prompt-building API
# ---------------------------------------------------------------------------

def build_prompts_for_species(label: SpeciesLabel, mode: PromptMode) -> list[str]:
    """Return a deduplicated list of text prompts for one species."""
    if mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode '{mode}'. Choose from: {PROMPT_MODES}")

    prompts: list[str] = []

    # Family A: scientific-name templates (always)
    for sci in label.all_scientific:
        prompts.extend(_prompts_sci(sci))

    if mode in ("scientific_common", "all"):
        # Family B: common-name templates
        for common in label.all_common:
            prompts.extend(_prompts_common(common))
        # Family D: scientific + common cross templates (canonical sci only)
        if label.primary_common and label.canonical_scientific:
            prompts.extend(_prompts_sci_common(label.canonical_scientific, label.primary_common))

    if mode in ("scientific_family", "all"):
        # Family C: scientific + family templates (canonical sci only)
        if label.family and label.canonical_scientific:
            prompts.extend(_prompts_sci_family(label.canonical_scientific, label.family))

    return _dedup(prompts)


def build_all_prompts(
    labels: list[SpeciesLabel],
    mode: PromptMode,
) -> list[list[str]]:
    """Return one prompt list per species in the same order as `labels`."""
    return [build_prompts_for_species(label, mode) for label in labels]


def prompt_stats(prompt_lists: list[list[str]]) -> dict:
    """Return summary statistics about a prompt table."""
    counts = [len(p) for p in prompt_lists]
    total = sum(counts)
    return {
        "n_species": len(prompt_lists),
        "total_prompts": total,
        "min_per_species": min(counts) if counts else 0,
        "max_per_species": max(counts) if counts else 0,
        "avg_per_species": round(total / len(counts), 2) if counts else 0,
    }
