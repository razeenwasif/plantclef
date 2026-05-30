"""
Tile-logit aggregation methods for zero-shot BioCLIP inference.

All functions accept (n_tiles, n_species) tile_logits tensors and return
a (n_species,) image-level logit vector.

Methods
-------
  max              SAHI-style: per-species max over all tiles          [baseline from 002]
  mean             Plain mean over all tiles
  topk_mean        Mean of the top-k tiles per species
  weighted_mean    SAM-weight-scaled mean over all tiles
  weighted_topk_mean  Weighted mean restricted to top-k tiles by composite score
"""

from __future__ import annotations

import numpy as np
import torch

from config import AGGREGATION_METHODS, DEFAULT_TOPK_TILES


# ---------------------------------------------------------------------------
# Individual aggregation functions
# ---------------------------------------------------------------------------

def aggregate_max(tile_logits: torch.Tensor) -> torch.Tensor:
    """Per-species max across tiles. Identical to the 002 baseline."""
    return tile_logits.max(dim=0).values


def aggregate_mean(tile_logits: torch.Tensor) -> torch.Tensor:
    """Uniform mean across all tiles."""
    return tile_logits.mean(dim=0)


def aggregate_topk_mean(
    tile_logits: torch.Tensor,
    k: int = DEFAULT_TOPK_TILES,
) -> torch.Tensor:
    """
    Mean of the top-k tile logits per species.

    For each species independently, take the k tiles with the highest logit
    and average them.  This is a soft version of max that is less sensitive
    to a single noisy tile.
    """
    k = min(k, tile_logits.shape[0])
    topk_vals, _ = tile_logits.topk(k, dim=0)  # (k, n_species)
    return topk_vals.mean(dim=0)


def aggregate_weighted_mean(
    tile_logits: torch.Tensor,
    weights: np.ndarray,
) -> torch.Tensor:
    """
    Vegetation-weighted mean across all tiles.

    weights : (n_tiles,) array of positive floats (e.g. from compute_tile_weights)
    The weights are normalised to sum to 1 before application.
    """
    w = torch.tensor(weights, dtype=tile_logits.dtype, device=tile_logits.device)
    w = w / w.sum().clamp(min=1e-8)          # normalise to probability simplex
    return (tile_logits * w.unsqueeze(1)).sum(dim=0)


def aggregate_weighted_topk_mean(
    tile_logits: torch.Tensor,
    weights: np.ndarray,
    k: int = DEFAULT_TOPK_TILES,
) -> torch.Tensor:
    """
    Weighted mean of the top-k tiles selected by a composite tile score.

    Tile selection score: weight_i × max_species_logit_i
    Selects the k tiles with the highest composite score, then takes
    their weight-normalised mean logit vector.

    This rewards tiles that are both vegetation-rich and confidently
    predicted by BioCLIP.
    """
    k = min(k, tile_logits.shape[0])
    w = torch.tensor(weights, dtype=tile_logits.dtype, device=tile_logits.device)

    # Composite score: vegetation weight × best logit for that tile
    composite = w * tile_logits.max(dim=1).values   # (n_tiles,)
    top_idx = composite.topk(k).indices              # (k,)

    top_logits  = tile_logits[top_idx]               # (k, n_species)
    top_weights = w[top_idx]                         # (k,)
    top_weights = top_weights / top_weights.sum().clamp(min=1e-8)

    return (top_logits * top_weights.unsqueeze(1)).sum(dim=0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def aggregate(
    tile_logits: torch.Tensor,
    method: str,
    weights: np.ndarray | None = None,
    topk_tiles: int = DEFAULT_TOPK_TILES,
) -> torch.Tensor:
    """
    Dispatch to the selected aggregation method.

    Parameters
    ----------
    tile_logits : (n_tiles, n_species) float tensor
    method      : one of AGGREGATION_METHODS
    weights     : (n_tiles,) array — required for 'weighted_mean' and 'weighted_topk_mean'
    topk_tiles  : k for topk_mean / weighted_topk_mean

    Returns
    -------
    (n_species,) float tensor
    """
    if method == "max":
        return aggregate_max(tile_logits)
    if method == "mean":
        return aggregate_mean(tile_logits)
    if method == "topk_mean":
        return aggregate_topk_mean(tile_logits, k=topk_tiles)
    if method == "weighted_mean":
        if weights is None:
            raise ValueError("weights required for method='weighted_mean'")
        return aggregate_weighted_mean(tile_logits, weights)
    if method == "weighted_topk_mean":
        if weights is None:
            raise ValueError("weights required for method='weighted_topk_mean'")
        return aggregate_weighted_topk_mean(tile_logits, weights, k=topk_tiles)
    raise ValueError(
        f"Unknown aggregation method '{method}'. Choose from {AGGREGATION_METHODS}."
    )
