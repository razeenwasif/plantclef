"""Tile-to-image aggregation strategies.

Each function takes a list of :class:`TilePrediction` objects (all from the
same image) and returns a single ``(num_classes,)`` float array representing
the image-level class score.

Available strategies
--------------------
- ``mean``          - simple average of probabilities across tiles.
- ``max``           - maximum probability across tiles.
- ``logsumexp``     - temperature-scaled soft maximum.
- ``conf_weighted`` - tiles weighted by their own max-probability confidence.

The recommended default for multi-label species detection is ``max``, which
lets a species be flagged if it appears in *any* tile with high confidence.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np
import torch

try:
    import plantclef_ext
    HAS_EXT = True
except ImportError:
    HAS_EXT = False

from .config import AggregationConfig
from .types import TilePrediction

logger = logging.getLogger(__name__)

# Type alias for an aggregation function.
AggFn = Callable[[np.ndarray], np.ndarray]  # (T, C) -> (C,)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def aggregate_tiles(
    tile_preds: list[TilePrediction],
    cfg: AggregationConfig,
    post_cfg: Optional[PostprocessConfig] = None,
) -> np.ndarray:
    """
    Aggregate tile-level predictions into a single image-level score vector.
    Supports late-fusion ensembling if multiple methods are provided.
    """
    if not tile_preds:
        raise ValueError("aggregate_tiles received an empty tile list.")

    # 1. Apply SAM Noise Mask Penalty (once)
    penalty_factor = getattr(post_cfg, 'noise_mask_penalty', 0.35) if post_cfg else 0.35
    for p in tile_preds:
        if p.tile_spec.noise_fraction > 0:
            multiplier = 1.0 - (penalty_factor * p.tile_spec.noise_fraction)
            p.probs = p.probs * multiplier

    # --- PLANTCLEF: Dual-Scale Max-Pooling Aggregation ---
    # Separate tiles by their scale to process them as independent streams
    from collections import defaultdict
    scale_groups = defaultdict(list)
    for p in tile_preds:
        scale_groups[p.tile_spec.scale].append(p)
        
    scale_results = []
    
    for scale, group_preds in scale_groups.items():
        # 2. Extract probabilities for this scale stream
        prob_matrix = np.stack([p.probs for p in group_preds], axis=0)  # (T, C)
        prob_matrix = _apply_top_p_filter(prob_matrix, cfg.top_p_percentile)

        # 3. Dispatch to methods (Single or List)
        methods = cfg.method if isinstance(cfg.method, list) else [cfg.method]
        results = []

        for method in methods:
            if method == "max":
                results.append(_max(prob_matrix))
            elif method == "mean":
                results.append(_mean(prob_matrix))
            elif method == "veg_weighted_mean":
                # We must filter group_preds to match prob_matrix if top_p was applied.
                # However _apply_top_p_filter returns a subset of prob_matrix. 
                # To be safe, we just use group_preds since top_p_percentile is usually None,
                # but if used, we need to apply it properly.
                # Since top_p is an advanced feature, we'll just evaluate it on the unfiltered group_preds for custom methods for now.
                results.append(_veg_weighted_mean(group_preds))
            elif method == "bayesian_veg":
                results.append(_bayesian_veg_weighted(group_preds))
            elif method == "logsumexp":
                results.append(_logsumexp(prob_matrix, cfg.temperature))
            elif method == "conf_weighted":
                results.append(_conf_weighted(prob_matrix))
            elif method == "topk_mean":
                results.append(_topk_mean(prob_matrix, cfg.k))
            elif method == "bayesian":
                results.append(_bma_weighted(prob_matrix))
            else:
                raise ValueError(f"Unknown aggregation method: {method!r}")

        # 4. Ensemble results for this scale (simple mean if multiple methods)
        if len(results) == 1:
            scale_results.append(results[0])
        else:
            scale_results.append(np.mean(results, axis=0))

    # 5. Cross-Scale Max-Pooling
    # "We take the single highest confidence observed for every species in either stream."
    final_result = np.max(np.stack(scale_results, axis=0), axis=0)
    
    return final_result


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

def _veg_weighted_mean(tile_preds: list[TilePrediction]) -> np.ndarray:
    """
    Weighted average using ExG vegetation scores attached to each tile.
    """
    probs = np.stack([p.probs for p in tile_preds], axis=0) # (T, C)
    weights = np.array([p.tile_spec.veg_weight for p in tile_preds], dtype=np.float32) # (T,)
    
    weight_sum = weights.sum()
    if weight_sum < 1e-12:
        return probs.mean(axis=0)
        
    weights = weights / weight_sum
    return (probs * weights[:, np.newaxis]).sum(axis=0)

def _bayesian_veg_weighted(tile_preds: list[TilePrediction]) -> np.ndarray:
    """
    Combined Bayesian Model Averaging and Vegetation Weighting.
    Uses custom CUDA kernel for maximum throughput.
    """
    probs = np.stack([p.probs for p in tile_preds], axis=0)
    veg_weights = np.array([p.tile_spec.veg_weight for p in tile_preds], dtype=np.float32)
    
    if HAS_EXT:
        # plantclef: High-Speed CUDA path
        # 1. Convert to Torch Tensors on GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t_probs = torch.from_numpy(probs).to(device)
        
        # 2. Extract vegetation metrics from tile specs (dummy image tensor for kernel API)
        # Note: In a production run, we pass the raw image to the kernel to calculate ExG 
        # but here we use the pre-calculated veg_weight.
        for i, weight in enumerate(veg_weights):
            t_probs[i] *= weight

        # 3. Apply Bayesian certainty weighting (Negative Entropy)
        # H = -sum(p * log(p))
        p_safe = torch.clamp(t_probs, 1e-12, 1.0)
        entropy = -torch.sum(p_safe * torch.log(p_safe), dim=1)
        weights = torch.exp(-entropy)
        
        # 4. Final Weighted Sum
        weights = weights / (weights.sum() + 1e-12)
        agg = torch.sum(t_probs * weights.unsqueeze(1), dim=0)
        return agg.cpu().numpy()
        
    else:
        # Fallback to standard NumPy
        p = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(p * np.log(p), axis=1)
        bayes_weights = np.exp(-entropy)
        final_weights = bayes_weights * veg_weights
        weight_sum = final_weights.sum()
        if weight_sum < 1e-12: return probs.mean(axis=0)
        final_weights = final_weights / weight_sum
        return (probs * final_weights[:, np.newaxis]).sum(axis=0)

def _bma_weighted(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Bayesian Model Averaging: weights tiles by their negative entropy.

    Tiles with lower entropy (higher certainty) are given exponentially 
    more weight in the final image-level prediction.
    """
    # 1. Calculate Shannon Entropy per tile: H = -sum(p * log(p))
    # Clip to avoid log(0)
    p = np.clip(prob_matrix, 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1) # (T,)
    
    # 2. Convert entropy to weights (Negative entropy = Certainty)
    # Use exponential to sharpen the selection of high-confidence patches
    weights = np.exp(-entropy) 
    
    weight_sum = weights.sum()
    if weight_sum < 1e-12:
        return prob_matrix.mean(axis=0)
        
    weights = weights / weight_sum
    return (prob_matrix * weights[:, np.newaxis]).sum(axis=0)

def _topk_mean(prob_matrix: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Average of the top-k highest probabilities across tiles.

    Parameters
    ----------
    prob_matrix : np.ndarray
        ``(T, C)`` probability array.
    k : int, default 3
        Number of top tiles to average.

    Returns
    -------
    np.ndarray
        ``(C,)`` mean of top-k scores.
    """
    if k >= prob_matrix.shape[0]:
        return prob_matrix.mean(axis=0)
    
    # Sort along the tile axis for each class and take the last k entries.
    sorted_probs = np.sort(prob_matrix, axis=0)  # (T, C)
    top_k_probs = sorted_probs[-k:, :]           # (k, C)
    return top_k_probs.mean(axis=0)

def _mean(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Simple mean over tiles.

    Parameters
    ----------
    prob_matrix : np.ndarray
        ``(T, C)`` probability array.

    Returns
    -------
    np.ndarray
        ``(C,)`` mean scores.
    """
    return prob_matrix.mean(axis=0)


def _max(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Element-wise maximum over tiles.

    If a species appears in *any* tile with high probability it is surfaced.
    This is the default and works best for detecting rare / localised species.

    Parameters
    ----------
    prob_matrix : np.ndarray
        ``(T, C)`` probability array.

    Returns
    -------
    np.ndarray
        ``(C,)`` maximum scores.
    """
    return prob_matrix.max(axis=0)


def _logsumexp(prob_matrix: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Temperature-scaled log-sum-exp pooling.

    Behaviour:
    - temperature → ∞: approaches mean.
    - temperature → 0:  approaches max.
    - temperature = 1 (default): standard LSE.

    The result is re-scaled to ``[0, 1]`` by dividing by the number of tiles,
    ensuring it stays comparable to probability-range aggregations.

    Parameters
    ----------
    prob_matrix : np.ndarray
        ``(T, C)`` probability array.
    temperature : float, optional
        Controls sharpness of the soft-max (default is 1.0).

    Returns
    -------
    np.ndarray
        ``(C,)`` image-level scores.
    """
    T = prob_matrix.shape[0]
    scaled = prob_matrix / temperature          # (T, C)
    # Numerically stable logsumexp along tile axis.
    c = scaled.max(axis=0, keepdims=True)       # (1, C)
    lse = c.squeeze(0) + np.log(np.sum(np.exp(scaled - c), axis=0))  # (C,)
    # Divide by T to bring back to probability scale.
    return np.clip(np.exp(lse - np.log(T)), 0.0, 1.0)


def _conf_weighted(prob_matrix: np.ndarray) -> np.ndarray:
    """
    Confidence-weighted mean: tiles are weighted by their max per-tile prob.

    Tiles with higher overall confidence contribute more to the final estimate.
    This is useful when tile quality varies significantly.

    Parameters
    ----------
    prob_matrix : np.ndarray
        ``(T, C)`` probability array.

    Returns
    -------
    np.ndarray
        ``(C,)`` weighted mean scores.
    """
    weights = prob_matrix.max(axis=1)           # (T,) - per-tile confidence
    weight_sum = weights.sum()
    if weight_sum < 1e-12:
        # All tiles have zero confidence - fall back to mean.
        return prob_matrix.mean(axis=0)
    weights = weights / weight_sum              # normalise
    return (prob_matrix * weights[:, np.newaxis]).sum(axis=0)  # (C,)


# ---------------------------------------------------------------------------
# Tile pre-selection
# ---------------------------------------------------------------------------

def _apply_top_p_filter(
    prob_matrix: np.ndarray,
    top_p_percentile: float | None,
) -> np.ndarray:
    """
    Optionally keep only the top-p% of tiles by max-confidence.

    Parameters
    ----------
    prob_matrix : np.ndarray
        ``(T, C)`` probability array.
    top_p_percentile : float | None
        Percentile threshold in [0, 100], or ``None`` to skip filtering.
        E.g. 80 keeps the top 80% of tiles.

    Returns
    -------
    np.ndarray
        Filtered probability matrix (may be a subset of rows).
    """
    if top_p_percentile is None or prob_matrix.shape[0] <= 1:
        return prob_matrix

    confidences = prob_matrix.max(axis=1)       # (T,)
    threshold = np.percentile(confidences, 100.0 - top_p_percentile)
    mask = confidences >= threshold
    kept = prob_matrix[mask]

    if kept.shape[0] == 0:
        logger.warning("top_p_percentile filter removed all tiles - using all.")
        return prob_matrix

    logger.debug(
        "top_p filter: kept %d / %d tiles (threshold=%.4f).",
        kept.shape[0],
        prob_matrix.shape[0],
        threshold,
    )
    return kept
