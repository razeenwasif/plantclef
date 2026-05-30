"""Final label selection from image-level class scores.

Postprocessing is the last step before submission generation.  It converts a
continuous score vector into a discrete set of predicted class indices.

Decision logic (applied in order)
----------------------------------
1. Apply per-class thresholds (or the global threshold for classes that have
   no per-class override).
2. Optionally cap at the top-k predictions.
3. Guarantee at least ``min_predictions`` labels (by promoting the
   highest-scoring classes even if they are below the threshold).
"""

from __future__ import annotations

import logging
import os
import json
from typing import TYPE_CHECKING, Literal, Union, List

import numpy as np

if TYPE_CHECKING:
    from .config import PostprocessConfig
from .types import ImagePrediction

logger = logging.getLogger(__name__)

# Global cache for the Rust filter and BP data to prevent re-loading
_TAXON_FILTER = None
_SPARSE_PAIRS = None
_SPARSE_WEIGHTS = None
_CONFORMAL_PREDICTOR = None

def _get_conformal_predictor(cfg: PostprocessConfig):
    """Lazily load the Hierarchical Conformal Predictor."""
    global _CONFORMAL_PREDICTOR
    if _CONFORMAL_PREDICTOR is None:
        try:
            from src.models.uncertainty.conformal import HierarchicalConformalPredictor
            import os
            calib_path = "models/conformal_calibration.pt"
            if not os.path.exists(calib_path):
                calib_path = "/workspace/PlantCLEF2026/models/conformal_calibration.pt"
            
            if os.path.exists(calib_path):
                _CONFORMAL_PREDICTOR = HierarchicalConformalPredictor()
                _CONFORMAL_PREDICTOR.load(calib_path)
                logger.info(f"[Postprocess] Loaded Conformal Predictor (q_hat={_CONFORMAL_PREDICTOR.q_hat:.4f}).")
            else:
                _CONFORMAL_PREDICTOR = False
        except Exception as e:
            logger.error(f"[Postprocess] Failed to load Conformal Predictor: {e}")
            _CONFORMAL_PREDICTOR = False
            
    return _CONFORMAL_PREDICTOR if _CONFORMAL_PREDICTOR is not False else None

_PAV_SOLVER = None
_FW_SOLVER = None

def _get_pav_solver():
    """Lazily initialize the Hierarchical Isotonic Regression solver."""
    global _PAV_SOLVER
    if _PAV_SOLVER is None:
        tree_path = "data/pav_taxonomy_tree.json"
        if os.path.exists(tree_path):
            try:
                from .pav_solver import PAVTreeSolver
                _PAV_SOLVER = PAVTreeSolver(tree_path)
                logger.info("[Postprocess] Initialized Hierarchical PAV-Tree Calibration.")
            except Exception as e:
                logger.error(f"[Postprocess] Failed to load PAV Solver: {e}")
                _PAV_SOLVER = False
    return _PAV_SOLVER if _PAV_SOLVER is not False else None

def _get_fw_solver():
    """Lazily initialize the Frank-Wolfe Constrained Solver."""
    global _FW_SOLVER
    if _FW_SOLVER is None:
        adj_path = "data/taxonomic_graph.json"
        if os.path.exists(adj_path):
            try:
                from .frank_wolfe import FrankWolfeSolver
                _FW_SOLVER = FrankWolfeSolver(adj_path)
                logger.info("[Postprocess] Initialized Frank-Wolfe Ecological Solver.")
            except Exception as e:
                logger.error(f"[Postprocess] Failed to load FW Solver: {e}")
                _FW_SOLVER = False
    return _FW_SOLVER if _FW_SOLVER is not False else None

def _get_taxon_filter(cfg: PostprocessConfig):
    """Lazily initialize the high-speed Rust taxonomic filter."""
    global _TAXON_FILTER
    if _TAXON_FILTER is None:
        try:
            import data_auditor
            # We assume a JSON file exists with the species adjacency/graph
            graph_path = "data/taxonomic_graph.json"
            genus_path = "data/species_to_genus.json"
            family_path = "data/genus_to_family.json"
            bioclim_path = "data/bioclim_data.json"
            
            if os.path.exists(graph_path) and os.path.exists(genus_path):
                with open(graph_path, 'r') as f:
                    adj = json.load(f)
                with open(genus_path, 'r') as f:
                    s2g = json.load(f)
                with open(family_path, 'r') as f:
                    g2f = json.load(f)
                with open(bioclim_path, 'r') as f:
                    bioclim = json.load(f)
                _TAXON_FILTER = data_auditor.TaxonomicFilter(adj, s2g, g2f, bioclim)
                logger.info("[Postprocess] Initialized high-speed Rust TaxonomicFilter.")
            else:
                logger.warning("[Postprocess] Taxonomic graph missing. Filtering disabled.")
        except ImportError:
            logger.error("[Postprocess] data_auditor not found. Using raw scores.")
    return _TAXON_FILTER

def _get_loopy_bp_data():
    """Lazily load and sparsify the ecological co-occurrence matrix."""
    global _SPARSE_PAIRS, _SPARSE_WEIGHTS
    if _SPARSE_PAIRS is None:
        try:
            adj_path = "data/ecological_adj.npy"
            if os.path.exists(adj_path):
                # Load the co-occurrence matrix
                adj = np.load(adj_path)
                # Sparsify: find non-zero entries above noise floor
                rows, cols = np.nonzero(adj > 1e-4)
                # Mask out diagonal (self-transition handled in Rust)
                mask = rows != cols
                rows, cols = rows[mask], cols[mask]
                
                _SPARSE_PAIRS = list(zip(rows.tolist(), cols.tolist()))
                _SPARSE_WEIGHTS = adj[rows, cols].astype(np.float32).tolist()
                logger.info(f"[Postprocess] Sparsified co-occurrence matrix: {len(_SPARSE_PAIRS)} edges.")
            else:
                logger.warning("[Postprocess] ecological_adj.npy missing. BP identity fallback.")
                _SPARSE_PAIRS = []
                _SPARSE_WEIGHTS = []
        except Exception as e:
            logger.error(f"[Postprocess] Failed to load BP data: {e}")
            _SPARSE_PAIRS = []
            _SPARSE_WEIGHTS = []
    return _SPARSE_PAIRS, _SPARSE_WEIGHTS

def solve_quadrat_consistency(
    tile_predictions: list[ImagePrediction],
    cfg: PostprocessConfig,
    threshold: float = 0.05
) -> list[ImagePrediction]:
    """
    PLANTCLEF: Apply Quadrat-Level Consistency Solver (AC-3 or Loopy BP).
    
    This ensures that species predicted across multiple tiles in a 
    single quadrat are ecologically and taxonomically compatible.
    """
    if len(tile_predictions) <= 1:
        return tile_predictions

    method = getattr(cfg, 'consistency_method', 'ac3')
    
    if method == "loopy_bp":
        return _solve_quadrat_loopy_bp(tile_predictions, cfg)
    elif method == "continuous_ac3":
        # The EIVE + GBIF Neuro-Symbolic Continuous AC-3 Solver
        taxon_filter = _get_taxon_filter(cfg)
        if taxon_filter is None:
            return tile_predictions
            
        # Load EIVE compatibility matrix (lazy load)
        global _EIVE_COMPATIBILITY
        if '_EIVE_COMPATIBILITY' not in globals() or _EIVE_COMPATIBILITY is None:
            comp_path = "data/eive_compatibility.npy"
            if os.path.exists(comp_path):
                _EIVE_COMPATIBILITY = np.load(comp_path).astype(np.float32)
            else:
                logger.error("[Postprocess] eive_compatibility.npy missing! Fallback to hard AC-3.")
                method = "ac3"
        
        if method == "continuous_ac3":
            prob_vectors = [p.class_scores.astype(np.float32) for p in tile_predictions]
            # Call the new Rust solver
            consistent_probs = taxon_filter.solve_ac3_continuous(
                prob_vectors, threshold, _EIVE_COMPATIBILITY
            )
            for i, pred in enumerate(tile_predictions):
                pred.class_scores = consistent_probs[i]
            return tile_predictions

    # Default to high-speed AC-3 (Hard Constraints)
    if method == "ac3":
        taxon_filter = _get_taxon_filter(cfg)
        if taxon_filter is None:
            return tile_predictions

        prob_vectors = [p.class_scores for p in tile_predictions]
        consistent_probs = taxon_filter.solve_ac3(prob_vectors, threshold)
        
        for i, pred in enumerate(tile_predictions):
            pred.class_scores = consistent_probs[i]
            
        return tile_predictions
    return tile_predictions

def _solve_quadrat_loopy_bp(
    tile_predictions: list[ImagePrediction],
    cfg: PostprocessConfig
) -> list[ImagePrediction]:
    """Elite-Tier Loopy Belief Propagation for soft quadrat consistency."""
    try:
        import data_auditor
        pairs, weights = _get_loopy_bp_data()
        
        # Stack probabilities: [num_tiles, num_species]
        unary = np.stack([p.class_scores for p in tile_predictions]).astype(np.float32)
        
        # Run Rust BP solver (Max-Product algorithm)
        num_iters = getattr(cfg, 'bp_iterations', 5)
        refined_marginals = data_auditor.loopy_belief_propagation(
            unary, pairs, weights, num_iters
        )
        
        # Update predictions
        for i, pred in enumerate(tile_predictions):
            pred.class_scores = refined_marginals[i]
            
        return tile_predictions
    except Exception as e:
        logger.error(f"[Postprocess] Loopy BP failed: {e}. Falling back to raw scores.")
        return tile_predictions

def postprocess(
    prediction: ImagePrediction,
    cfg: PostprocessConfig,
) -> ImagePrediction:
    """Apply thresholding, top-k selection, and Rust-based taxonomic filtering.

    Modifies ``prediction.predicted_class_indices`` and
    ``prediction.predicted_scores`` in place, then returns the same object.
    """
    scores = prediction.class_scores  # (C,)

    # --- HIERARCHICAL CALIBRATION (PAV-Tree) ---
    pav_solver = _get_pav_solver()
    if pav_solver:
        # Enforce P(Species) <= P(Genus) <= P(Family)
        # Smooths uncertain tail species toward stable genus estimates
        scores = pav_solver.solve(scores)

    # Step 0: Apply high-speed Rust Taxonomic Filtering (SIMD bitsets)
    taxon_filter = _get_taxon_filter(cfg)
    if taxon_filter is not None:
        scores = taxon_filter.filter_predictions(scores)
        prediction.class_scores = scores # Update scores with zeroed-out invalid species
        
    # Step 1: apply thresholds, Conformal Prediction, or Frank-Wolfe.
    fw_solver = _get_fw_solver()
    conformal_pred = _get_conformal_predictor(cfg)

    if fw_solver:
        # plantclef: Globally Optimal Selection in Ecological Polytope
        k = cfg.top_k or 3
        # FW expects torch tensor, handle numpy conversion if needed
        scores_torch = torch.from_numpy(scores).float()
        y_opt = fw_solver.solve(scores_torch, max_iters=5, sparsity_k=k)
        
        # Convert continuous FW solution back to hard indices (non-zero entries)
        selected_indices = torch.nonzero(y_opt > 1e-3).squeeze().tolist()
        if isinstance(selected_indices, int): selected_indices = [selected_indices]
        if not selected_indices: selected_indices = [int(torch.argmax(scores_torch))]
        
    elif conformal_pred is not None:
        # Apply Adaptive Prediction Sets (APS) logic on probabilities
        sorted_indices = np.argsort(scores)[::-1]
        sorted_probs = scores[sorted_indices]
        cum_probs = np.cumsum(sorted_probs)
        
        # Include all indices where cum_prob <= q_hat plus the first index that exceeds it
        cutoff_idx = np.searchsorted(cum_probs, conformal_pred.q_hat)
        num_to_keep = min(len(scores), cutoff_idx + 1)
        selected_indices = sorted_indices[:num_to_keep].tolist()
    else:
        # Fall back to Brent's or global thresholds
        selected_indices = _apply_thresholds(scores, cfg)

    # Step 2: cap at top-k.
    if cfg.top_k is not None and len(selected_indices) > cfg.top_k:
        selected_indices = _top_k(scores, selected_indices, cfg.top_k)

    # Step 3: guarantee minimum number of predictions.
    if len(selected_indices) < cfg.min_predictions:
        selected_indices = _fill_to_min(
            scores, selected_indices, cfg.min_predictions
        )

    # Sort by descending score for readability.
    selected_indices = sorted(
        selected_indices, key=lambda i: scores[i], reverse=True
    )

    prediction.predicted_class_indices = selected_indices
    prediction.predicted_scores = [float(scores[i]) for i in selected_indices]
    return prediction


def postprocess_batch(
    predictions: list[ImagePrediction],
    cfg: PostprocessConfig,
) -> list[ImagePrediction]:
    """Apply :func:`postprocess` to a list of predictions.

    Parameters
    ----------
    predictions : list[ImagePrediction]
        Ensemble-combined predictions for multiple images.
    cfg : PostprocessConfig
        Postprocessing configuration.

    Returns
    -------
    list[ImagePrediction]
        The same list with all predictions updated in place.
    """
    for pred in predictions:
        postprocess(pred, cfg)
    return predictions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_thresholds(
    scores: np.ndarray,
    cfg: PostprocessConfig,
) -> list[int]:
    """Return class indices that meet their configured threshold.

    Parameters
    ----------
    scores : np.ndarray
        ``(C,)`` image-level class scores.
    cfg : PostprocessConfig
        Postprocessing configuration.

    Returns
    -------
    list[int]
        List of class indices above the threshold.
    """
    # PLANTCLEF OPTIMIZATION: Structure of Arrays (SoA) and SIMD Masking
    # Instead of a scalar Python loop with branching (which destroys instruction pipelines),
    # we build a contiguous threshold array and use vectorized C/AVX instructions via NumPy.
    C = len(scores)
    
    if not hasattr(cfg, '_cached_thresholds') or len(getattr(cfg, '_cached_thresholds', [])) != C:
        # Precompute the SoA threshold array once
        thresholds = np.full(C, cfg.global_threshold, dtype=np.float32)
        if cfg.per_class_thresholds:
            for i, t in cfg.per_class_thresholds.items():
                thresholds[int(i)] = t
        cfg._cached_thresholds = thresholds
    
    # Branchless SIMD masked comparison
    mask = scores >= cfg._cached_thresholds
    selected = np.nonzero(mask)[0].tolist()
    
    return selected


def _top_k(
    scores: np.ndarray,
    indices: list[int],
    k: int,
) -> list[int]:
    """Keep only the top-k indices by score.

    Parameters
    ----------
    scores : np.ndarray
        Full ``(C,)`` score array.
    indices : list[int]
        Current candidate indices.
    k : int
        Maximum number to keep.

    Returns
    -------
    list[int]
        Subset of *indices* of size at most *k*.
    """
    if len(indices) <= k:
        return indices
    indices_arr = np.array(indices)
    top = np.argpartition(scores[indices_arr], -k)[-k:]
    return indices_arr[top].tolist()


def _fill_to_min(
    scores: np.ndarray,
    current: list[int],
    min_count: int,
) -> list[int]:
    """Promote highest-scoring classes until at least ``min_count`` predictions.

    Parameters
    ----------
    scores : np.ndarray
        Full ``(C,)`` score array.
    current : list[int]
        Already-selected class indices.
    min_count : int
        Minimum number of predictions required.

    Returns
    -------
    list[int]
        Extended list of class indices.
    """
    needed = min_count - len(current)
    if needed <= 0:
        return current

    current_set = set(current)
    # Candidates: all classes not already selected.
    all_indices = np.arange(len(scores))
    candidates = np.array([i for i in all_indices if i not in current_set])

    if len(candidates) == 0:
        return current

    # Take the top `needed` candidates by score.
    n_take = min(needed, len(candidates))
    top_local = np.argpartition(scores[candidates], -n_take)[-n_take:]
    promoted = candidates[top_local].tolist()

    logger.debug(
        "Promoted %d class(es) to meet min_predictions=%d.", len(promoted), min_count
    )
    return current + promoted
