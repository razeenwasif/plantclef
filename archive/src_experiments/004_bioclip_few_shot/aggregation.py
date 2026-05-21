"""
Multi-label tile-to-plot aggregation strategies.

Input
-----
tile_scores : (n_tiles, n_species) tensor of per-tile species scores.

Output
------
image_scores : (n_species,) aggregated image-level score.
pred_ids     : list of predicted species IDs after thresholding / top-N.
"""

from __future__ import annotations

import torch


# Valid mode strings exposed for CLI validation
AGG_MODES = ("max", "mean", "mean_top_m", "noisy_or")


def aggregate_scores(
    tile_scores: torch.Tensor,
    mode: str = "max",
    top_m: int = 3,
) -> torch.Tensor:
    """
    Aggregate per-tile species scores into a single image-level score.

    Parameters
    ----------
    tile_scores : (n_tiles, n_species) float tensor
    mode        : aggregation strategy - one of AGG_MODES
    top_m       : number of tiles to average for "mean_top_m" mode

    Modes
    -----
    max         SAHI-style: per-species max over tiles.
                Best for detecting rare species that appear in only a few tiles.

    mean        Simple mean over all tiles.
                Works well when species are spread across the quadrat.

    mean_top_m  Mean of the top-m tile scores per species.
                Balances max (noisy single tile) and mean (diluted by background).

    noisy_or    Probabilistic OR: 1 - prod(1 - sigmoid(score)).
                Models the probability that at least one tile contains the species.

    Returns
    -------
    (n_species,) float32 tensor of image-level scores
    """
    if mode == "max":
        return tile_scores.max(dim=0).values

    elif mode == "mean":
        return tile_scores.mean(dim=0)

    elif mode == "mean_top_m":
        n_tiles = tile_scores.shape[0]
        m = min(top_m, n_tiles)
        top_vals, _ = tile_scores.topk(m, dim=0)  # (m, n_species)
        return top_vals.mean(dim=0)

    elif mode == "noisy_or":
        # Treat scores as logits → sigmoid → noisy-OR across tiles
        probs = torch.sigmoid(tile_scores)   # (n_tiles, n_species)
        # log(1 - p) summed, then exp and complement
        log_comp = torch.log1p(-probs.clamp(max=1.0 - 1e-6))
        return 1.0 - torch.exp(log_comp.sum(dim=0))

    else:
        raise ValueError(
            f"Unknown aggregation mode: '{mode}'. "
            f"Valid choices: {AGG_MODES}"
        )


def apply_threshold(
    image_scores: torch.Tensor,
    species_ids: list[str],
    threshold: float = 0.0,
    top_n: int = 20,
    per_species_thresholds: dict[str, float] | None = None,
) -> tuple[list[str], list[float]]:
    """
    Apply threshold and top-N filter to produce final multi-label predictions.

    Parameters
    ----------
    image_scores            : (n_species,) image-level scores
    species_ids             : ordered species ID list matching axis-0 of image_scores
    threshold               : global minimum score; species below are excluded
    top_n                   : hard cap on number of predictions (0 = no cap)
    per_species_thresholds  : optional dict species_id -> threshold override

    Returns
    -------
    pred_ids    : list of species ID strings, sorted by score descending
    pred_scores : corresponding float scores
    """
    # Sort all species by score descending
    sorted_vals, sorted_idx = image_scores.sort(descending=True)

    # Apply top_n cap
    if top_n > 0:
        sorted_vals = sorted_vals[:top_n]
        sorted_idx = sorted_idx[:top_n]

    pred_ids: list[str] = []
    pred_scores: list[float] = []

    for val, idx in zip(sorted_vals.tolist(), sorted_idx.tolist()):
        sid = species_ids[idx]

        # Per-species threshold takes priority over global
        if per_species_thresholds is not None and sid in per_species_thresholds:
            thr = per_species_thresholds[sid]
        else:
            thr = threshold

        if val < thr:
            break  # sorted, so all remaining are below threshold too

        pred_ids.append(sid)
        pred_scores.append(val)

    return pred_ids, pred_scores
