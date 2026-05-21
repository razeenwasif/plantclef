"""Multi-model ensembling of image-level predictions.

After each model has produced an :class:`ImagePrediction` for a given image,
:func:`ensemble_predictions` combines the ``class_scores`` arrays into a
single consensus score vector.

Currently supported strategies
-------------------------------
- **Weighted average** (default): ``score = Σ w_i * scores_i`` where weights
  are normalised to sum to 1.  Set all weights equal for a simple mean.

Additional strategies (e.g. rank-averaging, geometric mean) can be added by
extending :func:`ensemble_predictions`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import EnsembleConfig
from .types import ImagePrediction

logger = logging.getLogger(__name__)


def ensemble_predictions(
    model_predictions: list[ImagePrediction],
    cfg: EnsembleConfig,
) -> ImagePrediction:
    """Combine per-model :class:`ImagePrediction` objects for one image.

    Parameters
    ----------
    model_predictions : list[ImagePrediction]
        One :class:`ImagePrediction` per model, all for the
        same ``image_id``.  Must be non-empty.
    cfg : EnsembleConfig
        Ensemble configuration (optional per-model weights).

    Returns
    -------
    ImagePrediction
        A new :class:`ImagePrediction` with ensembled ``class_scores``.
        The ``predicted_class_indices`` and ``predicted_scores`` fields are
        left empty - they are populated later by :mod:`postprocess`.

    Raises
    ------
    ValueError
        If ``model_predictions`` is empty or image IDs mismatch.
    """
    if not model_predictions:
        raise ValueError("ensemble_predictions received an empty list.")

    image_ids = {p.image_id for p in model_predictions}
    if len(image_ids) > 1:
        raise ValueError(
            f"ensemble_predictions received predictions from different images: "
            f"{image_ids}"
        )

    image_id = model_predictions[0].image_id
    n = len(model_predictions)

    weights = _resolve_weights(cfg.weights, n)
    logger.debug(
        "Ensembling %d model(s) for image %r with weights %s.",
        n,
        image_id,
        weights,
    )

    scores_list = [p.class_scores for p in model_predictions]
    combined = _weighted_average(scores_list, weights)

    return ImagePrediction(
        image_id=image_id,
        class_scores=combined,
        predicted_class_indices=[],
        predicted_scores=[],
    )


def _weighted_average(
    scores_list: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Compute a normalised weighted average of score arrays.

    Parameters
    ----------
    scores_list : list[np.ndarray]
        List of ``(C,)`` float arrays.
    weights : list[float]
        Per-array weights (must already sum to 1.0).

    Returns
    -------
    np.ndarray
        ``(C,)`` weighted average array.
    """
    result = np.zeros_like(scores_list[0], dtype=np.float64)
    for w, s in zip(weights, scores_list):
        result += w * s.astype(np.float64)
    return result.astype(np.float32)


def _resolve_weights(
    weights: list[float] | None,
    n: int,
) -> list[float]:
    """Return normalised weights for *n* models.

    If ``weights`` is ``None``, returns uniform weights summing to 1.
    Otherwise validates length and normalises to sum to 1.

    Parameters
    ----------
    weights : list[float] | None
        Raw per-model weights or ``None``.
    n : int
        Number of models.

    Returns
    -------
    list[float]
        Normalised weight list of length *n*.

    Raises
    ------
    ValueError
        If the provided weights list has the wrong length or
        all weights are zero.
    """
    if weights is None:
        return [1.0 / n] * n

    if len(weights) != n:
        raise ValueError(
            f"Expected {n} ensemble weights (one per checkpoint), "
            f"got {len(weights)}."
        )

    total = sum(weights)
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")

    return [w / total for w in weights]
