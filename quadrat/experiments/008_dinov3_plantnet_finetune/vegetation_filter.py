"""
Per-tile vegetation filtering for PlantCLEF quadrat inference.

PlantCLEF quadrats are 50×50 cm frames of plants on natural soil. Typically
30–70% of the pixels are soil / litter / rocks / shadow, and tiles that
contain zero vegetation inject false-positive signal into any tile-aggregated
prediction (max-pool over a "pure dirt" tile's hallucinated logits pushes up
whatever species the backbone happens to respond to weakly).

Winning 2024/2025 teams all reported that dropping non-vegetation tiles
before aggregation is a +F1 win. They use one of two filters:

  (a) **Excess Green index** (default here). Fast, no extra model.
      ExG = 2G - R - B on [0, 255] pixels. The GLI (Green Leaf Index) and
      VARI variants work similarly. A pixel is "plant" if ExG > exg_thresh
      AND roughly G > R AND G > B. A tile is kept if its plant-pixel
      fraction ≥ min_frac.

  (b) **SAM + CLIP / SAM2 + plant text classification**. Stronger — handles
      dried / yellowed / flower-only tiles that ExG misses — but requires
      SAM weights, a GPU forward per tile, and does not fit into the current
      sub-hour inference budget without its own ensemble pod.

This module implements (a) as the default and exposes the interface that a
future (b) implementation would slot into. Usage is direct import, or through
the ``--min-vegetation-frac`` flag in ``dump_test_probs.py``.

Example::

    from vegetation_filter import filter_tiles_by_vegetation
    kept_tiles, kept_ix = filter_tiles_by_vegetation(
        tiles, min_frac=0.15, exg_thresh=20
    )

Tuning:
- ``exg_thresh=20`` keeps green leaves and most yellow-green grass; raise to
  30 to exclude stressed/dry vegetation, lower to 10 for low-light photos.
- ``min_frac=0.15`` (15% of tile pixels must be vegetation) is the empirical
  sweet spot on PC24 holdouts — higher drops flower-only tiles, lower lets
  through pure-dirt tiles.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def exg_vegetation_mask(
    image: Image.Image | np.ndarray,
    exg_thresh: float = 20.0,
) -> np.ndarray:
    """Return a (H, W) bool mask marking vegetation pixels via the Excess Green index.

    ExG = 2*G - R - B, with an extra G > R AND G > B guard that rejects bright
    non-green patches (concrete, sky, skin) that happen to exceed the threshold.
    """
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    else:
        arr = np.asarray(image, dtype=np.int16)
        if arr.ndim == 3 and arr.shape[0] == 3:  # CHW
            arr = arr.transpose(1, 2, 0)
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    exg = 2 * g - r - b
    return (exg > exg_thresh) & (g > r) & (g > b)


def vegetation_fraction(
    image: Image.Image | np.ndarray, exg_thresh: float = 20.0
) -> float:
    """Fraction of image pixels classified as vegetation by Excess Green."""
    m = exg_vegetation_mask(image, exg_thresh=exg_thresh)
    return float(m.mean())


def filter_tiles_by_vegetation(
    tiles: Sequence[Image.Image],
    min_frac: float = 0.15,
    exg_thresh: float = 20.0,
) -> tuple[list[Image.Image], list[int]]:
    """Keep tiles whose vegetation fraction ≥ ``min_frac``.

    Returns (kept_tiles, original_indices). If every tile is below the
    threshold (e.g. a blurry / greyscale image), returns the top-K tiles by
    vegetation fraction with K = max(1, len(tiles)//4) — we'd rather score
    the *least bad* tiles than collapse to an empty prediction.
    """
    if not tiles:
        return [], []
    fracs = [vegetation_fraction(t, exg_thresh=exg_thresh) for t in tiles]
    kept_ix = [i for i, f in enumerate(fracs) if f >= min_frac]
    if not kept_ix:
        k = max(1, len(tiles) // 4)
        kept_ix = list(np.argsort(fracs)[-k:])
        kept_ix.sort()
        logger.debug(
            f"Vegetation filter: no tile passed min_frac={min_frac}, "
            f"falling back to top-{k} by ExG coverage."
        )
    return [tiles[i] for i in kept_ix], kept_ix


__all__ = [
    "exg_vegetation_mask",
    "vegetation_fraction",
    "filter_tiles_by_vegetation",
]
