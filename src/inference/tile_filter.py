"""Lightweight tile filter for skipping low-information tiles.

Three optional, independently configurable heuristics:

1. **Variance filter** - skip tiles with near-uniform pixel values
   (e.g. sky, white paper label, blown-out regions).
2. **Laplacian-variance filter** - skip blurry / out-of-focus tiles.
3. **Vegetation filter** - skip tiles with very little green vegetation
   using the VARI index: ``(G - R) / (G + R - B + ε)``.

All heuristics are cheap (operate on a small grayscale thumbnail) and can
be individually disabled via :class:`FilterConfig`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .config import FilterConfig
from .types import TileSpec

logger = logging.getLogger(__name__)

# Thumbnail size used for all filter computations - keeps filtering fast.
_THUMB_SIZE: Tuple[int, int] = (128, 128)


class TileFilter:
    """Stateless filter that decides whether a tile should be processed.

    Parameters
    ----------
    cfg : FilterConfig
        Filter configuration.
    """

    def __init__(self, cfg: FilterConfig) -> None:
        self.cfg = cfg
        self._skipped: int = 0
        self._total: int = 0

    def should_keep(self, tile: Image.Image, spec: TileSpec) -> bool:
        """Return ``True`` if the tile contains enough information to process.

        When :attr:`FilterConfig.enabled` is ``False`` this always returns
        ``True``.

        Parameters
        ----------
    tile : Image.Image
        PIL tile image (RGB).
    spec : TileSpec
        Spatial metadata for the tile (used for logging only).

        Returns
        -------
    bool
        ``True`` to keep the tile, ``False`` to skip it.
        """
        self._total += 1

        if not self.cfg.enabled:
            return True

        arr = _to_thumb_array(tile)

        if not _passes_variance(arr, self.cfg.min_variance):
            logger.debug("Tile %d skipped: low variance.", spec.tile_id)
            self._skipped += 1
            return False

        if not _passes_laplacian(arr, self.cfg.min_laplacian_var):
            logger.debug("Tile %d skipped: blurry (low Laplacian variance).", spec.tile_id)
            self._skipped += 1
            return False

        if self.cfg.min_vegetation_ratio > 0.0:
            arr_rgb = _to_thumb_array_rgb(tile)
            passes, veg_fraction = _passes_vegetation(arr_rgb, self.cfg.min_vegetation_ratio)
            if not passes:
                logger.debug("Tile %d skipped: insufficient vegetation.", spec.tile_id)
                self._skipped += 1
                return False
            
            # ORACLE: Calculate soft weight based on teammate's report (alpha=0.5, beta=1.0)
            # but using pure ExG instead of SAM
            spec.veg_weight = min(max(0.5 + 1.0 * veg_fraction, 0.1), 2.0)

        return True

    def filter_batch(
        self,
        tiles: list[tuple[TileSpec, Image.Image]],
    ) -> list[tuple[TileSpec, Image.Image]]:
        """Filter a list of ``(spec, tile)`` pairs and return those that pass.

        Parameters
        ----------
    tiles : list[tuple[TileSpec, Image.Image]]
        Candidate tiles from :func:`~tiling.generate_tiles`.

        Returns
        -------
    list[tuple[TileSpec, Image.Image]]
        Subset of *tiles* that passed all configured filters.
        """
        kept = [(spec, img) for spec, img in tiles if self.should_keep(img, spec)]
        return kept

    @property
    def skip_rate(self) -> float:
        """Fraction of tiles skipped so far (0.0–1.0).

        Returns
        -------
        float
            The skip rate as a fraction between 0.0 and 1.0.
        """
        if self._total == 0:
            return 0.0
        return self._skipped / self._total

    def log_summary(self) -> None:
        """Log the overall skip statistics at INFO level."""
        logger.info(
            "Tile filter: kept %d / %d tiles (%.1f%% skipped).",
            self._total - self._skipped,
            self._total,
            100.0 * self.skip_rate,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_thumb_array(tile: Image.Image) -> np.ndarray:
    """Convert tile to a small grayscale float array.

    Parameters
    ----------
    tile : Image.Image
        The input tile image.

    Returns
    -------
    np.ndarray
        Grayscale thumbnail array of shape _THUMB_SIZE.
    """
    gray = tile.convert("L").resize(_THUMB_SIZE, Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32)


def _to_thumb_array_rgb(tile: Image.Image) -> np.ndarray:
    """Convert tile to a small RGB float array, channels in [0, 255].

    Parameters
    ----------
    tile : Image.Image
        The input tile image.

    Returns
    -------
    np.ndarray
        RGB thumbnail array of shape (*_THUMB_SIZE, 3).
    """
    rgb = tile.resize(_THUMB_SIZE, Image.BILINEAR).convert("RGB")
    return np.asarray(rgb, dtype=np.float32)  # (H, W, 3)


def _passes_variance(gray: np.ndarray, threshold: float) -> bool:
    """Return True if the pixel standard deviation exceeds ``threshold``.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image array.
    threshold : float
        Minimum standard deviation threshold.

    Returns
    -------
    bool
        True if the image passes the variance test.
    """
    return float(gray.std()) >= threshold


def _passes_laplacian(gray: np.ndarray, threshold: float) -> bool:
    """Return True if the Laplacian variance exceeds ``threshold``.

    The Laplacian variance is a classic sharpness / blur metric.  A simple
    finite-difference approximation is used to keep the dependency list minimal
    (no scipy / cv2 required).

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image array.
    threshold : float
        Minimum Laplacian variance threshold.

    Returns
    -------
    bool
        True if the image passes the Laplacian variance test.
    """
    # 3x3 discrete Laplacian (approximation)
    lap = (
        -gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
        + 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var()) >= threshold


def _passes_vegetation(rgb: np.ndarray, min_ratio: float) -> tuple[bool, float]:
    """Return (True/False, ExG_veg_fraction) based on Excess Green scoring.

    Uses Excess Green (ExG) index:
        ExG = 2G - R - B

    Parameters
    ----------
    rgb : np.ndarray
        Float array shaped ``(H, W, 3)``, values in [0, 255].
    min_ratio : float
        Minimum fraction of pixels with ExG > 20 to pass.

    Returns
    -------
    tuple[bool, float]
        (Passes filter, Vegetation fraction)
    """
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    
    # ExG requires normalized values or careful scaling
    # We use a simple threshold > 20 on the 0-255 scale (equivalent to teammate's ExG > 20)
    exg = 2.0 * G - R - B
    veg_fraction = float((exg > 20.0).mean())
    
    return veg_fraction >= min_ratio, veg_fraction
