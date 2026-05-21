"""Core dataclasses for the inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TileSpec:
    """Spatial metadata for a single tile cut from a larger image.

    Coordinates are in the original image's pixel space (before any scaling).
    """

    tile_id: int
    image_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    scale: float = 1.0  # downscale factor applied before cropping (1.0 = native)
    veg_weight: float = 1.0  # Vegetation weight based on ExG scoring
    noise_fraction: float = 0.0  # Fraction of tile covered by SAM noise masks

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def __repr__(self) -> str:
        return (
            f"TileSpec(id={self.tile_id}, image={self.image_id!r}, "
            f"box=({self.x0},{self.y0},{self.x1},{self.y1}), scale={self.scale})"
        )


@dataclass
class TilePrediction:
    """Model output for a single image tile.

    Attributes:
        image_id: Identifier of the source image (e.g. quadrat_id).
        tile_id: Integer tile index within the image.
        tile_spec: Spatial metadata for this tile.
        logits: Raw model output, shape ``(num_classes,)``.
        probs: Sigmoid-activated probabilities, shape ``(num_classes,)``.
    """

    image_id: str
    tile_id: int
    tile_spec: TileSpec
    logits: np.ndarray  # (C,)
    probs: np.ndarray   # (C,)
    veg_weight: float = 1.0  # Vegetation weight based on ExG scoring

    def top_k(self, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-k class indices and their probabilities."""
        indices = np.argpartition(self.probs, -k)[-k:]
        indices = indices[np.argsort(self.probs[indices])[::-1]]
        return indices, self.probs[indices]


@dataclass
class ImagePrediction:
    """Aggregated prediction for one image after tile pooling.

    Attributes:
        image_id: Identifier of the source image.
        class_scores: Aggregated probability score per class, shape ``(num_classes,)``.
        predicted_class_indices: Class indices selected after postprocessing.
        predicted_scores: Corresponding scores for the selected classes.
    """

    image_id: str
    class_scores: np.ndarray       # (C,)
    predicted_class_indices: list[int] = field(default_factory=list)
    predicted_scores: list[float] = field(default_factory=list)

    @property
    def num_predictions(self) -> int:
        return len(self.predicted_class_indices)

    def top_k(self, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-k class indices and scores from the raw class_scores."""
        k = min(k, len(self.class_scores))
        indices = np.argpartition(self.class_scores, -k)[-k:]
        indices = indices[np.argsort(self.class_scores[indices])[::-1]]
        return indices, self.class_scores[indices]
