"""
PlantCLEF 2026 — Inference Pipeline
=====================================
Modular tile-based inference for high-resolution vegetation quadrat images.

Typical usage::

    from inference.config import InferenceConfig, load_config
    from inference.pipeline import InferencePipeline

    cfg = load_config("configs/inference.yaml")
    pipeline = InferencePipeline(cfg)
    pipeline.run()
"""

from .types import TileSpec, TilePrediction, ImagePrediction
from .config import (
    InferenceConfig,
    TilingConfig,
    FilterConfig,
    ModelConfig,
    AggregationConfig,
    PostprocessConfig,
    load_config,
)
from .pipeline import InferencePipeline

__all__ = [
    "TileSpec",
    "TilePrediction",
    "ImagePrediction",
    "InferenceConfig",
    "TilingConfig",
    "FilterConfig",
    "ModelConfig",
    "AggregationConfig",
    "PostprocessConfig",
    "load_config",
    "InferencePipeline",
]
