"""Inference Config Redirection Layer.
Redirects to the unified OracleConfig system in src/config/.
"""
from __future__ import annotations
import logging
from pathlib import Path
from src.config import (
    load_config as unified_load,
    OracleConfig, TilingConfig, FilterConfig, ModelConfig, 
    AggregationConfig, PostprocessConfig
)

# Export names for backward compatibility
InferenceConfig = OracleConfig

def load_config(config_path: str | Path) -> OracleConfig:
    """Legacy wrapper for the unified config loader."""
    return unified_load(str(config_path))

# The dataclasses are already exported from src.config.schema via src.config
# We maintain the names here so existing code doesn't break.
