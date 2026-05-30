"""
Model loader abstraction for BioCLIP variants.

Supports:
  - BioCLIP 1   : hf-hub:imageomics/bioclip
  - BioCLIP 2   : hf-hub:imageomics/bioclip-2
  - BioCLIP 2.5 : hf-hub:imageomics/bioclip-2.5-vith14

Shorthand aliases (e.g. "bioclip", "bioclip-2", "bioclip-2.5") are accepted.
"""

from __future__ import annotations

import sys

import torch
import open_clip


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Shorthand -> full HF-hub path
_SHORTHAND: dict[str, str] = {
    "bioclip":   "hf-hub:imageomics/bioclip",
    "bioclip-2": "hf-hub:imageomics/bioclip-2",
    "bioclip-2.5": "hf-hub:imageomics/bioclip-2.5-vith14",
}

# Full names supported as-is
_KNOWN_FULL: set[str] = {
    "hf-hub:imageomics/bioclip",
    "hf-hub:imageomics/bioclip-2",
    "hf-hub:imageomics/bioclip-2.5-vith14",
}

# Default image-tile batch sizes for ~16 GB VRAM
_DEFAULT_BATCH: dict[str, int] = {
    "hf-hub:imageomics/bioclip":            64,
    "hf-hub:imageomics/bioclip-2":          64,
    "hf-hub:imageomics/bioclip-2.5-vith14": 16,
}

ALL_KNOWN_NAMES: list[str] = sorted(_SHORTHAND.keys()) + sorted(_KNOWN_FULL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_model_name(name: str) -> str:
    """Resolve a shorthand alias or full path to the canonical HF-hub identifier."""
    return _SHORTHAND.get(name, name)


def load_model(model_name: str, device: str):
    """
    Load an OpenCLIP model, its preprocessing transform, and its tokenizer.

    Parameters
    ----------
    model_name : str
        Short alias (e.g. "bioclip-2") or full HF-hub path.
    device : str
        Torch device string (e.g. "cuda", "cpu").

    Returns
    -------
    model     : open_clip model in eval mode on `device`
    transform : image preprocessing callable
    tokenizer : model-matched text tokenizer
    """
    full_name = resolve_model_name(model_name)
    print(f"  Loading model '{full_name}' on {device} ...")
    try:
        model, _, transform = open_clip.create_model_and_transforms(full_name)
        tokenizer = open_clip.get_tokenizer(full_name)
    except Exception as exc:
        known = "\n    ".join(ALL_KNOWN_NAMES)
        sys.exit(
            f"ERROR: failed to load model '{full_name}'.\n"
            f"  Reason: {exc}\n"
            f"  Known model names:\n    {known}"
        )
    model = model.to(device)
    model.eval()
    return model, transform, tokenizer


def default_batch_size(model_name: str) -> int:
    """Return the recommended image-tile batch size for a given model."""
    full_name = resolve_model_name(model_name)
    return _DEFAULT_BATCH.get(full_name, 64)


def resolve_device(device_str: str) -> str:
    """Resolve 'auto' to 'cuda' if available, otherwise 'cpu'."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str
