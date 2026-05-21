"""
BioCLIP 2.5 linear probe model.

Architecture
------------
  frozen BioCLIP 2.5 ViT-H/14 image encoder
  + trainable nn.Linear(embed_dim, num_classes) head

The backbone is always kept in eval mode and produces no gradients.
Only the linear head is optimized during training.

Embedding dimension is probed dynamically at construction time so we do not
hard-code an assumption about the CLIP projection size.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BIOCLIP25_MODEL_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"


class BioCLIP25LinearProbe(nn.Module):
    """
    Frozen BioCLIP 2.5 image encoder + trainable linear species classifier.

    Parameters
    ----------
    num_classes : number of species classes (7,806 for PlantCLEF 2026)
    model_name  : OpenCLIP hub model identifier
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = BIOCLIP25_MODEL_NAME,
    ) -> None:
        super().__init__()

        import open_clip
        logger.info(f"Loading BioCLIP model: {model_name}")
        clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name)

        self.backbone = clip_model   # full OpenCLIP CLIP model
        self.preprocess = preprocess  # official val preprocessing (for reference)

        # Freeze every backbone parameter
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()

        # Dynamically probe output embedding dimension
        embed_dim = self._probe_embed_dim()
        logger.info(
            f"BioCLIP 2.5 embed_dim={embed_dim}, num_classes={num_classes:,}, "
            f"head_params={embed_dim * num_classes + num_classes:,}"
        )

        self.embed_dim = embed_dim
        self.head = nn.Linear(embed_dim, num_classes)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _probe_embed_dim(self) -> int:
        """Forward a dummy image to discover the output embedding dimension."""
        device = next(self.backbone.parameters()).device
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            feat = self._encode_raw(dummy)
        return feat.shape[-1]

    def _encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Call encode_image without L2 normalisation."""
        try:
            return self.backbone.encode_image(x, normalize=False)
        except TypeError:
            # Older OpenCLIP builds may not accept the normalize kwarg
            return self.backbone.encode_image(x)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw (un-normalised) backbone features.

        Shape: (B, embed_dim)  float32
        The backbone is always run under torch.no_grad().
        """
        return self._encode_raw(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass → class logits.

        Shape: (B, num_classes)
        Backbone runs under no_grad; gradients flow only through self.head.
        """
        feat = self.encode(x)
        return self.head(feat)

    # Keep backbone permanently in eval mode regardless of model.train() calls
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_bioclip25_probe(
    checkpoint_path: str,
    device: str = "cpu",
    model_name: Optional[str] = None,
) -> tuple[BioCLIP25LinearProbe, list[str]]:
    """
    Load a trained BioCLIP25LinearProbe from a .pt checkpoint.

    Supports two checkpoint formats:
    1. Full checkpoint (normal training):
       ``model_state_dict``  - full state dict, DDP module.* prefix stripped
    2. Head-only checkpoint (cache-mode training):
       ``head_state_dict``   - only the linear head weights

    Both formats must contain ``idx_to_species`` (list of species_id strings).

    Returns (model_on_device_in_eval_mode, idx_to_species).
    """
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    logger.info(f"Loading checkpoint: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)

    idx_to_species: list[str] = ckpt["idx_to_species"]
    config = ckpt.get("config", {})
    resolved_name = model_name or config.get("model_name", BIOCLIP25_MODEL_NAME)
    num_classes = len(idx_to_species)

    logger.info(
        f"  model_name={resolved_name}, "
        f"num_classes={num_classes:,}, "
        f"epoch={ckpt.get('epoch', '?')}"
    )

    model = BioCLIP25LinearProbe(num_classes=num_classes, model_name=resolved_name)

    if "model_state_dict" in ckpt:
        # Full model checkpoint (normal or DDP training)
        state_dict = ckpt["model_state_dict"]
        if all(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logger.info("  Loaded full model state dict")
    elif "head_state_dict" in ckpt:
        # Head-only checkpoint from cache-mode training
        model.head.load_state_dict(ckpt["head_state_dict"])
        logger.info("  Loaded head-only state dict (backbone from pretrained weights)")
    else:
        raise KeyError(
            "Checkpoint must contain 'model_state_dict' or 'head_state_dict'. "
            f"Found keys: {list(ckpt.keys())}"
        )

    model = model.to(device)
    model.eval()
    logger.info(f"Model ready on {device}")
    return model, idx_to_species
