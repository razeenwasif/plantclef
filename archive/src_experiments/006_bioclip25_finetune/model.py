"""
BioCLIP 2.5 model with optional partial backbone fine-tuning.

Architecture
------------
  BioCLIP 2.5 ViT-H/14 image encoder
  + trainable nn.Linear(embed_dim, num_classes) head

By default the entire backbone is frozen (linear probe mode).
Call unfreeze_last_n_blocks(n) after construction to unfreeze the last n
transformer blocks + ln_post/proj for partial fine-tuning.

Two-group optimizer pattern (handled in train.py)
--------------------------------------------------
  backbone params (unfrozen blocks)  →  lr * backbone_lr_scale  (e.g. 0.1×)
  head params                        →  lr  (full learning rate)
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
    BioCLIP 2.5 image encoder + linear species classifier.

    Parameters
    ----------
    num_classes    : number of species classes (7,806 for PlantCLEF 2026)
    model_name     : OpenCLIP hub model identifier
    unfreeze_blocks: number of trailing transformer blocks to unfreeze for
                     partial fine-tuning (0 = fully frozen linear probe)
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = BIOCLIP25_MODEL_NAME,
        unfreeze_blocks: int = 0,
    ) -> None:
        super().__init__()

        import open_clip
        logger.info(f"Loading BioCLIP model: {model_name}")
        clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name)

        self.backbone = clip_model
        self.preprocess = preprocess

        # Freeze every backbone parameter to start
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

        if unfreeze_blocks > 0:
            self.unfreeze_last_n_blocks(unfreeze_blocks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _probe_embed_dim(self) -> int:
        device = next(self.backbone.parameters()).device
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            feat = self._encode_raw(dummy)
        return feat.shape[-1]

    def _encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.backbone.encode_image(x, normalize=False)
        except TypeError:
            return self.backbone.encode_image(x)

    # ------------------------------------------------------------------
    # Partial fine-tuning
    # ------------------------------------------------------------------

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """
        Unfreeze the last *n* transformer blocks of the visual encoder plus
        the final layer-norm (ln_post) and projection (proj) so gradients
        can flow through them during fine-tuning.

        The rest of the backbone (patch embedding, positional embedding, and
        all earlier blocks) remains frozen.

        Call this once after construction; train.py will pick up the newly
        trainable parameters when building the optimizer.
        """
        if n <= 0:
            return

        visual = self.backbone.visual
        if not hasattr(visual, "transformer") or not hasattr(visual.transformer, "resblocks"):
            raise RuntimeError(
                "unfreeze_last_n_blocks: backbone.visual does not have the expected "
                "ViT structure (visual.transformer.resblocks). "
                "Check the OpenCLIP model variant."
            )

        resblocks = visual.transformer.resblocks
        n_total = len(resblocks)
        n_unfreeze = min(n, n_total)

        for i, block in enumerate(resblocks):
            if i >= n_total - n_unfreeze:
                for p in block.parameters():
                    p.requires_grad_(True)

        # Also unfreeze final layer-norm and projection
        for attr in ("ln_post", "proj"):
            obj = getattr(visual, attr, None)
            if obj is None:
                continue
            if isinstance(obj, nn.Parameter):
                obj.requires_grad_(True)
            elif isinstance(obj, nn.Module):
                for p in obj.parameters():
                    p.requires_grad_(True)

        n_backbone_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        logger.info(
            f"Unfroze last {n_unfreeze}/{n_total} backbone blocks + ln_post/proj  "
            f"→ {n_backbone_trainable:,} backbone params now trainable"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw (un-normalised) backbone features.  Shape: (B, embed_dim).
        Backbone always runs under no_grad at inference time; during training
        gradients flow through the unfrozen blocks via the forward() path.
        """
        return self._encode_raw(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass → class logits.  Shape: (B, num_classes)."""
        feat = self._encode_raw(x)
        return self.head(feat)

    def train(self, mode: bool = True):
        """
        Override train() so that:
          - The fully-frozen backbone stays in eval mode (avoids BatchNorm /
            Dropout issues and keeps inference deterministic for frozen layers).
          - Any modules that own trainable parameters are switched to train
            mode so dropout / BN in unfrozen blocks behave correctly.
        """
        super().train(mode)
        # Reset entire backbone to eval first
        self.backbone.eval()
        if mode:
            # Re-enable train mode for every module that owns trainable params
            for module in self.backbone.modules():
                own_trainable = any(
                    p.requires_grad
                    for p in module.parameters(recurse=False)
                )
                if own_trainable:
                    module.train(True)
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

    # At load time the backbone is always fully frozen (unfreeze_blocks=0).
    # We load weights as-is; fine-tuning config is set by train.py at resume.
    model = BioCLIP25LinearProbe(num_classes=num_classes, model_name=resolved_name)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if all(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logger.info("  Loaded full model state dict")
    elif "head_state_dict" in ckpt:
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
