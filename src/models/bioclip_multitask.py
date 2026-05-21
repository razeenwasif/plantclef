"""
BioCLIP 2.5 Multi-Task Architectures.

Includes:
  - Shared MLP variant (010)
  - Per-head MLP variant (i002)
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

BIOCLIP25_MODEL_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"


class HeadMLP(nn.Module):
    """LayerNorm → Linear → GELU → Dropout."""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BioCLIP25MultiTask(nn.Module):
    """
    BioCLIP 2.5 backbone with multi-task heads.
    Supports both shared-MLP (010) and per-head MLP (i002) architectures.
    """

    def __init__(
        self,
        num_species: int,
        num_genus:   int = 0,
        num_family:  int = 0,
        num_order:   int = 0,
        num_class:   int = 0,
        model_name:  str = BIOCLIP25_MODEL_NAME,
        hidden_dim:  int = 1024,
        dropout:     float = 0.2,
        use_taxonomy_heads: bool = True,
        shared_mlp:  bool = True,
    ) -> None:
        super().__init__()
        self.shared_mlp_mode = shared_mlp
        
        import open_clip
        logger.info(f"Loading BioCLIP 2.5: {model_name} (shared_mlp={shared_mlp})")
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.backbone = clip_model

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        embed_dim = self._probe_embed_dim()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        if shared_mlp:
            self.shared_mlp = HeadMLP(embed_dim, hidden_dim, dropout)
            self.species_mlp = None
            self.genus_mlp = None
            self.family_mlp = None
            self.order_mlp = None
            self.class_mlp = None
        else:
            self.shared_mlp = None
            self.species_mlp = HeadMLP(embed_dim, hidden_dim, dropout)
            self.genus_mlp = HeadMLP(embed_dim, hidden_dim, dropout) if (use_taxonomy_heads and num_genus > 0) else None
            self.family_mlp = HeadMLP(embed_dim, hidden_dim, dropout) if (use_taxonomy_heads and num_family > 0) else None
            self.order_mlp = HeadMLP(embed_dim, hidden_dim, dropout) if (use_taxonomy_heads and num_order > 0) else None
            self.class_mlp = HeadMLP(embed_dim, hidden_dim, dropout) if (use_taxonomy_heads and num_class > 0) else None

        self.species_head = nn.Linear(hidden_dim, num_species)
        
        use_aux = use_taxonomy_heads
        self.genus_head = nn.Linear(hidden_dim, num_genus) if (use_aux and num_genus > 0) else None
        self.family_head = nn.Linear(hidden_dim, num_family) if (use_aux and num_family > 0) else None
        self.order_head = nn.Linear(hidden_dim, num_order) if (use_aux and num_order > 0) else None
        self.class_head = nn.Linear(hidden_dim, num_class) if (use_aux and num_class > 0) else None

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

    def forward(self, x: torch.Tensor):
        feat = self._encode_raw(x)
        
        if self.shared_mlp_mode:
            hidden = self.shared_mlp(feat)
            return (
                self.species_head(hidden),
                self.genus_head(hidden) if self.genus_head else None,
                self.family_head(hidden) if self.family_head else None,
                self.order_head(hidden) if self.order_head else None,
                self.class_head(hidden) if self.class_head else None,
            )
        else:
            return (
                self.species_head(self.species_mlp(feat)),
                self.genus_head(self.genus_mlp(feat)) if self.genus_head else None,
                self.family_head(self.family_mlp(feat)) if self.family_head else None,
                self.order_head(self.order_mlp(feat)) if self.order_mlp else None,
                self.class_head(self.class_mlp(feat)) if self.class_mlp else None,
            )

    def configure_backbone(self, mode: str, n_blocks: int = 0) -> None:
        if mode == "freeze":
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            return
        if mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            return
        if mode == "last_n":
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            visual = self.backbone.visual
            resblocks = visual.transformer.resblocks
            n_total = len(resblocks)
            n_unfreeze = min(n_blocks, n_total)
            for block in resblocks[n_total - n_unfreeze:]:
                for p in block.parameters():
                    p.requires_grad_(True)
            for attr in ("ln_post", "proj"):
                obj = getattr(visual, attr, None)
                if obj is not None:
                    if isinstance(obj, nn.Parameter): obj.requires_grad_(True)
                    elif isinstance(obj, nn.Module):
                        for p in obj.parameters(): p.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if mode:
            for module in self.backbone.modules():
                if any(p.requires_grad for p in module.parameters(recurse=False)):
                    module.train(True)
        return self


def detect_arch(state_dict_keys) -> str:
    keys = list(state_dict_keys)
    has_shared = any(k.startswith("shared_mlp.") for k in keys)
    has_species_mlp = any(k.startswith("species_mlp.") for k in keys)
    if has_species_mlp: return "i002"
    if has_shared: return "010"
    raise ValueError("Cannot detect architecture from state_dict keys")


def load_checkpoint_model(checkpoint_path: str, device: str = "cpu"):
    p = Path(checkpoint_path)
    if not p.exists(): raise FileNotFoundError(f"Checkpoint not found: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    
    if all(k.startswith("module.") for k in state_dict):
        peek_keys = [k[len("module."):] for k in state_dict]
    else:
        peek_keys = list(state_dict)
    
    arch = detect_arch(peek_keys)
    logger.info(f"Detected architecture: {arch}")
    
    config = ckpt.get("config", {})
    encoders = ckpt.get("encoders", {})
    
    model = BioCLIP25MultiTask(
        num_species = len(encoders.get("idx_to_species", [])),
        num_genus   = len(encoders.get("idx_to_genus", [])),
        num_family  = len(encoders.get("idx_to_family", [])),
        num_order   = len(encoders.get("idx_to_order", [])),
        num_class   = len(encoders.get("idx_to_class", [])),
        model_name  = config.get("model_name", BIOCLIP25_MODEL_NAME),
        hidden_dim  = config.get("hidden_dim", 1024),
        dropout     = config.get("dropout", 0.2),
        use_taxonomy_heads = config.get("use_taxonomy_heads", True),
        shared_mlp = (arch == "010"),
    )
    
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, encoders, config
