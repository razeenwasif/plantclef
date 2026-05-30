"""
BioCLIP 2.5 + per-head MLPs + optional taxonomy auxiliary heads (genus, family).

Architecture
------------
  backbone     : BioCLIP 2.5 ViT-H/14 (frozen by default)
  species_mlp  : LayerNorm(embed_dim) → Linear(embed_dim→hidden_dim) → GELU → Dropout
  genus_mlp    : same structure (independent weights)
  family_mlp   : same structure (independent weights)
  species_head : Linear(hidden_dim → num_species)
  genus_head   : Linear(hidden_dim → num_genus)   [optional]
  family_head  : Linear(hidden_dim → num_family)  [optional]

Each head has its own dedicated MLP so gradients do not interfere across tasks.

Fine-tuning modes (set via configure_backbone)
-----------------------------------------------
  "freeze"  : entire backbone frozen
  "last_n"  : last N transformer blocks + ln_post/proj unfrozen
  "full"    : entire backbone unfrozen
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BIOCLIP25_MODEL_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"


class HeadMLP(nn.Module):
    """LayerNorm → Linear → GELU → Dropout.  One instance per prediction head."""

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
    BioCLIP 2.5 backbone with per-head MLPs and optional taxonomy heads.

    Parameters
    ----------
    num_species         : number of species classes (required)
    num_genus           : genus head size (0 = disabled)
    num_family          : family head size (0 = disabled)
    model_name          : OpenCLIP hub model identifier
    hidden_dim          : MLP output dimension (same for all heads)
    dropout             : dropout rate in each HeadMLP
    use_taxonomy_heads  : master switch; if False genus/family heads are skipped
    """

    def __init__(
        self,
        num_species:  int,
        num_genus:    int = 0,
        num_family:   int = 0,
        model_name:   str = BIOCLIP25_MODEL_NAME,
        hidden_dim:   int = 1024,
        dropout:      float = 0.2,
        use_taxonomy_heads: bool = True,
    ) -> None:
        super().__init__()

        import open_clip
        logger.info(f"Loading BioCLIP 2.5: {model_name}")
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.backbone = clip_model

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        embed_dim = self._probe_embed_dim()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        logger.info(
            f"embed_dim={embed_dim}  hidden_dim={hidden_dim}  "
            f"num_species={num_species:,}  dropout={dropout}"
        )

        # Per-head MLPs (independent weights)
        self.species_mlp  = HeadMLP(embed_dim, hidden_dim, dropout)
        self.species_head = nn.Linear(hidden_dim, num_species)

        use_aux = use_taxonomy_heads
        self.genus_mlp   = HeadMLP(embed_dim, hidden_dim, dropout) if (use_aux and num_genus  > 0) else None
        self.genus_head  = nn.Linear(hidden_dim, num_genus)         if (use_aux and num_genus  > 0) else None
        self.family_mlp  = HeadMLP(embed_dim, hidden_dim, dropout) if (use_aux and num_family > 0) else None
        self.family_head = nn.Linear(hidden_dim, num_family)        if (use_aux and num_family > 0) else None

        logger.info(f"  species : MLP({embed_dim}→{hidden_dim}) → Linear({hidden_dim},{num_species})")
        if self.genus_head is not None:
            logger.info(f"  genus   : MLP({embed_dim}→{hidden_dim}) → Linear({hidden_dim},{num_genus})")
        if self.family_head is not None:
            logger.info(f"  family  : MLP({embed_dim}→{hidden_dim}) → Linear({hidden_dim},{num_family})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_embed_dim(self) -> int:
        device = next(self.backbone.parameters()).device
        dummy  = torch.zeros(1, 3, 224, 224, device=device)
        with torch.no_grad():
            feat = self._encode_raw(dummy)
        return feat.shape[-1]

    def _encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.backbone.encode_image(x, normalize=False)
        except TypeError:
            return self.backbone.encode_image(x)

    # ------------------------------------------------------------------
    # Backbone fine-tuning configuration
    # ------------------------------------------------------------------

    def configure_backbone(self, mode: str, n_blocks: int = 0) -> None:
        """
        Set which backbone parameters receive gradients.

        mode
        ----
        "freeze" : all backbone params frozen
        "last_n" : last n_blocks transformer blocks + ln_post/proj unfrozen
        "full"   : entire backbone unfrozen
        """
        if mode == "freeze":
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            logger.info("Backbone: fully frozen")
            return

        if mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            n = sum(p.numel() for p in self.backbone.parameters())
            logger.info(f"Backbone: fully unfrozen ({n:,} params)")
            return

        if mode == "last_n":
            for p in self.backbone.parameters():
                p.requires_grad_(False)

            visual     = self.backbone.visual
            resblocks  = visual.transformer.resblocks
            n_total    = len(resblocks)
            n_unfreeze = min(n_blocks, n_total)

            for block in resblocks[n_total - n_unfreeze:]:
                for p in block.parameters():
                    p.requires_grad_(True)

            for attr in ("ln_post", "proj"):
                obj = getattr(visual, attr, None)
                if obj is None:
                    continue
                if isinstance(obj, nn.Parameter):
                    obj.requires_grad_(True)
                elif isinstance(obj, nn.Module):
                    for p in obj.parameters():
                        p.requires_grad_(True)

            n_trainable = sum(
                p.numel() for p in self.backbone.parameters() if p.requires_grad
            )
            logger.info(
                f"Backbone: last {n_unfreeze}/{n_total} blocks + ln_post/proj unfrozen "
                f"({n_trainable:,} params)"
            )
            return

        raise ValueError(f"Unknown backbone mode {mode!r}. Use 'freeze', 'last_n', or 'full'.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Returns (species_logits, genus_logits, family_logits).
        genus_logits / family_logits are None when that head is disabled.
        """
        feat = self._encode_raw(x)   # (B, embed_dim)

        sp_logits  = self.species_head(self.species_mlp(feat))
        gen_logits = self.genus_head(self.genus_mlp(feat))   if self.genus_head  else None
        fam_logits = self.family_head(self.family_mlp(feat)) if self.family_head else None

        return sp_logits, gen_logits, fam_logits

    def train(self, mode: bool = True):
        """Keep fully-frozen backbone layers in eval mode to avoid BN/Dropout issues."""
        super().train(mode)
        self.backbone.eval()
        if mode:
            for module in self.backbone.modules():
                if any(p.requires_grad for p in module.parameters(recurse=False)):
                    module.train(True)
        return self

    # ------------------------------------------------------------------
    # Optimizer param groups
    # ------------------------------------------------------------------

    def get_param_groups(
        self,
        backbone_lr:  float,
        head_lr:      float,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """
        Build optimizer param groups.
        Backbone params (if unfrozen) use backbone_lr; all MLP+head params use head_lr.
        Only includes parameters that require gradients.
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params: list[nn.Parameter] = []
        for m in [
            self.species_mlp,  self.species_head,
            self.genus_mlp,    self.genus_head,
            self.family_mlp,   self.family_head,
        ]:
            if m is not None:
                head_params.extend(p for p in m.parameters() if p.requires_grad)

        groups: list[dict] = []
        if backbone_params:
            groups.append({
                "params":       backbone_params,
                "lr":           backbone_lr,
                "weight_decay": weight_decay,
                "name":         "backbone",
            })
            n_bb = sum(p.numel() for p in backbone_params)
            logger.info(f"  Backbone group: {n_bb:,} params @ lr={backbone_lr:.2e}")

        if head_params:
            groups.append({
                "params":       head_params,
                "lr":           head_lr,
                "weight_decay": weight_decay,
                "name":         "head",
            })
            n_hd = sum(p.numel() for p in head_params)
            logger.info(f"  Head group:     {n_hd:,} params @ lr={head_lr:.2e}")

        return groups


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple["BioCLIP25MultiTask", dict, dict]:
    """
    Load BioCLIP25MultiTask from a checkpoint.

    Returns (model_in_eval_mode, encoders, config).
    """
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    ckpt     = torch.load(p, map_location="cpu", weights_only=False)
    config   = ckpt.get("config", {})
    encoders = ckpt.get("encoders", {})

    model = BioCLIP25MultiTask(
        num_species        = len(encoders.get("idx_to_species", [])),
        num_genus          = len(encoders.get("idx_to_genus",   [])),
        num_family         = len(encoders.get("idx_to_family",  [])),
        model_name         = config.get("model_name",  BIOCLIP25_MODEL_NAME),
        hidden_dim         = config.get("hidden_dim",  1024),
        dropout            = config.get("dropout",     0.2),
        use_taxonomy_heads = config.get("use_taxonomy_heads", True),
    )

    state_dict = ckpt["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {p}  epoch={ckpt.get('epoch', '?')}")
    return model, encoders, config
