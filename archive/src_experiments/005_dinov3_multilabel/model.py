"""
DINOv3 backbone wrapper + multi-label classifier head for PlantCLEF 2026.

Backbone loaders
----------------
Two paths supported, controlled by ``backbone_source``:
  - "torch_hub" (default) : facebookresearch/dinov3 via torch.hub.
                            Forward returns a dict with x_norm_clstoken and
                            x_norm_patchtokens (CLS + register tokens already
                            removed from the patch view). This is the format
                            used by 005_dinov3_vegetation_adapt's SSL pretrain.
  - "timm"                : timm.create_model('vit_large_patch16_dinov3.lvd1689m')
                            Forward returns (B, 1+4+n_patch, D); we slice past
                            the CLS + 4 register tokens to recover patches.

Architecture
------------
DINOv3 ViT-L/16
    -> CLS token            (1024-d)
    -> GeM-pooled patches   (1024-d)
    -> concat               (2048-d)
    -> LayerNorm -> Linear(2048, 1024) -> GELU -> Dropout(0.2) -> Linear(1024, n_classes)

Sigmoid output for multi-label compatibility (apply sigmoid in the loss / inference).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


N_REGISTER_TOKENS_TIMM = 4
N_PREFIX_TOKENS_TIMM = 1 + N_REGISTER_TOKENS_TIMM

DEFAULT_HUB_NAME = "dinov3_vitl16"
DEFAULT_TIMM_NAME = "vit_large_patch16_dinov3.lvd1689m"
DEFAULT_EMBED_DIM = 1024  # ViT-L


# ---------------------------------------------------------------------------
# GeM pooling
# ---------------------------------------------------------------------------

class GeM(nn.Module):
    """Generalised-mean pooling over the token dimension."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)  ->  (B, D)
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=1)
        return x.pow(1.0 / self.p)


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def _load_torch_hub_backbone(
    model_name: str,
    pretrained: bool,
    hub_repo: str = "facebookresearch/dinov3",
) -> nn.Module:
    logger.info(f"Loading DINOv3 via torch.hub: {hub_repo}/{model_name} (pretrained={pretrained})")
    try:
        return torch.hub.load(hub_repo, model_name, pretrained=pretrained, trust_repo=True)
    except Exception as exc:
        if pretrained:
            logger.warning(f"torch.hub pretrained load failed ({exc}); falling back to architecture only.")
            return torch.hub.load(hub_repo, model_name, pretrained=False, trust_repo=True)
        raise


def _load_timm_backbone(model_name: str, pretrained: bool) -> nn.Module:
    import timm
    logger.info(f"Loading DINOv3 via timm: {model_name} (pretrained={pretrained})")
    return timm.create_model(model_name, pretrained=pretrained, num_classes=0)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class DINOv3MultiLabel(nn.Module):
    """
    DINOv3 backbone + dual-pool feature extractor + multi-label classifier head.

    Parameters
    ----------
    n_classes        : number of species (PlantCLEF 2026 = 7806)
    backbone_source  : "torch_hub" or "timm"
    backbone_name    : model identifier within the chosen source
    pretrained       : load pretrained weights (ignored if ssl_checkpoint is set)
    ssl_checkpoint   : optional path to a SimCLR/SSL checkpoint produced by
                       005_dinov3_vegetation_adapt/train_ssl.py. Loaded with
                       strict=False so any minor key mismatches are tolerated.
    head_hidden      : hidden width inside the classifier head
    dropout          : dropout in the classifier head
    embed_dim        : per-token embedding dim (auto-detected if 0)
    """

    def __init__(
        self,
        n_classes: int,
        backbone_source: str = "torch_hub",
        backbone_name: Optional[str] = None,
        pretrained: bool = True,
        ssl_checkpoint: Optional[str] = None,
        head_hidden: int = 1024,
        dropout: float = 0.2,
        embed_dim: int = 0,
    ) -> None:
        super().__init__()

        self.backbone_source = backbone_source
        self.n_classes = n_classes

        if backbone_source == "torch_hub":
            name = backbone_name or DEFAULT_HUB_NAME
            self.backbone = _load_torch_hub_backbone(name, pretrained=pretrained and not ssl_checkpoint)
        elif backbone_source == "timm":
            name = backbone_name or DEFAULT_TIMM_NAME
            self.backbone = _load_timm_backbone(name, pretrained=pretrained and not ssl_checkpoint)
        else:
            raise ValueError(f"Unknown backbone_source: {backbone_source}")
        self.backbone_name = name

        if ssl_checkpoint:
            self.load_ssl_backbone(ssl_checkpoint)

        # Resolve embed_dim
        detected = (
            getattr(self.backbone, "embed_dim", None)
            or getattr(self.backbone, "num_features", None)
            or DEFAULT_EMBED_DIM
        )
        self.embed_dim = embed_dim or int(detected)

        self.gem = GeM(p=3.0)

        feat_dim = 2 * self.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

        logger.info(
            f"DINOv3MultiLabel: source={backbone_source}, name={name}, "
            f"embed_dim={self.embed_dim}, n_classes={n_classes}"
        )

    # ------------------------------------------------------------------
    # SSL checkpoint loading
    # ------------------------------------------------------------------

    def load_ssl_backbone(self, path: str | Path) -> None:
        """
        Load backbone weights from an SSL checkpoint.

        Supports two layouts:
          1. {"backbone_state_dict": {...}} — produced by train_ssl.py:338
          2. raw state_dict (for completed SSL final dumps)

        Also tolerates 'backbone.' prefixes (when the SSL model was a wrapper).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SSL checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
            sd = ckpt["backbone_state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt

        # Strip a 'backbone.' prefix if present (SimCLR wrapper layout)
        if any(k.startswith("backbone.") for k in sd.keys()):
            sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}

        result = self.backbone.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(result.unexpected_keys)
        logger.info(
            f"Loaded SSL backbone from {path}: "
            f"{n_loaded}/{len(sd)} keys matched, "
            f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}"
        )
        if result.missing_keys:
            logger.info(f"  first 5 missing: {result.missing_keys[:5]}")
        if result.unexpected_keys:
            logger.info(f"  first 5 unexpected: {result.unexpected_keys[:5]}")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Returns the 2048-d (CLS + GeM patches) feature for each input.

        Input H, W must be multiples of 16.
        """
        if self.backbone_source == "torch_hub":
            out = self.backbone.forward_features(pixel_values)
            if not isinstance(out, dict):
                raise RuntimeError(
                    f"torch.hub DINOv3 forward_features returned {type(out)}, expected dict."
                )
            cls = out["x_norm_clstoken"]              # (B, D)
            patches = out["x_norm_patchtokens"]        # (B, n_patch, D)
        else:
            tokens = self.backbone.forward_features(pixel_values)  # (B, 1+4+n_patch, D)
            if tokens.dim() != 3:
                raise RuntimeError(
                    f"timm DINOv3 forward_features returned shape {tokens.shape}, expected (B, N, D)."
                )
            cls = tokens[:, 0, :]
            patches = tokens[:, N_PREFIX_TOKENS_TIMM:, :]

        gem_patches = self.gem(patches)
        return torch.cat([cls, gem_patches], dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return raw logits (B, n_classes). Apply sigmoid downstream."""
        return self.head(self.forward_features(pixel_values))

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def head_parameters(self):
        yield from self.head.parameters()
        yield self.gem.p

    def backbone_parameters(self):
        yield from self.backbone.parameters()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def apply_lora(
        self,
        r: int = 32,
        alpha: int = 64,
        dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
    ) -> None:
        """
        Wrap backbone attention + MLP projections with LoRA via PEFT.

        Both torch.hub and timm DINOv3 use the names ``qkv, proj, fc1, fc2``,
        confirmed against the SSL backbone state_dict (e.g. blocks.0.attn.qkv,
        blocks.0.attn.proj, blocks.0.mlp.fc1, blocks.0.mlp.fc2).
        """
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["qkv", "proj", "fc1", "fc2"]

        # LoRA + gradient checkpointing don't compose well — disable first.
        if hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(False)
            except Exception:
                pass

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.backbone = get_peft_model(self.backbone, config)
        logger.info(
            f"LoRA applied to backbone: r={r}, alpha={alpha}, targets={target_modules}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_lora_target_candidates(model: nn.Module, sample_n: int = 30) -> list[str]:
    """Return a sample of nn.Linear module suffixes from the backbone."""
    backbone = model.backbone if hasattr(model, "backbone") else model
    names: list[str] = []
    for n, m in backbone.named_modules():
        if isinstance(m, nn.Linear):
            names.append(n)
    return names[:sample_n]


def build_default_transform(resolution: int = 384):
    """
    Standard CPU-side transform for inference / cached feature extraction.

    DINOv3 (both torch.hub and timm.lvd1689m) was pretrained with ImageNet stats.
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
