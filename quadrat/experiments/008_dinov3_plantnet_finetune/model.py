"""
DINOv3 PlantCLEF full fine-tune model — PlantNet-style recipe.

Two heads ship here:

  A) ``DINOv3SinglePlantClassifier`` — ViT-L/16 DINOv3 backbone + a *single*
     ``Linear(1024, 7806)`` classifier, with a thin ``LayerNorm`` on top of the
     CLS token. This intentionally mirrors PlantNet 2024's recipe: full fine-
     tune pushes taxonomic knowledge into the backbone itself, so the head is
     just a readout. Used for Phase A (1.4M single-plant CE pretrain).

  B) For Phase B we swap to 005's ``DINOv3MultiLabel`` (CLS + GeM patches ->
     wider MLP head, sigmoid), loading the Phase-A backbone into its own
     ``self.backbone`` before applying LoRA + ASL on collages. That logic
     lives in ``train_phase_b.py``.
"""
from __future__ import annotations

import importlib.util as _ilu
import logging
from pathlib import Path

import torch
import torch.nn as nn

# Reuse 005's DINOv3 wrapper + helpers without touching sys.path (that would
# shadow this module's own ``model`` name and cause a circular import).
_FIVE_MODEL = Path(__file__).resolve().parent.parent / "005_dinov3_multilabel" / "model.py"
_spec = _ilu.spec_from_file_location("dinov3_model_005", str(_FIVE_MODEL))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DINOv3MultiLabel = _mod.DINOv3MultiLabel
DEFAULT_EMBED_DIM = _mod.DEFAULT_EMBED_DIM
DEFAULT_TIMM_NAME = _mod.DEFAULT_TIMM_NAME
N_PREFIX_TOKENS_TIMM = _mod.N_PREFIX_TOKENS_TIMM
_load_timm_backbone = _mod._load_timm_backbone
build_default_transform = _mod.build_default_transform

logger = logging.getLogger(__name__)


class DINOv3SinglePlantClassifier(nn.Module):
    """DINOv3-L backbone + LayerNorm(CLS) + Linear(1024, n_classes).

    Design notes
    ------------
    * No GeM branch. Single-plant images are subject-centered; CLS alone gives
      a strong global descriptor, and the full-FT gradient flows into the
      backbone's attention (which is where taxonomic structure lives).
    * Head initialised with small weights so the pretrained DINOv3 features
      dominate the first forward pass and LR schedules remain stable.
    """

    def __init__(
        self,
        n_classes: int = 7806,
        backbone_name: str = DEFAULT_TIMM_NAME,
        pretrained: bool = True,
        embed_dim: int = DEFAULT_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.backbone = _load_timm_backbone(backbone_name, pretrained=pretrained)
        self.backbone_name = backbone_name
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        logger.info(
            f"DINOv3SinglePlantClassifier: {backbone_name}, "
            f"embed_dim={embed_dim}, n_classes={n_classes}"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(pixel_values)  # (B, 1+4+P, D)
        cls = tokens[:, 0, :]
        return self.head(self.norm(cls))

    def head_parameters(self):
        yield from self.norm.parameters()
        yield from self.head.parameters()

    def backbone_parameters(self):
        yield from self.backbone.parameters()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


def build_train_transform(resolution: int = 224):
    """Standard single-plant CE augmentation (torchvision CPU pipeline)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.RandomResizedCrop(
            resolution,
            scale=(0.65, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


__all__ = [
    "DINOv3SinglePlantClassifier",
    "DINOv3MultiLabel",
    "N_PREFIX_TOKENS_TIMM",
    "DEFAULT_TIMM_NAME",
    "build_default_transform",
    "build_train_transform",
]
