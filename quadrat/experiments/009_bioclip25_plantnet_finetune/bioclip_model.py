"""
BioCLIP-2.5 ViT-H/14 PlantCLEF full fine-tune model.

Mirrors ``008/model.py:DINOv3SinglePlantClassifier`` so the rest of the pipeline
(training loop, inference dump, fusion) can swap backbones with a one-line
import change. Differences vs DINOv3:

  * Backbone is loaded via ``open_clip.create_model_and_transforms`` and we
    keep only ``model.visual`` (ViT-H/14, 632M params, output dim 1024 after
    the cross-modal projection).
  * Patch size is 14 (not 16) — image_size must be a multiple of 14.
  * Normalization stats are CLIP/BioCLIP's
    ``mean=(0.481, 0.458, 0.408), std=(0.269, 0.261, 0.276)``,
    *not* ImageNet's. Wrong stats here silently hurt convergence.
"""
from __future__ import annotations

import logging

import open_clip
import torch
import torch.nn as nn
from torchvision import transforms

logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "hf-hub:imageomics/bioclip-2.5-vith14"
DEFAULT_EMBED_DIM = 1024  # output dim of the joint vision-text projection

BIOCLIP_MEAN = (0.481, 0.458, 0.408)
BIOCLIP_STD = (0.269, 0.261, 0.276)


class BioCLIP25SinglePlantClassifier(nn.Module):
    """BioCLIP-2.5 ViT-H/14 visual tower + LayerNorm + Linear(1024, n_classes)."""

    def __init__(
        self,
        n_classes: int = 7806,
        backbone_name: str = DEFAULT_MODEL_NAME,
        pretrained: bool = True,
        embed_dim: int = DEFAULT_EMBED_DIM,
    ) -> None:
        super().__init__()
        if not pretrained:
            raise ValueError("BioCLIP-2.5 must be loaded pretrained from HF hub")
        clip_model, _, _ = open_clip.create_model_and_transforms(backbone_name)
        self.backbone = clip_model.visual
        self.backbone_name = backbone_name
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        logger.info(
            f"BioCLIP25SinglePlantClassifier: {backbone_name}, "
            f"embed_dim={embed_dim}, n_classes={n_classes}, "
            f"backbone_params={sum(p.numel() for p in self.backbone.parameters())/1e6:.1f}M"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)  # (B, 1024) after proj+ln_post
        return self.head(self.norm(feats))

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
        transforms.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def build_default_transform(resolution: int = 224):
    return transforms.Compose([
        transforms.Resize(
            resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD),
    ])


__all__ = [
    "BioCLIP25SinglePlantClassifier",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_EMBED_DIM",
    "BIOCLIP_MEAN",
    "BIOCLIP_STD",
    "build_train_transform",
    "build_default_transform",
]
