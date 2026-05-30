"""
Transforms for BioCLIP 2.5 end-to-end fine-tuning.

Training: strong plant-specific augmentations (safe, no destructive ops)
Val/Test: standard deterministic preprocessing (resize + center crop)
"""
from __future__ import annotations

import torchvision.transforms as T

# OpenAI CLIP normalisation stats — used by all BioCLIP / OpenCLIP models
BIOCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOCLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def train_transform(img_size: int = 224) -> T.Compose:
    """Strong but safe plant-specific augmentation pipeline."""
    return T.Compose([
        T.RandomResizedCrop(
            img_size,
            scale=(0.5, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD),
    ])


def val_transform(img_size: int = 224) -> T.Compose:
    """Deterministic eval transform matching OpenCLIP's official preprocessing."""
    resize_to = int(img_size * (256 / 224))
    return T.Compose([
        T.Resize(resize_to, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD),
    ])
