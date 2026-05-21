"""
Transforms for BioCLIP 2.5 end-to-end fine-tuning and SSL pre-training.

Training (supervised): strong plant-specific augmentations (safe, no destructive ops)
Val/Test            : standard deterministic preprocessing (resize + center crop)
SSL                 : two independent strongly-augmented views of each image
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


def ssl_augmentation(img_size: int = 224) -> T.Compose:
    """
    Strong augmentation pipeline for one SSL view.

    Applied independently twice per image to produce two correlated-but-distinct
    views for SimSiam/BYOL-style self-supervised learning.
    """
    kernel_size = int(0.1 * img_size) | 1  # ensure odd

    aug_list: list = [
        T.RandomResizedCrop(
            img_size,
            scale=(0.2, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=kernel_size)], p=0.3),
    ]

    # Add RandomAutocontrast or RandomSolarize when available (torchvision >= 0.12)
    if hasattr(T, "RandomAutocontrast"):
        aug_list.append(T.RandomAutocontrast(p=0.2))
    elif hasattr(T, "RandomSolarize"):
        aug_list.append(T.RandomSolarize(threshold=128, p=0.1))

    aug_list += [
        T.ToTensor(),
        T.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD),
    ]
    return T.Compose(aug_list)


class SSLTwoViewTransform:
    """
    Wraps ssl_augmentation to produce two independent augmented views.

    __call__(img) → (view1, view2)
    Both views are drawn from the same image but with different random seeds,
    providing the positive pairs required by SimSiam/BYOL.
    """

    def __init__(self, img_size: int = 224) -> None:
        self.transform = ssl_augmentation(img_size)

    def __call__(self, img):
        return self.transform(img), self.transform(img)
