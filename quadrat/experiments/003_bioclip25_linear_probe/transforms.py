"""
Transforms for BioCLIP 2.5 linear probe.

Training
--------
  RandomResizedCrop + RandomHorizontalFlip + optional ColorJitter
  + OpenAI CLIP normalisation stats (same as all OpenCLIP CLIP models)

Validation / default inference
-------------------------------
  Resize shortest side to 256 → CenterCrop 224 (bicubic by default)
  + OpenAI CLIP normalisation

Inference-time preprocessing ablations
---------------------------------------
  InferencePreprocessor wraps the base val transform with:
  - configurable resize interpolation: 'bicubic' | 'lanczos' | 'bilinear'
  - optional margin crop (fraction of shorter side, e.g. 0.05 = 5% each side)
  - optional JPEG recompression before resizing (quality 85 or 94)
  - configurable JPEG chroma subsampling (0=4:4:4, 1=4:2:2, 2=4:2:0)

The training pipeline always uses standard OpenCLIP preprocessing (bicubic,
no crop, no JPEG) unless overridden.  Ablations are inference-only.
"""

from __future__ import annotations

import io
from typing import Optional

import torch
from PIL import Image
import torchvision.transforms as T

# OpenAI CLIP normalisation - used by all OpenCLIP CLIP models including BioCLIP 2.5
BIOCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOCLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

INTERP_MODES = {
    "bicubic":  T.InterpolationMode.BICUBIC,
    "bilinear": T.InterpolationMode.BILINEAR,
    "lanczos":  T.InterpolationMode.LANCZOS,
}


# ---------------------------------------------------------------------------
# Training transform
# ---------------------------------------------------------------------------

def bioclip_train_transform(img_size: int = 224) -> T.Compose:
    """
    Moderate training augmentation for supervised linear-probe training.

    Uses bicubic interpolation to match OpenCLIP's official val preprocessing.
    Augmentation is kept light to preserve discriminative plant features.
    """
    return T.Compose([
        T.RandomResizedCrop(
            img_size,
            scale=(0.2, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD),
    ])


# ---------------------------------------------------------------------------
# Validation / standard inference transform
# ---------------------------------------------------------------------------

def bioclip_val_transform(
    img_size: int = 224,
    interpolation: str = "bicubic",
) -> T.Compose:
    """
    Deterministic eval transform matching OpenCLIP's official preprocessing.

    Resize shorter side to ~256 (14.3% margin) then center-crop to img_size.
    """
    interp = INTERP_MODES.get(interpolation, T.InterpolationMode.BICUBIC)
    resize_to = int(img_size * (256 / 224))  # 256 when img_size=224
    return T.Compose([
        T.Resize(resize_to, interpolation=interp),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=BIOCLIP_MEAN, std=BIOCLIP_STD),
    ])


# ---------------------------------------------------------------------------
# Inference-time preprocessing ablations
# ---------------------------------------------------------------------------

class InferencePreprocessor:
    """
    Configurable inference-time preprocessing pipeline.

    Supports:
      interpolation  : resize interpolation mode ('bicubic', 'lanczos', 'bilinear')
      margin_crop    : crop a border margin before any other transform.
                       Value is a fraction of the shorter image side (e.g. 0.05 → 5%).
                       0.0 = no crop (default).
      jpeg_quality   : recompress image as JPEG at this quality level before resizing.
                       None = skip (default).  Typical ablation values: 85, 94.
      jpeg_subsampling: chroma subsampling for JPEG recompression.
                       0 = 4:4:4 (no subsampling, default)
                       1 = 4:2:2
                       2 = 4:2:0 (most aggressive, standard for web JPEG)
      img_size       : final crop side length (224 for BioCLIP 2.5)

    Usage
    -----
        preproc = InferencePreprocessor(
            interpolation='lanczos',
            jpeg_quality=85,
            margin_crop=0.05,
        )
        tensor = preproc(pil_image)  # (3, 224, 224)
    """

    def __init__(
        self,
        img_size: int = 224,
        interpolation: str = "bicubic",
        margin_crop: float = 0.0,
        jpeg_quality: Optional[int] = None,
        jpeg_subsampling: int = 0,
    ) -> None:
        if interpolation not in INTERP_MODES:
            raise ValueError(
                f"Unknown interpolation '{interpolation}'. "
                f"Choose from: {list(INTERP_MODES)}"
            )
        self.img_size = img_size
        self.interpolation = interpolation
        self.margin_crop = margin_crop
        self.jpeg_quality = jpeg_quality
        self.jpeg_subsampling = jpeg_subsampling
        self._base_transform = bioclip_val_transform(
            img_size=img_size, interpolation=interpolation
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # 1. Margin crop: remove a border fraction from all four sides
        if self.margin_crop > 0.0:
            w, h = image.size
            m = int(min(w, h) * self.margin_crop)
            if m > 0:
                image = image.crop((m, m, w - m, h - m))

        # 2. JPEG recompression (simulate compressed-image distribution shift)
        if self.jpeg_quality is not None:
            buf = io.BytesIO()
            image.save(
                buf,
                format="JPEG",
                quality=self.jpeg_quality,
                subsampling=self.jpeg_subsampling,
            )
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

        # 3. Standard resize + crop + normalise
        return self._base_transform(image)

    def __repr__(self) -> str:
        return (
            f"InferencePreprocessor("
            f"img_size={self.img_size}, "
            f"interpolation={self.interpolation!r}, "
            f"margin_crop={self.margin_crop}, "
            f"jpeg_quality={self.jpeg_quality}, "
            f"jpeg_subsampling={self.jpeg_subsampling})"
        )


def build_inference_preprocessor(args) -> InferencePreprocessor:
    """
    Build an InferencePreprocessor from parsed argparse arguments.

    Expected attributes on args:
      img_size, interp, margin_crop, jpeg_quality (int or 0 = no compression)
      jpeg_subsampling (optional, default 0)
    """
    jpeg_quality = args.jpeg_quality if args.jpeg_quality > 0 else None
    return InferencePreprocessor(
        img_size=args.img_size,
        interpolation=args.interp,
        margin_crop=args.margin_crop,
        jpeg_quality=jpeg_quality,
        jpeg_subsampling=getattr(args, "jpeg_subsampling", 0),
    )
