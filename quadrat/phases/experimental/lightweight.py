import torch
import torch.nn as nn
import timm
from typing import Optional
import contextlib

# plantclef: Robust SDPA Kernel Fallback & Context Wrapper (Blackwell Optimized)
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend
    _NEW_SDPA = True
except ImportError:
    _sdpa_kernel = None
    _NEW_SDPA = False

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False

def get_sdpa_context():
    """Returns a normalized context manager for SDPA backends."""
    if _sdpa_kernel is None:
        return contextlib.nullcontext()
    
    # Enable optimized backends for Blackwell/RTX 6000
    backends = []
    for b in ["FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION"]:
        if hasattr(SDPBackend, b):
            backends.append(getattr(SDPBackend, b))
    
    return _sdpa_kernel(backends) if backends else contextlib.nullcontext()

class LightweightBioCLIP(nn.Module):
    """
    Super lightweight BioCLIP2 variant.
    Uses ViT-Tiny (5.7M params) as the backbone.
    Optimized for Blackwell SDPA and FP8 (if available).
    """
    def __init__(self, num_classes: int = 7808, input_res: int = 224):
        super().__init__()
        print(f"[LightweightBioCLIP] Initializing with ViT-Tiny backbone at {input_res}px")
        
        # Pass img_size to ensure patch_embed matches input_res
        self.backbone = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, img_size=input_res)
        self.feature_dim = self.backbone.num_features # 192
        
        # Apply stochastic depth (10%)
        from timm.layers import DropPath
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks:
                block.drop_path = DropPath(0.1)

        # plantclef: Use TransformerEngine for the wide classification head (7.8k classes)
        if HAS_TE and torch.cuda.is_available():
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Dropout(0.3),
                te.Linear(self.feature_dim, num_classes, params_dtype=torch.bfloat16)
            )
            print("[Optim] Using TransformerEngine for classification head.")
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with get_sdpa_context():
            features = self.backbone(x)
            return self.classifier(features)


class LightweightDINOv3(nn.Module):
    """
    Super lightweight DINOv3 variant.
    Uses ViT-Small (22M params) with DINOv2 weights.
    """
    def __init__(self, num_classes: int = 7808, input_res: int = 224):
        super().__init__()
        # patch_size=16 is what makes arbitrary resolutions work cleanly. Patch-14
        # (DINOv2) silently ignores img_size when it's not a multiple of 14 and
        # falls back to its default 518, which then asserts on the first batch.
        assert input_res % 16 == 0, f"input_res={input_res} must be divisible by 16 (DINOv3 patch size)"
        print(f"[LightweightDINOv3] Initializing with ViT-Small (DINOv3) backbone at {input_res}px")

        self.backbone = timm.create_model(
            'vit_small_patch16_dinov3', pretrained=True, num_classes=0, img_size=input_res
        )
        self.feature_dim = self.backbone.num_features  # 384 for ViT-Small
        
        if HAS_TE and torch.cuda.is_available():
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Dropout(0.3),
                te.Linear(self.feature_dim, num_classes, params_dtype=torch.bfloat16)
            )
            print("[Optim] Using TransformerEngine for classification head.")
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with get_sdpa_context():
            features = self.backbone(x)
            return self.classifier(features)

def get_lightweight_model(model_type: str = 'bioclip', num_classes: int = 7808, input_res: int = 224) -> nn.Module:
    if model_type.lower() == 'bioclip':
        return LightweightBioCLIP(num_classes=num_classes, input_res=input_res)
    elif model_type.lower() == 'dinov3':
        return LightweightDINOv3(num_classes=num_classes, input_res=input_res)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
