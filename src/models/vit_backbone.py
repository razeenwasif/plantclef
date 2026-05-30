import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
from typing import Optional
import contextlib

# plantclef: Robust SDPA Kernel Fallback & Context Wrapper
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend
    _NEW_SDPA = True
except ImportError:
    try:
        from torch.backends.cuda import sdp_kernel as _sdpa_kernel, SDPBackend
        _NEW_SDPA = False
    except ImportError:
        _sdpa_kernel = None
        SDPBackend = None
        _NEW_SDPA = False

# plantclef: Determine available backends once
SDP_BACKENDS = []
if SDPBackend is not None:
    for b in ["FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION"]:
        if hasattr(SDPBackend, b):
            SDP_BACKENDS.append(getattr(SDPBackend, b))
    if not SDP_BACKENDS and hasattr(SDPBackend, "MATH"):
        SDP_BACKENDS.append(SDPBackend.MATH)

def get_sdpa_context():
    """Returns a normalized context manager for SDPA backends."""
    if _sdpa_kernel is None or not SDP_BACKENDS:
        return contextlib.nullcontext()
    
    if _NEW_SDPA:
        # Newer PyTorch 2.x signature: sdpa_kernel(List[SDPBackend])
        return _sdpa_kernel(SDP_BACKENDS)
    else:
        # Older PyTorch 2.x signature: sdp_kernel(enable_flash, enable_math, enable_mem_efficient)
        has_flash = any("FLASH" in str(b) for b in SDP_BACKENDS)
        has_mem = any("EFFICIENT" in str(b) or "MEM" in str(b) for b in SDP_BACKENDS)
        has_math = any("MATH" in str(b) for b in SDP_BACKENDS)
        return _sdpa_kernel(enable_flash=has_flash, enable_math=has_math, enable_mem_efficient=has_mem)

def apply_fa4_monkeypatch(model: nn.Module):
    """Surgically injects optimized SDPA into transformer blocks."""
    import types
    for name, module in model.named_modules():
        if hasattr(module, "num_heads") and hasattr(module, "qkv") and hasattr(module, "forward"):
            original_forward = module.forward
            def sdpa_forward(self, x, *args, **kwargs):
                if x.dim() == 3:
                    B, L, C = x.shape
                    H = self.num_heads
                    D = C // H
                    qkv = self.qkv(x).reshape(B, L, 3, H, D)
                    q, k, v = qkv.unbind(2)
                    
                    # Force Blackwell-optimized kernels (FA2 or cuDNN)
                    # This bypasses the slower 'math' kernel and ensures peak throughput.
                    with get_sdpa_context():
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
                        )
                    
                    out = out.reshape(B, L, C)
                    return self.proj_drop(self.proj(out))
                return original_forward(x, *args, **kwargs)
            
            module.forward = types.MethodType(sdpa_forward, module)
            if "attn" in name.lower():
                print(f"[Optim] Stable Blackwell SDPA Active for: {name}")

class PlantViTBackbone(nn.Module):
    def __init__(self, model_name: str = 'vit_large_patch16_dinov3.lvd1689m', input_res: int = 384) -> None:
        super(PlantViTBackbone, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=input_res)
        apply_fa4_monkeypatch(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def set_resolution(self, input_res: int) -> None:
        """Switch input resolution. Updates patch_embed metadata AND interpolates
        positional embedding to match the new patch grid (cls token preserved)."""
        if hasattr(self.backbone, 'patch_embed'):
            old_img_size = self.backbone.patch_embed.img_size
            self.backbone.patch_embed.img_size = (input_res, input_res)
            # Recompute grid metadata if timm exposes it
            patch_size = self.backbone.patch_embed.patch_size
            ps = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
            new_grid = input_res // ps
            if hasattr(self.backbone.patch_embed, 'grid_size'):
                self.backbone.patch_embed.grid_size = (new_grid, new_grid)
            if hasattr(self.backbone.patch_embed, 'num_patches'):
                self.backbone.patch_embed.num_patches = new_grid * new_grid

        # Interpolate pos_embed to match the new patch count (cls token kept).
        if not hasattr(self.backbone, 'pos_embed') or self.backbone.pos_embed is None:
            return
        pos = self.backbone.pos_embed                  # [1, N+1, D] or [1, N, D]
        old_n_total = pos.shape[1]
        D           = pos.shape[2]
        target_grid = new_grid
        target_n    = target_grid * target_grid

        # Detect cls/extra tokens by comparing total to a square + 0/1.
        old_grid_sq = int(round((old_n_total) ** 0.5))
        if old_grid_sq * old_grid_sq == old_n_total:
            num_extra = 0
        else:
            num_extra = old_n_total - int(round(((old_n_total - 1)) ** 0.5)) ** 2

        if old_n_total - num_extra == target_n:
            return  # already correct

        old_grid = int(round((old_n_total - num_extra) ** 0.5))
        extra    = pos[:, :num_extra, :]
        patches  = pos[:, num_extra:, :]               # [1, old_grid², D]
        patches_2d = patches.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
        patches_2d = torch.nn.functional.interpolate(
            patches_2d.float(), size=(target_grid, target_grid),
            mode='bicubic', align_corners=False, antialias=True,
        ).to(pos.dtype)
        new_patches = patches_2d.permute(0, 2, 3, 1).reshape(1, target_n, D)
        new_pos = torch.cat([extra, new_patches], dim=1) if num_extra else new_patches
        self.backbone.pos_embed = nn.Parameter(new_pos.contiguous(), requires_grad=pos.requires_grad)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Toggles gradient checkpointing on the underlying timm model."""
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)
