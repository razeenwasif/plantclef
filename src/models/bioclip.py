import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from typing import Optional, Union
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

class PlantBioCLIP(nn.Module):
    """
    BioCLIP model for plant classification.
    Optimized for Stable Blackwell SDPA (FA2/cuDNN).
    """

    def __init__(self, num_classes: int = 7800, checkpoint: str = 'hf-hub:imageomics/bioclip', input_res: int = 448) -> None:
        super(PlantBioCLIP, self).__init__()

        # Load BioCLIP
        print(f"[BioCLIP] Loading checkpoint: {checkpoint}")
        self.model, _, _ = open_clip.create_model_and_transforms(checkpoint)

        # Stock OpenCLIP forward path. Two earlier "PLANTCLEF" monkey-patches were
        # removed because they silently corrupted the visual transformer:
        #
        #   (1) `memory_efficient_attention` assumed seq-first input `[L, B, C]`,
        #       but BioCLIP-2.5's transformer is `batch_first=True` (input
        #       `[B, L, C]`). The first line `L, B, C = q_x.shape` swapped batch
        #       and sequence dims, scrambling Q/K/V across every block.
        #
        #   (2) `drop_path_residual_forward` referenced `self.ls1`/`self.ls2`
        #       (no underscore) and replaced them with `nn.Identity`, but the
        #       trained blocks have `ls_1`/`ls_2` LayerScale weights. The patch
        #       silently bypassed those weights for every forward.
        #
        # Combined effect: the visual transformer produced near-constant features
        # regardless of input. The 010 model was trained against stock OpenCLIP
        # so stock inference matches training exactly. PyTorch's default
        # MultiheadAttention already dispatches into SDPA on modern hardware,
        # so we lose nothing in throughput.

        self.backbone = self.model.visual
        self.tokenizer = open_clip.get_tokenizer(checkpoint)
        self.checkpoint = checkpoint
        
        if hasattr(self.backbone, 'output_dim'):
            self.feature_dim = self.backbone.output_dim
        elif hasattr(self.backbone, 'proj') and self.backbone.proj is not None:
            self.feature_dim = self.backbone.proj.shape[1]
        else:
            self.feature_dim = 768

        self.input_res = input_res
        self._interpolate_pos_embeddings(input_res)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)

    def _interpolate_pos_embeddings(self, input_res: int) -> None:
        pos_embed = self.backbone.positional_embedding
        if pos_embed.dim() == 3:
            pos_embed = pos_embed.squeeze(0)
            
        old_pos_embed = pos_embed
        D = old_pos_embed.shape[-1]
        num_extra_tokens = 1 
        class_token_embed = old_pos_embed[:num_extra_tokens]
        patch_embeds = old_pos_embed[num_extra_tokens:]
        
        old_grid_size = int(patch_embeds.shape[0]**0.5)
        if hasattr(self.backbone, 'patch_embed'):
            patch_size = self.backbone.patch_embed.patch_size[0]
        elif hasattr(self.backbone, 'conv1'):
            patch_size = self.backbone.conv1.stride[0]
        else:
            patch_size = 14 if "bioclip" in self.checkpoint else 16
            
        new_grid_size = input_res // patch_size
        
        if old_grid_size != new_grid_size:
            print(f"[BioCLIP] Interpolating pos_embed: {old_grid_size} -> {new_grid_size}")
            patch_embeds = patch_embeds.reshape(1, old_grid_size, old_grid_size, D).permute(0, 3, 1, 2)
            patch_embeds = F.interpolate(patch_embeds, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
            patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(-1, D)
            
            new_pos_embed = torch.cat([class_token_embed, patch_embeds], dim=0)
            if self.backbone.positional_embedding.dim() == 3:
                new_pos_embed = new_pos_embed.unsqueeze(0)
                
            self.backbone.positional_embedding = nn.Parameter(
                new_pos_embed.to(device=old_pos_embed.device, dtype=old_pos_embed.dtype),
                requires_grad=old_pos_embed.requires_grad,   # preserve frozen-ness
            )

    def set_resolution(self, input_res: int) -> None:
        self._interpolate_pos_embeddings(input_res)
        self.input_res = input_res

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Overrides load_state_dict to handle resolution surgery on the fly.
        """
        keys_to_check = [
            "backbone.positional_embedding",
            "model.visual.positional_embedding"
        ]
        
        for k in keys_to_check:
            if k in state_dict:
                ckpt_param = state_dict[k]
                # Try to find current shape
                current_shape = self.state_dict().get(k, ckpt_param).shape
                
                if ckpt_param.shape != current_shape:
                    print(f"[BioCLIP] StateDict Surgery: Resampling {k} "
                          f"({ckpt_param.shape} -> {current_shape})")
                    
                    old_pos = ckpt_param.clone()
                    if old_pos.dim() == 3: old_pos = old_pos.squeeze(0)
                    
                    D = old_pos.shape[-1]
                    num_extra = 1
                    class_token = old_pos[:num_extra]
                    patch_embeds = old_pos[num_extra:]
                    
                    old_grid = int(patch_embeds.shape[0]**0.5)
                    new_grid = int((current_shape[-2] - num_extra)**0.5)
                    
                    patch_embeds = patch_embeds.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
                    patch_embeds = F.interpolate(
                        patch_embeds, size=(new_grid, new_grid), 
                        mode='bicubic', align_corners=False
                    )
                    patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(-1, D)
                    
                    new_pos = torch.cat([class_token, patch_embeds], dim=0)
                    if len(current_shape) == 3:
                        new_pos = new_pos.unsqueeze(0)
                        
                    state_dict[k] = new_pos.to(dtype=ckpt_param.dtype, device=ckpt_param.device)

        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        if not return_features:
            return self.backbone(x)
        
        # Bypassing the projection head to get raw transformer output
        # visual is a VisualTransformer from OpenCLIP
        x = self.backbone.conv1(x.to(self.backbone.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.backbone.class_embedding.to(x.dtype) + torch.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.backbone.positional_embedding.to(x.dtype)
        x = self.backbone.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.backbone.transformer(x)
        x = x.permute(1, 0, 2)
        return self.backbone.ln_post(x[:, 0, :])

    def encode_text(self, text: Union[str, list], device: torch.device) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text).to(device)
        return self.model.encode_text(tokens)
