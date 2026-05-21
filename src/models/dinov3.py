import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional

class PlantDINOv3(nn.Module):
    """
    DINOv3 model for plant classification (2025 Release).

    This model uses a DINOv3 backbone from the `timm` library, which is a
    self-supervised Vision Transformer (ViT) pre-trained on 1.7 billion images.

    Parameters
    ----------
    model_name : str, optional
        Name of the DINOv3 model architecture to load from timm
        (default is 'vit_large_patch16_dinov3.lvd1689m').
    input_res : int, optional
        Target input resolution for the model (default is 448).
    """

    def __init__(self, model_name: str = 'vit_large_patch16_dinov3.lvd1689m', input_res: int = 448) -> None:
        super(PlantDINOv3, self).__init__()
        
        # Load DINOv3 from timm
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0, 
            img_size=input_res
        )
        
        # Determine feature dimension
        self.feature_dim = self.backbone.num_features

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """
        Enables or disables gradient checkpointing for the backbone.
        Uses Selective Checkpointing to balance VRAM vs Compute.
        """
        if not enable:
            self.backbone.set_grad_checkpointing(False)
            return

        # Selective Checkpointing for DINOv3
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            n_blocks = len(blocks)
            for i in range(n_blocks):
                blocks[i].grad_checkpointing = (i % 3 == 0)
            self.backbone.grad_checkpointing = True
            print(f"[DINOv3] Selective Checkpointing enabled (every 3rd block of {n_blocks}).")
        else:
            self.backbone.set_grad_checkpointing(True)
            print(f"[DINOv3] Standard Checkpointing enabled (fallback).")

    def set_resolution(self, input_res: int) -> None:
        """
        Updates the DINOv3 positional embeddings for a new input resolution.
        """
        # Reach through PeftModel wrapper if LoRA is applied
        base_model = self.backbone.base_model.model if hasattr(self.backbone, "base_model") else self.backbone
        
        # Robust attribute detection for timm ViT/EVA
        pos_embed = getattr(base_model, "pos_embed", None)
        if pos_embed is None:
            if hasattr(base_model, "rope"):
                # RoPE (Rotary Embeddings) handle dynamic resolution automatically
                if hasattr(base_model.patch_embed, 'img_size'):
                    base_model.patch_embed.img_size = (input_res, input_res)
                return
            else:
                print(f"[Warning] DINOv3: 'pos_embed' not found and no RoPE detected. Skipping interpolation.")
                return

        num_extra_tokens = 1 # [CLS] token
        old_pos_embed = pos_embed.detach()
        D = old_pos_embed.shape[-1]
        cls_token = old_pos_embed[:, :num_extra_tokens]
        patch_embeds = old_pos_embed[:, num_extra_tokens:]
        
        old_grid_size = int(patch_embeds.shape[1]**0.5)
        patch_size = 16 # DINOv3 default (patch16)
        new_grid_size = input_res // patch_size
        
        if old_grid_size != new_grid_size:
            print(f"[DINOv3] Interpolating pos_embed: {old_grid_size} -> {new_grid_size}")
            patch_embeds = patch_embeds.reshape(1, old_grid_size, old_grid_size, D).permute(0, 3, 1, 2)
            patch_embeds = F.interpolate(patch_embeds, size=(new_grid_size, new_grid_size), 
                                         mode='bicubic', align_corners=False)
            patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(1, -1, D)
            
            new_pos_embed = torch.cat([cls_token, patch_embeds], dim=1)
            
            # Explicitly delete to allow re-registration with new shape
            if hasattr(base_model, 'pos_embed'):
                del base_model.pos_embed
            
            base_model.pos_embed = nn.Parameter(
                new_pos_embed.to(device=pos_embed.device, dtype=pos_embed.dtype)
            )
            
            if hasattr(base_model.patch_embed, 'img_size'):
                base_model.patch_embed.img_size = (input_res, input_res)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DINOv3 model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch_size, 3, input_res, input_res).

        Returns
        -------
        output : torch.Tensor
            Extracted features from the backbone.
        """
        return self.backbone(x)
