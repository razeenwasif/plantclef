import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional

class PlantDINOv2(nn.Module):
    """
    DINOv2 model for plant classification.

    This model uses a DINOv2 backbone from the `timm` library, which is a
    self-supervised Vision Transformer (ViT) pre-trained on large-scale data.

    Parameters
    ----------
    model_name : str, optional
        Name of the DINOv2 model architecture to load from timm
        (default is 'vit_large_patch14_dinov2.lvd142m').
    input_res : int, optional
        Target input resolution for the model (default is 448).
    """

    def __init__(self, model_name: str = 'vit_large_patch14_dinov2.lvd142m', input_res: int = 448) -> None:
        super(PlantDINOv2, self).__init__()
        
        # Load DINOv2 from timm
        # vit_large_patch14_dinov2 is natively 1024-dim
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
        """
        self.backbone.set_grad_checkpointing(enable)
        print(f"[DINOv2] Full Gradient Checkpointing {'enabled' if enable else 'disabled'}.")

    def set_resolution(self, input_res: int) -> None:
        """
        Updates the DINOv2 positional embeddings for a new input resolution.
        """
        # Reach through PeftModel wrapper if LoRA is applied
        base_model = self.backbone.base_model.model if hasattr(self.backbone, "base_model") else self.backbone
        
        pos_embed = base_model.pos_embed
        num_extra_tokens = 1 # [CLS] token
        
        old_pos_embed = pos_embed.detach()
        D = old_pos_embed.shape[-1]
        cls_token = old_pos_embed[:, :num_extra_tokens]
        patch_embeds = old_pos_embed[:, num_extra_tokens:]
        
        old_grid_size = int(patch_embeds.shape[1]**0.5)
        patch_size = 14 # DINOv2 default
        new_grid_size = input_res // patch_size
        
        if old_grid_size != new_grid_size:
            print(f"[DINOv2] Interpolating pos_embed: {old_grid_size} -> {new_grid_size}")
            patch_embeds = patch_embeds.reshape(1, old_grid_size, old_grid_size, D).permute(0, 3, 1, 2)
            patch_embeds = F.interpolate(patch_embeds, size=(new_grid_size, new_grid_size), 
                                         mode='bicubic', align_corners=False)
            patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(1, -1, D)
            
            new_pos_embed = torch.cat([cls_token, patch_embeds], dim=1)
            
            # Re-register on the base model to avoid PeftModel attribute errors
            base_model.pos_embed = nn.Parameter(
                new_pos_embed.to(device=pos_embed.device, dtype=pos_embed.dtype)
            )
            
            if hasattr(base_model.patch_embed, 'img_size'):
                base_model.patch_embed.img_size = (input_res, input_res)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DINOv2 model.

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
