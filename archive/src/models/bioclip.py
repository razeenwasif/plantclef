import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip 

class PlantBioCLIP(nn.Module):
    def __init__(self, num_classes=7800, checkpoint='hf-hub:imageomics/bioclip', input_res=448):
        super(PlantBioCLIP, self).__init__()

        # Load BioCLIP
        print(f"[BioCLIP] Loading checkpoint: {checkpoint}")
        self.model, _, _ = open_clip.create_model_and_transforms(checkpoint)
        self.backbone = self.model.visual 
        
        # Dynamically determine feature dimension
        if hasattr(self.backbone, 'output_dim'):
            self.feature_dim = self.backbone.output_dim
        elif hasattr(self.backbone, 'proj') and self.backbone.proj is not None:
            self.feature_dim = self.backbone.proj.shape[1]
        else:
            self.feature_dim = 768

        self.input_res = input_res

        # Handle Positional Embedding Interpolation
        self._interpolate_pos_embeddings(input_res)

    def _interpolate_pos_embeddings(self, input_res):
        """
        Interpolates the positional embeddings of the Vision Transformer (ViT) backbone 
        to match a new input resolution.

        This allows the model to process images at a resolution different from its 
        pre-trained resolution (typically 224x224) by scaling the patch-based 
        positional embeddings using bicubic interpolation while preserving the 
        class token embedding.

        Args:
            input_res (int): The target input resolution (e.g., 448).
        """
        pos_embed = self.backbone.positional_embedding
        if pos_embed.dim() == 3:
            pos_embed = pos_embed.squeeze(0)
            
        old_pos_embed = pos_embed
        D = old_pos_embed.shape[-1]
        num_extra_tokens = 1 
        class_token_embed = old_pos_embed[:num_extra_tokens]
        patch_embeds = old_pos_embed[num_extra_tokens:]
        
        old_grid_size = int(patch_embeds.shape[0]**0.5)
        original_res = 224
        patch_size = original_res // old_grid_size
        new_grid_size = input_res // patch_size
        
        if old_grid_size != new_grid_size:
            print(f"[BioCLIP] Interpolating pos_embed: {old_grid_size} -> {new_grid_size}")
            patch_embeds = patch_embeds.reshape(1, old_grid_size, old_grid_size, D).permute(0, 3, 1, 2)
            patch_embeds = F.interpolate(patch_embeds, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
            patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(-1, D)
            
            new_pos_embed = torch.cat([class_token_embed, patch_embeds], dim=0)
            if self.backbone.positional_embedding.dim() == 3:
                new_pos_embed = new_pos_embed.unsqueeze(0)
                
            self.backbone.positional_embedding = nn.Parameter(new_pos_embed)

    def forward(self, x):
        return self.backbone(x)
