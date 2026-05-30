"""
Holy Trinity Architecture: Asymmetric Dual-Teacher DeiT.

Variants:
1. Standard: [CLS] + [DIST] (Averaged BioCLIP/DINO soft labels)
2. Elite:    [CLS] + [DIST_Bio] + [DIST_Dino] (Separate taxonomic and geometric distillation)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from functools import partial

class TrinityDeiT(VisionTransformer):
    """
    Modified Vision Transformer to support dual distillation tokens.
    Based on DeiT (Data-efficient Image Transformer).
    """
    def __init__(self, *args, variant="standard", num_classes=7806, **kwargs):
        super().__init__(*args, num_classes=num_classes, **kwargs)
        self.variant = variant
        
        # Standard DeiT has 1 dist_token. Elite needs 2.
        self.num_dist_tokens = 1 if variant == "standard" else 2
        
        # Add the extra distillation tokens
        self.dist_token = nn.Parameter(torch.zeros(1, self.num_dist_tokens, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1 + self.num_dist_tokens, self.embed_dim))
        
        # Define separate heads for each token
        if variant == "standard":
            self.head_dist = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head_dist_bio = nn.Linear(self.embed_dim, num_classes)
            self.head_dist_dino = nn.Linear(self.embed_dim, num_classes)
            # Remove standard head_dist
            if hasattr(self, 'head_dist'): del self.head_dist

        nn.init.trunc_normal_(self.dist_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        dist_token = self.dist_token.expand(x.shape[0], -1, -1)
        
        x = torch.cat((cls_token, dist_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        
        # Return the tokens [CLS, DIST_1, (DIST_2)]
        return x[:, 0], x[:, 1:1+self.num_dist_tokens]

    def forward(self, x):
        cls_t, dist_ts = self.forward_features(x)
        
        # [CLS] head is always for Ground Truth
        x_cls = self.head(cls_t)
        
        if self.variant == "standard":
            x_dist = self.head_dist(dist_ts[:, 0])
            return x_cls, x_dist
        else:
            # Elite variant: BioCLIP and DINO specific heads
            x_bio = self.head_dist_bio(dist_ts[:, 0])
            x_dino = self.head_dist_dino(dist_ts[:, 1])
            return x_cls, x_bio, x_dino

def create_trinity_student(variant="standard", model_size="large", img_size=224):
    """Factory to create the student model."""
    if model_size == "large":
        # ViT-L/16
        model = TrinityDeiT(
            variant=variant, patch_size=16, embed_dim=1024, depth=24, num_heads=16, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            img_size=img_size
        )
    else:
        # ViT-B/16 (more efficient for fast inference)
        model = TrinityDeiT(
            variant=variant, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            img_size=img_size
        )
    return model
