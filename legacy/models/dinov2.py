import torch
import torch.nn as nn
import timm

class PlantDINOv2(nn.Module):
    def __init__(self, model_name='vit_large_patch14_dinov2.lvd142m', input_res=448):
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

    def forward(self, x):
        return self.backbone(x)
