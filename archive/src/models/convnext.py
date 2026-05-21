import torch
import torch.nn as nn
import timm

class PlantConvNeXt(nn.Module):
    """
    A wrapper for ConvNeXt-V2 architectures provided by the timm library.
    ConvNeXt models are modernized CNNs that incorporate design elements 
    from Transformers while maintaining convolutional efficiency.
    """
    def __init__(self, model_name='convnextv2_huge.fcmae_ft_in22k_in1k_384', input_res=448):
        """
        Args:
            model_name (str): The specific ConvNeXt-V2 checkpoint to load.
            input_res (int): Target resolution (Note: ConvNeXt is fully convolutional 
                             and resolution-independent at initialization).
        """
        super(PlantConvNeXt, self).__init__()
        
        # Load ConvNeXt-V2 from timm
        # We set num_classes=0 to remove the classification head and return features.
        # Note: img_size is removed as ConvNeXt-V2 constructor in timm does not take it.
        print(f"[ConvNeXt] Loading backbone: {model_name}")
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0
        )
        
        # Capture the feature dimension for the fusion head
        self.feature_dim = self.backbone.num_features

    def forward(self, x):
        """ Returns the global pooled feature representation of the input. """
        return self.backbone(x)
