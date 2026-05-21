import torch
import torch.nn as nn
import timm
from typing import Optional

class PlantConvNeXt(nn.Module):
    """
    ConvNeXt model for plant classification.

    This model uses a ConvNeXt-V2 backbone from the `timm` library, which is a
    modernized Convolutional Neural Network (CNN) incorporating Transformer-like 
    design elements.

    Parameters
    ----------
    model_name : str, optional
        The specific ConvNeXt-V2 checkpoint to load from timm
        (default is 'convnextv2_huge.fcmae_ft_in22k_in1k_384').
    input_res : int, optional
        Target input resolution for the model. Note that ConvNeXt is fully
        convolutional and resolution-independent at initialization
        (default is 448).
    """

    def __init__(self, model_name: str = 'convnextv2_huge.fcmae_ft_in22k_in1k_384', input_res: int = 448) -> None:
        super(PlantConvNeXt, self).__init__()
        
        # Load ConvNeXt-V2 from timm
        print(f"[ConvNeXt] Loading backbone: {model_name}")
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0
        )
        
        # Capture the feature dimension for the fusion head
        self.feature_dim = self.backbone.num_features

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """
        Enables or disables gradient checkpointing for the backbone.
        Uses Full Checkpointing for Phase 2 stability.
        """
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)
            status = "enabled" if enable else "disabled"
            print(f"[ConvNeXt] Full Gradient Checkpointing {status}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvNeXt model."""
        return self.backbone(x)
