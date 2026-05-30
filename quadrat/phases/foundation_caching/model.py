"""Phase 1 model — feature extractor only.

Owned exclusively by p1_extract. DO NOT import from here in other phases.
If you need a backbone for another phase, that phase defines its own class.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Loads three frozen backbones and returns their concatenated L2-normalised features.

    Output shape: [B, 3328]  (768 BioCLIP + 1024 DINOv3 + 1536 ConvNeXt)
    No classification head. No LoRA. No training state.
    """

    def __init__(self, bioclip_name: str, dinov3_name: str, convnext_name: str,
                 input_res: int = 224) -> None:
        super().__init__()
        self.bioclip_name  = bioclip_name
        self.dinov3_name   = dinov3_name
        self.convnext_name = convnext_name
        self.input_res     = input_res

        # Lazy-initialised to keep Phase 1 VRAM budget predictable
        self.bioclip:  nn.Module | None = None
        self.dinov3:   nn.Module | None = None
        self.convnext: nn.Module | None = None

    # ------------------------------------------------------------------
    def load_backbones(self) -> None:
        if self.bioclip is not None:
            return
        from src.models.bioclip      import PlantBioCLIP
        from src.models.vit_backbone import PlantViTBackbone
        from src.models.convnext     import PlantConvNeXt

        # Use whichever device the dummy parameter lives on
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bioclip  = PlantBioCLIP(checkpoint=self.bioclip_name,  input_res=self.input_res).to(device)
        self.dinov3   = PlantViTBackbone(model_name=self.dinov3_name,   input_res=self.input_res).to(device)
        self.convnext = PlantConvNeXt(model_name=self.convnext_name, input_res=self.input_res).to(device)

        for bb in (self.bioclip, self.dinov3, self.convnext):
            bb.eval()
            for p in bb.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 3328] concatenated normalised features."""
        self.load_backbones()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            f_bio  = self.bioclip(x)
            f_dino = self.dinov3(x)
            f_conv = self.convnext(x)

        f_bio  = F.normalize(f_bio.float(),  p=2, dim=1)
        f_dino = F.normalize(f_dino.float(), p=2, dim=1)
        f_conv = F.normalize(f_conv.float(), p=2, dim=1)
        return torch.cat([f_bio, f_dino, f_conv], dim=1).contiguous()
