import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, List

class HotspotDetector(nn.Module):
    """
    A tiny model to identify 'hotspots' (regions with vegetation).

    Used to skip empty tiles (soil, rock, debris) in the main SAHI pass.

    Parameters
    ----------
    model_name : str, optional
        The tiny backbone to extract high-level feature maps
        (default is 'mobilenetv3_small_050').
    threshold : float, optional
        Activity threshold for hotspot detection (default is 0.15).
    """

    def __init__(self, model_name: str = 'mobilenetv3_small_050', threshold: float = 0.15) -> None:
        super().__init__()
        # Load a tiny backbone to extract high-level feature maps
        self.backbone = timm.create_model(model_name, pretrained=True,
                                          num_classes=0, global_pool='')
        self.threshold = threshold

        # Determine the reduction factor of the backbone
        # For MobileNetV3-Small, typical reduction is 32x
        self.reduction = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).
            Usually a downsampled global view.

        Returns
        -------
        output : torch.Tensor
            A hotspot mask (batch_size, 1, h, w) where h, w are feature map dims.
        """
        features = self.backbone(x)
        # Aggregate features across channels to get an 'activity' map
        # Using mean of absolute values as a proxy for 'interestingness'
        activity_map = torch.mean(torch.abs(features), dim=1, keepdim=True)

        # Normalize to [0, 1]
        batch_size = x.size(0)
        min_val = activity_map.view(batch_size, -1).min(dim=1)[0].view(-1, 1, 1, 1)
        max_val = activity_map.view(batch_size, -1).max(dim=1)[0].view(-1, 1, 1, 1)
        activity_map = (activity_map - min_val) / (max_val - min_val + 1e-6)

        return activity_map > self.threshold

    def get_hotspot_indices(self, img_tensor: torch.Tensor, tile_h: int, tile_w: int, overlap: float) -> List[int]:
        """
        Returns indices of tiles that contain hotspots.

        Parameters
        ----------
        img_tensor : torch.Tensor
            Full image tensor of shape (batch_size, channels, height, width).
        tile_h : int
            Height of each tile.
        tile_w : int
            Width of each tile.
        overlap : float
            Overlap factor between tiles.

        Returns
        -------
        keep_indices : List[int]
            A list of indices of tiles containing hotspots.
        """
        batch_size, _, height, width = img_tensor.shape

        # 1. Prepare low-res global view for the tiny model
        global_view = F.interpolate(img_tensor, size=(448, 448),
                                                      mode='bilinear')

        # 2. Get hotspot mask
        with torch.no_grad():
            mask = self.forward(global_view)  # (batch_size, 1, 14, 14) for 448px/32

        # 3. Calculate grid steps
        y_step = int(tile_h * (1 - overlap))
        x_step = int(tile_w * (1 - overlap))
        num_tiles_y = (height - tile_h) // y_step + 1
        num_tiles_x = (width - tile_w) // x_step + 1

        # 4. Map mask cells to tile indices
        keep_indices = []
        mask_h, mask_w = mask.shape[2], mask.shape[3]

        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                # Tile boundaries in normalized coordinates [0, 1]
                y0, y1 = (tile_y * y_step) / height, (tile_y * y_step + tile_h) / height
                x0, x1 = (tile_x * x_step) / width, (tile_x * x_step + tile_w) / width

                # Corresponding mask coordinates
                mask_y0, mask_y1 = int(y0 * mask_h), int(y1 * mask_h)
                mask_x0, mask_x1 = int(x0 * mask_w), int(x1 * mask_w)

                # Check if any cell in this range is True
                if mask[0, 0, mask_y0:mask_y1+1, mask_x0:mask_x1+1].any():
                    keep_indices.append(tile_y * num_tiles_x + tile_x)

        return keep_indices
