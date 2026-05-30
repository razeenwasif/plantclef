import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
from .types import TileSpec

# Attempt to load SAM
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

def generate_instance_masks(image: Image.Image, image_id: str, cfg: Any) -> List[Tuple[TileSpec, Image.Image]]:
    """
    ORACLE SOTA: Use SAM to identify actual plant bodies instead of blind grid tiling.
    """
    if not HAS_SAM:
        print("[Warning] SAM not installed. Falling back to dummy full-image mask.")
        spec = TileSpec(tile_id=0, image_id=image_id, x0=0, y0=0, x1=image.width, y1=image.height, scale=1.0)
        return [(spec, image)]

    # 1. Initialize SAM on current device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth").to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 2. Generate Masks
    img_np = np.array(image)
    masks = mask_generator.generate(img_np)
    
    results = []
    for i, mask in enumerate(masks):
        # x, y, w, h
        bbox = mask['bbox'] 
        x0, y0, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Crop actual plant instance
        crop = image.crop((x0, y0, x0+w, y0+h))
        
        spec = TileSpec(
            tile_id=i,
            image_id=image_id,
            x0=x0, y0=y0, x1=x0+w, y1=y0+h,
            scale=1.0
        )
        results.append((spec, crop))
        
    return results
