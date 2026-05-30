"""Interactive 3D Visualization tool for PlantCLEF Quadrats.

Loads a 2D image, estimates depth, and opens an Open3D window.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import argparse

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.biomass.depth_pipeline import MonocularBiomassEstimator

def visualize(image_path: str) -> None:
    """
    Loads a 2D image, estimates relative depth, and generates a 3D point cloud visualization.

    This function uses a MonocularBiomassEstimator to predict the depth of the 
    provided image and opens an interactive 3D visualization window using Open3D.

    Parameters
    ----------
    image_path : str
        The file path to the input image (quadrat).

    Returns
    -------
    None
        Opens an interactive Open3D visualization window.
    """
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # 1. Load Image
    img_pil = Image.open(image_path).convert("RGB")
    img_np  = np.array(img_pil)
    
    # 2. Prepare for depth estimation
    # (Simplified preprocessing for visualization)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.cuda() if torch.cuda.is_available() else img_tensor
    
    # 3. Estimate Depth
    estimator = MonocularBiomassEstimator()
    print("[3D] Estimating relative depth...")
    depth_map = estimator.estimate_relative_depth(img_tensor)
    depth_np  = depth_map.cpu().numpy()
    
    # 4. Generate Point Cloud
    print("[3D] Projecting to point cloud...")
    pcd = estimator.generate_point_cloud(img_np, depth_np)
    
    # 5. Visualize
    estimator.visualize_3d(pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a quadrat in 3D.")
    parser.add_argument("image", type=str, help="Path to the image file.")
    args = parser.parse_args()
    
    visualize(args.image)
