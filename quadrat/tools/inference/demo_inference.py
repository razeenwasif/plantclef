import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import src.config as config
from src.models.ensemble import PlantEnsemble
from src.training.checkpoints import get_raw_model
from typing import List, Optional

def load_species_mapping() -> Optional[List[str]]:
    """
    Loads the species mapping from a CSV file.

    Parameters
    ----------
    None

    Returns
    -------
    Optional[List[str]]
        A list of species IDs indexed by their class index, or None if the mapping file 
        is not found.
    """
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    if os.path.exists(mapping_path):
        df = pd.read_csv(mapping_path, header=None)
        return df[0].tolist()
    return None

def preprocess_image(image_path: str, resolution: int = 384) -> torch.Tensor:
    """
    Preprocesses a single image for inference.

    This function loads the image from the specified path, resizes it, and 
    applies normalization compatible with the DALI dataloader.

    Parameters
    ----------
    image_path : str
        The file path to the input image.
    resolution : int, optional
        The resolution to resize the image to (default is 384).

    Returns
    -------
    torch.Tensor
        The preprocessed image tensor with shape (1, 3, resolution, resolution).
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((resolution, resolution), Image.Resampling.BILINEAR)
    img_np = np.array(img)
    
    # Normalization (DALI matching)
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std  = np.array([0.26862954, 0.26130258, 0.27577711])
    
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    
    return img_tensor.unsqueeze(0)

def run_demo(image_path: str) -> None:
    """
    Runs a complete inference demo on a single image and prints the top 5 predictions.

    This function loads the model, applies LoRA, loads the best weights, performs
    inference with mixed precision, and displays the top 5 species predictions 
    with their confidence levels.

    Parameters
    ----------
    image_path : str
        The file path to the input image.

    Returns
    -------
    None
        Prints the top 5 predictions to the console.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # 1. Load Mapping
    species_ids = load_species_mapping()
    if not species_ids:
        print("[!] Warning: Species mapping not found. Will display indices instead.")

    # 2. Initialize Model
    model = PlantEnsemble(
        num_classes=7806, 
        input_res=config.RESOLUTION,
        bioclip_name=config.BIOCLIP_NAME,
        dinov2_name=config.DINOV2_NAME,
        convnext_name=config.CONVNEXT_NAME
    )
    
    # Apply LoRA (Must match R=64 for the push run)
    model.apply_lora(r=config.LORA_R, lora_alpha=config.LORA_ALPHA)
    
    # 3. Load Weights
    checkpoint_path = "models/blackwell_v2/phase2_checkpoint/checkpoint_latest/mp_rank_00_model_states.pt"
    if not os.path.exists(checkpoint_path):
        # Fallback to a backup if exists
        checkpoint_path = "models/blackwell_v2/swa_model_final.pth"
        
    if os.path.exists(checkpoint_path):
        print(f"[*] Loading weights from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['module'] if 'module' in ckpt else ckpt
        
        # Strip prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "").replace("module.", "")
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("[!] Error: No checkpoint found!")
        return

    model.to(device)
    model.eval()

    # 4. Inference
    print(f"[*] Processing image: {os.path.basename(image_path)}")
    input_tensor = preprocess_image(image_path, resolution=config.RESOLUTION).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

    # 5. Results
    top_probs, top_indices = torch.topk(probs, 5)
    
    print("\n" + "="*40)
    print(" TOP 5 PREDICTIONS")
    print("="*40)
    for i in range(5):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        sid = species_ids[idx] if species_ids else idx
        print(f"{i+1}. Species ID: {sid:<8} | Confidence: {prob*100:>6.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-find an image to test if none provided
        import glob
        test_images = glob.glob(os.path.join(config.IMG_DIR, "**/*.jpg"), recursive=True)
        if test_images:
            image_to_test = test_images[0]
            run_demo(image_to_test)
        else:
            print("Usage: python demo_inference.py <path_to_image>")
    else:
        run_demo(sys.argv[1])
