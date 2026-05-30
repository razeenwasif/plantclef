import torch
import pandas as pd
import os
import sys
from tqdm import tqdm

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../src")

from models.bioclip import PlantBioCLIP
from src import config as scfg
#  CLEANED_CSV, RAW_CSV

def generate_zero_shot_anchors():
    """
    Generate BioCLIP text embeddings for all species in the dataset.
    These 'anchors' can be used to initialize the classifier or for 
    zero-shot inference.
    """
    device = torch.device("cpu") # Force CPU due to sm_120 compatibility issues
    print("[Zero-Shot] Using CPU for text encoding.")
    
    # 1. Load species mapping (Global Master)
    name_map_path = "/workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt"
    print(f"[Zero-Shot] Loading names from {name_map_path}...")
    df_names = pd.read_csv(name_map_path, sep=';', quotechar='"')
    df_names.columns = [c.strip('"') for c in df_names.columns]
    
    # Sort by species_id to match the mapping used in trainer.py/dataloader.py
    df_names = df_names.sort_values('species_id')
    num_classes = len(df_names)
    print(f"[Zero-Shot] Aligning to {num_classes} master classes.")
    
    # 2. Format prompts
    prompts = []
    for _, row in df_names.iterrows():
        name = row['species']
        # Simple cleaning: remove author info if present (e.g., "Taxus baccata L." -> "Taxus baccata")
        clean_name = " ".join(name.split()[:2])
        prompts.append(f"a photo of {clean_name}, a type of plant")
            
    # 4. Load BioCLIP and encode
    # BioCLIP 1 (v1) = 512-d, BioCLIP 2 (v2) = 768-d
    checkpoint = os.environ.get("BIOCLIP_CKPT", "hf-hub:imageomics/bioclip-2")
    version = "v2" if "bioclip-2" in checkpoint else "v1"
    
    print(f"[Zero-Shot] Loading BioCLIP ({version}) for encoding...")
    model = PlantBioCLIP(checkpoint=checkpoint, input_res=224).to(device)
    model.eval()
    
    feat_dim = model.feature_dim
    print(f"[Zero-Shot] Encoding {len(prompts)} species prompts into {feat_dim}-d space...")
    anchors = []
    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            batch_anchors = model.encode_text(batch_prompts, device)
            anchors.append(batch_anchors.cpu())
            
    anchors_tensor = torch.cat(anchors, dim=0)
    
    # 5. Save anchors
    out_path = f"models/zero_shot_anchors_{version}.pt"
    # Also save to main path for backward compatibility
    torch.save(anchors_tensor, out_path)
    torch.save(anchors_tensor, "models/zero_shot_anchors.pt")
    print(f"[Zero-Shot] Successfully saved {anchors_tensor.shape} anchors to {out_path}")

if __name__ == "__main__":
    generate_zero_shot_anchors()
