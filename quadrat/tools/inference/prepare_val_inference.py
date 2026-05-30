import os
import sys
from pathlib import Path

# --- PLANTCLEF: Path Sync ---
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io
import numpy as np
from src.data.dataloader import get_dali_loaders

def build_val_set(num_samples=1000):
    """
    Extracts N samples from the DALI val_loader and saves them as a 
    standalone validation directory and CSV for inference testing.
    """
    out_dir = "data/mini_val_images"
    out_csv = "data/mini_val_ground_truth.csv"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[*] Initializing DALI for validation extraction (N={num_samples})...")
    
    # We set batch_size=1 for easy extraction
    _, val_loader, _, _ = get_dali_loaders(
        batch_size=1,
        resolution=224,
        device_id=0,
        training=True
    )
    
    # Load species mapping
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    species_list = []
    with open(mapping_path) as f:
        for line in f:
            species_list.append(line.strip())

    records = []
    
    pbar = tqdm(total=num_samples, desc="Extracting Val Shards")
    for i in range(num_samples):
        try:
            data = next(val_loader)
            img_tensor = data[0]["data"][0] # [3, 224, 224]
            label_idx = int(data[0]["label"][0].item())
            
            # Convert tensor back to image
            # DALI output is usually normalized or float32 [0,1] or [0,255]
            # We'll just save it as a clean JPEG
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            # Re-scale if needed
            if img_np.max() <= 1.01:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
                
            img_id = f"val_sample_{i:05d}"
            img_path = os.path.join(out_dir, f"{img_id}.jpg")
            
            Image.fromarray(img_np).save(img_path)
            
            # Store the real species ID from our mapping
            real_species_id = species_list[label_idx]
            records.append({
                "quadrat_id": img_id,
                "species_ids": f"[{real_species_id}]"
            })
            
            pbar.update(1)
        except StopIteration:
            break
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            continue
            
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"\n[Done] Saved {len(df)} images to {out_dir}")
    print(f"[Done] Saved ground truth to {out_csv}")

if __name__ == "__main__":
    # We'll need a way to pass this to the inference pipeline as a 'test_csv'
    # but the pipeline expects ';' delimiter and specific columns.
    # I'll create a second CSV in the format the pipeline expects.
    build_val_set(1000)
    
    # Create the 'inference' style CSV (quadrat_id;image_name)
    df = pd.read_csv("data/mini_val_ground_truth.csv")
    inf_df = pd.DataFrame({
        "quadrat_id": df["quadrat_id"],
        "image_name": df["quadrat_id"] + ".jpg"
    })
    inf_df.to_csv("data/mini_val_inference.csv", sep=";", index=False)
    print("[Done] Created data/mini_val_inference.csv for the pipeline.")
