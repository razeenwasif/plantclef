import os
import sys
from pathlib import Path
import struct

# --- ORACLE: Path Sync ---
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

import webdataset as wds
import pandas as pd
from tqdm import tqdm
import torch

def build_val_set_debug(num_samples=1000):
    """
    DEBUG version: Extract samples and report label indices.
    """
    out_dir = "data/mini_val_images"
    out_csv = "data/mini_val_ground_truth.csv"
    os.makedirs(out_dir, exist_ok=True)
    
    shard_path = "/workspace/plantclef/shards/val_00000.tar"
    dataset = wds.WebDataset(shard_path)
    
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    species_list = []
    with open(mapping_path) as f:
        for line in f:
            species_list.append(line.strip())
    
    print(f"[*] Mapping length: {len(species_list)}")

    records = []
    count = 0
    
    for sample in dataset:
        try:
            img_id = sample['__key__']
            cls_bin = sample['cls']
            
            # Try different unpackings
            idx_le = struct.unpack('<I', cls_bin)[0]
            
            if count < 5:
                print(f"Sample {count}: id={img_id}, raw_bin={cls_bin.hex()}, idx_le={idx_le}")
            
            if idx_le < len(species_list):
                real_species_id = species_list[idx_le]
                img_bin = sample['jpg']
                img_path = os.path.join(out_dir, f"{img_id}.jpg")
                with open(img_path, "wb") as f:
                    f.write(img_bin)
                
                records.append({
                    "quadrat_id": img_id,
                    "species_ids": f"[{real_species_id}]"
                })
                count += 1
            else:
                if count < 20: # Don't spam too much
                    print(f" [!] Rogue Index: {idx_le} (Map Size: {len(species_list)}) at {img_id}")
            
            if count >= num_samples: break
            
        except Exception as e:
            print(f"Error: {e}")
            continue
            
    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_csv, index=False)
        inf_df = pd.DataFrame({
            "quadrat_id": df["quadrat_id"],
            "image_name": df["quadrat_id"] + ".jpg"
        })
        inf_df.to_csv("data/mini_val_inference.csv", sep=";", index=False)
        print(f"\n[Success] Created val set with {len(df)} samples.")
    else:
        print("\n[Failure] No valid samples found.")

if __name__ == "__main__":
    build_val_set_debug(100)
