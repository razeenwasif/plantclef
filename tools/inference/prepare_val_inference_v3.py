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

def build_val_set_final(num_samples=1000):
    """
    Extracts samples and uses the RAW species ID from the shard.
    """
    out_dir = "data/mini_val_images"
    out_csv = "data/mini_val_ground_truth.csv"
    os.makedirs(out_dir, exist_ok=True)
    
    shard_path = "/workspace/plantclef/shards/val_00000.tar"
    print(f"[*] Extracting from {shard_path}...")
    dataset = wds.WebDataset(shard_path)
    
    records = []
    count = 0
    
    pbar = tqdm(total=num_samples, desc="Building Val Set")
    for sample in dataset:
        try:
            img_id = sample['__key__']
            cls_bin = sample['cls']
            img_bin = sample['jpg']
            
            # The label in the shard is the RAW species ID (e.g. 1355868)
            raw_species_id = struct.unpack('<I', cls_bin)[0]
            
            # Save Image
            img_path = os.path.join(out_dir, f"{img_id}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_bin)
            
            records.append({
                "quadrat_id": img_id,
                "species_ids": f"[{raw_species_id}]"
            })
            
            count += 1
            pbar.update(1)
            if count >= num_samples: break
            
        except Exception as e:
            continue
            
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    
    # Create the 'inference' style CSV (quadrat_id;image_name)
    inf_df = pd.DataFrame({
        "quadrat_id": df["quadrat_id"],
        "image_name": df["quadrat_id"] + ".jpg"
    })
    inf_df.to_csv("data/mini_val_inference.csv", sep=";", index=False)
    
    print(f"\n[Success] Created Mini-Val set with {len(df)} samples.")
    print(f" - Images: {out_dir}/")
    print(f" - Ground Truth: {out_csv}")
    print(f" - Inference Map: data/mini_val_inference.csv")

if __name__ == "__main__":
    build_val_set_final(1000)
