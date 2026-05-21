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
from PIL import Image
import io

def build_val_set_wds(num_samples=1000):
    """
    Extracts N samples from WebDataset shards directly to avoid DALI complexity.
    Decodes the custom binary 'cls' format.
    """
    out_dir = "data/mini_val_images"
    out_csv = "data/mini_val_ground_truth.csv"
    os.makedirs(out_dir, exist_ok=True)
    
    shard_path = "/workspace/plantclef/shards/val_00000.tar"
    print(f"[*] Extracting from {shard_path}...")
    
    # We use wds without automatic decoding to handle 'cls' manually
    dataset = wds.WebDataset(shard_path)
    
    # Load species mapping
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    species_list = []
    with open(mapping_path) as f:
        for line in f:
            species_list.append(line.strip())

    records = []
    
    pbar = tqdm(total=num_samples, desc="Extracting Val Shards")
    count = 0
    for sample in dataset:
        try:
            # sample is a dict: {'__key__': ..., 'jpg': binary, 'cls': binary}
            img_id = sample['__key__']
            img_bin = sample['jpg']
            cls_bin = sample['cls']
            
            # ORACLE: Decode 4-byte little-endian integer (INT32)
            # This matches the fn.reinterpret(labels, dtype=types.INT32) in DALI
            label_idx = struct.unpack('<I', cls_bin)[0]
            
            # Save Image
            img_path = os.path.join(out_dir, f"{img_id}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_bin)
            
            # Map index to Real ID
            real_species_id = species_list[label_idx]
            records.append({
                "quadrat_id": img_id,
                "species_ids": f"[{real_species_id}]"
            })
            
            count += 1
            pbar.update(1)
            if count >= num_samples: break
            
        except Exception as e:
            print(f"Error at sample: {e}")
            continue
            
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    
    # Create the 'inference' style CSV (quadrat_id;image_name)
    inf_df = pd.DataFrame({
        "quadrat_id": df["quadrat_id"],
        "image_name": df["quadrat_id"] + ".jpg"
    })
    inf_df.to_csv("data/mini_val_inference.csv", sep=";", index=False)
    
    print(f"\n[Done] Saved {len(df)} images to {out_dir}")
    print(f"[Done] Created data/mini_val_inference.csv for the pipeline.")

if __name__ == "__main__":
    build_val_set_wds(1000)
