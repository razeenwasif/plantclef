"""High-speed sharding and preprocessing tool for TPU/Multi-GPU data pipelines.

This script transforms a raw image dataset into WebDataset (.tar) shards.
It performs pre-resizing to 384px and metadata packing to eliminate CPU 
bottlenecks during training on high-throughput hardware like TPU v4.
"""

import os
import sys
import json
import tarfile
import io
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path for config access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from src import config as scfg

def process_sample(args: tuple[str, int, str]) -> tuple[bool, bytes | None, bytes | None, str]:
    """
    Resize a single image and return bytes for tar packing.

    Parameters
    ----------
    args : tuple[str, int, str]
        A tuple containing (image_path, species_id, image_name).

    Returns
    -------
    tuple[bool, bytes | None, bytes | None, str]
        A tuple containing (success, image_bytes, json_bytes, image_name).
    """
    img_path, species_id, img_name = args
    try:
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            # Pre-resize to training resolution
            img = img.resize((config.RESOLUTION, config.RESOLUTION), Image.LANCZOS)
            
            # Save to buffer
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=95)
            img_bytes = buf.getvalue()
            
            # Create metadata
            meta = {
                "species_id": int(species_id),
                "image_name": img_name
            }
            json_bytes = json.dumps(meta).encode('utf-8')
            
            return True, img_bytes, json_bytes, img_name
    except Exception as e:
        return False, None, None, img_name

def create_shards(df: pd.DataFrame, output_dir: str, samples_per_shard: int = 1000, num_workers: int | None = None) -> None:
    """
    Parallel orchestrator with Rust pre-filtering and optimized I/O.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. High-Speed Pre-Filtering (Rust)
    print(f"[Sharder] Constructing paths for integrity check...")
    tasks = []
    full_paths = []
    for _, row in df.iterrows():
        sid = row['species_id']
        name = row['image_name']
        img_path = os.path.join(config.IMG_DIR, str(sid), name)
        tasks.append((img_path, sid, name))
        full_paths.append(img_path)

    try:
        import data_auditor
        print(f"[Sharder] Running Rust-Accelerated data audit...")
        valid_indices = set(data_auditor.audit_dataset(full_paths, True))
        tasks = [tasks[i] for i in range(len(tasks)) if i in valid_indices]
        print(f"[Sharder] Rust Audit complete. Filtered out {len(full_paths) - len(tasks):,} broken images.")
    except ImportError:
        print("[Warning] data_auditor (Rust) not found. Proceeding with slow internal filtering.")

    num_shards = int(np.ceil(len(tasks) / samples_per_shard))
    if num_workers is None:
        num_workers = mp.cpu_count()
        
    print(f"[Sharder] Launching {num_workers} workers to create {num_shards} shards...")
    
    pool = mp.Pool(num_workers)
    
    for shard_idx in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard-{shard_idx:05d}.tar")
        
        # Resume Check: Skip if shard already exists
        if os.path.exists(shard_path):
            print(f"  --> Skipping {shard_path} (already exists)")
            continue

        shard_tasks = tasks[shard_idx * samples_per_shard : (shard_idx + 1) * samples_per_shard]
        
        with tarfile.open(shard_path, "w") as tar:
            # Use imap_unordered for smoother CPU utilization
            results = pool.map(process_sample, shard_tasks)
            
            for success, img_bytes, json_bytes, name in results:
                if not success: continue
                
                # Add Image
                img_info = tarfile.TarInfo(name=f"{name}.jpg")
                img_info.size = len(img_bytes)
                tar.addfile(img_info, io.BytesIO(img_bytes))
                
                # Add JSON
                json_info = tarfile.TarInfo(name=f"{name}.json")
                json_info.size = len(json_bytes)
                tar.addfile(json_info, io.BytesIO(json_bytes))
                
        print(f"  --> Completed {shard_path} ({len(shard_tasks)} samples)")

    pool.close()
    pool.join()

if __name__ == "__main__":
    # Load metadata
    csv_path = config.CLEANED_CSV if os.path.exists(config.CLEANED_CSV) else config.RAW_CSV
    print(f"[Sharder] Loading metadata from {csv_path}...")
    master_df = pd.read_csv(csv_path, sep=';')
    
    # Optional: Filter for SW Europe if needed or use full dataset
    # We'll shard the full cleaned dataset for maximum coverage
    
    OUTPUT_PATH = "data/shards_384px"
    create_shards(master_df, OUTPUT_PATH, samples_per_shard=1000)
    print(f"\n[Success] Sharding complete. Data ready for TPU at {OUTPUT_PATH}")
