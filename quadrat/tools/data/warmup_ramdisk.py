import os
import sys
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

# PLANTCLEF RAM Disk Config
SHM_DIR = "/dev/shm/plantclef_train"
IMG_DIR = "/workspace/plantclef/raw/train/images_max_side_800/"
METADATA = "/workspace/plantclef/raw/PlantCLEF2024_single_plant_training_metadata.csv"

def warmup():
    print(f"\n" + "="*60)
    print(f" PLANTCLEF I/O: Populating Hybrid RAM Disk (EPYC Edition)")
    print(f"="*60)
    
    if not os.path.exists(METADATA):
        print(f"[Error] Metadata not found: {METADATA}")
        return

    df = pd.read_csv(METADATA, sep=';', low_memory=False)
    img_names = df['image_name'].tolist()
    sids = df['species_id'].astype(str).tolist()
    
    tasks = []
    for name, sid in zip(img_names, sids):
        src = os.path.join(IMG_DIR, sid, name)
        dst = os.path.join(SHM_DIR, sid, name)
        tasks.append((src, dst))
        
    print(f"Transferring {len(tasks):,} images to {SHM_DIR}...")
    
    def copy_file(pair):
        src, dst = pair
        if os.path.exists(dst): return
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        except: pass

    # Use 96 threads to leave some headroom for the OS/Training
    with ThreadPoolExecutor(max_workers=96) as executor:
        list(tqdm(executor.map(copy_file, tasks), total=len(tasks), desc="Filling RAM Disk"))

    print(f"\n=== WARMUP COMPLETE ===")
    print(f"Your dataset is now resident in system memory.")

if __name__ == "__main__":
    warmup()
