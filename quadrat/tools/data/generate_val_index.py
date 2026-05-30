import os
import glob
import subprocess
from pathlib import Path
from tqdm import tqdm
import shutil

# --- CONFIG ---
SHARD_DIR = "/workspace/plantclef/shards"
INDEX_DIR = "/workspace/plantclef/shards/.index"
# Official DALI utility path
WDS2IDX_PATH = "/workspace/pytorch_env/lib/python3.12/site-packages/wds2idx.py"
PYTHON_BIN = "/workspace/pytorch_env/bin/python3"

def main():
    if not os.path.exists(SHARD_DIR):
        print(f"[Error] Shard directory not found: {SHARD_DIR}")
        return

    # Create index directory if missing
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # Target only Validation shards
    tar_files = sorted(glob.glob(os.path.join(SHARD_DIR, "val_*.tar")))
    
    if not tar_files:
        print("[Error] No validation shards (val_*.tar) found. Run val_packer first.")
        return

    print(f"[*] Found {len(tar_files)} validation shards. Generating official DALI indices...")

    for tar_f in tqdm(tar_files, desc="Indexing Val Shards"):
        shard_name = Path(tar_f).stem
        idx_f = os.path.join(INDEX_DIR, shard_name + ".idx")
        
        cmd = [PYTHON_BIN, WDS2IDX_PATH, tar_f, idx_f]
        
        try:
            # We don't wipe the whole folder here, just overwrite the specific index
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[!] Failed to index {shard_name}: {e.stderr.decode()}")

    print(f"\n[+] SUCCESS: Validation indices generated in {INDEX_DIR}")

if __name__ == "__main__":
    main()
