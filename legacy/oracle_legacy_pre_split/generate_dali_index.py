import os
import tarfile
import glob
from pathlib import Path
from tqdm import tqdm
import shutil
import subprocess

# --- CONFIG ---
SHARD_DIR = "/workspace/plantclef/shards"
INDEX_DIR = "/workspace/plantclef/shards/.index"

def generate_index(tar_path, idx_path):
    """
    Creates a DALI-compatible binary index file for a WebDataset tarball using the official wds2idx tool.
    """
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    if os.path.exists(idx_path):
        return
        
    tmp_idx_path = f"{idx_path}.{os.getpid()}.tmp"
    # Use the official DALI index generator
    subprocess.run(["/workspace/pytorch_env/bin/wds2idx", tar_path, tmp_idx_path], check=True)
    
    # Atomic rename guarantees that an incomplete file is never read by DALI
    os.replace(tmp_idx_path, idx_path)

def main():
    if not os.path.exists(SHARD_DIR):
        print(f"[Error] Shard directory not found: {SHARD_DIR}")
        return

    # Clean Room: Wipe old indices
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    tar_files = sorted(glob.glob(os.path.join(SHARD_DIR, "*.tar")))
    print(f"[*] Found {len(tar_files)} shards. Generating official DALI binary indices...")

    for tar_f in tqdm(tar_files, desc="Indexing"):
        shard_name = Path(tar_f).stem
        idx_f = os.path.join(INDEX_DIR, shard_name + ".idx")
        try:
            generate_index(tar_f, idx_f)
        except Exception as e:
            print(f"\n[!] Failed to index {shard_name}: {e}")

    print(f"\n[+] SUCCESS: {len(tar_files)} indices generated in {INDEX_DIR}")
    print("[*] Your cluster is now ready for zero-latency startup.")

if __name__ == "__main__":
    main()
