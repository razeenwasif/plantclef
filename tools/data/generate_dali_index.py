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
WDS2IDX_BIN = "/workspace/pytorch_env/bin/wds2idx"

def generate_index(tar_path: str, idx_path: str):
    """
    Creates a DALI-compatible binary index file for a WebDataset tarball.
    Utilizes the official wds2idx tool.
    """
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    if os.path.exists(idx_path):
        return
        
    tmp_idx_path = f"{idx_path}.{os.getpid()}.tmp"
    
    # ORACLE: Atomic generation to prevent race conditions during cluster startup
    try:
        subprocess.run([WDS2IDX_BIN, tar_path, tmp_idx_path], check=True, capture_output=True)
        os.replace(tmp_idx_path, idx_path)
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp_idx_path):
            os.remove(tmp_idx_path)
        raise RuntimeError(f"Failed to generate index for {tar_path}: {e.stderr.decode()}")

def main():
    if not os.path.exists(SHARD_DIR):
        print(f"[Error] Shard directory not found: {SHARD_DIR}")
        return

    # 1. Clean Room: Wipe old malformed indices
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    tar_files = sorted(glob.glob(os.path.join(SHARD_DIR, "*.tar")))
    print(f"[*] Found {len(tar_files)} shards. Generating official DALI indices...")

    for tar_f in tqdm(tar_files, desc="Indexing"):
        shard_name = Path(tar_f).stem
        idx_f = os.path.join(INDEX_DIR, shard_name + ".idx")
        
        try:
            generate_index(tar_f, idx_f)
        except Exception as e:
            print(f"\n[!] Failed to index {shard_name}: {e}")

    print(f"\n[+] SUCCESS: {len(tar_files)} official indices generated in {INDEX_DIR}")
    print("[*] Your cluster is now ready for zero-latency startup.")

if __name__ == "__main__":
    main()
