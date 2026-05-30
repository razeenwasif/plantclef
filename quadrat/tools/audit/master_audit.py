import os
import multiprocessing as mp
import cudf
import time
from tqdm import tqdm
from typing import Tuple, List, Union, Any

# Master list of ALL images
CSV_PATH = "/workspace/plantclef/raw/PlantCLEF2024_single_plant_training_metadata.csv"
IMG_DIR  = "/workspace/plantclef/raw/train/images_max_side_800/"
CORRUPTED_LIST_FILE = "/workspace/plantclef/processed/master_corrupted_images.txt"

def check_file(args: Tuple[str, int]) -> Union[bool, Tuple[str, str]]:
    """
    Checks the integrity of a single image file.

    Parameters
    ----------
    args : Tuple[str, int]
        A tuple containing (image_name, species_id).

    Returns
    -------
    Union[bool, Tuple[str, str]]
        True if the file is valid and readable.
        A tuple ("MISSING", path) if the file is missing.
        A tuple ("CORRUPTED", path) if the file cannot be read.
    """
    fname, sid = args
    path = os.path.join(IMG_DIR, str(sid), fname)
    if not os.path.exists(path):
        return ("MISSING", path)
    try:
        # Read 10KB to trigger filesystem/corruption errors
        with open(path, 'rb') as f:
            f.read(10240)
        return True
    except Exception as e:
        return ("CORRUPTED", path)

def master_audit() -> None:
    """
    Performs a global audit of all images using a Hybrid-Audit strategy:
    1. Rust (Rayon) performs a 20s sweep to find all broken files.
    2. Python classifies the results as MISSING or CORRUPTED.
    """
    print(f"Loading Master Metadata: {CSV_PATH}...")
    df = cudf.read_csv(CSV_PATH, sep=';')
    image_names = df['image_name'].to_arrow().to_pylist()
    species_ids = df['species_id'].to_arrow().to_pylist()
    
    # Construct full paths for the Rust engine
    print(f"[Audit] Constructing {len(image_names):,} full paths...")
    full_paths = [os.path.join(IMG_DIR, str(sid), fname) for fname, sid in zip(image_names, species_ids)]
    total = len(full_paths)

    corrupted_paths = []
    missing_paths = []
    
    try:
        import data_auditor
        print(f"[Audit] Starting Rust-Accelerated Sweep (Parallel-Rayon)...")
        t0 = time.time()
        # Rust returns indices of VALID files
        valid_indices = set(data_auditor.audit_dataset(full_paths, True))
        dt = time.time() - t0
        print(f"[Audit] Rust sweep complete in {dt:.1f}s.")
        
        # Identify broken indices
        broken_indices = [i for i in range(total) if i not in valid_indices]
        print(f"[Audit] Classifying {len(broken_indices):,} broken files...")
        
        for idx in tqdm(broken_indices, desc="Classifying"):
            path = full_paths[idx]
            if not os.path.exists(path):
                missing_paths.append(path)
            else:
                corrupted_paths.append(path)
                
    except ImportError:
        print("[Warning] data_auditor (Rust) not found. Falling back to slow multi-processing...")
        data = list(zip(image_names, species_ids))
        with mp.Pool(mp.cpu_count()) as pool:
            for res in tqdm(pool.imap(check_file, data, chunksize=1024), total=total):
                if res is not True:
                    status, path = res
                    if status == "CORRUPTED": corrupted_paths.append(path)
                    else: missing_paths.append(path)
    
    print(f"\nAudit Complete.")
    print(f"Found {len(corrupted_paths)} corrupted files.")
    print(f"Found {len(missing_paths)} missing files.")
    
    with open(CORRUPTED_LIST_FILE, "w") as f:
        f.write("CORRUPTED:\n")
        for p in corrupted_paths: f.write(f"{p}\n")
        f.write("\nMISSING:\n")
        for p in missing_paths: f.write(f"{p}\n")
    
    print(f"Results saved to {CORRUPTED_LIST_FILE}")

if __name__ == "__main__":
    master_audit()
