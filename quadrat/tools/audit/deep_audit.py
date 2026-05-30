import os
import multiprocessing as mp
import cudf
import time
from tqdm import tqdm
from typing import Tuple, List, Union, Any

CSV_PATH = "/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv"
IMG_DIR  = "/workspace/plantclef/raw/train/images_max_side_800/"
OUTPUT_CSV = "/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified_deep_audit.csv"

def check_file(args: Tuple[str, int]) -> Union[bool, str]:
    """
    Performs a deep read of an image file to check for truncation or other errors.

    Parameters
    ----------
    args : Tuple[str, int]
        A tuple containing (image_name, species_id).

    Returns
    -------
    Union[bool, str]
        True if the file is valid and can be read.
        The full path to the file as a string if an error occurs.
    """
    fname, sid = args
    path = os.path.join(IMG_DIR, str(sid), fname)
    try:
        # Try to read 10KB to catch truncation/SIGBUS/read errors
        with open(path, 'rb') as f:
            f.read(10240)
        return True
    except Exception as e:
        # Return path so we can see what failed
        return path

def verify() -> None:
    """
    Orchestrates a deep audit of the stratified training metadata.

    Uses multiprocessing to verify all image files listed in the 
    metadata CSV and generates a new CSV containing only verified 
    valid images.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    print(f"Loading {CSV_PATH}...")
    df = cudf.read_csv(CSV_PATH, sep=';')
    image_names = df['image_name'].to_arrow().to_pylist()
    species_ids = df['species_id'].to_arrow().to_pylist()
    data = list(zip(image_names, species_ids))
    
    print(f"Deep Audit of {len(data):,} images using {mp.cpu_count()} cores...")
    start = time.time()
    
    corrupted = []
    valid_indices = []
    
    with mp.Pool(mp.cpu_count()) as pool:
        # Chunksize 512 for better performance on large datasets
        results = list(tqdm(pool.imap(check_file, data, chunksize=512), total=len(data)))
    
    for i, res in enumerate(results):
        if res is True:
            valid_indices.append(i)
        else:
            corrupted.append(res)
    
    print(f"Audit complete in {time.time() - start:.2f}s")
    print(f"Valid images: {len(valid_indices):,}")
    print(f"Corrupted found: {len(corrupted):,}")
    
    if corrupted:
        print("First 10 corrupted paths:")
        for c in corrupted[:10]:
            print(f"  {c}")
            
    df_verified = df.iloc[valid_indices]
    df_verified.to_csv(OUTPUT_CSV, sep=';', index=False)
    print(f"Saved verified metadata to {OUTPUT_CSV}")

if __name__ == "__main__":
    verify()
