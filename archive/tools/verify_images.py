import os
import multiprocessing as mp
import cudf
import time
from tqdm import tqdm

CSV_PATH = "/workspace/plantclef/processed/train_metadata_cleaned.csv"
IMG_DIR  = "/workspace/plantclef/raw/train/images_max_side_800/"
OUTPUT_CSV = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"

def check_file(args):
    fname, sid = args
    path = os.path.join(IMG_DIR, str(sid), fname)
    try:
        fd = os.open(path, os.O_RDONLY)
        os.read(fd, 10)
        os.close(fd)
        return True
    except Exception:
        # Don't print for every failure to avoid flooding stdout and slowing down
        return False

def verify():
    print(f"Loading {CSV_PATH}...")
    df = cudf.read_csv(CSV_PATH, sep=';')
    image_names = df['image_name'].to_arrow().to_pylist()
    species_ids = df['species_id'].to_arrow().to_pylist()
    data = list(zip(image_names, species_ids))
    
    print(f"Verifying {len(data):,} images using {mp.cpu_count()} cores...")
    start = time.time()
    
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        # Use imap to get results as they finish for the progress bar
        for res in tqdm(pool.imap(check_file, data, chunksize=256), total=len(data)):
            results.append(res)
    
    valid_indices = [i for i, ok in enumerate(results) if ok]
    df_verified = df.iloc[valid_indices]
    
    print(f"Verification complete in {time.time() - start:.2f}s")
    print(f"Valid images: {len(df_verified):,} (Removed {len(df) - len(df_verified)})")
    
    df_verified.to_csv(OUTPUT_CSV, sep=';', index=False)
    print(f"Saved verified metadata to {OUTPUT_CSV}")

if __name__ == "__main__":
    verify()
