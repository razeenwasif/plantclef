"""Script to verify the integrity of images in the dataset.

This script checks if image files exist and can be partially read,
filtering out any corrupted or missing images from the metadata CSV.
"""

import multiprocessing as mp
import os
import time

import cudf
from tqdm import tqdm

CSV_PATH = "/workspace/plantclef/processed/train_metadata_cleaned.csv"
IMG_DIR = "/workspace/plantclef/raw/train/images_max_side_800/"
OUTPUT_CSV = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"


from typing import Tuple, List

def check_file(args: Tuple[str, int]) -> bool:
    """
    Checks if an image file exists and its first few bytes are readable.

    This function attempts to open the image file in read-only mode and read 
    a small number of bytes to verify its accessibility and basic integrity.

    Parameters
    ----------
    args : Tuple[str, int]
        A tuple containing (image_name, species_id).

    Returns
    -------
    bool
        True if the file is successfully opened and read, False otherwise.
    """
    image_name, species_id = args
    path = os.path.join(IMG_DIR, str(species_id), image_name)
    try:
        fd = os.open(path, os.O_RDONLY)
        os.read(fd, 10)
        os.close(fd)
        return True
    except OSError:
        # Don't print for every failure to avoid flooding stdout and slowing down
        return False


def verify() -> None:
    """
    Verifies the integrity of all images listed in the metadata CSV.

    This function loads the metadata, uses a multiprocessing pool to parallelize 
    image verification, and filters out any entries corresponding to missing or 
    unreadable files. The resulting verified metadata is saved to a new CSV.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Saves the filtered metadata to 'OUTPUT_CSV'.
    """
    print(f"Loading {CSV_PATH}...")
    df = cudf.read_csv(CSV_PATH, sep=';')
    image_names = df['image_name'].to_arrow().to_pylist()
    species_ids = df['species_id'].to_arrow().to_pylist()
    data = list(zip(image_names, species_ids))

    print(f"Verifying {len(data):,} images using {mp.cpu_count()} cores...")
    start_time = time.time()

    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        # Use imap to get results as they finish for the progress bar
        for res in tqdm(
            pool.imap(check_file, data, chunksize=256), total=len(data)):
            results.append(res)

    valid_indices = [i for i, ok in enumerate(results) if ok]
    df_verified = df.iloc[valid_indices]

    elapsed_time = time.time() - start_time
    print(f"Verification complete in {elapsed_time:.2f}s")
    print(f"Valid images: {len(df_verified):,} "
          f"(Removed {len(df) - len(df_verified)})")

    df_verified.to_csv(OUTPUT_CSV, sep=';', index=False)
    print(f"Saved verified metadata to {OUTPUT_CSV}")


if __name__ == "__main__":
    verify()
