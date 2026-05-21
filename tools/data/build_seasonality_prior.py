import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm

def main():
    manifest_path = "/workspace/plantclef/processed/pc24_inat_combined_manifest.csv"
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    output_path = "models/species_month_priors.pt"
    
    print(f"Loading mapping...")
    mapping = pd.read_csv(mapping_path, header=None)
    species_to_idx = {int(id): i for i, id in enumerate(mapping[0])}
    num_species = len(mapping)
    
    # [Species, 12 months]
    counts = np.zeros((num_species, 12), dtype=np.float32)
    
    print(f"Streaming manifest...")
    # Read in chunks to save RAM
    chunk_size = 100000
    reader = pd.read_csv(manifest_path, sep=';', chunksize=chunk_size, low_memory=False)
    
    for chunk in tqdm(reader, desc="Processing Chunks"):
        # We need date to get month
        # Wait, manifest didn't have date? Let me check headers again
        pass

if __name__ == "__main__":
    main()
