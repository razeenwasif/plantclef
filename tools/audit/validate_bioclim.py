import os
import rasterio
import numpy as np
import torch
from typing import Optional

def validate_bioclim() -> None:
    """
    Validates WorldClim GeoTIFFs and generated ecological traits.

    Iterates through WorldClim bioclimatic rasters to ensure their 
    validity and coverage. Also checks the generated ecological traits 
    database for consistency and data availability.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    bioclim_dir = "/workspace/plantclef/raw/worldclim"
    traits_path = "data/ecological_traits.npy"
    
    print("--- Validating WorldClim GeoTIFFs ---")
    for i in range(1, 20):
        tif_path = os.path.join(bioclim_dir, f"wc2.1_2.5m_bio_{i}.tif")
        if not os.path.exists(tif_path):
            print(f"[ERROR] {tif_path} is missing!")
            continue
            
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            # Filter out nodata values
            valid_data = data[data != src.nodata]
            
            if len(valid_data) == 0:
                print(f"[ERROR] {tif_path} has no valid data!")
            else:
                v_min = valid_data.min()
                v_max = valid_data.max()
                v_mean = valid_data.mean()
                v_std = valid_data.std()
                print(f"BIO{i:02d}: Min={v_min:8.2f}, Max={v_max:8.2f}, Mean={v_mean:8.2f}, Std={v_std:8.2f}")
                
                # Sanity check for BIO1 (Annual Mean Temperature)
                # WorldClim temperatures are in degrees Celsius.
                if i == 1:
                    if v_mean < -50 or v_mean > 50:
                        print(f"  [WARNING] BIO1 mean ({v_mean:.2f}) seems unusual for Celsius.")

    print("\n--- Validating Generated Ecological Traits ---")
    if os.path.exists(traits_path):
        traits = np.load(traits_path)
        print(f"Shape: {traits.shape}")
        
        # Count non-zero rows (species with sampled data)
        non_zero_rows = np.any(traits != 0, axis=1).sum()
        print(f"Species with data: {non_zero_rows} / {traits.shape[0]}")
        
        if non_zero_rows == 0:
            print("[ERROR] Ecological traits database is empty (all zeros)!")
        else:
            for i in range(min(5, traits.shape[1])):
                col_data = traits[:, i]
                col_valid = col_data[col_data != 0]
                if len(col_valid) > 0:
                    print(f"Trait {i+1:02d}: Min={col_valid.min():8.2f}, Max={col_valid.max():8.2f}, Mean={col_valid.mean():8.2f}")
    else:
        print(f"[ERROR] {traits_path} is missing!")

if __name__ == "__main__":
    validate_bioclim()
