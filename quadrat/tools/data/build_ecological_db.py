"""Optimized Ecological Niche Database Builder for PlantCLEF 2026.

This version implements:
1. Global GBIF Search (Removes SW Europe restriction)
2. Author-free species name cleaning for higher search success.
3. Real Ecological Adjacency calculation based on bioclimatic niche similarity.
"""

import os
import torch
import numpy as np
import pandas as pd
from pygbif import occurrences
from tqdm import tqdm
import rasterio
from sklearn.metrics.pairwise import cosine_similarity

class OptimizedEcologicalBuilder:
    """
    Builder for an optimized ecological niche database for plant species.

    Uses GBIF occurrence data and WorldClim bioclimatic variables to construct
    ecological niche profiles and similarity-based adjacency matrices.

    Parameters
    ----------
    name_map_path : str
        Path to the species name mapping file (CSV).
    bioclim_dir : str
        Directory containing WorldClim .tif files.
    """

    def __init__(self, name_map_path: str, bioclim_dir: str) -> None:
        """Initialize the builder."""
        self.bioclim_dir = bioclim_dir
        # Load scientific names and strip author names for better GBIF matching
        print("[DB Builder] Loading and cleaning species names...")
        df = pd.read_csv(name_map_path, sep=';')
        # Simple cleaning: take first two words (Genus species)
        df['clean_name'] = df['species'].apply(lambda x: " ".join(x.split()[:2]))
        self.species_list = df['clean_name'].tolist()
        self.num_classes = 7806
        self.transform = None
        self.climate_map = None

    def load_climate(self) -> None:
        """
        Load 19 Bioclimatic layers into memory.

        Returns
        -------
        None
        """
        print("[DB Builder] Loading 19 Bioclim layers...")
        layers = []
        for i in range(1, 20):
            path = os.path.join(self.bioclim_dir, f"wc2.1_2.5m_bio_{i}.tif")
            with rasterio.open(path) as src:
                if self.transform is None:
                    self.transform = ~src.transform
                data = src.read(1)
                data[data == src.nodata] = 0
                layers.append(torch.from_numpy(data))
        self.climate_map = torch.stack(layers, dim=0)

    def get_global_coords(self, name: str, limit: int = 30) -> list[tuple[float, float]]:
        """
        Search GBIF for occurrence coordinates of a given species.

        Parameters
        ----------
        name : str
            Scientific name of the species.
        limit : int, optional
            Maximum number of occurrences to retrieve (default is 30).

        Returns
        -------
        list of tuple
            A list of (latitude, longitude) tuples.
        """
        try:
            # Removed geometry restriction for global coverage
            res = occurrences.search(scientificName=name, limit=limit, hasCoordinate=True)
            return [(r['decimalLatitude'], r['decimalLongitude']) for r in res.get('results', [])]
        except:
            return []

    def build(self, output_dir: str = "data") -> None:
        """
        Construct the niche profiles and the ecological adjacency matrix.

        Samples climate variables at occurrence coordinates and computes
        cosine similarity between species to build the adjacency matrix.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save the resulting .npy files (default is "data").

        Returns
        -------
        None
        """
        self.load_climate()
        traits_db = np.zeros((self.num_classes, 19))
        
        print(f"[DB Builder] Sampling climate for {self.num_classes} species...")
        for i, name in enumerate(tqdm(self.species_list)):
            coords = self.get_global_coords(name)
            if not coords: continue
            
            lats, lons = zip(*coords)
            # Fix: Affine transform requires numpy arrays, not raw tuples/lists
            lons_arr, lats_arr = np.array(lons), np.array(lats)
            cols, rows = self.transform * (lons_arr, lats_arr)
            rows = np.clip(np.floor(rows).astype(int), 0, self.climate_map.shape[1]-1)
            cols = np.clip(np.floor(cols).astype(int), 0, self.climate_map.shape[2]-1)
            
            samples = self.climate_map[:, rows, cols].float()
            traits_db[i] = samples.mean(dim=1).numpy()

        # 1. Save Traits
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "ecological_traits.npy"), traits_db)
        
        # 2. Calculate Real Adjacency Matrix
        print("[DB Builder] Calculating niche similarity adjacency...")
        # Normalize traits for cosine similarity
        norm_traits = (traits_db - traits_db.mean(axis=0)) / (traits_db.std(axis=0) + 1e-6)
        adj = cosine_similarity(norm_traits)
        # Apply threshold and sparsify to keep only meaningful relationships
        adj[adj < 0.7] = 0
        np.save(os.path.join(output_dir, "ecological_adj.npy"), adj)
        
        print(f"[DB Builder] Success! Traits and Adjacency saved to {output_dir}")

if __name__ == "__main__":
    builder = OptimizedEcologicalBuilder(
        "/workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt",
        "/workspace/plantclef/raw/worldclim"
    )
    builder.build()
