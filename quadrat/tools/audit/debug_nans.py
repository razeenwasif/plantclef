import os
import torch
import pandas as pd
import numpy as np
from typing import Optional

def debug() -> None:
    """
    Performs a stability diagnostic for Phase 6 of the training process.

    Checks training metadata CSV for label distributions, and validates 
    ecological traits and adjacency matrices for NaNs or unusual values 
    that could lead to training instability.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    print("--- Phase 6 Stability Diagnostic ---")
    
    # 1. Check CSV and Counts
    csv_path = "/workspace/plantclef/processed/student_train_final.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=';')
        all_labels = []
        for s in df['species_ids']:
            if isinstance(s, str) and ',' in s:
                all_labels.extend([int(x) for x in s.split(',')])
            else:
                all_labels.append(int(s))
        
        counts_ser = pd.Series(all_labels).value_counts().sort_index()
        counts = counts_ser.tolist()
        print(f"Num Classes: {len(counts)}")
        print(f"Min Samples: {min(counts)}")
        
        priors = np.array(counts) / sum(counts)
        logit_adj = np.log(priors)
        print(f"Logit Adj - Inf count: {np.isinf(logit_adj).sum()}")
        print(f"Logit Adj - NaN count: {np.isnan(logit_adj).sum()}")
    
    # 2. Check Ecological Assets
    traits_path = "data/ecological_traits.npy"
    if os.path.exists(traits_path):
        traits = np.load(traits_path)
        print(f"Traits - NaN count: {np.isnan(traits).sum()}")
        print(f"Traits - Max value: {np.max(traits)}")
        print(f"Traits - Min value: {np.min(traits)}")

    # 3. Check Adjacency
    adj_path = "data/ecological_adj.npy"
    if os.path.exists(adj_path):
        adj = np.load(adj_path)
        print(f"Adj - NaN count: {np.isnan(adj).sum()}")
        print(f"Adj - Max value: {np.max(adj)}")

if __name__ == "__main__":
    debug()
