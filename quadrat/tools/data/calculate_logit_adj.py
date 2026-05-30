import pandas as pd
import numpy as np
import os

def calculate_logit_adj() -> None:
    """
    Calculate the logit adjustment for species based on their training frequency.

    Scans the training metadata, counts unique species occurrences, and calculates
    the prior probability distribution for logit-adjustment loss.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    csv_path = '/workspace/plantclef/processed/student_train_final.csv'
    df = pd.read_csv(csv_path, sep=';', dtype={'species_ids': str})
    
    all_labels = []
    for s in df['species_ids']:
        if isinstance(s, str) and ',' in s:
            all_labels.extend([int(x) for x in s.split(',')])
        else:
            all_labels.append(int(s))
            
    # Unique IDs sorted numerically
    unique_counts = pd.Series(all_labels).value_counts().sort_index()
    counts_list = unique_counts.tolist()
    
    print(f"Found {len(counts_list)} unique species IDs.")
    
    if len(counts_list) != 7806:
        print(f"Warning: Found {len(counts_list)} unique IDs, but expected 7806. Padding...")
        while len(counts_list) < 7806:
            counts_list.append(1)
            
    priors = np.array(counts_list) / sum(counts_list)
    logit_adj = np.log(priors)
    
    out_path = '/workspace/plantclef/processed/logit_adj.npy'
    np.save(out_path, logit_adj)
    print(f"Logit adjustment saved to {out_path}")
    print(f"Min count: {min(counts_list)} | Max count: {max(counts_list)}")

if __name__ == "__main__":
    calculate_logit_adj()
