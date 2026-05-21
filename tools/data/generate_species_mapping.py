import pandas as pd
import os

def generate_species_mapping() -> None:
    """
    Generate a CSV mapping of unique species IDs from training metadata.

    Scans the training metadata, extracts all unique species IDs, and saves
    them as a sorted list in a mapping file.

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
            
    unique_ids = sorted(list(set(all_labels)))
    print(f"Found {len(unique_ids)} unique species IDs.")
    
    # Pad to 7806 if needed (using dummy IDs or continuing the sequence)
    # We used 1 as count for padded classes, so they are the "extra" 2 classes.
    # In training we just padded the counts list. 
    # Here we should just save the real ones.
    
    out_path = '/workspace/plantclef/processed/species_ids_mapping.csv'
    with open(out_path, 'w') as f:
        for sid in unique_ids:
            f.write(f"{sid}\n")
            
    print(f"Species mapping saved to {out_path}")

if __name__ == "__main__":
    generate_species_mapping()
