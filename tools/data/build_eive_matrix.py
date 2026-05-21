import os
import json
import pandas as pd
import numpy as np

def build_eive_matrix():
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    names_path = "/workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt"
    eive_path = "/workspace/plantclef/raw/eive/EIVE_1.0_taxonomy_main.csv"
    output_path = "data/eive_compatibility.npy"
    
    # 1. Load Species Indices (0 to 7805)
    df_map = pd.read_csv(mapping_path, header=None, names=["species_id"])
    species_id_to_idx = {int(sid): idx for idx, sid in enumerate(df_map["species_id"])}
    num_classes = len(species_id_to_idx)
    
    # 2. Load Species Names
    df_names = pd.read_csv(names_path, sep=';')
    idx_to_name = {}
    for _, row in df_names.iterrows():
        sid = int(row["species_id"])
        if sid in species_id_to_idx:
            idx = species_id_to_idx[sid]
            idx_to_name[idx] = str(row["species"]).strip()
            
    # 3. Load EIVE Data
    df_eive = pd.read_csv(eive_path)
    df_eive = df_eive.drop_duplicates(subset=["TaxonConcept"]).set_index("TaxonConcept")
    
    indicators = ['M', 'N', 'R', 'L', 'T']
    mu = np.zeros((num_classes, len(indicators)), dtype=np.float32)
    sigma = np.ones((num_classes, len(indicators)), dtype=np.float32) * 1.5
    
    for idx in range(num_classes):
        name = idx_to_name.get(idx, "")
        if name in df_eive.index:
            row = df_eive.loc[name]
            for i, ind in enumerate(indicators):
                m_val = row.get(f'EIVEres-{ind}')
                s_val = row.get(f'EIVEres-{ind}.nw3')
                
                if pd.notna(m_val) and pd.notna(s_val) and s_val > 0:
                    mu[idx, i] = m_val
                    sigma[idx, i] = s_val
                    
    print("Computing BC matrix...")
    
    mu_A = mu[:, np.newaxis, :]
    mu_B = mu[np.newaxis, :, :]
    sig_A = sigma[:, np.newaxis, :]
    sig_B = sigma[np.newaxis, :, :]
    
    sig2_A = sig_A ** 2
    sig2_B = sig_B ** 2
    sig2_sum = sig2_A + sig2_B
    
    term1 = np.sqrt((2 * sig_A * sig_B) / sig2_sum)
    term2 = np.exp(- ((mu_A - mu_B) ** 2) / (4 * sig2_sum))
    
    bc_all = term1 * term2
    compatibility_matrix = np.prod(bc_all, axis=2)
    
    np.fill_diagonal(compatibility_matrix, 1.0)
    
    print(f"Matrix shape: {compatibility_matrix.shape}")
    print(f"Min BC: {compatibility_matrix.min():.4f}, Max BC: {compatibility_matrix.max():.4f}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, compatibility_matrix.astype(np.float32))
    print(f"Saved compatibility matrix to {output_path}")

if __name__ == "__main__":
    build_eive_matrix()
