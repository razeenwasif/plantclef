#!/usr/bin/env python3
"""
PLANTCLEF Phase: Elite SOTA Pre-Compilation
Builds offline environmental tensors (Phenology & Allelopathy) 
from the raw dataset metadata, so the inference loop runs at O(1) latency.
"""
import os
import json
import torch
import csv
from tqdm import tqdm

def load_mapping(mapping_path):
    mapping = {}
    with open(mapping_path, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if row:
                mapping[int(row[0].strip())] = idx
    return mapping

def build_phenology_matrix(mapping, bloom_path, out_path):
    print(f"[*] Building Thermodynamic Phenology Matrix from {bloom_path}...")
    num_classes = len(mapping)
    
    # [num_classes, 12 months]
    # 1.0 means blooming/visible, 0.0 means completely out of season
    pheno_matrix = torch.zeros((num_classes, 12), dtype=torch.float32)
    
    if not os.path.exists(bloom_path):
        print("[!] Bloom calendar not found, falling back to neutral (1.0).")
        pheno_matrix += 1.0
        torch.save(pheno_matrix, out_path)
        return
        
    with open(bloom_path, 'r') as f:
        bloom_data = json.load(f)
        
    mapped_count = 0
    for raw_id_str, bitmask in bloom_data.items():
        raw_id = int(raw_id_str)
        if raw_id in mapping:
            cls_idx = mapping[raw_id]
            # Bitmask: Month 1 (Jan) is bit 0, Month 12 (Dec) is bit 11
            for m in range(12):
                if (bitmask & (1 << m)):
                    pheno_matrix[cls_idx, m] = 1.0
            mapped_count += 1
            
    # Fill missing species with 1.0 (neutral) so they aren't unfairly penalized
    missing = num_classes - mapped_count
    for i in range(num_classes):
        if pheno_matrix[i].sum() == 0:
            pheno_matrix[i] += 1.0
            
    # Apply a slight Gaussian smoothing (thermal tolerance) 
    # Plants don't instantly disappear on the 1st of the month
    smoothed_matrix = torch.zeros_like(pheno_matrix)
    for m in range(12):
        prev_m = (m - 1) % 12
        next_m = (m + 1) % 12
        smoothed_matrix[:, m] = pheno_matrix[:, m] + 0.3 * pheno_matrix[:, prev_m] + 0.3 * pheno_matrix[:, next_m]
    
    smoothed_matrix = torch.clamp(smoothed_matrix, 0.0, 1.0)
    
    torch.save(smoothed_matrix, out_path)
    print(f"[*] Saved Phenology Matrix [C, 12] to {out_path} ({mapped_count} mapped, {missing} defaulted).")


def build_allelopathy_matrix(mapping, graph_path, out_path):
    print(f"[*] Building Structural Allelopathy (Repulsion) Matrix from {graph_path}...")
    num_classes = len(mapping)
    
    if not os.path.exists(graph_path):
        print("[!] Taxonomic graph not found. Saving neutral matrix.")
        torch.save(torch.zeros((num_classes, num_classes)), out_path)
        return
        
    with open(graph_path, 'r') as f:
        adj = json.load(f)
        
    # We build a dense repulsion matrix
    # 0.0 = Neutral or positive co-occurrence
    # -1.0 = Repulsion (Species never co-occur and are topologically distant)
    repulsion = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    
    # Since checking graph distance > 2 for an 7806^2 matrix is heavy, 
    # we do a 2-hop neighborhood expansion.
    print("  -> Computing 2-hop neighborhoods...")
    
    # 1. Pad graph if it's slightly smaller than mapping
    while len(adj) < num_classes:
        adj.append([len(adj)])
        
    # 2. Compute
    for i in tqdm(range(num_classes)):
        neighbors = set(adj[i])
        hop2 = set()
        for n in neighbors:
            if n < num_classes:
                hop2.update(adj[n])
                
        # The "Safe Zone" is the 2-hop neighborhood. 
        # Anything outside the safe zone gets a slight negative penalty (-0.1)
        # to gently push the Frank-Wolfe solver toward tighter ecological niches.
        safe_zone = neighbors.union(hop2)
        
        # Apply penalty to everything
        repulsion[i, :] = -0.1
        # Remove penalty for safe zone
        safe_list = [x for x in safe_zone if x < num_classes]
        repulsion[i, safe_list] = 0.0
        
    torch.save(repulsion, out_path)
    print(f"[*] Saved Allelopathic Repulsion Matrix [C, C] to {out_path}.")

def build_geochem_matrix(mapping, eive_path, out_path):
    """Builds a [C, C] ecological-incompatibility (geochem/edaphic) penalty matrix
    from precomputed EIVE pairwise compatibility scores.

    EIVE indicator values cover light, temperature, moisture, soil reaction (pH),
    nutrients, and salinity — i.e., the same axes that geochem/edaphic priors care
    about. The compatibility matrix is mostly 1.0 (compatible) with a sparse
    minority of strongly-incompatible pairs (down to ~0.003).

    We invert to a repulsion matrix:  J = -(1.0 - compat)   (∈ [-1, 0])
    """
    import numpy as np
    print(f"[*] Building Geochem Repulsion Matrix from {eive_path}...")
    num_classes = len(mapping)

    if not os.path.exists(eive_path):
        print(f"[!] EIVE compat matrix not found at {eive_path}. Saving zero matrix.")
        torch.save(torch.zeros((num_classes, num_classes)), out_path)
        return

    compat = np.load(eive_path).astype(np.float32)  # may be (7804, 7804) if 2 species are unmapped
    src_n = compat.shape[0]
    print(f"  -> Loaded EIVE compat: shape={compat.shape}, "
          f"frac_incompat={(compat < 1.0).mean():.4%}, min={compat.min():.3f}")

    # Pad to (num_classes, num_classes) with full-compat (1.0) so unmapped species are neutral.
    if src_n != num_classes:
        padded = np.ones((num_classes, num_classes), dtype=np.float32)
        n = min(src_n, num_classes)
        padded[:n, :n] = compat[:n, :n]
        compat = padded
        print(f"  -> Padded to ({num_classes}, {num_classes}) with neutral 1.0 for missing species.")

    # Repulsion: 0 for compatible pairs, negative for incompatible.
    # Diagonal stays 0 (self-interaction is meaningless for repulsion).
    J = -(1.0 - compat)
    np.fill_diagonal(J, 0.0)

    J_t = torch.from_numpy(J)
    torch.save(J_t, out_path)
    nz = (J_t != 0).sum().item()
    print(f"[*] Saved Geochem Repulsion Matrix [C, C] to {out_path} "
          f"({nz:,} non-zero entries; mean penalty={J_t[J_t < 0].mean().item():.4f}).")


if __name__ == "__main__":
    os.makedirs("models/priors", exist_ok=True)
    mapping = load_mapping("/workspace/plantclef/processed/species_ids_mapping.csv")

    build_phenology_matrix(
        mapping,
        bloom_path="/workspace/plantclef/processed/species_bloom_calendar.json",
        out_path="models/priors/phenology_matrix.pt"
    )

    build_allelopathy_matrix(
        mapping,
        graph_path="data/taxonomic_graph.json",
        out_path="models/priors/allelopathy_matrix.pt"
    )

    build_geochem_matrix(
        mapping,
        eive_path="data/eive_compatibility.npy",
        out_path="models/priors/geochem_matrix.pt"
    )
    print("\n✅ Environmental Priors Compilation Complete.")
