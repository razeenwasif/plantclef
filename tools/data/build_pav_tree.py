import os
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from pygbif import species as sp
from pathlib import Path

# --- CONFIG ---
MAPPING_PATH = "/workspace/plantclef/processed/species_ids_mapping.csv"
NAMES_PATH = "/workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt"
EIVE_DATA_PATH = "/workspace/plantclef/raw/eive/EIVE_1.0_taxonomy_main.csv"
OUTPUT_PATH = "data/pav_taxonomy_tree.json"

def get_taxonomy_for_genus(genus_name):
    """Fetches family and order for a genus from GBIF."""
    try:
        result = sp.name_backbone(name=genus_name, rank='genus', kingdom='Plantae')
        return {
            'genus': genus_name,
            'family': result.get('family', 'Unknown Family'),
            'order': result.get('order', 'Unknown Order')
        }
    except Exception:
        return {'genus': genus_name, 'family': 'Unknown Family', 'order': 'Unknown Order'}

def build_pav_tree():
    print("🌿 ULTRA-ORACLE: Building Hierarchical PAV-Tree...")
    
    # 1. Load Species List
    df_names = pd.read_csv(NAMES_PATH, sep=';')
    df_map = pd.read_csv(MAPPING_PATH, header=None, names=["species_id"])
    
    # Model ID to Name Mapping
    id_to_name = {row['species_id']: row['species'] for _, row in df_names.iterrows()}
    model_species = []
    for sid in df_map['species_id']:
        model_species.append(id_to_name.get(sid, "Unknown Species"))

    # 2. Extract Unique Genera
    unique_genera = sorted(list(set(name.split()[0] for name in model_species)))
    print(f"[*] Found {len(unique_genera)} unique genera. Fetching higher taxonomy...")

    # 3. GBIF Lookup for Higher Levels (with caching to avoid redundant calls)
    genus_to_higher = {}
    for genus in tqdm(unique_genera, desc="GBIF Lookup"):
        genus_to_higher[genus] = get_taxonomy_for_genus(genus)
        # Be nice to GBIF API
        time.sleep(0.1)

    # 4. Ingest EIVE Ecological Vitals (Niche Widths)
    # We use SD (Standard Deviation) as an inverse reliability weight
    print("[*] Ingesting EIVE Ecological Indicator Values...")
    eive = pd.read_csv(EIVE_DATA_PATH)
    # Map species name to its niche stability (mean of SDs across L, M, R, N)
    # Lower SD = higher reliability (specialist)
    species_reliability = {}
    for _, row in eive.iterrows():
        s_name = row['TaxonConcept']
        # Extract SD columns
        sds = [row.get('EIVEres-L.nw3', 1.0), row.get('EIVEres-M.nw3', 1.0), 
               row.get('EIVEres-R.nw3', 1.0), row.get('EIVEres-N.nw3', 1.0)]
        avg_sd = np.nanmean(sds) if not np.all(np.isnan(sds)) else 1.0
        # Reliability is inverse of niche width
        species_reliability[s_name] = round(1.0 / (avg_sd + 0.1), 4)

    # 5. Assemble Hierarchy
    # Structure: Root -> Order -> Family -> Genus -> Species
    tree = {"name": "Plantae", "children": {}}
    
    for idx, s_name in enumerate(model_species):
        genus = s_name.split()[0]
        higher = genus_to_higher.get(genus, {})
        family = higher.get('family', 'Unknown Family')
        order = higher.get('order', 'Unknown Order')
        
        # Build Path
        if order not in tree["children"]: tree["children"][order] = {"children": {}}
        if family not in tree["children"][order]["children"]: tree["children"][order]["children"][family] = {"children": {}}
        if genus not in tree["children"][order]["children"][family]["children"]: 
            tree["children"][order]["children"][family]["children"][genus] = {"children": {}}
            
        # Add Species Leaf
        tree["children"][order]["children"][family]["children"][genus]["children"][s_name] = {
            "idx": idx,
            "weight": species_reliability.get(s_name, 0.5) # Default weight if no EIVE data
        }

    # 6. Save Tree
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(tree, f, indent=4)
    
    print(f"✅ PAV-Tree Built: {OUTPUT_PATH}")
    print(f"[*] Total Levels: 4 (Species -> Genus -> Family -> Order)")

if __name__ == "__main__":
    build_pav_tree()
