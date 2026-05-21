import os
import json
import pandas as pd

def build_taxonomic_data():
    print("Building Taxonomic Data for Rust AC-3 Solver...")
    
    # Paths
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    names_path = "/workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt"
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Species Indices (0 to 7805)
    # The model outputs probabilities in this exact order
    df_map = pd.read_csv(mapping_path, header=None, names=["species_id"])
    species_id_to_idx = {sid: idx for idx, sid in enumerate(df_map["species_id"])}
    num_classes = len(species_id_to_idx)
    
    # Load Species Names and extract Genus
    df_names = pd.read_csv(names_path, sep=';')
    
    # Map each model index (0-7805) to its Genus string
    idx_to_genus = {}
    for _, row in df_names.iterrows():
        sid = row["species_id"]
        if sid in species_id_to_idx:
            idx = species_id_to_idx[sid]
            # The first word of the scientific name is the Genus
            genus = str(row["species"]).split()[0]
            idx_to_genus[idx] = genus
            
    # Assign integer IDs to each unique Genus
    unique_genera = sorted(list(set(idx_to_genus.values())))
    genus_str_to_id = {g: i for i, g in enumerate(unique_genera)}
    
    # 1. species_to_genus (List of ints)
    species_to_genus = [0] * num_classes
    genus_to_species_idxs = {g_id: [] for g_id in genus_str_to_id.values()}
    
    for idx in range(num_classes):
        genus_str = idx_to_genus.get(idx, "Unknown")
        genus_id = genus_str_to_id.get(genus_str, 0)
        species_to_genus[idx] = genus_id
        genus_to_species_idxs[genus_id].append(idx)
        
    # 2. allowed_neighbors (AC-3 Graph)
    # A species is allowed to co-occur with any other species in the same Genus.
    # In a real ecological database, this would include cross-genus symbiotic relationships.
    allowed_neighbors = [[] for _ in range(num_classes)]
    for idx in range(num_classes):
        genus_id = species_to_genus[idx]
        # Add all species in the same genus as allowed neighbors
        allowed_neighbors[idx] = genus_to_species_idxs[genus_id]
        
    # 3. genus_to_family (List of ints)
    # We don't have family data in this basic mapping, so we map all genera to Family 0
    genus_to_family = [0] * len(unique_genera)
    
    # 4. bioclim_data (List of lists)
    # We don't have hard climate boundaries here, so we create 1 universal climate cluster
    # that allows all 7806 species, effectively disabling the spatial bloom filter.
    bioclim_data = [list(range(num_classes))]
    
    # Save all 4 JSON files required by the Rust TaxonomicFilter
    files_to_save = {
        "taxonomic_graph.json": allowed_neighbors,
        "species_to_genus.json": species_to_genus,
        "genus_to_family.json": genus_to_family,
        "bioclim_data.json": bioclim_data
    }
    
    for filename, data in files_to_save.items():
        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'w') as f:
            json.dump(data, f)
        print(f"Saved {out_path} ({len(data)} elements)")
        
    print("Taxonomic Data Generation Complete! The inference pipeline will now use AC-3 consistency.")

if __name__ == "__main__":
    build_taxonomic_data()
