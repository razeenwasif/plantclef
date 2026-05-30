import torch
import pandas as pd
import os
from typing import Dict, Any

def build_species_traits() -> None:
    """
    Builds a mapping of species indices to their most frequent botanical traits.

    This function reads processed botanical traits and train metadata, aggregates
    traits per species, and saves the most frequent trait for each category
    as the ground truth for distillation.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The resulting mapping is saved to 'data/species_traits.pth'.
    """
    processed_traits_path = "data/processed_botanical_traits.pth"
    csv_path = "/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv"
    output_path = "data/species_traits.pth"
    
    if not os.path.exists(processed_traits_path):
        print("Error: Processed traits not found.")
        return

    processed_data: Dict[str, Dict[str, int]] = torch.load(processed_traits_path)
    df: pd.DataFrame = pd.read_csv(csv_path, sep=';')
    
    # Mapping image_name -> species_id
    img_to_sid: Dict[str, int] = dict(zip(df['image_name'], df['species_id']))
    
    # Get unique species and their indices (matching dataloader)
    unique_species = sorted(df['species_id'].unique())
    species_to_idx: Dict[int, int] = {s: i for i, s in enumerate(unique_species)}
    num_classes: int = 7806 # Force consistency
    
    categories = ["leaf_shape", "phyllotaxy", "flower_color", "inflorescence", "stem_type"]
    
    # species_idx -> {category: {trait_idx: count}}
    species_traits: Dict[int, Dict[str, Dict[int, int]]] = {i: {cat: {} for cat in categories} for i in range(num_classes)}
    
    print("Aggregating traits per species...")
    for img_name, traits in processed_data.items():
        if img_name in img_to_sid:
            sid = img_to_sid[img_name]
            if sid in species_to_idx:
                s_idx = species_to_idx[sid]
                for cat, trait_idx in traits.items():
                    species_traits[s_idx][cat][trait_idx] = species_traits[s_idx][cat].get(trait_idx, 0) + 1
    
    # Take the most frequent trait per species as the "ground truth" for distillation
    final_species_traits: Dict[int, Dict[str, torch.Tensor]] = {}
    for s_idx, cats in species_traits.items():
        s_cats = {}
        for cat, counts in cats.items():
            if counts:
                # Get trait with max count
                best_trait = max(counts, key=counts.get)
                s_cats[cat] = torch.tensor(best_trait)
            else:
                s_cats[cat] = torch.tensor(-1) # Ignore index
        final_species_traits[s_idx] = s_cats

    print(f"Saving traits for {len(final_species_traits)} species to {output_path}...")
    torch.save(final_species_traits, output_path)

if __name__ == "__main__":
    build_species_traits()
