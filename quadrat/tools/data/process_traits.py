import json
import os
import torch
from tqdm import tqdm

def process_traits() -> None:
    """
    Process raw JSONL botanical traits into a consolidated PyTorch format.

    Parses the extraction outputs, maps textual labels to indices using a
    predefined vocabulary, and saves the resulting dictionary as a .pth file.

    Returns
    -------
    None
    """
    trait_json = "/workspace/plantclef/processed/botanical_traits.jsonl"
    output_path = "data/processed_botanical_traits.pth"
    
    if not os.path.exists(trait_json):
        print(f"Error: {trait_json} not found.")
        return

    vocabs = {
        "leaf_shape": ["ovate", "lanceolate", "cordate", "elliptic", "linear", "oblong", "peltate", "palmate", "sagittate", "spatulate", "deltoid", "needle-like"],
        "phyllotaxy": ["alternate", "opposite", "whorled", "basal"],
        "flower_color": ["yellow", "white", "blue", "red", "purple", "pink", "orange", "green"],
        "inflorescence": ["spike", "umbel", "cyme", "raceme", "panicle", "solitary"],
        "stem_type": ["woody", "herbaceous", "succulent", "vine", "climbing"]
    }
    
    # Create reverse mapping for fast lookup
    vocab_maps = {cat: {trait: i for i, trait in enumerate(traits)} for cat, traits in vocabs.items()}
    
    processed_data = {}
    
    print(f"Processing {trait_json}...")
    with open(trait_json, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                img_name = item.get('image_name')
                if not img_name: continue
                
                # Model output can be in 'traits' or 'raw_output'
                traits_raw = item.get('traits', {})
                if not traits_raw and 'raw_output' in item:
                    # Try to find JSON in raw text
                    txt = item['raw_output'].lower()
                    traits_raw = {}
                    for cat, keywords in vocabs.items():
                        for k in keywords:
                            if k in txt:
                                traits_raw[cat] = k
                                break
                
                # Map to indices
                img_traits = {}
                for cat, val in traits_raw.items():
                    if cat in vocab_maps and isinstance(val, str):
                        val_clean = val.lower().strip()
                        if val_clean in vocab_maps[cat]:
                            img_traits[cat] = vocab_maps[cat][val_clean]
                
                if img_traits:
                    processed_data[img_name] = img_traits
                    
            except Exception as e:
                continue

    print(f"Saving {len(processed_data)} processed items to {output_path}...")
    torch.save(processed_data, output_path)

if __name__ == "__main__":
    process_traits()
