import torch
import pandas as pd
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Alignment Surgery for PlantCLEF 2026")
    parser.add_argument("--input", type=str, required=True, help="Path to misaligned .pth")
    parser.add_argument("--output", type=str, required=True, help="Path to save aligned .pth")
    parser.add_argument("--old_mapping", type=str, default="/workspace/plantclef/processed/species_ids_mapping.csv.bak")
    parser.add_argument("--new_mapping", type=str, default="/workspace/plantclef/processed/species_ids_mapping.csv")
    args = parser.parse_args()

    print(f"[Surgery] Loading mappings...")
    # Old mapping (can have header or not)
    df_old = pd.read_csv(args.old_mapping)
    # If the first value is not an ID-like integer, re-read without header
    if not str(df_old.iloc[0,0]).isdigit():
        df_old = pd.read_csv(args.old_mapping, header=None)
    old_ids = [int(x) for x in df_old.iloc[:,0].tolist() if str(x).isdigit()]
    
    # New mapping (no header)
    df_new = pd.read_csv(args.new_mapping, header=None)
    new_ids = [int(x) for x in df_new[0].tolist() if str(x).isdigit()]

    print(f"[Surgery] Old: {len(old_ids)} IDs | New: {len(new_ids)} IDs")

    # Create mapping: new_index -> old_index
    old_id_to_idx = {id: i for i, id in enumerate(old_ids)}
    
    # new_to_old[new_idx] = old_idx (or -1 if not found)
    new_to_old = []
    found_count = 0
    for nid in new_ids:
        if nid in old_id_to_idx:
            new_to_old.append(old_id_to_idx[nid])
            found_count += 1
        else:
            new_to_old.append(-1)
            
    print(f"[Surgery] Found {found_count} common species IDs.")

    print(f"[Surgery] Loading checkpoint: {args.input}")
    sd = torch.load(args.input, map_location="cpu")
    if "model_state" in sd:
        is_wrapped = True
        weights = sd["model_state"]
    else:
        is_wrapped = False
        weights = sd

    # 1. Align Classifier Weights & Biases
    # Patterns: species_classifier.weight, species_classifier.bias, species_classifier.traits
    # Also handles _orig_mod prefixes and Exp 008 'head.4' patterns.
    
    aligned_count = 0
    for k in list(weights.keys()):
        # We only touch keys that have 'species_classifier' or 'head.4' and are related to the class dimension
        is_classifier = "species_classifier" in k or "head.4" in k
        is_val_related = any(x in k for x in [".weight", ".bias", ".traits", ".adj"])
        
        if is_classifier and is_val_related:
            v = weights[k]
            # The class dimension is usually the first dimension [7808, ...] or [7806, ...]
            if v.shape[0] >= 7804:
                print(f"  -> Aligning {k} (shape: {list(v.shape)})")
                new_v = v.clone()
                new_v.zero_()
                
                for new_idx, old_idx in enumerate(new_to_old):
                    if old_idx != -1 and old_idx < v.shape[0]:
                        new_v[new_idx] = v[old_idx]
                
                weights[k] = new_v
                aligned_count += 1
                
        # 2. Align Warmup/Phase1 heads if they exist
        if any(x in k for x in ["warmup_head.final", "phase1_head.final"]):
             v = weights[k]
             if v.shape[0] >= 7804:
                print(f"  -> Aligning {k} (shape: {list(v.shape)})")
                new_v = v.clone()
                new_v.zero_()
                for new_idx, old_idx in enumerate(new_to_old):
                    if old_idx != -1:
                        new_v[new_idx] = v[old_idx]
                weights[k] = new_v
                aligned_count += 1

    print(f"[Surgery] Aligned {aligned_count} parameter tensors.")
    
    if is_wrapped:
        sd["model_state"] = weights
        torch.save(sd, args.output)
    else:
        torch.save(weights, args.output)
        
    print(f"[Surgery] Saved aligned model to: {args.output}")

if __name__ == "__main__":
    main()
