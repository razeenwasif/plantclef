import pandas as pd
import numpy as np
import os
import torch
import json
from collections import defaultdict

# --- CONFIG ---
LOG_DIR = "models/ensemble_members/expert_launcher_s42/"
SPECIES_MAP_CSV = "/workspace/plantclef/processed/species_ids_mapping.csv"
OUTPUT_REPORT = "reports/species_difficulty_map.json"

def main():
    os.makedirs("reports", exist_ok=True)
    
    print("[*] Initializing Species Audit...")
    
    if not os.path.exists(SPECIES_MAP_CSV):
        print(f"[Error] Species mapping not found at {SPECIES_MAP_CSV}")
        return

    # Load mapping
    df_map = pd.read_csv(SPECIES_MAP_CSV)
    idx_to_name = dict(zip(df_map['species_id'], df_map.get('name', df_map['species_id'])))

    # 1. Scan for Log Files
    val_results_path = os.path.join(LOG_DIR, "latest_val_results.pt")
    
    if not os.path.exists(val_results_path):
        print(f"[!] Validation results not found at {val_results_path}.")
        print("[!] Tip: The audit data is saved after each validation phase (every 5 epochs).")
        return

    try:
        data = torch.load(val_results_path, map_location='cpu')
        y_true = data['targets']
        y_pred = data['preds']
    except Exception as e:
        print(f"[Error] Failed to load results: {e}")
        return
    
    # 2. Calculate Error Density per Species
    species_errors = defaultdict(int)
    species_counts = defaultdict(int)
    
    for t, p in zip(y_true, y_pred):
        # Convert to standard python int
        t_id = int(t)
        p_id = int(p)
        species_counts[t_id] += 1
        if t_id != p_id:
            species_errors[t_id] += 1
            
    # 3. Rank "Problematic" Species
    difficulty_map = []
    for sid in species_counts:
        acc = 1.0 - (species_errors[sid] / species_counts[sid])
        difficulty_map.append({
            "species_id": sid,
            "name": str(idx_to_name.get(sid, "Unknown")),
            "accuracy": float(acc),
            "samples": int(species_counts[sid]),
            "difficulty": "HARD" if acc < 0.3 else "MEDIUM" if acc < 0.7 else "EASY"
        })
        
    difficulty_map.sort(key=lambda x: x['accuracy'])

    # 4. Export Focused Report
    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(difficulty_map, f, indent=4)
        
    print(f"[+] AUDIT COMPLETE: {len(difficulty_map)} species analyzed.")
    print(f"[+] Top 5 HARDEST species (Candidate for FAISS boosting):")
    for s in difficulty_map[:5]:
        print(f"  - ID: {s['species_id']} | Acc: {s['accuracy']:.1%} | Name: {s['name']}")

if __name__ == "__main__":
    main()
