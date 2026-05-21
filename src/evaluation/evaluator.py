from __future__ import annotations

import os
import sys
from typing import Any

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import glob
import json

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# Also add src/
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src import config
from src.models.ensemble import PlantEnsemble
from src.models.sahi import CUDASahiEngine

def load_competition_model(
    model_path: str | os.PathLike,
    num_classes: int,
    device: str = "cuda"
) -> PlantEnsemble:
    """
    Initializes and loads the ensemble model from a checkpoint.

    Parameters
    ----------
    model_path : str | os.PathLike
        Path to the model checkpoint.
    num_classes : int
        Number of output classes.
    device : str, optional
        Device to load the model on (default is 'cuda').

    Returns
    -------
    PlantEnsemble
        The loaded ensemble model.
    """
    model = PlantEnsemble(num_classes=num_classes, input_res=config.RESOLUTION)
    model.apply_lora(r=config.LORA_R, lora_alpha=config.LORA_ALPHA)
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt['module'] if isinstance(ckpt, dict) and 'module' in ckpt else ckpt
        # Clean state dict keys
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {model_path}")
    
    return model

def main() -> None:
    """
    Main inference script for PlantCLEF 2026.

    Returns
    -------
    None
    """
    base_dir = "/workspace/PlantCLEF2026"
    train_csv = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    test_dir = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/"
    gt_csv = "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test_labels.csv"
    output_csv = os.path.join(base_dir, "submission_plantclef2026.csv")

    # 1. Setup Species Mapping
    df_train = pd.read_csv(train_csv, sep=';')
    species_ids = sorted(df_train['species_id'].unique())
    idx_to_species = {i: s for i, s in enumerate(species_ids)}
    
    # 2. Initialize Model and Engine
    model_path = os.path.join(base_dir, "models/best_calibrated/mp_rank_00_model_states.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "models/final/mp_rank_00_model_states.pt")
        
    model = load_competition_model(model_path, len(species_ids))
    
    # Initialize Engine with Bloom Calendar
    bloom_json = "/workspace/plantclef/processed/species_bloom_calendar.json"
    engine = CUDASahiEngine(model, resolution=config.RESOLUTION, bloom_calendar_path=bloom_json)
    engine.set_pheno_mapping(species_to_idx={int(sid): i for i, sid in enumerate(species_ids)})
    
    # 3. Process Images
    test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    
    # Load Test Metadata for Month extraction
    df_test_meta = pd.read_csv(os.path.join("/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/", "PlantCLEF2025_test.csv"), sep=';')
    quadrat_to_month = {row['quadrat_id']: int(str(row['date']).split('-')[1]) for _, row in df_test_meta.iterrows()}

    # Load Optimized Thresholds (Brent's Method result)
    thresh_path = "models/optimized_thresholds.json"
    class_thresholds = None
    if os.path.exists(thresh_path):
        with open(thresh_path, 'r') as f:
            class_thresholds = json.load(f)
        print(f"Loaded optimized thresholds for {len(class_thresholds)} species.")
    else:
        print("Optimized thresholds not found. Using default (0.05).")

    results = []
    print(f"Starting CUDA-Accelerated SAHI with Phenological Filtering on {len(test_images)} images...")
    
    for img_path in tqdm(test_images):
        quadrat_id = os.path.splitext(os.path.basename(img_path))[0]
        # Get month for this quadrat
        month = quadrat_to_month.get(quadrat_id, None)
        
        probs = engine.predict_image(img_path, month=month)
        
        if probs is not None:
            # Predict top species using per-class thresholds
            top_indices = np.argsort(probs)[::-1]
            preds = [top_indices[0]] # Always keep Top-1
            
            for i in top_indices[1:20]: # Check top 20 candidates
                s_id = str(idx_to_species[i])
                thresh = class_thresholds.get(s_id, 0.05) if class_thresholds else 0.05
                if probs[i] > thresh:
                    preds.append(i)
            
            species = [idx_to_species[p] for p in preds]
            results.append({"quadrat_id": quadrat_id, "species_ids": species})

    # 4. Save and Report
    df = pd.DataFrame(results)
    df_csv = df.copy()
    df_csv['species_ids'] = df_csv['species_ids'].apply(lambda x: "[" + ", ".join(map(str, x)) + "]")
    df_csv.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"Submission saved to {output_csv}")

    # Accuracy check
    if os.path.exists(gt_csv):
        df_gt = pd.read_csv(gt_csv, sep=';')
        gt_dict = df_gt.groupby('quadrat_id')['species_id'].apply(set).to_dict()
        f1s = []
        for res in results:
            qid = res['quadrat_id']
            true_set = gt_dict.get(qid, set())
            pred_set = set(res['species_ids'])
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            f1s.append((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0)
        print(f"\n ► Final Macro-F1 Score: {np.mean(f1s):.4f}")

if __name__ == "__main__":
    main()
