"""Inference script for PlantCLEF 2026 using SAHI and ensemble models.

This script performs inference on a single image using a pre-trained ensemble model
and the SAHI (Slicing Aided Hyper Inference) engine for high-resolution processing.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

# Add workspace root to path to find local packages
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
src_dir = os.path.join(root_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src import config as scfg
from models.ensemble import PlantEnsemble
from models.sahi import CUDASahiEngine


def main() -> None:
    """
    Main execution function for PlantCLEF 2026 image prediction using SAHI.

    This function sets up the inference environment, loads the species mapping 
    from training metadata, initializes the ensemble model with LoRA, loads 
    the best available weights, and uses the CUDASahiEngine to perform 
    high-resolution sliced inference on a single input image.

    Parameters
    ----------
    None
        Arguments are parsed from the command line using argparse.

    Returns
    -------
    None
        Prints the top-k predictions and their confidence levels to the console.
    """
    parser = argparse.ArgumentParser(
        description="PlantCLEF 2026 - Quick Image Predictor")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "--topk", type=int, default=5, help="Number of top species to show")
    args = parser.parse_args()

    # 1. Setup
    train_csv_path = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    df_train = pd.read_csv(train_csv_path, sep=';')
    species_ids = sorted(df_train['species_id'].unique())
    idx_to_species = {i: s for i, s in enumerate(species_ids)}

    # 2. Model
    model_path = "models/best_calibrated/mp_rank_00_model_states.pt"
    if not os.path.exists(model_path):
        model_path = "models/final/mp_rank_00_model_states.pt"

    model = PlantEnsemble(
        num_classes=len(species_ids), input_res=config.RESOLUTION)
    model.apply_lora(r=config.LORA_R, lora_alpha=config.LORA_ALPHA)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location='cuda')
        state_dict = ckpt['module'] if isinstance(
            ckpt, dict) and 'module' in ckpt else ckpt
        new_state_dict = {
            k[7:] if k.startswith('module.') else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {model_path}")

    # 3. Engine
    engine = CUDASahiEngine(model, resolution=config.RESOLUTION)

    # 4. Predict
    print(f"Running inference on: {args.image}...")
    probs = engine.predict_image(args.image)

    if probs is not None:
        top_indices = np.argsort(probs)[-args.topk:][::-1]
        print("\n" + "=" * 40)
        print(f"      TOP {args.topk} SPECIES PREDICTIONS")
        print("=" * 40)
        for i, idx in enumerate(top_indices):
            s_id = idx_to_species[idx]
            conf = probs[idx]
            print(f" {i+1}. ID: {s_id:<10} | Confidence: {conf:.4f}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
