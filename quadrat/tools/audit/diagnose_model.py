import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.inference.model_runner import PlantEnsembleAdapter
from typing import Optional

def diagnose() -> None:
    """
    Performs a single-sample forward pass diagnostic on a loaded model.

    Loads the PlantEnsemble model, a sample image, and logit adjustments 
    to verify that the prediction pipeline is functioning correctly, 
    calculating top-5 predictions and comparing them to ground truth.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    model_path = "/workspace/PlantCLEF2026/models/best_calibrated/mp_rank_00_model_states.pt"
    train_csv = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    adj_path = "/workspace/plantclef/processed/logit_adj.npy"
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    
    print("--- Model Diagnostic Pass ---")
    
    # 1. Load Assets
    adapter = PlantEnsembleAdapter(model_path, num_classes_=7806, input_res=384)
    model = adapter.get_model().to("cuda")
    logit_adj = np.load(adj_path)
    
    with open(mapping_path, 'r') as f:
        species_list = [int(line.strip()) for line in f if line.strip()]
    species_to_idx = {sid: i for i, sid in enumerate(species_list)}

    # 2. Pick one sample
    df = pd.read_csv(train_csv, sep=';')
    sample = df.iloc[0]
    img_path = f"/workspace/plantclef/raw/train/images_max_side_800/{sample['species_id']}/{sample['image_name']}"
    
    correct_sid = sample['species_id']
    correct_idx = species_to_idx[correct_sid]
    
    print(f"Sample: {sample['image_name']}")
    print(f"Ground Truth Species: {correct_sid}")
    print(f"Expected Index:       {correct_idx}")

    # 3. Predict
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to("cuda")
    
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, tuple): logits = logits[0]
        logits = logits.cpu().numpy()[0]

    # 4. Analyze
    print("\nLogit Stats:")
    print(f"  - Min: {logits.min():.4f}, Max: {logits.max():.4f}, Mean: {logits.mean():.4f}")
    print(f"  - Logit at Correct Index ({correct_idx}): {logits[correct_idx]:.4f}")
    print(f"  - Adjustment at Correct Index:      {logit_adj[correct_idx]:.4f}")
    
    adjusted = logits + logit_adj
    top_5_indices = np.argsort(adjusted)[:7804][::-1][:5]
    
    print("\nTop 5 Predictions (Calibrated):")
    for rank, idx in enumerate(top_5_indices):
        sid = species_list[idx]
        print(f"  {rank+1}. Index {idx}: ID {sid}, Score: {adjusted[idx]:.4f} (Raw: {logits[idx]:.4f})")

if __name__ == "__main__":
    diagnose()
