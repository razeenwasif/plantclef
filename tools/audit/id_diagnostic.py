import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import joblib

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.ensemble import PlantEnsemble

def identification_diagnostic() -> None:
    """
    Performs a detailed diagnostic of the species identification model.

    This function loads the ensemble model, its trained weights, PCA transform,
    and species mapping. It then selects a sample from the training metadata,
    preprocesses it (including a center crop), runs inference, and displays
     the top 5 predicted species along with their raw logits and indices.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Prints the diagnostic results to the console.
    """
    model_path = "/workspace/PlantCLEF2026/models/phase2_checkpoint_ep21_step_final.pth"
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    pca_path = "/workspace/PlantCLEF2026/models/blackwell_v2/pca_transform.pkl"
    train_csv = "/workspace/plantclef/processed/train_metadata_cleaned_verified.csv"
    
    print("--- Species Identification Diagnostic ---")
    
    # 1. Load Everything
    model = PlantEnsemble(num_classes=7806, input_res=384).to("cuda")
    ckpt = torch.load(model_path, map_location="cuda", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    
    pca = joblib.load(pca_path)
    with open(mapping_path, 'r') as f:
        species_list = [int(line.strip()) for line in f if line.strip()]

    # 2. Pick a "Strong" sample (first one)
    df = pd.read_csv(train_csv, sep=';')
    sample = df.iloc[0]
    img_path = f"/workspace/plantclef/raw/train/images_max_side_800/{sample['species_id']}/{sample['image_name']}"
    
    print(f"Sample Image: {sample['image_name']}")
    print(f"CSV Species ID: {sample['species_id']}")

    # 3. Inference
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    center_crop = transforms.CenterCrop(224)
    
    img = Image.open(img_path).convert("RGB")
    t_full = preprocess(img).unsqueeze(0).to("cuda")
    t_crop = preprocess(center_crop(img).resize((384, 384), Image.BILINEAR)).unsqueeze(0).to("cuda")
    
    model.eval()
    with torch.no_grad():
        f1 = torch.cat([model.bioclip(t_full), model.dinov3(t_full), model.convnext(t_full)], dim=1)
        f2 = torch.cat([model.bioclip(t_crop), model.dinov3(t_crop), model.convnext(t_crop)], dim=1)
        f_fused = torch.cat([f1, f2], dim=1)
        f_pca = torch.from_numpy(pca.transform(f_fused.cpu().numpy())).to("cuda").float()
        logits = model.phase1_head(f_pca)[0].cpu().numpy()

    # 4. Results
    top_5_idx = np.argsort(logits)[::-1][:5]
    print("\nModel's Top 5 Guesses:")
    for i, idx in enumerate(top_5_idx):
        # We check both the mapping file AND the raw index
        sid = species_list[idx] if idx < len(species_list) else "OUT_OF_BOUNDS"
        print(f"  {i+1}. Index {idx}: ID {sid}, Raw Logit: {logits[idx]:.4f}")

if __name__ == "__main__":
    identification_diagnostic()
