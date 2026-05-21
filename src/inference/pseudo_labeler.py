from __future__ import annotations

"""Generates pseudo-labels for unlabelled plant images using a teacher model."""

import glob
import os
import sys
from typing import Any

# Add workspace root to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(root_dir, 'src'))
# Also add project root
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src import config
import numpy as np
import pandas as pd
import torch
from src.models.ensemble import PlantEnsemble
from src.models.sahi import CUDASahiEngine
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PseudoDataset(Dataset):
    """
    Dataset for loading images for pseudo-labeling.

    Parameters
    ----------
    image_paths : list[str]
        List of paths to the images.
    """

    def __init__(self, image_paths: list[str]) -> None:
        """
        Initializes the dataset.

        Parameters
        ----------
        image_paths : list[str]
            List of paths to the images.
        """
        self.image_paths = image_paths

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns
        -------
        int
            The number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        Loads and returns an image and its path.

        Parameters
        ----------
        idx : int
            The index of the image to load.

        Returns
        -------
        tuple[torch.Tensor, str]
            A tuple containing the image tensor (C, H, W) and the file path.
        """
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            # Pre-resize to a reasonable size to save PCIe bandwidth
            # But keep it large enough for SAHI (e.g. 1024)
            img.thumbnail((1024, 1024))
            img_tensor = torch.from_numpy(
                np.array(img)).permute(2, 0, 1).float() / 255.0
            return img_tensor, path
        except (IOError, ValueError, RuntimeError) as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((3, 224, 224)), "ERROR"


def load_teacher_model(
    model_path: str | os.PathLike,
    num_classes: int,
    device: str = 'cuda'
) -> PlantEnsemble:
    """
    Loads the teacher ensemble model from a checkpoint.

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
        The loaded teacher model.
    """
    model = PlantEnsemble(num_classes=num_classes, input_res=config.RESOLUTION)
    model.apply_lora(r=config.LORA_R, lora_alpha=config.LORA_ALPHA)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt['module'] if isinstance(
            ckpt, dict) and 'module' in ckpt else ckpt
        new_state_dict = {
            k[7:] if k.startswith('module.') else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(new_state_dict, strict=False)
    return model


def main() -> None:
    """Unified Pseudo-Labeler and LUCAS Salvage engine."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--salvage_lucas", action="store_true")
    parser.add_argument("--test_set", action="store_true")
    parser.add_argument("--confidence_threshold", type=float, default=0.95)
    args_cli = parser.parse_args()

    base_dir = "/workspace/PlantCLEF2026"
    train_csv = config.CLEANED_CSV if os.path.exists(config.CLEANED_CSV) else config.RAW_CSV
    output_csv = "data/champion_train_set.csv"

    # 1. Load Model
    df_train = pd.read_csv(train_csv, sep=';', low_memory=False)
    num_classes = 7806
    model_path = config.SWA_CKPT_PATH
    
    print(f"[Champion] Loading Teacher Model from {model_path}...")
    model = PlantEnsemble(num_classes=num_classes, input_res=config.RESOLUTION).to('cuda').eval()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=False)
    
    engine = CUDASahiEngine(model, resolution=config.RESOLUTION)

    # 2. Target Identification
    image_paths = []
    
    if args_cli.salvage_lucas:
        print("[Champion] Mode: LUCAS Salvage Pass")
        lucas_df = df_train[df_train['image_name'].str.startswith("LUCAS")].copy()
        # Find physical paths for LUCAS
        for _, row in tqdm(lucas_df.iterrows(), total=len(lucas_df), desc="Verifying LUCAS"):
            sid = str(row['species_ids']).split(',')[0].strip()
            p = os.path.join(config.IMG_DIR, sid, row['image_name'])
            if os.path.exists(p): image_paths.append(p)
            
    if args_cli.test_set:
        print("[Champion] Mode: Competition Test Pass")
        test_images = glob.glob(os.path.join(config.IMG_DIR.replace("train", "test"), "**/*.jpg"), recursive=True)
        image_paths.extend(test_images)

    # 3. Mass Pseudo-Labeling
    dataset = PseudoDataset(image_paths)
    loader = DataLoader(dataset, batch_size=1, num_workers=16, pin_memory=True)
    
    new_samples = []
    for img_tensors, paths in tqdm(loader, desc="Pseudo-Labeling"):
        img_tensor = img_tensors[0].to('cuda', non_blocking=True)
        path = paths[0]
        
        with torch.no_grad():
            probs, _ = engine.predict_tensor(img_tensor, gated_threshold=0.90)
            conf = np.max(probs)
            if conf >= args_cli.confidence_threshold:
                new_samples.append({
                    "image_name": os.path.basename(path),
                    "species_ids": str(np.argmax(probs)), # Temporary internal index
                    "is_pseudo": True
                })

    # 4. Create Champion Dataset
    df_pseudo = pd.DataFrame(new_samples)
    df_final = pd.concat([df_train[~df_train['image_name'].str.startswith("LUCAS")], df_pseudo])
    df_final.to_csv(output_csv, sep=';', index=False)
    print(f"[Champion] SUCCESS: Unified train set saved to {output_csv} ({len(df_final):,} samples)")


if __name__ == "__main__":
    main()
