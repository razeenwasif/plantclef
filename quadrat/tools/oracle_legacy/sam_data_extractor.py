"""Extracts plant stickers from images using the Segment Anything Model (SAM)."""

import os
import sys
import urllib.request
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

# Add src to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root_dir, 'src'))


def main() -> None:
    """
    Main function to extract stickers using SAM (Segment Anything Model).
    Identifies rare species and uses point-prompted segmentation to create transparent RGBA stickers.
    """
    # 1. Setup
    base_dir = "/workspace/plantclef"
    train_csv = os.path.join(base_dir,
                             "processed/train_metadata_cleaned_verified.csv")
    img_dir = os.path.join(base_dir, "raw/train/images_max_side_800/")
    output_dir = os.path.join(base_dir, "processed/stickers")
    os.makedirs(output_dir, exist_ok=True)

    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        print("Downloading SAM weights...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(url, sam_checkpoint)

    # 2. Identify Rare Species
    df = pd.read_csv(train_csv, sep=';')
    counts = df['species_id'].value_counts()
    rare_species = counts[counts < 20].index.tolist()
    rare_df = df[df['species_id'].isin(rare_species)]
    print(f"Found {len(rare_df)} images for point-prompted extraction.")

    # 3. Load SAM Predictor (Faster than MaskGenerator)
    print("Loading SAM Predictor on GPU...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    # 4. Extraction Loop
    for _, row in tqdm(
            rare_df.iterrows(), total=len(rare_df), desc="Fast Extraction"):
        img_name = row['image_name']
        species_id = row['species_id']
        img_path = os.path.join(img_dir, str(species_id), img_name)
        save_path = os.path.join(output_dir, str(species_id),
                                 img_name.replace('.jpg', '.png'))

        if os.path.exists(save_path):
            continue
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        # Star-pattern prompts in the center
        h, w, _ = image.shape
        input_points = np.array([[w // 2, h // 2], [w // 3, h // 3],
                                 [2 * w // 3, h // 3], [w // 3, 2 * h // 3],
                                 [2 * w // 3, 2 * h // 3]])
        input_labels = np.array([1, 1, 1, 1, 1])

        # Near-instant prediction
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True)

        # Pick the most confident mask
        best_mask = masks[np.argmax(scores)]

        if best_mask.any():
            rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = (best_mask * 255).astype(np.uint8)
            coords = np.argwhere(best_mask)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            cv2.imwrite(
                save_path,
                cv2.cvtColor(rgba[y0:y1, x0:x1], cv2.COLOR_RGBA2BGRA))



if __name__ == "__main__":
    main()
