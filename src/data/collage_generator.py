"""Generates collages by overlaying species stickers on background images."""

import glob
import os
import random
import sys
from typing import List, Dict, Any, Union, Set

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/data
src_dir = os.path.dirname(current_dir)  # src
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def gpu_overlay(background_tensor: torch.Tensor, sticker_paths: List[str]) -> torch.Tensor:
    """
    Overlays multiple RGBA stickers onto a background tensor on GPU.

    Stickers are loaded from disk and moved to GPU on-the-fly to save System RAM.

    Parameters
    ----------
    background_tensor : torch.Tensor
        The background image tensor of shape (C, H, W).
    sticker_paths : List[str]
        List of absolute paths to the sticker (RGBA) images.

    Returns
    -------
    torch.Tensor
        The resulting background tensor with stickers overlaid.
    """
    _, h, w = background_tensor.shape

    for s_path in sticker_paths:
        img = cv2.imread(s_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Move sticker to GPU
        sticker = torch.from_numpy(img).permute(2, 0, 1).float().to('cuda') / 255.0
        _, sh, sw = sticker.shape

        # 1. Random Scale
        scale = random.uniform(0.4, 1.2)
        target_h, target_w = int(sh * scale), int(sw * scale)
        if target_h >= h or target_w >= w or target_h < 10 or target_w < 10:
            scale = min(h / sh, w / sw) * 0.8
            target_h, target_w = int(sh * scale), int(sw * scale)

        if target_h < 10 or target_w < 10:
            continue

        sticker_res = F.interpolate(
            sticker.unsqueeze(0),
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False).squeeze(0)

        # 2. Random Position
        y_off = random.randint(0, h - target_h)
        x_off = random.randint(0, w - target_w)

        # 3. Alpha Blending
        rgb = sticker_res[:3, :, :]
        alpha = sticker_res[3:4, :, :]

        bg_crop = background_tensor[:, y_off:y_off + target_h,
                                    x_off:x_off + target_w]
        blended = (rgb * alpha) + (bg_crop * (1.0 - alpha))
        background_tensor[:, y_off:y_off + target_h,
                          x_off:x_off + target_w] = blended

    return background_tensor


def main() -> None:
    """
    Main function to generate a synthetic dataset of plant collages.
    """
    base_dir = "/workspace/plantclef"
    sticker_dir = os.path.join(base_dir, "processed/stickers")
    # Instead of glob, use the existing metadata
    train_csv = os.path.join(base_dir,
                             "processed/train_metadata_cleaned_verified.csv")
    background_dir = os.path.join(base_dir, "raw/train/images_max_side_800/")
    output_dir = os.path.join(base_dir, "processed/collages")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Index Sticker Paths (Fast, low RAM)
    print("Indexing sticker paths...")
    sticker_paths = glob.glob(
        os.path.join(sticker_dir, "**/*.png"), recursive=True)
    species_to_stickers: Dict[str, List[str]] = {}
    for p in sticker_paths:
        sid = os.path.basename(os.path.dirname(p))
        if sid not in species_to_stickers:
            species_to_stickers[sid] = []
        species_to_stickers[sid].append(p)
    print(f"Indexed {len(sticker_paths)} stickers across "
          f"{len(species_to_stickers)} species.")

    # 2. Load Backgrounds from CSV (Instant)
    print("Loading background paths from metadata...")
    df_bg = pd.read_csv(train_csv, sep=';')
    all_background_rows = df_bg.to_dict('records')

    # 3. Generation Loop
    num_collages = 50000
    results: List[Dict[str, str]] = []

    print(f"Generating {num_collages} Memory-Efficient GPU Collages...")
    for i in tqdm(range(num_collages)):
        bg_row = random.choice(all_background_rows)
        bg_name = bg_row['image_name']
        bg_sid = str(bg_row['species_id'])
        bg_path = os.path.join(background_dir, bg_sid, bg_name)

        bg_img = cv2.imread(bg_path)
        if bg_img is None:
            continue

        # Background to GPU
        bg_tensor = torch.from_numpy(bg_img).permute(
            2, 0, 1).float().to('cuda') / 255.0
        labels: Set[str] = {bg_sid}

        # Pick 3-6 random rare species
        num_stickers_to_add = random.randint(3, 6)
        available_sids = list(species_to_stickers.keys())
        pasting_paths: List[str] = []

        for _ in range(num_stickers_to_add):
            sid = random.choice(available_sids)
            labels.add(sid)
            pasting_paths.append(random.choice(species_to_stickers[sid]))

        # GPU Synthesis
        with torch.no_grad():
            collage_tensor = gpu_overlay(bg_tensor, pasting_paths)

        # Save
        collage_np = (collage_tensor.permute(
            1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        collage_name = f"collage_{i:06d}.jpg"
        cv2.imwrite(os.path.join(output_dir, collage_name), collage_np)

        results.append({
            "image_name": collage_name,
            "species_ids": ",".join(list(labels))
        })

    pd.DataFrame(results).to_csv(
        os.path.join(base_dir, "processed/synthetic_collages.csv"),
        index=False,
        sep=';')
    print(f"Success! 50,000 collages saved to {output_dir}")



if __name__ == "__main__":
    main()
