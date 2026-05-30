"""
BioCLIP zero-shot tile classification on a single image, with SAHI-style max-pool aggregation.
Usage:
    python run_one_image.py
    python run_one_image.py --image_path /path/to/image.jpg --tile_size 224 --stride 112 --top_k 5
"""

import argparse
import os
import csv
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import (
    load_species,
    get_tiles,
    encode_text_features,
    encode_image_tiles,
    compute_tile_logits,
    aggregate_tile_logits,
    tile_top_k,
    image_top_k,
)

IMAGES_DIR = "/workspace/plantclef/kaggle_uploads/test/images"
SPECIES_MAPPING = "/workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt"
OUTPUT_DIR = "./outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to image. Defaults to first image in IMAGES_DIR.")
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def draw_tile_grid(image, coords, top1_names, top1_scores):
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), name, score in zip(coords, top1_names, top1_scores):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = f"{name.split()[0]} {score:.2f}"
        draw.text((x1 + 2, y1 + 2), label, fill="yellow", font=font)
    return vis


def draw_confidence_heatmap(image, coords, top1_scores):
    w, h = image.size
    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2), score in zip(coords, top1_scores):
        heatmap[y1:y2, x1:x2] += score
        counts[y1:y2, x1:x2] += 1

    counts = np.maximum(counts, 1)
    heatmap /= counts

    norm = plt.Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    colored = cm.viridis(norm(heatmap))
    colored = (colored[:, :, :3] * 255).astype(np.uint8)

    base = np.array(image.convert("RGB"))
    blended = (0.5 * base + 0.5 * colored).astype(np.uint8)
    return Image.fromarray(blended)


def save_tile_csv(output_path, image_name, coords, top_indices, top_scores, species_ids, id_to_name):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name", "tile_index",
            "x1", "y1", "x2", "y2",
            "top1_species_id", "top1_species_name", "top1_score",
            "top2_species_id", "top2_species_name", "top2_score",
            "top3_species_id", "top3_species_name", "top3_score",
        ])
        for idx, ((x1, y1, x2, y2), indices, scores) in enumerate(zip(coords, top_indices, top_scores)):
            row = [image_name, idx, x1, y1, x2, y2]
            for rank in range(3):
                sid = species_ids[indices[rank]]
                sname = id_to_name[sid]
                score = float(scores[rank])
                row += [sid, sname, f"{score:.6f}"]
            writer.writerow(row)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image_path:
        image_path = args.image_path
    else:
        images = sorted(Path(IMAGES_DIR).glob("*.jpg")) or sorted(Path(IMAGES_DIR).glob("*.png"))
        image_path = str(images[0])
    image_name = Path(image_path).stem
    print(f"Image: {image_path}")

    print("Loading species names...")
    species_ids, species_names, id_to_name = load_species(SPECIES_MAPPING)
    print(f"  {len(species_ids)} species loaded")

    print("Loading BioCLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    model = model.to(device)
    model.eval()
    print(f"  Model on {device}")

    print("Encoding species text prompts...")
    text_feats = encode_text_features(model, tokenizer, species_names, device)
    print(f"  text_feats shape: {text_feats.shape}")

    print("Loading image and generating tiles...")
    image = Image.open(image_path).convert("RGB")
    print(f"  Image size: {image.size}")
    tiles, coords = get_tiles(image, args.tile_size, args.stride)
    print(f"  {len(tiles)} tiles (tile_size={args.tile_size}, stride={args.stride})")

    print("Encoding tile image features...")
    image_feats = encode_image_tiles(model, transform, tiles, device, batch_size=args.batch_size)
    print(f"  image_feats shape: {image_feats.shape}")

    logit_scale = model.logit_scale.exp().item()
    print(f"  logit_scale: {logit_scale:.4f}")

    print("Computing tile logits and aggregating...")
    tile_logits = compute_tile_logits(image_feats, text_feats, logit_scale)  # (n_tiles, n_species)
    image_logits = aggregate_tile_logits(tile_logits)                         # (n_species,)

    # Per-tile top-3 for debug CSV (uses softmax probs as scores)
    tile_indices, tile_scores = tile_top_k(tile_logits, k=3)

    # Image-level top-K from aggregated logits
    top_species_ids, top_scores_img = image_top_k(image_logits, species_ids, k=args.top_k)

    top1_names = [id_to_name[species_ids[tile_indices[i, 0]]] for i in range(len(tiles))]
    top1_tile_scores = tile_scores[:, 0]

    print("Saving outputs...")

    tile_csv_path = os.path.join(args.output_dir, f"{image_name}_tile_predictions.csv")
    save_tile_csv(tile_csv_path, image_name, coords, tile_indices, tile_scores, species_ids, id_to_name)
    print(f"  Tile CSV saved: {tile_csv_path}")

    grid_vis = draw_tile_grid(image, coords, top1_names, top1_tile_scores)
    grid_path = os.path.join(args.output_dir, f"{image_name}_tile_grid.jpg")
    grid_vis.save(grid_path)
    print(f"  Tile grid saved: {grid_path}")

    heatmap_vis = draw_confidence_heatmap(image, coords, top1_tile_scores)
    heatmap_path = os.path.join(args.output_dir, f"{image_name}_confidence_heatmap.jpg")
    heatmap_vis.save(heatmap_path)
    print(f"  Confidence heatmap saved: {heatmap_path}")

    print(f"\nImage-level top-{args.top_k} predictions (SAHI max-pool aggregation):")
    for rank, (sid, score) in enumerate(zip(top_species_ids, top_scores_img)):
        sname = id_to_name[sid]
        print(f"  #{rank+1}: {sname} (id={sid}, logit={score:.4f})")

    print(f"\n  species_ids list: {[int(sid) for sid in top_species_ids]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
