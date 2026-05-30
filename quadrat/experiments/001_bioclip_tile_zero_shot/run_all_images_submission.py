"""
BioCLIP zero-shot inference over all test images with SAHI-style max-pool aggregation.
Outputs a submission CSV in the PlantCLEF format.

Usage:
    python run_all_images_submission.py
    python run_all_images_submission.py --images_dir /path/to/images --top_k 5 --output submission.csv
"""

import argparse
import os
import csv
from pathlib import Path

import torch
import open_clip
from PIL import Image

from utils import (
    load_species,
    get_tiles,
    encode_text_features,
    encode_image_tiles,
    compute_tile_logits,
    aggregate_tile_logits,
    image_top_k,
)

IMAGES_DIR = "/workspace/plantclef/kaggle_uploads/test/images"
SPECIES_MAPPING = "/workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt"
DEFAULT_OUTPUT = "./outputs/submission.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--mapping", type=str, default=SPECIES_MAPPING)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def find_images(images_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for ext in exts:
        paths.extend(Path(images_dir).glob(ext))
    return sorted(set(paths))


def process_image(image_path, model, transform, text_feats, species_ids, logit_scale,
                  tile_size, stride, batch_size, top_k, device):
    image = Image.open(image_path).convert("RGB")
    tiles, _ = get_tiles(image, tile_size, stride)
    image_feats = encode_image_tiles(model, transform, tiles, device, batch_size=batch_size)
    image_feats = image_feats.to(device)
    tile_logits = compute_tile_logits(image_feats, text_feats, logit_scale)
    image_logits = aggregate_tile_logits(tile_logits)
    top_species_ids, _ = image_top_k(image_logits, species_ids, k=top_k)
    return top_species_ids


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("Loading species names...")
    species_ids, species_names, id_to_name = load_species(args.mapping)
    print(f"  {len(species_ids)} species loaded")

    print("Loading BioCLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    model = model.to(device)
    model.eval()
    print(f"  Model on {device}")

    print("Encoding species text prompts (one-time)...")
    text_feats = encode_text_features(model, tokenizer, species_names, device)
    text_feats = text_feats.to(device)
    print(f"  text_feats shape: {text_feats.shape}")

    logit_scale = model.logit_scale.exp().item()
    print(f"  logit_scale: {logit_scale:.4f}")

    image_paths = find_images(args.images_dir)
    print(f"\nFound {len(image_paths)} images in {args.images_dir}")
    print(f"tile_size={args.tile_size}, stride={args.stride}, top_k={args.top_k}\n")

    rows = []
    for i, image_path in enumerate(image_paths):
        quadrat_id = image_path.stem
        print(f"[{i+1}/{len(image_paths)}] {quadrat_id} ...", end=" ", flush=True)

        top_species_ids = process_image(
            image_path, model, transform, text_feats, species_ids, logit_scale,
            args.tile_size, args.stride, args.batch_size, args.top_k, device,
        )

        species_ids_str = "[" + ", ".join(top_species_ids) + "]"
        rows.append({"quadrat_id": quadrat_id, "species_ids": species_ids_str})
        print(species_ids_str)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quadrat_id", "species_ids"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSubmission saved to: {args.output}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
