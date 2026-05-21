import os
if "HF_HUB_OFFLINE" in os.environ:
    del os.environ["HF_HUB_OFFLINE"]
if "TRANSFORMERS_OFFLINE" in os.environ:
    del os.environ["TRANSFORMERS_OFFLINE"]
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from transformers import SamModel, SamProcessor
except ImportError:
    raise ImportError("Please install required packages: pip install transformers accelerate")

def get_config_paths(config_path="configs/inference.yaml"):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Path(cfg['paths']['image_dir']), Path(cfg['paths']['output_dir']) / "noise_masks"

def main():
    parser = argparse.ArgumentParser(description="Generate GroundingDINO + SAM noise masks.")
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to inference config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="GroundingDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold")
    args = parser.parse_args()

    image_dir, mask_dir = get_config_paths(args.config)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing GroundingDINO on {args.device}...")
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(args.device)

    print(f"[*] Initializing SAM on {args.device}...")
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(args.device)

    # The exact noise classes identified in the paper
    text_prompt = "stone. shell. roulette. plastic. metal. hand. ice. snow. measure. ruler. wood. board. paper."
    
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
    print(f"[*] Found {len(image_paths)} images in {image_dir}")

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Generating Masks"):
            mask_out_path = mask_dir / f"{img_path.stem}_mask.png"
            if mask_out_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

            # 1. GroundingDINO Detection
            inputs = dino_processor(images=image, text=text_prompt, return_tensors="pt").to(args.device)
            outputs = dino_model(**inputs)
            
            results = dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[image.size[::-1]]
            )[0]

            boxes = results["boxes"]
            
            # 2. SAM Segmentation (Only if boxes were found)
            if len(boxes) > 0:
                # SAM expects boxes in format [[[x1, y1, x2, y2], ...]]
                sam_inputs = sam_processor(images=image, input_boxes=[[boxes.tolist()]], return_tensors="pt").to(args.device)
                sam_outputs = sam_model(**sam_inputs)
                
                # SAM returns masks of shape (batch_size, num_boxes, 3, H, W)
                # We take the best mask (index 0 of the 3 generated per box) and combine them
                masks = sam_processor.image_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu()
                )[0]
                
                # Combine all box masks into one binary mask
                combined_mask = torch.any(masks[:, 0, :, :], dim=0).numpy()
                mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
            else:
                # No noise detected, blank mask
                mask_img = Image.new('L', image.size, 0)

            # Save the full-resolution mask
            mask_img.save(mask_out_path)

if __name__ == "__main__":
    main()
