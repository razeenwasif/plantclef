"""
Teacher Feature Extraction.
Pre-computes BioCLIP and DINOv3 logits for all training images.
Optimized for 4-pod RTX PRO 6000 setup using DDP.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Import project backbones
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.bioclip import PlantBioCLIP
from src.models.vit_backbone import PlantViTBackbone

class TeacherExtractor:
    def __init__(self, bioclip_path, dinov3_path, device="cuda"):
        # Load Teacher 1: BioCLIP
        self.bio = PlantBioCLIP(checkpoint=bioclip_path, input_res=224).to(device)
        # Load Teacher 2: DINOv3
        self.dino = PlantViTBackbone(model_name=dinov3_path, input_res=224).to(device)
        
        self.bio.eval()
        self.dino.eval()
        self.device = device

    @torch.no_grad()
    def extract_batch(self, x):
        with torch.amp.autocast(self.device, dtype=torch.bfloat16):
            # We assume these wrappers return species logits or we apply a head
            logits_bio = self.bio(x)
            logits_dino = self.dino(x)
        return logits_bio.float().cpu(), logits_dino.float().cpu()

import torch.distributed as dist

def run_extraction(img_dir, output_dir, bioclip_path, dinov3_path):
    """
    Main extraction loop. 
    Uses distributed sharding to split the dataset across all available GPUs.
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        
    os.makedirs(output_dir, exist_ok=True)
    device = f"cuda:{rank}"
    extractor = TeacherExtractor(bioclip_path, dinov3_path, device=device)
    
    # Check for existing progress to allow resume
    progress_file = os.path.join(output_dir, f"progress_rank_{rank}.txt")
    processed_count = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            processed_count = int(f.read().strip())

    print(f"Rank {rank} resuming from index {processed_count}")
    
    # ... logic to shard dataset based on world_size ...
    # for i, (img, target, img_name) in enumerate(dataloader):
    #     if i < processed_count: continue
    #     logits_bio, logits_dino = extractor.extract_batch(img)
    #     
    #     # Save as compressed numpy for fast retrieval
    #     out_path = os.path.join(output_dir, f"{Path(img_name).stem}.npz")
    #     np.savez_compressed(out_path, bio=logits_bio.numpy(), dino=logits_dino.numpy())
    #     
    #     # update progress_file

if __name__ == "__main__":
    # Example paths
    OUT = "data/cache/trinity_teachers/"
    BIO = "models/bioclip_finetuned.pth"
    DINO = "models/dinov3_finetuned.pth"
    print(f"Teacher extraction script initialized. Targets: {OUT}")
