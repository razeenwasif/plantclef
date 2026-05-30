"""
Holy Trinity Training: Asymmetric Dual-Teacher Distillation.
Trains the DeiT student using pre-computed teacher logits.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .trinity_deit import create_trinity_student

def setup_hardware_context():
    """
    Hooks into the existing system hardware scanner.
    Detects if we are on Blackwell (RTX 6000) or Lovelace (4090).
    """
    # Placeholder for your existing hardware scanner call
    # e.g., from tools.infrastructure.scanner import get_gpu_arch
    # arch = get_gpu_arch() 
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        
    device = torch.device(f"cuda:{local_rank}")
    
    # Check for Blackwell FP8 support
    gpu_name = torch.cuda.get_device_name(device).lower()
    is_blackwell = "rtx 6000" in gpu_name or "b200" in gpu_name
    
    return {
        "device": device,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_blackwell": is_blackwell,
        "dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    }

class TrinityLoss(nn.Module):
    """
    Implements Asymmetric Distillation for both Standard and Elite variants.
    """
    def __init__(self, variant="standard", alpha=0.3, beta=0.7, temp=2.0):
        super().__init__()
        self.variant = variant
        self.alpha = alpha   # Weight for Ground Truth
        self.beta = beta     # Weight for Distillation
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_outputs, labels, teacher_bio, teacher_dino):
        # student_outputs: (x_cls, x_dist) or (x_cls, x_bio, x_dino)
        x_cls = student_outputs[0]
        
        # 1. Ground Truth Loss (anchored to CLS token)
        loss_gt = self.ce(x_cls, labels)
        
        if self.variant == "standard":
            x_dist = student_outputs[1]
            # Average teachers for standard distillation
            teacher_soft = (F.softmax(teacher_bio / self.temp, dim=1) + 
                            F.softmax(teacher_dino / self.temp, dim=1)) / 2.0
            
            loss_dist = self.kl(F.log_softmax(x_dist / self.temp, dim=1), teacher_soft) * (self.temp ** 2)
            return self.alpha * loss_gt + self.beta * loss_dist

        else:
            x_s_bio = student_outputs[1]
            x_s_dino = student_outputs[2]
            
            # Independent distillation per token
            t_bio_soft = F.softmax(teacher_bio / self.temp, dim=1)
            t_dino_soft = F.softmax(teacher_dino / self.temp, dim=1)
            
            loss_bio = self.kl(F.log_softmax(x_s_bio / self.temp, dim=1), t_bio_soft)
            loss_dino = self.kl(F.log_softmax(x_s_dino / self.temp, dim=1), t_dino_soft)
            
            loss_dist = (loss_bio + loss_dino) / 2.0 * (self.temp ** 2)
            return self.alpha * loss_gt + self.beta * loss_dist

import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.data.hash_indexer import resolve_external_path

class FusedTeacherDataset(Dataset):
    """
    Bridges images from disk and pre-computed teacher logits from NVMe cache.
    """
    def __init__(self, img_dir: str, teacher_cache_dir: str, transform=None):
        self.img_dir = resolve_external_path(img_dir)
        self.teacher_cache_dir = teacher_cache_dir
        self.transform = transform
        
        # Discover images (using the same logic as our dynamic loader)
        self.samples = []
        classes = sorted([d for d in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir, d))])
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        for cls in classes:
            cls_dir = os.path.join(self.img_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Path to the pre-computed logits (saved as .npy during extraction)
                    logit_path = os.path.join(teacher_cache_dir, f"{Path(img_name).stem}.npz")
                    if os.path.exists(logit_path):
                        self.samples.append({
                            'img_path': os.path.join(cls_dir, img_name),
                            'logit_path': logit_path,
                            'label': class_to_idx[cls]
                        })

        print(f"[AD-TD] Fused Dataset ready: {len(self.samples)} valid samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        from PIL import Image
        img = Image.open(s['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        # Load teacher knowledge
        teachers = np.load(s['logit_path'])
        t_bio = torch.from_numpy(teachers['bio'])
        t_dino = torch.from_numpy(teachers['dino'])
        
        return img, s['label'], t_bio, t_dino

def train_one_epoch(model, loader, optimizer, criterion, hw_context):
    model.train()
    device = hw_context["device"]
    is_blackwell = hw_context["is_blackwell"]
    
    for batch in loader:
        images, labels, t_bio, t_dino = batch
        images, labels = images.to(device), labels.to(device)
        t_bio, t_dino = t_bio.to(device), t_dino.to(device)
        
        optimizer.zero_grad()
        
        # Blackwell Optimization: Use TransformerEngine for FP8 if available
        # Otherwise fall back to standard AMP
        with torch.amp.autocast("cuda", dtype=hw_context["dtype"]):
            outputs = model(images)
            loss = criterion(outputs, labels, t_bio, t_dino)
            
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    hw = setup_hardware_context()
    
    # Adaptive Batching: 4090 (24GB) gets smaller batches, RTX 6000 (96GB) gets huge ones
    batch_size = 128 if hw["is_blackwell"] else 32
    
    print(f"Hardware-Aware Training Initialized.")
    print(f"GPU: {torch.cuda.get_device_name(hw['device'])} | Blackwell: {hw['is_blackwell']}")
    print(f"Precision: {hw['dtype']} | Batch Size: {batch_size}")
    
    model = create_trinity_student(variant="elite", model_size="large").to(hw["device"])
    if hw["world_size"] > 1:
        model = DDP(model, device_ids=[hw["local_rank"]])
        
    criterion = TrinityLoss(variant="elite")
