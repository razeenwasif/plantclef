"""Threshold Optimizer for PlantCLEF 2026.

Optimized for Dual-GPU (2x RTX 5090 / PRO 6000) with Multi-GPU Inference.
Uses vectorized GPU operations and Brent's Method to find per-class probability
thresholds that maximize the Macro-F1 score.
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize_scalar
import json
from pathlib import Path

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.inference.model_runner import PlantEnsembleAdapter, ModelRunner
from src.inference.config import ModelConfig
from src.inference.types import TileSpec
from src.config import load_config

def optimize() -> None:
    """
    Find per-class probability thresholds that maximize the Macro-F1 score.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to specific model checkpoint")
    args, _ = parser.parse_known_args()

    # plantclef: Unified Config Loading
    config = load_config(args.config)

    # 1. Setup Distributed Environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
            
    local_rank = config.hardware.local_rank
    world_size = config.hardware.world_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_master = (local_rank == 0)

    # plantclef: Allow manual checkpoint override
    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = config.SWA_CKPT_PATH if os.path.exists(config.SWA_CKPT_PATH) else config.P1_CKPT_PATH
    
    mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    output_json = "models/optimized_thresholds.json"
    
    # Load Species Mapping (to convert IDs to indices)
    df_map = pd.read_csv(mapping_path, header=None)
    species_to_idx = {int(row[0]): i for i, row in df_map.iterrows()}
    
    if is_master:
        print("="*50)
        print(" PLANTCLEF 2026: THRESHOLD OPTIMIZATION (Dual-RTX 5090 / PRO 6000)")
        print("="*50)
    
    # 2. Initialize Optimized Model Runner
    inference_batch = 256
    m_cfg = ModelConfig(
        num_classes=config.model.num_classes,
        input_size=config.model.resolution,
        batch_size=inference_batch,
        device=str(device),
        use_compile=config.hardware.use_compile
    )
    # The adapter handles the weight loading
    adapter = PlantEnsembleAdapter(Path(model_path), num_classes_=config.model.num_classes, input_res=config.model.resolution)
    # Ensure memory format is channels_last for RTX 5090 / PRO 6000 Tensor Cores
    adapter._model.to(device).to(memory_format=torch.channels_last)
    runner = ModelRunner(adapter, m_cfg)

    # 3. Collect Raw Probabilities for a Large Val Subset
    num_samples = 20000 # Increased for dual-GPU stability
    if is_master: print(f"Collecting validation probabilities ({num_samples} images)...")
    
    from tools.infrastructure.pulsar import PulsarHeartbeat
    pulsar = PulsarHeartbeat()
    start_time = time.time()
    
    # Only load master CSV on rank 0 then shard indices
    csv_path = getattr(config.dataset, "raw_csv", "/workspace/plantclef/processed/student_train_final.csv")
    if not os.path.exists(csv_path):
        csv_path = "/workspace/plantclef/processed/student_train_final.csv"

    if is_master:
        df = pd.read_csv(csv_path, sep=';', low_memory=False)
        val_df = df.sample(min(len(df), num_samples))
        # Broadcast the selected indices to all ranks
        selected_indices = val_df.index.tolist()
    else:
        selected_indices = None
        df = pd.read_csv(csv_path, sep=';', low_memory=False)

    # Simple sharding of indices
    if world_size > 1:
        # Broadcast from master
        obj_list = [selected_indices] if is_master else [None]
        dist.broadcast_object_list(obj_list, src=0)
        selected_indices = obj_list[0]
        
        # Take local shard
        chunk_size = (len(selected_indices) + world_size - 1) // world_size
        local_indices = selected_indices[local_rank * chunk_size : (local_rank + 1) * chunk_size]
        local_val_df = df.loc[local_indices]
    else:
        local_val_df = val_df

    all_probs = []
    all_labels = []
    
    # Safely retrieve image directory
    image_dir = getattr(config.dataset, "img_dir", "/workspace/plantclef/raw/train/images_max_side_800/")
    if not os.path.exists(image_dir):
        image_dir = "/workspace/plantclef/raw/train/images_max_side_800/"
        
    img_list = []
    
    pbar = tqdm(local_val_df.iterrows(), total=len(local_val_df), desc=f"Rank {local_rank}")
    first_path = True
    for _, row in pbar:
        sid = row['species_ids'] if 'species_ids' in row else row['species_id']
        # Handle multiple IDs (use primary for optimization)
        sid_primary = str(sid).split(',')[0].strip()
        path = os.path.join(image_dir, sid_primary, row['image_name'])
        
        if first_path:
            print(f"\n[Debug] First path: {path}")
            first_path = False
        
        if os.path.exists(path):
            from PIL import Image
            try:
                img = Image.open(path).convert("RGB")
                spec = TileSpec(
                    image_id=str(row.get('image_id', row['image_name'])), 
                    tile_id=0, 
                    x0=0, y0=0, x1=img.width, y1=img.height
                )
                img_list.append((spec, img))
                # Map raw ID to class index
                sid_int = int(sid_primary)
                idx = species_to_idx.get(sid_int)
                if idx is not None:
                    all_labels.append(idx)
                else:
                    # Skip if not in mapping
                    img_list.pop()
                    continue
            except Exception as e:
                if first_path: print(f"[Error] Failed to process first path: {e}")
                continue
            
            if len(img_list) >= inference_batch:
                preds = runner.predict(img_list)
                all_probs.extend([p.probs for p in preds])
                img_list = []
                
                # plantclef: Pulse metrics
                if len(all_probs) % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = len(all_probs) * world_size / max(0.1, elapsed)
                    pulsar.pulse(0, len(all_probs), 0.0, fps, status="optimizing")

    if img_list:
        preds = runner.predict(img_list)
        all_probs.extend([p.probs for p in preds])
    
    if is_master:
        print(f"Rank {local_rank}: Collected {len(all_probs)} probabilities.")

    local_probs_t = torch.from_numpy(np.stack(all_probs)).to(device)
    local_labels_t = torch.tensor(all_labels).to(device)

    # Aggregate results to Master Rank
    if world_size > 1:
        if is_master: print(f"GATHERING: Syncing {world_size} GPU shards...")
        
        # plantclef: High-speed GPU Gather
        # First, ensure all shards have the same size (DDP requirement)
        local_count = torch.tensor([local_probs_t.size(0)], device=device)
        all_counts = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)
        max_count = max(c.item() for c in all_counts)
        
        # Pad local tensors to max_count
        pad_size = max_count - local_probs_t.size(0)
        if pad_size > 0:
            padded_probs = torch.cat([local_probs_t, torch.zeros(pad_size, 7806, device=device)])
            padded_labels = torch.cat([local_labels_t, torch.full((pad_size,), -1, device=device)])
        else:
            padded_probs = local_probs_t
            padded_labels = local_labels_t
            
        # All-gather the padded data
        gathered_probs = [torch.zeros(max_count, 7806, device=device) for _ in range(world_size)]
        gathered_labels = [torch.zeros(max_count, dtype=torch.long, device=device) for _ in range(world_size)]
        
        dist.all_gather(gathered_probs, padded_probs)
        dist.all_gather(gathered_labels, padded_labels)
        
        if is_master:
            all_probs_list = []
            all_labels_list = []
            for i in range(world_size):
                # Slice off the padding
                actual_len = all_counts[i].item()
                all_probs_list.append(gathered_probs[i][:actual_len])
                all_labels_list.append(gathered_labels[i][:actual_len])
            
            all_probs_t = torch.cat(all_probs_list).cuda()
            all_labels_t = torch.cat(all_labels_list).cuda()
    else:
        all_probs_t = local_probs_t
        all_labels_t = local_labels_t

    # 4. Vectorized Per-Class F1 Optimization (Master Rank only)
    if is_master:
        print(f"\nPerforming Vectorized Per-Class optimization on {len(all_labels_t)} samples...")
        
        # plantclef: Per-Class Thresholding
        # We find the best threshold for EVERY species independently
        thresholds = torch.linspace(0.005, 0.5, steps=50).cuda()
        best_thresholds = {}
        
        # Convert labels to one-hot for fast vectorized F1 calculation
        # Shape: [N, C]
        y_true = torch.nn.functional.one_hot(all_labels_t, num_classes=7806).float()
        
        for c in tqdm(range(7806), desc="Optimizing Species"):
            c_probs = all_probs_t[:, c]
            c_true = y_true[:, c]
            
            if c_true.sum() == 0:
                best_thresholds[str(c)] = 0.1 # Default for unseen classes
                continue
                
            # Vectorized F1 over all 50 threshold candidates
            # c_probs: [N], thresholds: [50] -> preds: [N, 50]
            preds = (c_probs.unsqueeze(1) > thresholds.unsqueeze(0))
            
            tp = (preds * c_true.unsqueeze(1)).sum(dim=0)
            fp = (preds * (1 - c_true).unsqueeze(1)).sum(dim=0)
            fn = ((~preds) * c_true.unsqueeze(1)).sum(dim=0)
            
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            best_idx = torch.argmax(f1)
            best_thresholds[str(c)] = float(thresholds[best_idx].cpu())

        # 5. Export (Diversity-Aware)
        # Use species ID mapping if available
        if os.path.exists(mapping_path):
            species_list = pd.read_csv(mapping_path, header=None)[0].tolist()
            final_json = {str(sid): best_thresholds.get(str(i), 0.1) for i, sid in enumerate(species_list)}
        else:
            final_json = best_thresholds

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(final_json, f, indent=4)
            
        print(f"Success: Per-class thresholds saved to {output_json}")

if __name__ == "__main__":
    optimize()
