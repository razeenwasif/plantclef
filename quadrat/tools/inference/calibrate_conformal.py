"""Calibration tool for Hierarchical Conformal Prediction.

Optimized for Dual-GPU (2x RTX PRO 6000) with Multi-GPU Inference.
Runs inference on validation data to compute the APS quantile (q_hat).
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import load_config
from src.models.ensemble import PlantEnsemble
from src.models.uncertainty.conformal import HierarchicalConformalPredictor
from src.data.dataloader import get_dali_loaders

def calibrate() -> None:
    """
    Calibrates the Hierarchical Conformal Prediction system on a validation subset.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args, _ = parser.parse_known_args()

    # plantclef: Unified Config Loading
    config = load_config(args.config)

    # 1. Distributed Setup
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
            
    local_rank = config.hardware.local_rank
    world_size = config.hardware.world_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_master = (local_rank == 0)

    alpha = 0.05 
    genus_map = "/workspace/plantclef/processed/genus_ids.pt"
    
    # 2. Load Model (PLANTCLEF: Use latest SWA weights)
    if is_master: print("[Calibration] Loading model...")
    # plantclef: Padded to 7808 for Blackwell FP8 alignment
    model = PlantEnsemble(num_classes=config.model.num_classes, input_res=config.model.resolution).to(device).to(memory_format=torch.channels_last)
    
    # Safely retrieve best checkpoint or fallback
    best_ckpt = getattr(config, "SWA_CKPT_PATH", "models/cuda_deep_sat/plantclef_s9999/swa_model_final.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False), strict=False)
        if is_master: print(f"[Calibration] Loaded weights from {best_ckpt}")
    else:
        if is_master: print(f"[Calibration] Warning: Checkpoint {best_ckpt} not found. Calibrating with random weights.")
    model.eval()
    
    # 3. Load Validation Data (PLANTCLEF: Use high-speed DALI)
    _, val_loader, _, _ = get_dali_loaders(
        batch_size=128,
        resolution=config.model.resolution,
        device_id=config.hardware.local_rank,
        num_shards=config.hardware.world_size,
        shard_id=config.hardware.rank,
        csv_path=config.dataset.raw_csv if hasattr(config.dataset, "raw_csv") else "/workspace/plantclef/processed/student_train_final.csv",
        img_dir=config.dataset.img_dir
    )
    
    # 4. Collect Logits and Labels
    all_logits = []
    all_labels = []
    
    if is_master: print("[Calibration] Running inference on validation set...")
    
    # Build species-ID → class-index lookup tensor for shard label remapping.
    _mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    _df_map  = pd.read_csv(_mapping_path, header=None, dtype={0: int})
    _id_remap = torch.zeros(int(_df_map[0].max()) + 1, dtype=torch.long, device=device)
    _id_remap[_df_map[0].values] = torch.arange(len(_df_map), device=device)

    try:
        from tools.infrastructure.pulsar import PulsarHeartbeat
        pulsar = PulsarHeartbeat()
        has_pulsar = True
    except ImportError:
        has_pulsar = False

    start_time = time.time()
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, disable=not is_master)):
            images = data[0]['data'].to(device, dtype=torch.bfloat16, memory_format=torch.channels_last)
            raw_lbl = data[0]['label'].squeeze().long().to(device)
            labels = _id_remap[raw_lbl.clamp(0, _id_remap.shape[0] - 1)]
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
                
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            
            # plantclef: Pulse every 5 batches
            if has_pulsar and (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) * 128 * world_size / max(0.1, elapsed)
                pulsar.pulse(0, i + 1, 0.0, fps, status="calibrating")

            # Use ~16k samples for calibration speed (8k per GPU)
            if i > (128 // world_size): break 
            
    local_logits = torch.cat(all_logits)
    local_labels = torch.cat(all_labels)

    # 5. Gather and Calibrate
    out_dir = getattr(config, "BASE_MODEL_DIR", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/conformal_calibration.pt"

    if world_size > 1:
        # plantclef: Use gather_object for non-tensor collections
        gathered_data = [None for _ in range(world_size)] if is_master else None
        dist.gather_object((local_logits, local_labels), gathered_data, dst=0)
        
        if is_master:
            final_logits = torch.cat([d[0] for d in gathered_data])
            final_labels = torch.cat([d[1] for d in gathered_data])
            
            predictor = HierarchicalConformalPredictor(alpha=alpha, genus_map_path=genus_map)
            predictor.calibrate(final_logits, final_labels)
            
            predictor.save(out_path)
            print(f"[Calibration] Saved global conformal state to {out_path}")
    else:
        predictor = HierarchicalConformalPredictor(alpha=alpha, genus_map_path=genus_map)
        predictor.calibrate(local_logits, local_labels)
        predictor.save(out_path)
        print(f"[Calibration] Saved conformal state to {out_path}")

if __name__ == "__main__":
    calibrate()
