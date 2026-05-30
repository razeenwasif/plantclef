import os
import sys
from pathlib import Path

# plantclef: ATOMIC ENVIRONMENT HARDENING
# 1. Path Sync
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. DeepSpeed Environment Sync
# DeepSpeed Stage 1/2 REQUIRES these variables even on 1 GPU.
if "LOCAL_RANK" not in os.environ: os.environ["LOCAL_RANK"] = "0"
if "RANK" not in os.environ: os.environ["RANK"] = "0"
if "WORLD_SIZE" not in os.environ: os.environ["WORLD_SIZE"] = "1"
if "MASTER_ADDR" not in os.environ: os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ: os.environ["MASTER_PORT"] = "29505"

import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from tqdm import tqdm
from datetime import timedelta

# plantclef: Standard Configuration Interface
from src import config

def get_pca_module(pca, device):
    if pca is None: return None
    layer = nn.Linear(pca.components_.shape[1], pca.n_components_)
    layer.weight.data = torch.from_numpy(pca.components_).float()
    layer.bias.data = torch.from_numpy(-pca.mean_ @ pca.components_.T).float()
    return layer.to(device)

def wrap_loader(loader, device):
    return loader

def get_ds_config(phase='p1', micro_batch_size=2048):
    config_dict = {
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e7,
            "overlap_comm": True
        },
        "bf16": {"enabled": True},
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    return config_dict

def train():
    """Universal 2-Stage Trainer for 4x RTX PRO 6000."""
    t_start = time.time()
    
    # 1. Framework Load
    t0 = time.time()
    import deepspeed
    import wandb
    import pandas as pd
    from src.models.ensemble import PlantEnsemble, ResidualMLP
    from src.data.dataloader import get_dali_loaders
    from src.training import (
        AsymmetricLoss, EarlyStopping,
        extract_and_cache_features, run_phase1_cached,
        load_phase1_checkpoint, robust_load_state_dict, get_raw_model
    )
    print(f"[Init] Framework Load: {time.time() - t0:.2f}s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--samples_limit", type=int, default=None)
    args, _ = parser.parse_known_args()
    
    phase_target = args.phase

    # plantclef: Accelerator setup (cuda | tpu | cpu, controlled by CLUSTER_MODE).
    # Replaces the old direct torch.cuda.* / NCCL init dance.
    from src.training.accelerator import make_accelerator, set_accelerator
    accel = make_accelerator(os.environ.get("CLUSTER_MODE", "auto"))
    set_accelerator(accel)
    accel.init_distributed()
    device = accel.device
    world_size = accel.world_size
    rank = accel.rank
    local_rank = accel.local_rank
    if accel.is_cuda:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    if rank == 0:
        print(f"[Init] Accelerator: {accel.mode}  World Size: {world_size}  Device: {device}")

    # plantclef: Structured telemetry — one JSONL per rank under reports/telemetry/.
    # The liveness probe + (future) sweep coordinator consume these events.
    from src.training.telemetry import telemetry, start_heartbeat
    telemetry.bind(
        run_id=os.environ.get("PLANTCLEF_RUN_ID") or None,  # let auto-generate if unset
        rank=rank, world=world_size, phase=phase_target,
        host_id=os.environ.get("CLUSTER_HOST_ID") or None,
    )
    telemetry.emit("run.start",
                   accelerator=accel.mode,
                   device=str(device),
                   seed=int(os.environ.get("PLANTCLEF_SEED", "42")),
                   name=os.environ.get("PLANTCLEF_NAME", ""))
    start_heartbeat(interval_sec=float(os.environ.get("PLANTCLEF_HEARTBEAT_INTERVAL", "15")))

    # 3. Metadata & Priors
    t0 = time.time()
    csv_path = config.CLEANED_CSV if os.path.exists(config.CLEANED_CSV) else config.RAW_CSV
    num_classes = 7808
    genus_ids = torch.arange(num_classes).int().to(device)
    genus_path = "/workspace/plantclef/processed/genus_ids.pt"
    if os.path.exists(genus_path):
        genus_ids = torch.load(genus_path, map_location=device, weights_only=True).int()[:num_classes]

    # plantclef: Calculate Class Frequencies for Duality-Derived ASL
    class_counts = None
    if os.path.exists(csv_path) and rank == 0:
        df = pd.read_csv(csv_path, sep=';', low_memory=False)
        # Assuming column name is 'species_id' and we have a mapping to indices
        counts_map = df['species_id'].value_counts().to_dict()
        # Load mapping to convert IDs to indices
        mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
        if os.path.exists(mapping_path):
            df_map = pd.read_csv(mapping_path, header=None)
            id_to_idx = {int(row[0]): i for i, row in df_map.iterrows()}
            class_counts = torch.zeros(num_classes)
            for sid, count in counts_map.items():
                if sid in id_to_idx:
                    class_counts[id_to_idx[sid]] = count
    
    # Broadcast counts from master
    if torch.distributed.is_initialized():
        if class_counts is None: class_counts = torch.zeros(num_classes)
        # plantclef: NCCL requires GPU tensors
        class_counts = class_counts.to(device)
        torch.distributed.broadcast(class_counts, src=0)

    criterion = AsymmetricLoss(
        logit_adjustments=None, 
        genus_ids=genus_ids,
        class_counts=class_counts,
        taxon_smoothing=0.1, 
        use_fused=False 
    ).to(device)
    print(f"[Init] Metadata/Loss Ready: {time.time() - t0:.2f}s")

    # 4. Model Build (Lazy)
    t0 = time.time()
    if phase_target in ["p1", "p2a"]:
        from src.models.ensemble import ResidualMLP
        if phase_target == "p1":
            model = ResidualMLP(in_features=config.PCA_COMPONENTS, hidden_features=2048, out_features=num_classes).to(device)
        else:
            model = ResidualMLP(in_features=3328, hidden_features=2048, out_features=num_classes).to(device)
    else:
        model = PlantEnsemble(num_classes=num_classes, input_res=config.RESOLUTION).to(device)
    print(f"[Init] Model Build: {time.time() - t0:.2f}s")

    # --- PHASE 2A ---
    if phase_target == "p2a":
        if rank == 0: print("\n--- PHASE 2A: HARDENED WARMUP ---")
        feature_cache_path = config.FEATURE_CACHE_PATH
        
        if os.path.exists(feature_cache_path):
            from src.data.dataloader import CachedFeatureDataset
            from torch.utils.data import DataLoader
            
            cache_data = torch.load(feature_cache_path, weights_only=False, mmap=True)
            if 'bio' in cache_data:
                features = torch.cat([cache_data['bio'], cache_data['dino'], cache_data['conv']], dim=1)
                labels = cache_data['labels']
            else:
                features = cache_data['features']
                labels = cache_data['labels']
                
            ds = CachedFeatureDataset(features, labels)
            warmup_loader = DataLoader(ds, batch_size=2048, shuffle=True, num_workers=2, pin_memory=True)
            
            for param in model.parameters(): param.requires_grad = True
            from lion_pytorch import Lion
            optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.01)
            
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=model, optimizer=optimizer, 
                config=get_ds_config(micro_batch_size=2048),
                dist_init_required=False 
            )
            
            for epoch in range(config.EPOCHS_PHASE1):
                run_phase1_cached(model_engine, warmup_loader, criterion, f"W-{epoch}", num_classes, device)
                if rank == 0:
                    torch.save({'model_state': get_raw_model(model_engine).state_dict()}, "models/warmup_hardened.pth")
                    print(f"[Checkpoint] Saved Phase 2A warm weights (Epoch {epoch})")

    # --- PHASE 2B ---
    if phase_target in ["all", "p2b"]:
        if rank == 0: print("\n--- PHASE 2B: FULL ENSEMBLE FINE-TUNING (LoRA R=512) ---")
        if not isinstance(model, PlantEnsemble):
            model = PlantEnsemble(num_classes=num_classes, input_res=config.RESOLUTION).to(device)
            warm_ckpt = "models/warmup_hardened.pth"
            if os.path.exists(warm_ckpt):
                robust_load_state_dict(model, torch.load(warm_ckpt, map_location=device)['model_state'], strict=False)

        model.apply_lora(r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT)
        model.freeze_backbones()

        from src.data.dataloader import get_dali_loaders
        train_loader, val_loader, _ = get_dali_loaders(
            csv_path=csv_path, img_dir=config.IMG_DIR, 
            batch_size=config.P2_BATCH_SIZE, resolution=config.RESOLUTION,
            device_id=local_rank, num_shards=world_size, shard_id=rank
        )
        
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=1e-5, weight_decay=0.01)
        ds_config = get_ds_config(phase='p2', micro_batch_size=config.P2_BATCH_SIZE)
        ds_config["zero_optimization"]["stage"] = 2

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model, optimizer=optimizer, 
            config=ds_config,
            dist_init_required=False
        )

        from src.training.loops import run_epoch
        for epoch in range(config.EPOCHS_PHASE2):
            run_epoch(model_engine, train_loader, val_loader, criterion, None, epoch, num_classes, device)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    train()
