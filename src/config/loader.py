import os
import yaml
import subprocess
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from .schema import (
    OracleConfig, HardwareConfig, DatasetConfig, ModelConfig,
    TrainingConfig, MemoryConfig, TilingConfig, FilterConfig, AggregationConfig, PostprocessConfig
)

def _probe_gpu_name() -> str:
    try:
        res = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
        return res.strip().split('\n')[0]
    except:
        return "Unknown GPU"

def _load_yaml(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def load_config(yaml_path: Optional[str] = None) -> OracleConfig:
    """
    Unified entry point for configuration.
    Loads YAML, probes hardware, and returns a typed OracleConfig object.
    """
    # 1. Start with Default Schema
    config = OracleConfig()
    
    # ORACLE: Enforce Blackwell Tensor Core utilization for FP32 operations
    torch.set_float32_matmul_precision('high')
    
    # 2. Load User YAML (if provided)
    user_cfg = _load_yaml(yaml_path)
    
    # Simple manual mapping
    if 'dataset' in user_cfg:
        ds = user_cfg['dataset']
        mode = ds.get('mode', 'plantclef')
        # Mode-specific sub-block (e.g. dataset.plantclef.raw_csv)
        mode_block = ds.get(mode, {})
        if 'raw_csv'     in mode_block: config.dataset.raw_csv     = mode_block['raw_csv']
        if 'cleaned_csv' in mode_block: config.dataset.cleaned_csv = mode_block['cleaned_csv']
        if 'img_dir'     in mode_block: config.dataset.img_dir     = mode_block['img_dir']
        if 'shard_dir'   in mode_block: config.dataset.shard_dir   = mode_block['shard_dir']
        # Also accept flat keys directly under dataset:
        if 'raw_csv'     in ds: config.dataset.raw_csv     = ds['raw_csv']
        if 'cleaned_csv' in ds: config.dataset.cleaned_csv = ds['cleaned_csv']
        if 'img_dir'     in ds: config.dataset.img_dir     = ds['img_dir']

    if 'architecture' in user_cfg:
        arch = user_cfg['architecture']
        if 'resolution' in arch: config.model.resolution = arch['resolution']
        if 'bioclip_name' in arch: config.model.bioclip_name = arch['bioclip_name']
        if 'dinov3_name' in arch: config.model.dinov3_name = arch['dinov3_name']
        if 'convnext_name' in arch: config.model.convnext_name = arch['convnext_name']
    
    _yaml_set_p2_batch = False
    for hp_key in ['hyperparameters', 'training']:
        if hp_key in user_cfg:
            hp = user_cfg[hp_key]
            if 'batch_size' in hp: config.training.batch_size = hp['batch_size']
            if 'p2_batch_size' in hp:
                config.training.p2_batch_size = hp['p2_batch_size']
                _yaml_set_p2_batch = True
            if 'accumulation_steps' in hp: config.training.accumulation_steps = hp['accumulation_steps']
            if 'epochs_p2' in hp: config.training.epochs_p2 = hp['epochs_p2']
            if 'p2_samples_per_epoch' in hp: config.training.p2_samples_per_epoch = hp['p2_samples_per_epoch']
            if 'lr' in hp: config.training.learning_rate = hp['lr']
            if 'weight_decay' in hp: config.training.weight_decay = hp['weight_decay']
            if 'val_every_n_epochs' in hp: config.training.val_every_n_epochs = hp['val_every_n_epochs']
            if 'max_val_batches' in hp: config.training.max_val_batches = hp['max_val_batches']
            if 'use_swa' in hp: config.training.use_swa = hp['use_swa']
            if 'swa_start_epoch' in hp: config.training.swa_start_epoch = hp['swa_start_epoch']
            if 'swa_lr' in hp: config.training.swa_lr = hp['swa_lr']
            if 'curriculum' in hp: config.training.curriculum = hp['curriculum']

    if 'lora' in user_cfg:
        lora = user_cfg['lora']
        if 'r' in lora: config.model.lora_r = lora['r']
        if 'alpha' in lora: config.model.lora_alpha = lora['alpha']
        if 'dropout' in lora: config.model.lora_dropout = lora['dropout']

    if 'research' in user_cfg:
        res = user_cfg['research']
        if 'use_swa' in res: config.training.use_swa = res['use_swa']
        if 'swa_start_epoch' in res: config.training.swa_start_epoch = res['swa_start_epoch']
        if 'swa_lr' in res: config.training.swa_lr = res['swa_lr']
        if 'use_flash_attn' in res: config.model.use_flash_attn = res['use_flash_attn']
        if 'use_zero_shot' in res: config.model.use_zero_shot = res['use_zero_shot']
        if 'zero_shot_weight' in res: config.model.zero_shot_weight = res['zero_shot_weight']
        if 'zero_shot_temp' in res: config.model.zero_shot_temp = res['zero_shot_temp']
        if 'use_region_features' in res: config.model.use_region_features = res['use_region_features']
        if 'use_fp8' in res: config.hardware.use_fp8 = res['use_fp8']
        if 'use_compile' in res: config.hardware.use_compile = res['use_compile']

    if 'memory' in user_cfg:
        mem = user_cfg['memory']
        if 'extract_chunk_size' in mem: config.memory.extract_chunk_size = mem['extract_chunk_size']
        if 'chunk_size' in mem: config.memory.chunk_size = mem['chunk_size']
        if 'p2_chunk_size' in mem: config.memory.p2_chunk_size = mem['p2_chunk_size']
        if 'prefetch_depth' in mem: config.memory.prefetch_depth = mem['prefetch_depth']

    # 3. Hardware Probing & Dynamic Overrides
    probed = _probe_gpu_name()
    config.hardware.probed_gpu = probed
    
    # Distributed Environment
    rank_offset = int(os.environ.get("RANK_OFFSET", 0))
    config.hardware.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config.hardware.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.hardware.rank = int(os.environ.get("RANK", 0)) + rank_offset
    
    # Blackwell vs Lovelace (Ada) vs Legacy
    if "Blackwell" in probed or "6000" in probed or "5090" in probed or "B200" in probed:
        config.hardware.preset = "blackwell"
        config.hardware.use_fp8 = True
        config.hardware.use_compile = True
        config.hardware.extreme_mode = (config.hardware.world_size >= 2)
        config.memory.prefetch_depth = 8
        
        # ORACLE: Global Blackwell optimizations
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        os.environ["TORCH_CUDA_MATMUL_TF32"] = "1"
        
        # Scale parameters for Blackwell
        if not _yaml_set_p2_batch:
            config.training.p2_batch_size = 1280
            
    elif "4090" in probed or "Ada" in probed or "Lovelace" in probed:
        config.hardware.preset = "lovelace"
        config.hardware.use_fp8 = False  # 4090 FP8 is slower than BF16 in some Torch versions
        config.hardware.use_compile = True
        config.hardware.extreme_mode = (config.hardware.world_size >= 2)
        config.memory.prefetch_depth = 4
        
        # Scale parameters for Lovelace
        if not _yaml_set_p2_batch:
            config.training.p2_batch_size = 640  # 4090 has less VRAM (24GB) than PRO 6000 (96GB)
    else:
        config.hardware.preset = "generic_cuda"
        config.hardware.use_fp8 = False
        config.hardware.use_compile = True

    # ORACLE: Apply YAML Presets if they match the detected hardware
    # This allows users to provide architecture-specific overrides in the YAML file.
    _preset_set_accum = False
    if 'presets' in user_cfg and config.hardware.preset in user_cfg['presets']:
        preset_cfg = user_cfg['presets'][config.hardware.preset]
        for hp_key in ['hyperparameters', 'training']:
            if hp_key in preset_cfg:
                php = preset_cfg[hp_key]
                if 'batch_size' in php: config.training.batch_size = php['batch_size']
                if 'p2_batch_size' in php: config.training.p2_batch_size = php['p2_batch_size']
                if 'accumulation_steps' in php:
                    config.training.accumulation_steps = php['accumulation_steps']
                    _preset_set_accum = True
        if 'lora' in preset_cfg:
            plora = preset_cfg['lora']
            if 'r' in plora: config.model.lora_r = plora['r']
            if 'alpha' in plora: config.model.lora_alpha = plora['alpha']
        if 'memory' in preset_cfg:
            pmem = preset_cfg['memory']
            if 'extract_chunk_size' in pmem: config.memory.extract_chunk_size = pmem['extract_chunk_size']
            if 'chunk_size' in pmem: config.memory.chunk_size = pmem['chunk_size']
            if 'p2_chunk_size' in pmem: config.memory.p2_chunk_size = pmem['p2_chunk_size']

    # ORACLE: Global Batch Scaling Logic
    # We target a stable global batch size for consistent LoRA convergence.
    target_global_batch = 15360 if config.hardware.preset == "blackwell" else 5120
    
    if not _preset_set_accum:
        # Calculate accumulation steps to hit target global batch
        # global = batch * world_size * accum
        config.training.accumulation_steps = max(1, round(target_global_batch / (config.training.p2_batch_size * config.hardware.world_size)))

    if config.hardware.extreme_mode and config.hardware.local_rank == 0:
        print(f"[ORACLE] Scaling for {config.hardware.world_size} GPUs: "
              f"Batch/GPU={config.training.p2_batch_size}, "
              f"Accum={config.training.accumulation_steps} "
              f"(Global ~{config.training.p2_batch_size * config.hardware.world_size * config.training.accumulation_steps})")

    # 5. Path Management (Cluster-Aware)
    hardware_target = os.environ.get("PLANTCLEF_HARDWARE", "CUDA").lower()
    if config.hardware.extreme_mode:
        pod_id = os.environ.get("ORACLE_NAME", "cluster_final")
    else:
        pod_id = os.environ.get("ORACLE_NAME", "pod_b" if rank_offset > 0 else "pod_a")

    config.base_model_dir = f"models/{hardware_target}_deep_sat/{pod_id}"
    
    # 6. Legacy compatibility for Inference Configs
    for section in ['tiling', 'filter', 'aggregation', 'postprocess']:
        if section in user_cfg:
            section_obj = getattr(config, section)
            for k, v in user_cfg[section].items():
                if hasattr(section_obj, k):
                    setattr(section_obj, k, v)
    
    # Dynamic Checkpoint Strings
    # We add these as properties to the OracleConfig object for convenience
    setattr(config, "SWA_CKPT_PATH", f"{config.base_model_dir}/swa_model_final.pth")
    setattr(config, "P1_CKPT_PATH", f"{config.base_model_dir}/phase1_checkpoint.pth")
    setattr(config, "P2_CKPT_DIR", f"{config.base_model_dir}/phase2_checkpoint")
    setattr(config, "P2_EPOCH_CKPT", f"{config.base_model_dir}/phase2_epoch_checkpoint.pth")
    setattr(config, "FEATURE_CACHE_PATH", f"models/{hardware_target}_deep_sat/phase1_feature_cache.pt")

    # Global Diagnostic Log (Rank 0 only)
    if config.hardware.local_rank == 0:
        mode_str = f"{config.hardware.world_size}x {config.hardware.probed_gpu} Cluster" if config.hardware.extreme_mode else config.hardware.preset
        print(f"[OracleConfig] Mode: {mode_str} | LoRA R={config.model.lora_r} | Batch={config.training.p2_batch_size}")

    return config
