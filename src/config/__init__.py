"""Unified Config Package for ORACLE.
Exports both the new typed system and legacy global variables.
"""
import os
import sys
from .loader import load_config
from .schema import OracleConfig, TrainingConfig, ModelConfig, HardwareConfig, TilingConfig, FilterConfig, AggregationConfig, PostprocessConfig

# 1. Load the unified config object
# If a YAML config was passed in CLI, we try to find it (best effort for legacy)
_yaml_path = None
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        _yaml_path = sys.argv[i+1]
        break

_cfg = load_config(_yaml_path)

# 2. Export Global Variables (Legacy Support)
# All scripts doing 'from src import config' will get these.

# Hardware
USE_FP8        = _cfg.hardware.use_fp8
USE_COMPILE    = _cfg.hardware.use_compile
COMPILE_MODE   = _cfg.hardware.compile_mode
EXTREME_MODE   = _cfg.hardware.extreme_mode
LOCAL_RANK     = _cfg.hardware.local_rank
WORLD_SIZE     = _cfg.hardware.world_size
RANK           = _cfg.hardware.rank
NUM_GPUS       = _cfg.hardware.world_size

# Dataset
DATASET_MODE   = _cfg.dataset.mode
RAW_CSV        = _cfg.dataset.raw_csv
CLEANED_CSV    = _cfg.dataset.cleaned_csv
TRAIN_CSV      = CLEANED_CSV if os.path.exists(CLEANED_CSV) else RAW_CSV
IMG_DIR        = _cfg.dataset.img_dir

# Architecture
RESOLUTION          = _cfg.model.resolution
BIOCLIP_NAME        = _cfg.model.bioclip_name
DINOV3_NAME         = _cfg.model.dinov3_name
CONVNEXT_NAME       = _cfg.model.convnext_name
USE_REGION_FEATURES = _cfg.model.use_region_features

# Hyperparameters
BATCH_SIZE         = _cfg.training.batch_size
P2_BATCH_SIZE      = _cfg.training.p2_batch_size
ACCUMULATION_STEPS = _cfg.training.accumulation_steps
LORA_R             = _cfg.model.lora_r
LORA_ALPHA         = _cfg.model.lora_alpha
LORA_DROPOUT       = _cfg.model.lora_dropout

# Paths (Dynamic)
BASE_MODEL_DIR     = _cfg.base_model_dir
SWA_CKPT_PATH      = getattr(_cfg, "SWA_CKPT_PATH", f"{BASE_MODEL_DIR}/swa_model_final.pth")
P1_CKPT_PATH       = getattr(_cfg, "P1_CKPT_PATH", f"{BASE_MODEL_DIR}/phase1_checkpoint.pth")
P2_CKPT_DIR        = getattr(_cfg, "P2_CKPT_DIR", f"{BASE_MODEL_DIR}/phase2_checkpoint")
P2_EPOCH_CKPT      = getattr(_cfg, "P2_EPOCH_CKPT", f"{BASE_MODEL_DIR}/phase2_epoch_checkpoint.pth")
FEATURE_CACHE_PATH = getattr(_cfg, "FEATURE_CACHE_PATH", f"models/cuda_deep_sat/phase1_feature_cache.pt")

# Research
USE_FLASH_ATTN         = _cfg.model.use_flash_attn
USE_ZERO_SHOT          = _cfg.model.use_zero_shot
ZERO_SHOT_WEIGHT       = _cfg.model.zero_shot_weight
ZERO_SHOT_TEMP         = _cfg.model.zero_shot_temp
ZERO_SHOT_PATH         = _cfg.model.zero_shot_path
USE_SWA                = _cfg.training.use_swa
SWA_START_EPOCH        = _cfg.training.swa_start_epoch
SWA_LR                 = _cfg.training.swa_lr
USE_TRAIT_DISTILLATION = _cfg.training.use_trait_distillation
KD_ALPHA               = _cfg.training.kd_alpha

# Memory chunking (per-backbone forward pass chunk sizes)
EXTRACT_CHUNK_SIZE = _cfg.memory.extract_chunk_size
CHUNK_SIZE         = _cfg.memory.chunk_size
P2_CHUNK_SIZE      = _cfg.memory.p2_chunk_size
PREFETCH_DEPTH     = _cfg.memory.prefetch_depth
PCA_COMPONENTS = 1024
VAL_EVERY_N_EPOCHS = _cfg.training.val_every_n_epochs
START_VAL_EPOCH = _cfg.training.start_val_epoch
MAX_VAL_BATCHES = _cfg.training.max_val_batches
PATIENCE = _cfg.training.patience
EPOCHS_PHASE2 = _cfg.training.epochs_p2

def get_raw_config():
    return _cfg

__all__ = [
    "load_config",
    "OracleConfig",
    "TrainingConfig",
    "ModelConfig",
    "HardwareConfig",
    "TilingConfig",
    "FilterConfig",
    "AggregationConfig",
    "PostprocessConfig",
    "RESOLUTION",
    "BATCH_SIZE",
    "P2_BATCH_SIZE",
    "USE_FP8",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "BASE_MODEL_DIR"
]
