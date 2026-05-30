import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_MODE = "plantclef"  # or "kaggle_test"

DATASETS = {
    "plantclef": {
        "raw_csv":     "/workspace/plantclef/raw/PlantCLEF2024_single_plant_training_metadata.csv",
        "img_dir":     "/workspace/plantclef/raw/train/images_max_side_800/",
        "cleaned_csv": "/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv"
    },
    "kaggle_test": {
        "raw_csv":     "/workspace/plantclef/processed/kaggle_test_metadata.csv",
        "img_dir":     "/workspace/plantclef/raw/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/",
        "cleaned_csv": "/workspace/plantclef/processed/kaggle_test_metadata_cleaned.csv"
    }
}

RAW_CSV     = DATASETS[DATASET_MODE]["raw_csv"]
IMG_DIR     = DATASETS[DATASET_MODE]["img_dir"]
CLEANED_CSV = DATASETS[DATASET_MODE]["cleaned_csv"]

# ---------------------------------------------------------------------------
# Resolution & batch sizes
# ---------------------------------------------------------------------------
RESOLUTION    = 384   
BATCH_SIZE    = 384   # Phase 1
P2_BATCH_SIZE = 64    # Safe stability limit for Triple-Parallel Ensemble + torch.compile

# ---------------------------------------------------------------------------
# Chunked forward chunk sizes
# ---------------------------------------------------------------------------
EXTRACT_CHUNK_SIZE = 64   
CHUNK_SIZE         = 32
P2_CHUNK_SIZE      = 4    # Ultra-low ceiling to prevent parallel backbone collisions

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
ACCUMULATION_STEPS = 8    # Effective Batch = 512 (64 * 8)

EPOCHS_PHASE1      = 20
EPOCHS_PHASE2      = 5    
P2_SAMPLES_PER_EPOCH = 1000000 

VAL_EVERY_N_EPOCHS = 1    
MAX_VAL_BATCHES    = 150  
PATIENCE           = 3    

# ---------------------------------------------------------------------------
# LoRA (Phase 2)
# ---------------------------------------------------------------------------
LORA_R       = 64    
LORA_ALPHA   = 128   
LORA_DROPOUT = 0.05

# ---------------------------------------------------------------------------
# Backbone selection (Blackwell Optimized)
# ---------------------------------------------------------------------------
MODE = "5090"   

if MODE == "5090":
    BIOCLIP_NAME  = "hf-hub:imageomics/bioclip"
    DINOV2_NAME   = "vit_large_patch14_dinov2"
    CONVNEXT_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    USE_FP8       = False  
    USE_COMPILE   = True
    LOAD_CACHE_TO_GPU = False 
elif MODE == "4090":
    BIOCLIP_NAME  = "hf-hub:imageomics/bioclip"
    DINOV2_NAME   = "vit_large_patch14_dinov2"
    CONVNEXT_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    USE_FP8       = False
    USE_COMPILE   = False
    LOAD_CACHE_TO_GPU = False
else:
    BIOCLIP_NAME  = "hf-hub:imageomics/bioclip"
    DINOV2_NAME   = "vit_giant_patch14_dinov2.lvd142m"
    CONVNEXT_NAME = "convnextv2_huge.fcmae_ft_in22k_in1k_384"
    USE_FP8       = False
    USE_COMPILE   = False
    LOAD_CACHE_TO_GPU = False

# ---------------------------------------------------------------------------
# Region features + PCA
# ---------------------------------------------------------------------------
USE_REGION_FEATURES = True
PCA_COMPONENTS      = 1024  
# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
FEATURE_CACHE_PATH = "models/blackwell_v2/phase1_feature_cache.pt"
P1_CKPT_PATH       = "models/blackwell_v2/phase1_checkpoint.pth"
P2_CKPT_DIR        = "models/blackwell_v2/phase2_checkpoint"
P2_EPOCH_CKPT      = "models/blackwell_v2/phase2_epoch_checkpoint.pth"
PCA_TRANSFORM_PATH  = "models/blackwell_v2/pca_transform.pkl"

# ---------------------------------------------------------------------------
# DINOv3 experiment (src_experiments/005_dinov3_multilabel/)
# ---------------------------------------------------------------------------
DINOV3_NAME        = "vit_large_patch16_dinov3.lvd1689m"  # 300M, 1024-d, patch-16
DINOV3_RES         = 384                                   # multiple of 16
DINOV3_LORA_R      = 32
DINOV3_LORA_ALPHA  = 64
DINOV3_LORA_DROPOUT = 0.05
# Probability of K=1..5 plants per synthetic mosaic; matches typical quadrat density
MOSAIC_K_DIST      = [0.30, 0.30, 0.20, 0.12, 0.08]

DINOV3_FEATURE_CACHE_PATH = "models/dinov3_v1/phase1_feature_cache.pt"
DINOV3_P1_HEAD_PATH       = "models/dinov3_v1/phase1_head.pth"
DINOV3_P2_CKPT_PATH       = "models/dinov3_v1/phase2_lora.pth"
