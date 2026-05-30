"""
Defaults and constants for experiment 007: BioCLIP + SAM vegetation-weighted inference.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# BioCLIP model defaults
# ---------------------------------------------------------------------------

# Default batch sizes keyed by model name (tuned for ~16 GB VRAM)
MODEL_DEFAULT_BATCH: dict[str, int] = {
    "hf-hub:imageomics/bioclip":              64,
    "hf-hub:imageomics/bioclip-2":            64,
    "hf-hub:imageomics/bioclip-2.5-vith14":   16,
}

DEFAULT_MODEL_NAME    = "hf-hub:imageomics/bioclip"
DEFAULT_TILE_SIZE     = 224
DEFAULT_TILE_OVERLAP  = 112
DEFAULT_TOP_K         = 5
DEFAULT_PROMPT_MODE   = "scientific"
DEFAULT_TEXT_BATCH    = 256

# ---------------------------------------------------------------------------
# Data paths  (override at CLI)
# ---------------------------------------------------------------------------

DEFAULT_SPECIES_CSV = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "002_bioclip_tile_zero_shot_v2/data/"
    "species_lookup_with_gbif_cleaned_names.csv"
)
DEFAULT_IMAGES_ROOT = "/workspace/plantclef/kaggle_uploads/test/images"
DEFAULT_OUTPUT_DIR  = "./outputs"

# ---------------------------------------------------------------------------
# SAM defaults
# ---------------------------------------------------------------------------

# Where to store the downloaded SAM checkpoint
DEFAULT_SAM_CHECKPOINT = "./checkpoints/sam_vit_b_01ec64.pth"
DEFAULT_SAM_MODEL_TYPE = "vit_b"       # smallest SAM model; vit_l/vit_h also work

# SamAutomaticMaskGenerator tuning for 224x224 tiles
SAM_POINTS_PER_SIDE          = 16     # fewer points → faster; 32 for higher recall
SAM_PRED_IOU_THRESH          = 0.70
SAM_STABILITY_SCORE_THRESH   = 0.80
SAM_MIN_MASK_REGION_AREA     = 50     # pixels; filters tiny mask fragments

# Greenness thresholds (Excess Green Index)
DEFAULT_EXG_THRESHOLD        = 20.0   # raw ExG value; pixels above = vegetation
DEFAULT_SAM_MIN_GREENNESS    = 0.25   # fraction of mask pixels that must be green

# ---------------------------------------------------------------------------
# Tile weighting defaults
# ---------------------------------------------------------------------------

# w_i = clip(alpha + beta * veg_ratio_i, w_min, w_max)
DEFAULT_WEIGHT_ALPHA = 0.5    # base weight for zero-veg tile (keeps it in play)
DEFAULT_WEIGHT_BETA  = 1.0    # slope: full-veg tile gets alpha+beta weight
DEFAULT_WEIGHT_MIN   = 0.1    # never fully suppress a tile
DEFAULT_WEIGHT_MAX   = 2.0    # cap on vegetation-rich tile boost

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

AGGREGATION_METHODS = ("max", "mean", "topk_mean", "weighted_mean", "weighted_topk_mean")
DEFAULT_AGGREGATION  = "weighted_mean"
DEFAULT_TOPK_TILES   = 3      # tiles used in topk_mean / weighted_topk_mean

# ---------------------------------------------------------------------------
# Scoring method
# ---------------------------------------------------------------------------

SCORING_METHODS = ("rgb", "sam")
DEFAULT_SCORING  = "rgb"      # "rgb" = pure ExG; "sam" = SAM masks + ExG

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

DEFAULT_N_VISUALIZE     = 10     # images to generate detailed visualizations for
MAX_TILE_VIZ_PER_IMAGE  = 16     # max tiles shown in the heatmap grid
MAX_SAM_VIZ_TILES       = 4      # max tiles with SAM mask overlays per image
VIZ_THUMBNAIL_SIZE      = 112    # px for tile thumbnails in grid panels
VIZ_DPI                 = 120
