#!/usr/bin/env bash
# Adaptive tile inference using the best SSL-supervised checkpoint.
#
# Sweeps all four selection modes (fixed_topk, prob_threshold,
# relative_threshold, gap) in one pass — model forward runs only once per image.
#
# Usage:
#   bash scripts/infer_best_adaptive.sh [checkpoint] [image_dir] [output_dir]
#
# bash scripts/infer_best_adaptive.sh outputs/head_only/checkpoints/best.pt /workspace/plantclef/raw/test outputs/head_only_infer
# bash scripts/infer_best_adaptive.sh outputs/last_blocks/checkpoints/best.pt /workspace/plantclef/raw/test outputs/last_blocks_infer
# bash scripts/infer_best_adaptive.sh outputs/last_blocks_8/checkpoints/best.pt /workspace/plantclef/raw/test outputs/last_blocks_8_infer
# 
# Defaults:
#   checkpoint : outputs/ssl_last_blocks/checkpoints/best.pt
#   image_dir  : /workspace/plantclef/raw/test
#   output_dir : outputs/ssl_adaptive_infer
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CKPT="${1:-outputs/last_blocks_12/checkpoints/best.pt}"
IMAGE_DIR="${2:-/workspace/plantclef/raw/test}"
OUT_DIR="${3:-outputs/last_blocks_12_tile_sweep_maxk10}"

echo "=========================================="
echo "  BioCLIP 2.5 — Adaptive Tile Inference"
echo "  checkpoint : $CKPT"
echo "  image_dir  : $IMAGE_DIR"
echo "  output_dir : $OUT_DIR"
echo "=========================================="

python infer_tiles_adaptive.py \
  --checkpoint "$CKPT" \
  --image-dir "$IMAGE_DIR" \
  --tile-mode grid_4x4 \
  --overlap 0.0 \
  --agg-modes softmax_mean \
  --top-ks 1 2 3 4 5 \
  --selection-modes fixed_topk relative_threshold prob_threshold gap \
  --min-k 2 \
  --max-k 10 \
  --relative-thresholds 0.15 0.20 0.25 0.30 \
  --prob-thresholds 0.02 0.03 0.05 \
  --gap-ratios 0.40 0.50 0.60 \
  --save-logits \
  --logit-adj-tau 0.25 \
  --metadata-csv /root/workspace/PlantCLEF2026/src_experiments/i003_bioclip25_cap_image_extra500/data/combined_old_extra_max500_train_manifest.csv \
  --precision bf16 \
  --batch-size 512 \
  --output-dir "$OUT_DIR"

echo ""
echo "Inference complete. Outputs: $OUT_DIR"
