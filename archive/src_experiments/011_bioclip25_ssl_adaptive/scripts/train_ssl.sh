#!/usr/bin/env bash
# SSL pre-training with SimSiam on unlabeled pseudo-quadrat images.
# Trains last 4 BioCLIP 2.5 transformer blocks + projector/predictor.
#
# Usage:
#   bash scripts/train_ssl.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

OUT_DIR="outputs/ssl_bioclip25_simsiam_e20"

echo "=========================================="
echo "  BioCLIP 2.5 — SSL Pre-Training (SimSiam)"
echo "=========================================="

python train_ssl.py \
  --image-dirs /workspace/plantclef/raw/pseudo_quadrats \
  --epochs 20 \
  --warmup-epochs 2 \
  --batch-size 128 \
  --num-workers 16 \
  --precision bf16 \
  --unfreeze-last-n-blocks 4 \
  --backbone-lr 1e-6 \
  --head-lr 1e-4 \
  --weight-decay 0.05 \
  --grad-clip 1.0 \
  --output-dir "$OUT_DIR"

echo ""
echo "SSL pre-training complete."
echo "Backbone checkpoint: $OUT_DIR/checkpoints/backbone.pt"
