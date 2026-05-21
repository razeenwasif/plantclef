#!/usr/bin/env bash
# Stage 1: Head-only training capped at 1000 images/species.
#
# Ideal for a fast first-pass or hyperparameter search when the full dataset
# is too large.  Remove --max-images-per-species to train on all images.
#
# Usage:
#   bash scripts/train_cap1000_head_only.sh          # single GPU
#   bash scripts/train_cap1000_head_only.sh 2        # 2 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NGPU="${1:-1}"

META_CSV="/root/workspace/PlantCLEF2026/src_experiments/i001_data_download/data/training_usage/metadata_filled_genus_family.csv"

echo "=========================================="
echo "  BioCLIP 2.5 i002 — Head Only (cap=1000/sp)"
echo "  GPUs: $NGPU"
echo "=========================================="

ARGS=(
  --metadata-csv "$META_CSV"
  --max-images-per-species 1000
  --cap-seed 42
  --epochs 10
  --batch-size 512
  --grad-accum-steps 2
  --precision bf16
  --use-taxonomy-heads
  --freeze-backbone
  --head-lr 1e-4
  --backbone-lr 1e-6
  --weight-decay 1e-4
  --label-smoothing 0.1
  --warmup-epochs 1
  --num-workers 16
  --log-every 100
  --val-every 1
  --output-dir ./outputs/cap1000_head_only
)

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" train.py "${ARGS[@]}"
else
  python train.py "${ARGS[@]}"
fi

echo ""
echo "Done. Best checkpoint: ./outputs/cap1000_head_only/checkpoints/best.pt"
