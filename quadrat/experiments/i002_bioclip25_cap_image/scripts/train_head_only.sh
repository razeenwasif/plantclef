#!/usr/bin/env bash
# Stage 1: Train head only (frozen backbone) with new metadata CSV.
# Usage:
#   bash scripts/train_head_only.sh          # single GPU
#   bash scripts/train_head_only.sh 2        # 2 GPUs (batch_size is per-GPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NGPU="${1:-1}"

META_CSV="/root/workspace/PlantCLEF2026/src_experiments/i001_data_download/data/training_usage/metadata_filled_genus_family.csv"

echo "=========================================="
echo "  BioCLIP 2.5 i002 — Stage 1: Head Only"
echo "  GPUs: $NGPU"
echo "=========================================="

ARGS=(
  --metadata-csv "$META_CSV"
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
  --num-workers 8
  --log-every 100
  --val-every 1
  --output-dir ./outputs/head_only
)

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" train.py "${ARGS[@]}"
else
  python train.py "${ARGS[@]}"
fi

echo ""
echo "Stage 1 complete. Best checkpoint: ./outputs/head_only/checkpoints/best.pt"
