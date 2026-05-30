#!/usr/bin/env bash
# Stage 2a: Unfreeze last 4 transformer blocks + projection.
# Usage:
#   bash scripts/train_last_blocks.sh                                            # 1 GPU
#   bash scripts/train_last_blocks.sh outputs/head_only/checkpoints/best.pt 2   # 2 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

RESUME="${1:-./outputs/head_only/checkpoints/best.pt}"
NGPU="${2:-1}"

META_CSV="/root/workspace/PlantCLEF2026/src_experiments/i001_data_download/data/training_usage/metadata_filled_genus_family.csv"

echo "=========================================="
echo "  BioCLIP 2.5 i002 — Stage 2a: Last 4 Blocks"
echo "  Resume: $RESUME"
echo "  GPUs: $NGPU"
echo "=========================================="

ARGS=(
  --metadata-csv "$META_CSV"
  --epochs 10
  --batch-size 128
  --grad-accum-steps 4
  --precision bf16
  --use-taxonomy-heads
  --unfreeze-last-n-blocks 4
  --head-lr 1e-4
  --backbone-lr 1e-6
  --weight-decay 1e-4
  --label-smoothing 0.1
  --warmup-epochs 1
  --num-workers 8
  --log-every 100
  --val-every 1
  --resume "$RESUME"
  --resume-weights-only
  --output-dir ./outputs/last_blocks
)

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" train.py "${ARGS[@]}"
else
  python train.py "${ARGS[@]}"
fi

echo ""
echo "Stage 2a complete. Best checkpoint: ./outputs/last_blocks/checkpoints/best.pt"
