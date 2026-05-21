#!/usr/bin/env bash
# Stage 2: Unfreeze last 4 transformer blocks + projection.
# Usage:
#   bash scripts/train_last_blocks.sh [checkpoint] [n_gpus]
#
# Examples:
#   bash scripts/train_last_blocks.sh                                    # 1 GPU, auto resume
#   bash scripts/train_last_blocks.sh outputs/head_only/checkpoints/best.pt 2   # 2 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

RESUME="${1:-./outputs/last_blocks_8/checkpoints/best.pt}"
NGPU="${2:-1}"

echo "=========================================="
echo "  BioCLIP 2.5 — Stage 2: Last 12 Blocks"
echo "  Resume: $RESUME"
echo "  GPUs: $NGPU"
echo "=========================================="

ARGS=(
  --epochs 5
  --batch-size 64
  --grad-accum-steps 4
  --precision bf16
  --use-taxonomy-heads
  --unfreeze-last-n-blocks 12
  --head-lr 1e-4
  --backbone-lr 1e-6
  --weight-decay 1e-4
  --label-smoothing 0.1
  --warmup-epochs 1
  --num-workers 8
  --log-every 50
  --val-every 1
  --resume "$RESUME"
  --resume-weights-only
  --output-dir ./outputs/last_blocks_12
)

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" train.py "${ARGS[@]}"
else
  python train.py "${ARGS[@]}"
fi

echo ""
echo "Stage 2 complete. Best checkpoint: ./outputs/last_blocks/checkpoints/best.pt"
