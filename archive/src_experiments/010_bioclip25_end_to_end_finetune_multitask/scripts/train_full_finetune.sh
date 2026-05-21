#!/usr/bin/env bash
# Stage 3: Full backbone fine-tuning at very low LR.
# Usage:
#   bash scripts/train_full_finetune.sh [checkpoint] [n_gpus]
#
# Examples:
#   bash scripts/train_full_finetune.sh                                       # 1 GPU, auto resume
#   bash scripts/train_full_finetune.sh outputs/last_blocks/checkpoints/best.pt 2  # 2 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

RESUME="${1:-./outputs/last_blocks/checkpoints/best.pt}"
NGPU="${2:-1}"

echo "=========================================="
echo "  BioCLIP 2.5 — Stage 3: Full Fine-Tune"
echo "  Resume: $RESUME"
echo "  GPUs: $NGPU"
echo "=========================================="

ARGS=(
  --epochs 10
  --batch-size 32
  --grad-accum-steps 8
  --precision bf16
  --use-taxonomy-heads
  --full-finetune
  --head-lr 5e-4
  --backbone-lr 5e-5
  --weight-decay 1e-4
  --label-smoothing 0.1
  --warmup-epochs 1
  --num-workers 8
  --log-every 50
  --val-every 1
  --resume "$RESUME"
  --resume-weights-only
  --output-dir ./outputs/full_finetune
)

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" train.py "${ARGS[@]}"
else
  python train.py "${ARGS[@]}"
fi

echo ""
echo "Stage 3 complete. Best checkpoint: ./outputs/full_finetune/checkpoints/best.pt"
