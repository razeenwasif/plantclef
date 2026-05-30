#!/usr/bin/env bash
# Stage 2c: Unfreeze last 12 transformer blocks + projection.
# Usage:
#   bash scripts/train_last_blocks_12.sh                                               # 1 GPU
#   bash scripts/train_last_blocks_12.sh outputs/last_blocks_8/checkpoints/best.pt 2  # 2 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

RESUME="${1:-./outputs/last_blocks_8/checkpoints/best.pt}"
NGPU="${2:-1}"

META_CSV="/root/workspace/PlantCLEF2026/src_experiments/i003_bioclip25_cap_image_extra500/data/combined_old_extra_max500_train_manifest.csv"

echo "=========================================="
echo "  BioCLIP 2.5 i003 — Stage 2c: Last 12 Blocks (extra500)"
echo "  Resume: $RESUME"
echo "  GPUs: $NGPU"
echo "=========================================="

ARGS=(
  --metadata-csv "$META_CSV"
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
echo "Stage 2c complete. Best checkpoint: ./outputs/last_blocks_12/checkpoints/best.pt"
