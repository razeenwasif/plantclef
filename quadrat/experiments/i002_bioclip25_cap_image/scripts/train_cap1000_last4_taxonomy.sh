#!/usr/bin/env bash
# Stage 2: Last-4-block fine-tuning with cap=1000/species + taxonomy heads.
#
# Resumes from cap1000_head_only best checkpoint by default.
# Also enables WeightedRandomSampler if sample_weight is in the CSV.
#
# Usage:
#   bash scripts/train_cap1000_last4_taxonomy.sh                                          # 1 GPU
#   bash scripts/train_cap1000_last4_taxonomy.sh outputs/cap1000_head_only/.../best.pt 2  # 2 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

RESUME="${1:-./outputs/cap1000_head_only/checkpoints/best.pt}"
NGPU="${2:-1}"

META_CSV="/root/workspace/PlantCLEF2026/src_experiments/i001_data_download/data/training_usage/metadata_filled_genus_family.csv"

echo "=========================================="
echo "  BioCLIP 2.5 i002 — Last 4 Blocks (cap=1000/sp + taxonomy)"
echo "  Resume: $RESUME"
echo "  GPUs: $NGPU"
echo "=========================================="

ARGS=(
  --metadata-csv "$META_CSV"
  --max-images-per-species 1000
  --cap-seed 42
  --use-sample-weights
  --epochs 5
  --batch-size 64
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
  --log-every 50
  --val-every 1
  --resume "$RESUME"
  --resume-weights-only
  --output-dir ./outputs/cap1000_last4_taxonomy
)

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" train.py "${ARGS[@]}"
else
  python train.py "${ARGS[@]}"
fi

echo ""
echo "Done. Best checkpoint: ./outputs/cap1000_last4_taxonomy/checkpoints/best.pt"
