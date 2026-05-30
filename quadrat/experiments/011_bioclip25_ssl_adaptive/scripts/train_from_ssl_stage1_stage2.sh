#!/usr/bin/env bash
# Two-stage supervised fine-tuning warm-started from an SSL backbone.
#
# Stage 1: Train head only (frozen backbone) from SSL backbone weights.
#          Produces outputs/ssl_head_only/checkpoints/best.pt
#
# Stage 2: Unfreeze last 8 blocks starting from Stage 1 best weights.
#          Produces outputs/ssl_last_blocks/checkpoints/best.pt
#
# Usage:
#   bash scripts/train_from_ssl_stage1_stage2.sh [ssl_backbone_ckpt] [n_gpus]
#
# Examples:
#   bash scripts/train_from_ssl_stage1_stage2.sh
#   bash scripts/train_from_ssl_stage1_stage2.sh outputs/ssl_bioclip25/checkpoints/backbone.pt 2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SSL_CKPT="${1:-outputs/ssl_bioclip25/checkpoints/backbone.pt}"
NGPU="${2:-1}"

echo "=========================================="
echo "  BioCLIP 2.5 SSL → Supervised Training"
echo "  SSL backbone : $SSL_CKPT"
echo "  GPUs         : $NGPU"
echo "=========================================="

# ---------------------------------------------------------------------------
# Stage 1: Head-only, warm-started from SSL backbone
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 1: Head Only (SSL warm-start) ---"

STAGE1_ARGS=(
  --ssl-backbone-checkpoint "$SSL_CKPT"
  --freeze-backbone
  --epochs 10
  --batch-size 512
  --grad-accum-steps 2
  --precision bf16
  --use-taxonomy-heads
  --head-lr 1e-4
  --backbone-lr 1e-6
  --weight-decay 1e-4
  --label-smoothing 0.1
  --warmup-epochs 1
  --num-workers 16
  --log-every 100
  --val-every 1
  --output-dir ./outputs/ssl_head_only
)

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" train.py "${STAGE1_ARGS[@]}"
else
  python train.py "${STAGE1_ARGS[@]}"
fi

echo "Stage 1 complete. Best checkpoint: ./outputs/ssl_head_only/checkpoints/best.pt"

# ---------------------------------------------------------------------------
# Stage 2: Unfreeze last 8 blocks, resume weights-only from Stage 1 best
# ---------------------------------------------------------------------------
echo ""
echo "--- Stage 2: Last 4 Blocks (from Stage 1 best) ---"

STAGE1_BEST="/root/workspace/PlantCLEF2026/src_experiments/011_bioclip25_ssl_adaptive/outputs/ssl_head_only/checkpoints/best.pt"

STAGE2_ARGS=(
  --unfreeze-last-n-blocks 8
  --resume "$STAGE1_BEST"
  --resume-weights-only
  --epochs 10
  --batch-size 64
  --grad-accum-steps 4
  --precision bf16
  --use-taxonomy-heads
  --head-lr 1e-4
  --backbone-lr 1e-6
  --weight-decay 1e-4
  --label-smoothing 0.1
  --warmup-epochs 1
  --num-workers 8
  --log-every 50
  --val-every 1
  --output-dir ./outputs/ssl_last_blocks
)

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" train.py "${STAGE2_ARGS[@]}"
else
  python train.py "${STAGE2_ARGS[@]}"
fi

echo ""
echo "Stage 2 complete. Best checkpoint: ./outputs/ssl_last_blocks/checkpoints/best.pt"
echo ""
echo "Run inference with:"
echo "  bash scripts/infer_best_adaptive.sh"
