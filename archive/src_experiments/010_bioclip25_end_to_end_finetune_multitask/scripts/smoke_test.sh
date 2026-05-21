#!/usr/bin/env bash
# Smoke test: ~200 samples, 1 epoch, frozen backbone, verify end-to-end pipeline.
# Usage:
#   bash scripts/smoke_test.sh          # single GPU
#   bash scripts/smoke_test.sh 2        # 2 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NGPU="${1:-1}"

echo "=========================================="
echo "  BioCLIP 2.5 Multi-Task — Smoke Test"
echo "  GPUs: $NGPU"
echo "=========================================="

TRAIN_CMD=(
  python train.py
  --smoke-test
  --epochs 1
  --batch-size 16
  --grad-accum-steps 1
  --precision fp16
  --use-taxonomy-heads
  --head-lr 1e-4
  --backbone-lr 1e-6
  --num-workers 2
  --log-every 5
  --val-every 1
  --output-dir ./outputs/smoke_test
)

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" "${TRAIN_CMD[@]:1}"
else
  "${TRAIN_CMD[@]}"
fi

echo ""
echo "Smoke test finished. Check ./outputs/smoke_test/"
