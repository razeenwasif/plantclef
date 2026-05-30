#!/usr/bin/env bash
# Smoke test: ~200 samples, 1 epoch, frozen backbone, verify end-to-end pipeline.
# Usage:
#   bash scripts/smoke_test.sh          # single GPU
#   bash scripts/smoke_test.sh 2        # 2 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NGPU="${1:-1}"

META_CSV="/root/workspace/PlantCLEF2026/src_experiments/i003_bioclip25_cap_image_extra500/data/combined_old_extra_max500_train_manifest.csv"

echo "=========================================="
echo "  BioCLIP 2.5 i003 — Smoke Test (extra500)"
echo "  GPUs: $NGPU"
echo "=========================================="

ARGS=(
  --metadata-csv "$META_CSV"
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
  torchrun --standalone --nproc_per_node="$NGPU" train.py "${ARGS[@]}"
else
  python train.py "${ARGS[@]}"
fi

echo ""
echo "Smoke test finished. Check ./outputs/smoke_test/"
