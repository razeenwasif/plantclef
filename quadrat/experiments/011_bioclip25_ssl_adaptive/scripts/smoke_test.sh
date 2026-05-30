#!/usr/bin/env bash
# Smoke tests for experiment 011 — verifies end-to-end pipeline.
#
# Test 1: SSL pre-training (32 images, 1 epoch)
# Test 2: Supervised training warm-started from SSL backbone (~200 samples, 1 epoch)
# Test 3: Plain supervised training without SSL (baseline check)
#
# Usage:
#   bash scripts/smoke_test.sh          # single GPU
#   bash scripts/smoke_test.sh 2        # 2 GPUs (test 3 only; SSL is single-GPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

NGPU="${1:-1}"

echo "=========================================="
echo "  BioCLIP 2.5 SSL+Supervised — Smoke Test"
echo "  GPUs: $NGPU"
echo "=========================================="

# ---------------------------------------------------------------------------
# Test 1: SSL pre-training
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 1: SSL pre-training (32 images, 1 epoch) ---"

python train_ssl.py \
  --image-dirs /workspace/plantclef/raw/pseudo_quadrats \
  --epochs 1 \
  --batch-size 8 \
  --limit 32 \
  --num-workers 2 \
  --precision bf16 \
  --output-dir outputs/ssl_smoke

echo "SSL pre-training smoke test passed."

# ---------------------------------------------------------------------------
# Test 2: Supervised training with SSL backbone warm-start
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 2: Supervised training (SSL warm-start, ~200 samples, 1 epoch) ---"

python train.py \
  --smoke-test \
  --ssl-backbone-checkpoint outputs/ssl_smoke/checkpoints/backbone.pt \
  --epochs 1 \
  --batch-size 16 \
  --precision bf16 \
  --use-taxonomy-heads \
  --head-lr 1e-4 \
  --backbone-lr 1e-6 \
  --num-workers 2 \
  --log-every 5 \
  --val-every 1 \
  --output-dir outputs/smoke_ssl_supervised

echo "SSL-supervised smoke test passed."

# ---------------------------------------------------------------------------
# Test 3: Plain supervised training (no SSL, verifies baseline still works)
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 3: Plain supervised training (~200 samples, 1 epoch) ---"

TRAIN_CMD=(
  python train.py
  --smoke-test
  --epochs 1
  --batch-size 16
  --grad-accum-steps 1
  --precision bf16
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
echo "All smoke tests passed."
echo "Outputs: outputs/ssl_smoke/  outputs/smoke_ssl_supervised/  outputs/smoke_test/"
