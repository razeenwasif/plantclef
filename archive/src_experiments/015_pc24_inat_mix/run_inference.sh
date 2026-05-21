#!/usr/bin/env bash
# 015 PC24+iNat mix inference — team-best recipe (grid_4x4 ov=0 tile=448 img=224
# softmax_mean top-3 bf16) on the epoch-1 best.pt checkpoint.

set -euo pipefail

REPO=/workspace/working/workspace/PlantCLEF2026
EXP=$REPO/src_experiments/010_bioclip25_end_to_end_finetune_multitask
OUTROOT=$REPO/src_experiments/015_pc24_inat_mix
TEST_IMAGES=/workspace/plantclef/processed/test_images_jpeg85_max800
PY=/workspace/pytorch_env/bin/python
LOGDIR=$OUTROOT/logs
mkdir -p "$LOGDIR"

cd "$EXP"

CKPT=$EXP/outputs/pc24_inat_mix_unfreeze4/checkpoints/best.pt
OUTDIR=$OUTROOT/inference/ep5
LOG=$LOGDIR/infer_ep5.log

if [ ! -f "$CKPT" ]; then
  echo "MISSING: $CKPT" >&2; exit 1
fi

echo "=========================================="
echo "  015 inference  →  $OUTDIR"
echo "=========================================="

"$PY" infer_tiles.py \
    --checkpoint "$CKPT" \
    --image-dir  "$TEST_IMAGES" \
    --tile-mode  grid_4x4 \
    --tile-size  448 \
    --overlap    0.0 \
    --img-size   224 \
    --agg-mode   softmax_mean \
    --top-k      3 \
    --batch-size 64 \
    --precision  bf16 \
    --num-workers 4 \
    --output-dir "$OUTDIR" 2>&1 | tee "$LOG"

echo "Done. Submission at $OUTDIR/submission.csv"
