#!/usr/bin/env bash
# 014 unfreeze-sweep inference — runs team-best recipe (grid_4x4 ov=0 tile=448
# img=224 softmax_mean top-3 bf16) on unfreeze3/best.pt and unfreeze5/best.pt
# to produce two Kaggle submission CSVs.
#
# Output:
#   .../014_unfreeze_sweep/inference/unfreeze3/submission.csv
#   .../014_unfreeze_sweep/inference/unfreeze5/submission.csv

set -euo pipefail

REPO=/workspace/working/workspace/PlantCLEF2026
EXP=$REPO/src_experiments/010_bioclip25_end_to_end_finetune_multitask
SWEEP=$REPO/src_experiments/014_unfreeze_sweep
TEST_IMAGES=/workspace/plantclef/processed/test_images_jpeg85_max800
PY=/workspace/pytorch_env/bin/python
LOGDIR=$SWEEP/logs
mkdir -p "$LOGDIR"

cd "$EXP"

infer_one () {
  local n="$1"
  local ckpt="$EXP/outputs/last_blocks_unfreeze${n}/checkpoints/best.pt"
  local outdir="$SWEEP/inference/unfreeze${n}"
  local log="$LOGDIR/infer_unfreeze${n}.log"

  if [ ! -f "$ckpt" ]; then
    echo "MISSING: $ckpt" >&2; exit 1
  fi

  echo "=========================================="
  echo "  infer unfreeze_n=$n  →  $outdir"
  echo "=========================================="

  "$PY" infer_tiles.py \
      --checkpoint "$ckpt" \
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
      --output-dir "$outdir" 2>&1 | tee "$log"
}

infer_one 3
infer_one 5
echo "Inference done. Submissions under $SWEEP/inference/unfreeze{3,5}/"
