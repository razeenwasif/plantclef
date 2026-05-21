#!/usr/bin/env bash
set -euo pipefail

CKPT="./outputs/train/checkpoints/best.pt"
BASE_OUT="./outputs/infer_sweep_top5"
mkdir -p "$BASE_OUT"

COMMON_ARGS=(
  --checkpoint "$CKPT"
  --top-n 5
  --tile-batch-size 256
)

run_exp () {
  local GPU="$1"
  local NAME="$2"
  shift 2

  echo "=================================================="
  echo "GPU ${GPU} | Running: ${NAME}"
  echo "=================================================="

  CUDA_VISIBLE_DEVICES="$GPU" \
  python infer.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --device cuda \
    --output-dir "${BASE_OUT}/${NAME}" \
    2>&1 | tee "${BASE_OUT}/${NAME}.log"
}

run_pair () {
  run_exp 0 "$1" "${@:2}" &
  PID1=$!

  shift
  # separator between jobs is ::
  while [[ "$1" != "::" ]]; do shift; done
  shift

  run_exp 1 "$1" "${@:2}" &
  PID2=$!

  wait $PID1
  wait $PID2
}

# pair 1
run_exp 0 "whole_max_bicubic" \
  --tile-mode whole \
  --agg-mode max \
  --interp bicubic &
PID1=$!

run_exp 1 "grid2x2_max_bicubic" \
  --tile-mode grid_2x2 \
  --agg-mode max \
  --interp bicubic &
PID2=$!

wait $PID1
wait $PID2

# pair 2
run_exp 0 "grid4x4_max_bicubic" \
  --tile-mode grid_4x4 \
  --agg-mode max \
  --interp bicubic &
PID1=$!

run_exp 1 "multiscale_max_bicubic" \
  --tile-mode multiscale \
  --agg-mode max \
  --interp bicubic &
PID2=$!

wait $PID1
wait $PID2

# pair 3
run_exp 0 "multiscale_topk5_lanczos" \
  --tile-mode multiscale \
  --agg-mode topk_mean \
  --topk-agg 5 \
  --interp lanczos &
PID1=$!

run_exp 1 "multiscale_topk10_lanczos" \
  --tile-mode multiscale \
  --agg-mode topk_mean \
  --topk-agg 10 \
  --interp lanczos &
PID2=$!

wait $PID1
wait $PID2

echo "All experiments finished."