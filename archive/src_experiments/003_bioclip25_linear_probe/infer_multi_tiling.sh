#!/bin/bash
set -euo pipefail

cd /root/workspace/PlantCLEF2026/src_experiments/003_bioclip25_linear_probe

CHECKPOINT="./outputs/train/checkpoints/best.pt"
TEST_DIR="/workspace/plantclef/raw/test"
BASE_OUT="./outputs/infer_sweeps_interp_k35"
LOG_DIR="./outputs/infer_sweeps_interp_k35_logs"

mkdir -p "$BASE_OUT" "$LOG_DIR"

NAMES=()
CMDS=()

add_run() {
  local name="$1"
  shift
  NAMES+=("$name")
  CMDS+=("python infer.py --checkpoint \"$CHECKPOINT\" --test-dir \"$TEST_DIR\" --agg-mode topk_mean $* --output-dir \"$BASE_OUT/$name\"")
}

for INTERP in bilinear bicubic; do
  for K in 3 5; do
    add_run "${INTERP}_whole_k${K}" \
      --interp "$INTERP" \
      --tile-mode whole \
      --top-n "$K"

    add_run "${INTERP}_grid3x3_k${K}" \
      --interp "$INTERP" \
      --tile-mode grid_3x3 \
      --top-n "$K"

    add_run "${INTERP}_grid5x5_k${K}" \
      --interp "$INTERP" \
      --tile-mode grid_5x5 \
      --top-n "$K"

    add_run "${INTERP}_fivecrop_k${K}" \
      --interp "$INTERP" \
      --tile-mode five_crop \
      --tile-size 224 \
      --top-n "$K"

    add_run "${INTERP}_sliding224s112_k${K}" \
      --interp "$INTERP" \
      --tile-mode sliding \
      --tile-size 224 \
      --stride 112 \
      --top-n "$K"

    add_run "${INTERP}_multiscale_k${K}" \
      --interp "$INTERP" \
      --tile-mode multiscale \
      --top-n "$K"

    add_run "${INTERP}_multiscale_dense_1234_k${K}" \
      --interp "$INTERP" \
      --tile-mode multiscale_dense \
      --scales 1,2,3,4 \
      --overlap 0.25 \
      --top-n "$K"
  done
done

run_worker() {
  local gpu="$1"
  local start_idx="$2"

  for ((i=start_idx; i<${#NAMES[@]}; i+=2)); do
    local name="${NAMES[$i]}"
    local cmd="${CMDS[$i]}"
    local log_file="$LOG_DIR/${name}.log"

    echo "[START][GPU $gpu] $name"
    CUDA_VISIBLE_DEVICES="$gpu" bash -lc "$cmd" 2>&1 | tee "$log_file"
    echo "[DONE ][GPU $gpu] $name"
  done
}

echo "Launching ${#NAMES[@]} experiments across GPUs 0 and 1"
run_worker 0 0 &
PID0=$!
run_worker 1 1 &
PID1=$!

wait $PID0
wait $PID1

echo "All experiments finished."