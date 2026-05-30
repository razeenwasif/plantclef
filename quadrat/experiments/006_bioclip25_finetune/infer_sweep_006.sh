#!/bin/bash
# Inference sweep for experiment 006.
#
# Covers three groups of experiments building on 003 results:
#
#   Group A - Grid progression (main focus)
#             bilinear + grid_{5..8}x{5..8} × k={1,3}
#             bicubic  + grid_{5..8}x{5..8} × k=3  (k=3 only, k=1 if promising)
#
#   Group B - k=1 ablation on known-good configs
#             bilinear/bicubic × multiscale × k={1,3}
#             bilinear × sliding224s112     × k={1,3}
#
#   Group C - New multiscale_dense combos with larger grids
#             bilinear × scales={1,2,4,6} or {1,2,4,6,8} or {2,4,6,8} × k=3
#
# 003 reference scores (for comparison):
#   bilinear_grid5x5_k3       = 0.24182  ← previous best
#   bicubic_multiscale_k3     = 0.23740
#   bilinear_multiscale_k3    = 0.23167
#   bicubic_sliding224s112_k3 = 0.22640
#   bilinear_grid3x3_k3       = 0.21167
#   bilinear_whole_k3         = 0.15386
#
# Usage:
#   bash infer_sweep_006.sh                        # uses defaults below
#   CHECKPOINT=./outputs/train_finetune_b2/checkpoints/best.pt bash infer_sweep_006.sh
#   N_GPUS=4 bash infer_sweep_006.sh               # spread across 4 GPUs
#
set -euo pipefail

cd /root/workspace/PlantCLEF2026/src_experiments/006_bioclip25_finetune

# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-/root/workspace/PlantCLEF2026/src_experiments/006_bioclip25_finetune/outputs/train_finetune_b4/checkpoints/best.pt}"
TEST_DIR="${TEST_DIR:-/workspace/plantclef/kaggle_uploads/test/images}"
BASE_OUT="${BASE_OUT:-./outputs/infer_sweep_006}"
LOG_DIR="${LOG_DIR:-./outputs/infer_sweep_006_logs}"
N_GPUS="${N_GPUS:-2}"

mkdir -p "$BASE_OUT" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------
NAMES=()
CMDS=()

add_run() {
  local name="$1"; shift
  NAMES+=("$name")
  CMDS+=("python infer.py \
    --checkpoint \"$CHECKPOINT\" \
    --test-dir \"$TEST_DIR\" \
    --agg-mode topk_mean \
    --topk-agg 5 \
    --tile-batch-size 64 \
    $* \
    --output-dir \"$BASE_OUT/$name\"")
}

# ---------------------------------------------------------------------------
# Group A - Grid progression  (bilinear × grid_{5..8} × k={1,3})
# ---------------------------------------------------------------------------
for GRID in 5 6 7 8; do
  for K in 1 3; do
    add_run "bilinear_grid${GRID}x${GRID}_k${K}" \
      --interp bilinear \
      --tile-mode "grid_${GRID}x${GRID}" \
      --top-n "$K"
  done
done

# Bicubic grid progression × k=3  (include k=1 for the winner after scoring)
for GRID in 5 6 7 8; do
  add_run "bicubic_grid${GRID}x${GRID}_k3" \
    --interp bicubic \
    --tile-mode "grid_${GRID}x${GRID}" \
    --top-n 3
done

# ---------------------------------------------------------------------------
# Group B - k=1 ablation on previously strong configs
# ---------------------------------------------------------------------------
for INTERP in bilinear bicubic; do
  for K in 1 3; do
    add_run "${INTERP}_multiscale_k${K}" \
      --interp "$INTERP" \
      --tile-mode multiscale \
      --top-n "$K"
  done
done

for K in 1 3; do
  add_run "bilinear_sliding224s112_k${K}" \
    --interp bilinear \
    --tile-mode sliding \
    --tile-size 224 \
    --stride 112 \
    --top-n "$K"
done

# ---------------------------------------------------------------------------
# Group C - New multiscale_dense combos with larger grid scales
# ---------------------------------------------------------------------------
# scales 1,2,4,6 → whole + 2x2 + 4x4 + 6x6  (29 tiles)
add_run "bilinear_msdense_1246_k3" \
  --interp bilinear \
  --tile-mode multiscale_dense \
  --scales 1,2,4,6 \
  --overlap 0.0 \
  --top-n 3

# scales 1,2,4,6,8 → whole + 2x2 + 4x4 + 6x6 + 8x8  (86 tiles)
add_run "bilinear_msdense_12468_k3" \
  --interp bilinear \
  --tile-mode multiscale_dense \
  --scales 1,2,4,6,8 \
  --overlap 0.0 \
  --top-n 3

# scales 2,4,6,8 (no whole, larger grids only) → 4+16+36+64 = 120 tiles
add_run "bilinear_msdense_2468_k3" \
  --interp bilinear \
  --tile-mode multiscale_dense \
  --scales 2,4,6,8 \
  --overlap 0.0 \
  --top-n 3

# scales 1,3,5,7 (odd sizes) → whole + 3x3 + 5x5 + 7x7
add_run "bilinear_msdense_1357_k3" \
  --interp bilinear \
  --tile-mode multiscale_dense \
  --scales 1,3,5,7 \
  --overlap 0.0 \
  --top-n 3

# scales 1,2,3,4,5 (fine-grained) with 25% overlap → more coverage
add_run "bilinear_msdense_12345_ov25_k3" \
  --interp bilinear \
  --tile-mode multiscale_dense \
  --scales 1,2,3,4,5 \
  --overlap 0.25 \
  --top-n 3

# ---------------------------------------------------------------------------
# Worker dispatch - round-robin across N_GPUS
# ---------------------------------------------------------------------------
run_worker() {
  local gpu="$1"
  local start_idx="$2"

  for ((i=start_idx; i<${#NAMES[@]}; i+=N_GPUS)); do
    local name="${NAMES[$i]}"
    local cmd="${CMDS[$i]}"
    local log_file="$LOG_DIR/${name}.log"

    echo "[START][GPU $gpu] $name"
    CUDA_VISIBLE_DEVICES="$gpu" bash -lc "$cmd" 2>&1 | tee "$log_file"
    echo "[DONE ][GPU $gpu] $name"
  done
}

echo "============================================================"
echo "Sweep: ${#NAMES[@]} runs across ${N_GPUS} GPU(s)"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $BASE_OUT"
echo "============================================================"
for i in "${!NAMES[@]}"; do
  printf "  [%02d] %s\n" "$i" "${NAMES[$i]}"
done
echo "------------------------------------------------------------"

PIDS=()
for ((gpu=0; gpu<N_GPUS; gpu++)); do
  run_worker "$gpu" "$gpu" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "============================================================"
echo "All ${#NAMES[@]} experiments finished."
echo "Outputs: $BASE_OUT"
echo "Logs:    $LOG_DIR"
echo "============================================================"
