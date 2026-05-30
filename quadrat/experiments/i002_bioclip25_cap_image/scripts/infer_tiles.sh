#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/PlantCLEF2026/src_experiments/010_bioclip25_end_to_end_finetune_multitask

IMAGE_DIR="/workspace/plantclef/raw/test"

HEAD_ONLY_CKPT="outputs/head_only/checkpoints/best.pt"
LAST_BLOCKS_CKPT="outputs/last_blocks_8/checkpoints/best.pt"
FULL_FINETUNE_CKPT="outputs/full_finetune/checkpoints/best.pt"

# Sweep design:
# - whole: run once
# - five_crop: run once with tile_size=448
# - grid_2x2/grid_3x3/grid_4x4/multiscale: sweep overlap only
# - sliding: sweep tile_size + overlap
# FIXED_MODES=()
# OVERLAP_MODES=(grid_2x2 grid_3x3 grid_4x4 multiscale)
# TILE_SIZES=(224 448 672)
# OVERLAPS=(0.0 0.25 0.5 0.75)
FIXED_MODES=()
OVERLAP_MODES=(grid_4x4 grid_5x5 grid_6x6)
TILE_SIZES=(672 448 224)
OVERLAPS=(0.0 0.25)

GPUS=(0 1)
MAX_JOBS=${#GPUS[@]}
JOBS=0
NEXT_GPU_IDX=0

mkdir -p logs/tile_sweeps

run_one() {
  local gpu="$1"
  local name="$2"
  local ckpt="$3"
  local mode="$4"
  local tile_size="$5"
  local overlap="$6"

  local tag="${mode}_ts${tile_size}_ov${overlap//./p}"

  echo "Starting: GPU=${gpu} | ${name} | ${tag}"

  CUDA_VISIBLE_DEVICES="$gpu" python infer_tiles.py \
    --checkpoint "$ckpt" \
    --image-dir "$IMAGE_DIR" \
    --tile-mode "$mode" \
    --tile-size "$tile_size" \
    --overlap "$overlap" \
    --agg-modes max softmax_mean \
    --top-ks 2 3 4 \
    --precision fp32 \
    --batch-size 128 \
    --device cuda \
    --save-logits \
    --num-workers 16 \
    --output-dir "outputs/tile_sweeps/${name}/${tag}" \
    > "logs/tile_sweeps/${name}_${tag}_gpu${gpu}.log" 2>&1

  echo "Finished: GPU=${gpu} | ${name} | ${tag}"
}

launch_job() {
  local name="$1"
  local ckpt="$2"
  local mode="$3"
  local tile_size="$4"
  local overlap="$5"

  local gpu="${GPUS[$NEXT_GPU_IDX]}"
  NEXT_GPU_IDX=$(( (NEXT_GPU_IDX + 1) % MAX_JOBS ))

  run_one "$gpu" "$name" "$ckpt" "$mode" "$tile_size" "$overlap" &

  JOBS=$((JOBS + 1))

  if [ "$JOBS" -ge "$MAX_JOBS" ]; then
    wait -n
    JOBS=$((JOBS - 1))
  fi
}

for MODEL in last_blocks_8; do
  if [ "$MODEL" = "head_only" ]; then
    CKPT="$HEAD_ONLY_CKPT"
  elif [ "$MODEL" = "last_blocks_8" ]; then
    CKPT="$LAST_BLOCKS_CKPT"
  elif [ "$MODEL" = "full_finetune" ]; then
    CKPT="$FULL_FINETUNE_CKPT"
  else
    echo "Unknown model: $MODEL"
    exit 1
  fi

  echo "=========================================="
  echo "Running model: $MODEL"
  echo "Checkpoint: $CKPT"
  echo "=========================================="

  # whole + five_crop: run once
  for MODE in "${FIXED_MODES[@]}"; do
    launch_job "$MODEL" "$CKPT" "$MODE" 448 0.0
  done

  # grid/multiscale: sweep overlap only
  for MODE in "${OVERLAP_MODES[@]}"; do
    for OVERLAP in "${OVERLAPS[@]}"; do
      launch_job "$MODEL" "$CKPT" "$MODE" 448 "$OVERLAP"
    done
  done

  # sliding: sweep tile size + overlap
  for TILE_SIZE in "${TILE_SIZES[@]}"; do
    for OVERLAP in "${OVERLAPS[@]}"; do
      launch_job "$MODEL" "$CKPT" "sliding" "$TILE_SIZE" "$OVERLAP"
    done
  done
done

wait

echo ""
echo "All tile sweeps complete."
echo "Outputs: outputs/tile_sweeps/"
echo "Logs: logs/tile_sweeps/"