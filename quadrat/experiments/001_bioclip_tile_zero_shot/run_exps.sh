#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="/root/workspace/PlantCLEF2026/src_experiments/bioclip_tile_zero_shot"
PY_SCRIPT="$SCRIPT_DIR/run_all_images_submission.py"
OUT_DIR="$SCRIPT_DIR/outputs/multi_settings"

mkdir -p "$OUT_DIR"

IMAGES_DIR="/workspace/plantclef/kaggle_uploads/test/images"
MAPPING="/workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt"
BATCH_SIZE=64

TILE_SIZES=(224 336 448)
STRIDES=(224 112 56 168)
TOPKS=(5 10 20)

for TILE in "${TILE_SIZES[@]}"; do
  for STRIDE in "${STRIDES[@]}"; do
    if (( STRIDE > TILE )); then
      continue
    fi

    for TOPK in "${TOPKS[@]}"; do
      NAME="tile${TILE}_stride${STRIDE}_topk${TOPK}"
      OUTPUT_CSV="$OUT_DIR/submission_${NAME}.csv"
      LOG_FILE="$OUT_DIR/${NAME}.log"

      echo "=================================================="
      echo "Running: tile_size=$TILE stride=$STRIDE top_k=$TOPK"
      echo "=================================================="

      python "$PY_SCRIPT" \
        --images_dir "$IMAGES_DIR" \
        --mapping "$MAPPING" \
        --batch_size "$BATCH_SIZE" \
        --tile_size "$TILE" \
        --stride "$STRIDE" \
        --top_k "$TOPK" \
        --output "$OUTPUT_CSV" \
        2>&1 | tee "$LOG_FILE"
    done
  done
done

echo "All runs finished."
