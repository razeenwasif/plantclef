#!/bin/bash
set -euo pipefail

cd /home/coder137wastaken/plantclef_bucket/repo/PlantCLEF2026/src_experiments_tpu/002_bioclip_tile_zero_shot_v2

MODEL_NAME="hf-hub:imageomics/bioclip-2.5-vith14"
SPECIES_CSV="/home/coder137wastaken/plantclef_bucket/repo/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv"
IMAGES_ROOT="/home/coder137wastaken/PlantCLEF2025_test_images/PlantCLEF2025_test_images"
BASE_OUTPUT_DIR="./outputs_prompt_tile_sweep"

TPU_BATCH_SIZE=256
ACCUMULATE_TILES=1024
NUM_WORKERS=8
PREFETCH_FACTOR=2

for PROMPT_MODE in scientific scientific_common scientific_family all; do
  for MAX_COMMON_NAMES in 1 3 5; do
    for MAX_SYNONYMS in 0 2 4; do
      for TOP_K in 3 5 10; do
        for TILE_SIZE in 160 192 224 256 320 384 448 512; do
          for OVERLAP_RATIO in 0.25 0.375 0.5 0.625 0.75; do

            TILE_OVERLAP=$(python3 - <<PY
tile_size = ${TILE_SIZE}
ratio = ${OVERLAP_RATIO}
overlap = int(round(tile_size * ratio))
if overlap >= tile_size:
    overlap = tile_size - 1
print(overlap)
PY
)

            RUN_NAME="pm_${PROMPT_MODE}_mc${MAX_COMMON_NAMES}_ms${MAX_SYNONYMS}_k${TOP_K}_ts${TILE_SIZE}_to${TILE_OVERLAP}"
            OUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"
            SUBMISSION_PATH="${OUT_DIR}/bioclip-2-5-vith14_${PROMPT_MODE}/submission.csv"

            if [ -f "${SUBMISSION_PATH}" ]; then
              echo "============================================================"
              echo "Skipping: ${RUN_NAME}"
              echo "Found existing submission: ${SUBMISSION_PATH}"
              echo "============================================================"
              echo
              continue
            fi

            echo "============================================================"
            echo "Starting: ${RUN_NAME}"
            echo "============================================================"

            python run_inference.py \
              --model-name "${MODEL_NAME}" \
              --species-csv "${SPECIES_CSV}" \
              --images-root "${IMAGES_ROOT}" \
              --prompt-mode "${PROMPT_MODE}" \
              --max-common-names "${MAX_COMMON_NAMES}" \
              --max-synonyms "${MAX_SYNONYMS}" \
              --top-k "${TOP_K}" \
              --tile-size "${TILE_SIZE}" \
              --tile-overlap "${TILE_OVERLAP}" \
              --device tpu \
              --output-dir "${OUT_DIR}" \
              --tpu-batch-size "${TPU_BATCH_SIZE}" \
              --num-workers "${NUM_WORKERS}" \
              --prefetch-factor "${PREFETCH_FACTOR}" \
              --log-interval 1 \
              --pad-final-batch \
              --accumulate-tiles "${ACCUMULATE_TILES}"

            echo "Finished: ${RUN_NAME}"
            echo
          done
        done
      done
    done
  done
done

echo "All runs completed."