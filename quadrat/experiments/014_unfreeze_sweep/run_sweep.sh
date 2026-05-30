#!/usr/bin/env bash
# 014 unfreeze-N sweep — runs unfreeze_n=3 then n=5 sequentially under
# torchrun DDP on 2 GPUs. Anchor (n=4) already at 0.38333.
#
# Usage:
#   bash run_sweep.sh
#
# Designed to live at:
#   /workspace/working/workspace/PlantCLEF2026/src_experiments/014_unfreeze_sweep/run_sweep.sh
# but the actual training entrypoint is 010's train.py, so we cd there.

set -euo pipefail

REPO=/workspace/working/workspace/PlantCLEF2026
EXP=$REPO/src_experiments/010_bioclip25_end_to_end_finetune_multitask
LOGDIR=$REPO/src_experiments/014_unfreeze_sweep/logs
mkdir -p "$LOGDIR"

PY=/workspace/pytorch_env/bin/python
HEAD_ONLY_CKPT=$EXP/outputs/head_only/checkpoints/best.pt

if [ ! -f "$HEAD_ONLY_CKPT" ]; then
  echo "MISSING: $HEAD_ONLY_CKPT" >&2; exit 1
fi

cd "$EXP"

run_one () {
  local n="$1"
  local outdir="./outputs/last_blocks_unfreeze${n}"
  local log="$LOGDIR/unfreeze${n}.log"
  echo "=========================================="
  echo "  unfreeze_n=$n  →  $outdir"
  echo "  log: $log"
  echo "=========================================="

  /workspace/pytorch_env/bin/torchrun \
    --standalone --nproc_per_node=2 \
    train.py \
      --epochs 5 \
      --batch-size 64 \
      --grad-accum-steps 4 \
      --precision bf16 \
      --use-taxonomy-heads \
      --unfreeze-last-n-blocks "$n" \
      --head-lr 1e-4 \
      --backbone-lr 1e-6 \
      --weight-decay 1e-4 \
      --label-smoothing 0.1 \
      --warmup-epochs 1 \
      --num-workers 8 \
      --log-every 50 \
      --val-every 1 \
      --train-meta-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/PlantCLEF2024_single_plant_training_metadata.csv \
      --taxonomy-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
      --train-image-root /workspace/plantclef/raw/train/images_max_side_800 \
      --resume "$HEAD_ONLY_CKPT" \
      --resume-weights-only \
      --output-dir "$outdir" 2>&1 | tee "$log"
}

run_one 3
run_one 5
echo ""
echo "Sweep done. best.pt under outputs/last_blocks_unfreeze{3,5}/checkpoints/"
