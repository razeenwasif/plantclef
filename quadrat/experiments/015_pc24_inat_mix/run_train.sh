#!/usr/bin/env bash
# 015 — BioCLIP-2.5 multitask fine-tune on PC24 + iNat Research-grade combined.
# Same recipe as the 014 anchor (n=4 unfreeze, 5 ep, taxonomy heads), but with
# the combined manifest as training data.
#
# RESUME MODE (post-crash 2026-05-03): epoch 1 finished and last.pt saved
# before NCCL watchdog killed the group on a 10-min ALLREDUCE timeout during
# validation. Resuming from last.pt with full optimizer/scheduler state via
# the run_with_long_timeout.py wrapper, which bumps the DDP timeout to 2h
# without touching shared 010 code.

set -euo pipefail

REPO=/workspace/working/workspace/PlantCLEF2026
EXP=$REPO/src_experiments/010_bioclip25_end_to_end_finetune_multitask
EXP015=$REPO/src_experiments/015_pc24_inat_mix
LOGDIR=$EXP015/logs
mkdir -p "$LOGDIR"

PY=/workspace/pytorch_env/bin/python
WRAPPER=$EXP015/run_with_long_timeout.py
COMBINED_CSV=/workspace/plantclef/processed/pc24_inat_combined_manifest.csv
TAXONOMY_CSV=/workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv
PC24_IMAGE_ROOT=/workspace/plantclef/raw/train/images_max_side_800
OUTDIR=$EXP/outputs/pc24_inat_mix_unfreeze4
RESUME_CKPT=$OUTDIR/checkpoints/last.pt

if [ ! -f "$WRAPPER" ]; then
  echo "MISSING: $WRAPPER" >&2; exit 1
fi
if [ ! -f "$COMBINED_CSV" ]; then
  echo "MISSING: $COMBINED_CSV — run build_combined_manifest.py first" >&2; exit 1
fi
if [ ! -f "$RESUME_CKPT" ]; then
  echo "MISSING: $RESUME_CKPT — was epoch 1 ever saved?" >&2; exit 1
fi

cd "$EXP"

LOG=$LOGDIR/train_resume.log
echo "==========================================" | tee "$LOG"
echo "  015 PC24+iNat mix  unfreeze_n=4  RESUME  →  $OUTDIR" | tee -a "$LOG"
echo "  resume from: $RESUME_CKPT" | tee -a "$LOG"
echo "  wrapper:     $WRAPPER  (DDP timeout = 2h)" | tee -a "$LOG"
echo "  log:         $LOG" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

/workspace/pytorch_env/bin/torchrun \
  --standalone --nproc_per_node=2 \
  "$WRAPPER" "$EXP/train.py" \
    --epochs 5 \
    --batch-size 64 \
    --grad-accum-steps 4 \
    --precision bf16 \
    --use-taxonomy-heads \
    --unfreeze-last-n-blocks 4 \
    --head-lr 1e-4 \
    --backbone-lr 1e-6 \
    --weight-decay 1e-4 \
    --label-smoothing 0.1 \
    --warmup-epochs 1 \
    --num-workers 8 \
    --log-every 50 \
    --val-every 1 \
    --train-meta-csv "$COMBINED_CSV" \
    --taxonomy-csv "$TAXONOMY_CSV" \
    --train-image-root "$PC24_IMAGE_ROOT" \
    --resume "$RESUME_CKPT" \
    --output-dir "$OUTDIR" 2>&1 | tee -a "$LOG"
