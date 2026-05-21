# 009 — BioCLIP-2.5 PlantNet-style full fine-tune

## Why

The 008 PhaseA × 004-frozen-BioCLIP-2.5 RRF ensemble cracked **0.34642** on
the leaderboard (α=0.70 top-3, +0.016 over Arjun's solo BioCLIP best of 0.33).
The fusion gain comes from two models making *different* errors. To push
the ensemble ceiling further, the second leg has to get stronger — currently
it's a frozen prototype matcher. This experiment fine-tunes BioCLIP-2.5
ViT-H/14 on the same 1.4M PC24 single-plant corpus that 008 used, so the
ensemble becomes "two strong species classifiers" instead of "one strong +
one frozen prototype."

## Recipe

Identical to `008_dinov3_plantnet_finetune/train_phase_a.py` — only the
backbone changes:

- DINOv3 ViT-L/16 → BioCLIP-2.5 ViT-H/14 (loaded via open_clip).
- Patch 16 → patch 14, so `--img-size` must be a multiple of 14.
- ImageNet normalization → BioCLIP/CLIP normalization (mean=(0.481,0.458,0.408),
  std=(0.269,0.261,0.276)). Wrong stats here silently kill convergence.
- Backbone params 300M → 632M. Default batch is halved (24 vs 48) and accum
  is doubled (2 vs 1) to keep effective batch ≈ PhaseA's 96.

Everything else (CE+label-smooth+logit-adjust, OneCycleLR, AdamW, two LR
groups, RandomResizedCrop+ColorJitter+RandomErasing, head-only warmup epoch)
matches 008 deliberately so the ensemble is genuinely apples-to-apples.

## Files

- `bioclip_model.py` — `BioCLIP25SinglePlantClassifier` + transforms.
- `train_phase_a.py` — training script (mirrors 008's).
- Reuses `008_dinov3_plantnet_finetune/single_plant_dataset.py` and
  `004_bioclip_few_shot/dataset.py:load_species_ids` via `sys.path` shim.

## Run

### Single 5090 (default — ~30-40h for 8 epochs)

```bash
cd src_experiments/009_bioclip25_plantnet_finetune

python3 train_phase_a.py \
  --train-csv   /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
  --images-root /workspace/plantclef/raw/train/images_max_side_800 \
  --species-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
  --img-size 224 --batch-size 24 --accum 2 --bf16 --grad-checkpoint \
  --epochs 8 --warmup-epochs 1 --num-workers 10 \
  --output-dir /workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1
```

### 2× 5090 DDP (~12-15h)

```bash
torchrun --standalone --nproc_per_node=2 train_phase_a.py \
  --train-csv   /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
  --images-root /workspace/plantclef/raw/train/images_max_side_800 \
  --species-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
  --img-size 224 --batch-size 24 --accum 1 --bf16 --grad-checkpoint \
  --epochs 8 --warmup-epochs 1 --num-workers 8 \
  --output-dir /workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1
```

`--grad-checkpoint` slows each step ~25% but saves ~30% VRAM, letting batch
24 fit comfortably alongside open_clip's text tower (which we leave loaded
even though we don't use it). Drop the flag if VRAM headroom is large.

## Targets

- Val top-1 ≥ 50% on the 1% single-plant holdout (PlantNet's DINOv2-B hit
  76% with 75 epochs — 8 epochs of BioCLIP-2.5 should land 50-60%).
- Tiled multi-label inference + RRF fusion with 008 PhaseA: ≥ 0.36 on the
  Kaggle public board (current best 0.34642). Stretch: 0.38+.

## Inference (post-training)

Reuse 008's `dump_test_probs_phase_a.py` flow with the new checkpoint —
needs a small adapter to load `BioCLIP25SinglePlantClassifier` instead of
`DINOv3SinglePlantClassifier`. Will land in this dir as `dump_test_probs.py`
once Phase A finishes.

Then fuse with `008/fuse_phase_a_bioclip.py --phase-a <new_009_npz>
--bioclip <existing_004_npz>` — same fusion pipeline, just two strong
classifiers feeding it instead of one strong + one prototype matcher.
