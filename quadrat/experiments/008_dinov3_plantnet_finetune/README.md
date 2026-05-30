# 008 — DINOv3 PlantNet-style full fine-tune

## Why

The 007 PlantNet-DINOv2 baseline (`plantnet_collage_v1`) scored **0.17476** on the
Kaggle leaderboard — better than 005 DINOv3 (0.13) but behind 006 BioCLIP-2.5
(0.31). The gap between 007 and 005 is *not* the DINOv3 architecture (which is
strictly more expressive thanks to Gram-anchored patch tokens) — it's the
**plant prior**. PlantNet released DINOv2-B after weeks of GPU time fine-tuning
on the 1.4M PC24 single-plant corpus (76.16% PC24 top-1). DINOv3-L's own
pretraining is generic (LVD-1689M), with no plant-specific signal.

This experiment trains DINOv3-L the same way PlantNet trained DINOv2-B:

- **Phase A** — full fine-tune of DINOv3-L on the 1.38M cleaned single-plant
  PC24 images, 7806-way softmax with cross-entropy + logit adjustment. This is
  the expensive step that embeds taxonomic knowledge into the backbone itself.
- **Phase B** — swap in a CLS+GeM multi-label head, apply LoRA, fine-tune on
  the 50k labelled collages with ASL. Same recipe as 007 Phase B.

## Files

- `model.py` — `DINOv3SinglePlantClassifier` (CLS + LayerNorm + `Linear(1024, 7806)`),
  plus re-exports of 005's `DINOv3MultiLabel` for Phase B.
- `single_plant_dataset.py` — CSV-backed `(image, class_idx)` dataset.
- `train_phase_a.py` — head-only warmup → full FT loop, OneCycleLR, CE + logit
  adjustment, bf16, two LR groups.
- `train_phase_b.py` — loads the Phase-A backbone into `DINOv3MultiLabel`,
  applies LoRA, trains on the 50k labelled collages with ASL (mirrors 007/train.py).
- `dump_test_probs.py` — tiled multi-label inference → fp16 probs for max/mean/noisy-or
  (schema matches 005/007 so `ensemble_with_other.py` works unchanged).
  Supports **multi-scale** (`--tile-sizes 336 448 560`), **hflip TTA**
  (`--hflip-tta`), and **ExG vegetation filtering** (`--min-vegetation-frac 0.15`).
- `preprocess_test_quadrats.py` — Lanczos resize + JPEG recompression
  to match the PC24 training-corpus byte distribution (TheHeartOfNoise 2025
  winner recipe). Run once before `dump_test_probs.py`.
- `vegetation_filter.py` — Excess-Green-based per-tile plant-coverage filter,
  imported by `dump_test_probs.py`.
- `enhance_probs.py` — post-processes a `test_probs.npz` with post-hoc logit
  adjustment (corrects for the training-time class-frequency shift), BMA
  aggregation (weighted geometric mean of max / noisy_or / mean), and a
  dynamic-threshold search to match a target avg preds/quadrat.
- `make_submission.py` — threshold + top-K → PlantCLEF submission CSV.
- `REMOTE_CONTEXT.md` — handoff brief for monitoring Phase A on the training pod.

## Recommended inference pipeline (winner-recipe stack)

```bash
# 1. Match training JPEG/chroma distribution
python preprocess_test_quadrats.py \
  --input-dir  /workspace/plantclef/raw/test/PlantCLEF2026_test_images \
  --output-dir /workspace/plantclef/processed/test_images_jpeg85_max800

# 2. Multi-scale + hflip TTA + vegetation filter
python dump_test_probs.py \
  --checkpoint /workspace/working/PlantCLEF2026/models/dinov3_plantnet_v1_ddp/phase_b_best.pth \
  --images-root /workspace/plantclef/processed/test_images_jpeg85_max800 \
  --output       outputs/test_probs_multiscale.npz \
  --tile-sizes 336 448 560 --hflip-tta --min-vegetation-frac 0.15 \
  --batch-size 16 --bf16

# 3. Post-hoc logit adjustment + BMA aggregation + dynamic threshold
python enhance_probs.py \
  --probs-npz outputs/test_probs_multiscale.npz \
  --class-freq-csv /workspace/plantclef/processed/species_train_counts.csv \
  --logit-adjust 1.0 --agg bma --bma-weights 0.5 0.3 0.2 \
  --dynamic-threshold --target-avg-preds 5.0 \
  --output outputs/test_probs_enhanced.npz

# 4. Build submission using the suggested threshold from step 3
python make_submission.py \
  --probs-npz outputs/test_probs_enhanced.npz --probs-key probs_max \
  --threshold <tau_from_step_3> --top-k 10 --min-preds 1 \
  --output outputs/submission_v1.csv
```

## Run (Phase A)

`--batch-size` is **per GPU**. With DDP the effective batch is
`batch_size × accum × world_size`. Target an effective batch of 96–144 for
stable ViT-L full FT.

### Single 5090

```bash
cd src_experiments/008_dinov3_plantnet_finetune

python3 train_phase_a.py \
  --train-csv   /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
  --images-root /workspace/plantclef/raw/train/images_max_side_800 \
  --species-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
  --img-size 224 --batch-size 48 --accum 2 --bf16 \
  --epochs 8 --warmup-epochs 1 --num-workers 12 \
  --output-dir /workspace/working/PlantCLEF2026/models/dinov3_plantnet_v1
```

### 2× 5090 DDP (recommended)

```bash
cd src_experiments/008_dinov3_plantnet_finetune

torchrun --standalone --nproc_per_node=2 train_phase_a.py \
  --train-csv   /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
  --images-root /workspace/plantclef/raw/train/images_max_side_800 \
  --species-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
  --img-size 224 --batch-size 48 --accum 1 --bf16 \
  --epochs 8 --warmup-epochs 1 --num-workers 10 \
  --output-dir /workspace/working/PlantCLEF2026/models/dinov3_plantnet_v1
```

### 3× 5090 DDP

```bash
cd src_experiments/008_dinov3_plantnet_finetune

torchrun --standalone --nproc_per_node=3 train_phase_a.py \
  --train-csv   /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
  --images-root /workspace/plantclef/raw/train/images_max_side_800 \
  --species-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
  --img-size 224 --batch-size 48 --accum 1 --bf16 \
  --epochs 8 --warmup-epochs 1 --num-workers 8 \
  --output-dir /workspace/working/PlantCLEF2026/models/dinov3_plantnet_v1
```

### Wall-time budget

| Config | Epoch | 8 epochs + warmup |
|--------|-------|-------------------|
| 1× 5090 | ~90 min | ~13–14 h |
| 2× 5090 DDP | ~48 min | ~7 h |
| 3× 5090 DDP | ~33 min | ~5 h |

Per-GPU VRAM @ batch 48, 224 px, bf16: ~15 GB of 32 GB.

## Targets

- End of Phase A: val top-1 ≥ 55% on the 1% single-plant holdout (PlantNet
  DINOv2 hit 76% at PC24's full scale with 75 epochs — an 8-epoch run should
  land in the 55–65% range).
- End of Phase B: val F1@0.5 ≥ 0.40 on the collage split (007 hit 0.358 with
  the weaker PlantNet DINOv2 backbone; DINOv3-L should exceed it).
- Kaggle: target ≥ 0.20, stretch to ≥ 0.23 (beat team best 0.222).
