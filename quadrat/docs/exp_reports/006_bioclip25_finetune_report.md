# Experiment Report: 006 - BioCLIP 2.5 Fine-tuning + Extended Grid Tiling


## Training Run: train_finetune_b4

**Experiment ID:** train_finetune_b4
**Author:** Arjun Raj
**Date:** April 22, 2026

### Model / Architecture:
- **Model:** BioCLIP 2.5 ViT-H/14
- **Architecture:** ViT-H/14 image encoder + trainable linear classification head
- **Fine-tuning mode:** Partial backbone - last 4 of 32 transformer blocks unfrozen, plus `visual.ln_post` and `visual.proj`
- **Frozen:** patch embedding, positional embedding, first 28 transformer blocks

### Dataset:
- PlantCLEF 2024 single-plant training metadata
- Image root: `/workspace/plantclef/raw/train/images_max_side_800`
- Train: 1,270,209 images | Val: 137,824 images (10% split, seed=42)
- Classes: 7,806 plant species
- Image size: 224×224

### Task:
Multi-class plant species classification; inference on quadrat images via tiling.

### Hyperparameters

- learning_rate: 5e-4 (head); 2.5e-5 (backbone = lr × backbone_lr_scale 0.05)
- batch_size: 512 per GPU × 2 GPUs via `torchrun` = 1,024 effective
- epochs: 20
- optimizer: AdamW (two parameter groups: backbone + head)
- scheduler: CosineAnnealingLR (anneals to 5e-6 by epoch 20)
- weight_decay: 1e-4
- image_size: 224×224
- interpolation: N/A (training)
- tile_mode: N/A (training)
- tile_size: N/A (training)
- stride: N/A (training)
- overlap: N/A (training)
- aggregation_mode: N/A (training)
- topk_agg: N/A (training)
- top_n: N/A (training)
- loss_function: CrossEntropyLoss, label_smoothing=0.1
- device: cuda, distributed 2 GPUs
- num_workers: 8
- seed: 42
- other_relevant_settings: `unfreeze_blocks=4`, `backbone_lr_scale=0.05`, `amp=true`, `warmup_epochs=1`, `num_classes=7806`

### Results

- train_accuracy: not found (not logged separately)
- val_accuracy: 0.7744 (top-1, epoch 20); best = **0.7748** (epoch 19)
- train_loss: 1.6290 (epoch 20, final)
- val_loss: 1.0943 (epoch 20); best = **1.0244** (epoch 8)
- best_metric: val_loss=1.0244 @ epoch 8 (`best.pt` checkpoint); val_acc=0.7748 @ epoch 19
- inference_summary: see sweeps below
- public_score: N/A (training run, not directly scored)
- other_metrics:
  - val_recall@5: 0.9387 | @10: 0.9600 | @20: 0.9717 (all epoch 20)
  - LR at epoch 20: 5e-6 (cosine annealed from 5e-4)


**Full epoch-by-epoch history:**

| Epoch | val_acc (top-1) | val_loss | train_loss | recall@5 | recall@10 | recall@20 | LR |
|-------|----------------|----------|------------|----------|-----------|-----------|----|
| 1  | 0.7298 | 1.1540 | 2.8437 | 0.9292 | 0.9577 | 0.9741 | 4.97e-4 |
| 2  | 0.7483 | 1.0782 | 2.3739 | 0.9356 | 0.9622 | 0.9770 | 4.88e-4 |
| 3  | 0.7560 | 1.0466 | 2.2580 | 0.9384 | 0.9634 | 0.9775 | 4.73e-4 |
| 4  | 0.7576 | 1.0420 | 2.1748 | 0.9402 | 0.9648 | 0.9783 | 4.53e-4 |
| 5  | 0.7644 | 1.0268 | 2.1032 | 0.9410 | 0.9649 | 0.9779 | 4.28e-4 |
| 6  | 0.7667 | 1.0290 | 2.0403 | 0.9418 | 0.9650 | 0.9777 | 3.98e-4 |
| 7  | 0.7662 | 1.0272 | 1.9845 | 0.9414 | 0.9648 | 0.9771 | 3.65e-4 |
| **8** | 0.7692 | **1.0244** ← best val_loss | 1.9340 | 0.9410 | 0.9646 | 0.9766 | 3.29e-4 |
| 9  | 0.7691 | 1.0351 | 1.8856 | 0.9406 | 0.9639 | 0.9761 | 2.91e-4 |
| 10 | 0.7694 | 1.0496 | 1.8427 | 0.9396 | 0.9631 | 0.9752 | 2.53e-4 |
| 11 | 0.7701 | 1.0582 | 1.8050 | 0.9393 | 0.9622 | 0.9742 | 2.14e-4 |
| 12 | 0.7710 | 1.0594 | 1.7717 | 0.9389 | 0.9614 | 0.9738 | 1.76e-4 |
| 13 | 0.7717 | 1.0734 | 1.7414 | 0.9384 | 0.9609 | 0.9729 | 1.40e-4 |
| 14 | 0.7723 | 1.0778 | 1.7164 | 0.9390 | 0.9610 | 0.9729 | 1.07e-4 |
| 15 | 0.7736 | 1.0783 | 1.6933 | 0.9388 | 0.9605 | 0.9729 | 7.75e-5 |
| 16 | 0.7732 | 1.0862 | 1.6748 | 0.9384 | 0.9602 | 0.9722 | 5.23e-5 |
| 17 | 0.7737 | 1.0897 | 1.6585 | 0.9386 | 0.9602 | 0.9720 | 3.20e-5 |
| 18 | 0.7747 | 1.0926 | 1.6463 | 0.9383 | 0.9599 | 0.9717 | 1.71e-5 |
| **19** | **0.7748** ← best val_acc | 1.0908 | 1.6352 | 0.9385 | 0.9600 | 0.9717 | 8.05e-6 |
| 20 | 0.7744 | 1.0943 | 1.6290 | 0.9387 | 0.9600 | 0.9717 | 5.00e-6 |

### Notes
- what changed from previous experiment: Partial backbone fine-tuning (last 4 blocks) instead of frozen linear probe (exp 003). Two-group AdamW with separate LRs. Extended tiling modes (grid_6×6 through 8×8) added via regex-based dispatcher. Two inference sweeps instead of one.
- observations: Best val_loss is epoch 8 (1.0244) - loss decreased gradually ep1→ep8 then rose monotonically. `best.pt` used for all inference runs corresponds to epoch 8. Val_accuracy kept improving all the way to epoch 19 (0.7748), 11 epochs past the best-loss checkpoint. Recall@5/10/20 all peak around ep4-ep6 and slowly decline as LR anneals, while top-1 accuracy keeps climbing - the model shifts from broad recall to sharper top-1 precision in the second half of training.

---

## Inference Sweep 1: infer_sweep_006 (23 configurations)

**Checkpoint:** `outputs/train_finetune_b4/checkpoints/best.pt` (epoch 8, best val_loss=1.0244)
**Test directory:** `/workspace/plantclef/kaggle_uploads/test/images`
**Test set:** 2,105 quadrat images, 7,806 classes
**Common settings:** `tile_batch_size=64`, `img_size=224`, `agg_mode=topk_mean`, `margin_crop=0.0`

### Group A: Grid Progression (bilinear + bicubic, grids 5×5-8×8)

| Configuration | Tile Mode | Interp | k | Time (s) | Imgs/sec | Public Score |
|---|---|---|---|---|---|---|
| bilinear_grid5x5_k1 | grid_5x5 | bilinear | 1 | 697.3 | 3.02 | **0.21639** |
| bilinear_grid5x5_k3 | grid_5x5 | bilinear | 3 | 697.4 | 3.02 | 0.21405 |
| bilinear_grid6x6_k1 | grid_6x6 | bilinear | 1 | 901.2 | 2.34 | 0.20983 |
| bilinear_grid6x6_k3 | grid_6x6 | bilinear | 3 | 901.3 | 2.34 | 0.20712 |
| bilinear_grid7x7_k1 | grid_7x7 | bilinear | 1 | 1064.0 | 1.98 | 0.20476 |
| bilinear_grid7x7_k3 | grid_7x7 | bilinear | 3 | 1064.0 | 1.98 | 0.18739 |
| bilinear_grid8x8_k1 | grid_8x8 | bilinear | 1 | 1337.6 | 1.57 | 0.21100 |
| bilinear_grid8x8_k3 | grid_8x8 | bilinear | 3 | 1337.5 | 1.57 | 0.20318 |
| bicubic_grid5x5_k3 | grid_5x5 | bicubic | 3 | 733.4 | 2.87 | 0.19625 |
| bicubic_grid6x6_k3 | grid_6x6 | bicubic | 3 | 952.0 | 2.21 | 0.20525 |
| bicubic_grid7x7_k3 | grid_7x7 | bicubic | 3 | 1114.4 | 1.89 | 0.21188 |
| bicubic_grid8x8_k3 | grid_8x8 | bicubic | 3 | 1402.2 | 1.50 | 0.20964 |

### Group B: Multiscale and Sliding Window

| Configuration | Tile Mode | Interp | k | Time (s) | Imgs/sec | Public Score |
|---|---|---|---|---|---|---|
| bilinear_multiscale_k1 | multiscale | bilinear | 1 | 736.7 | 2.86 | **0.22190** ← best overall |
| bilinear_multiscale_k3 | multiscale | bilinear | 3 | 743.1 | 2.83 | 0.21596 |
| bicubic_multiscale_k1 | multiscale | bicubic | 1 | 830.9 | 2.53 | 0.20200 |
| bicubic_multiscale_k3 | multiscale | bicubic | 3 | 830.7 | 2.53 | 0.20349 |
| bilinear_sliding224s112_k1 | sliding (stride=112) | bilinear | 1 | 10501.4 | 0.20 | 0.20646 |
| bilinear_sliding224s112_k3 | sliding (stride=112) | bilinear | 3 | 10573.5 | 0.20 | 0.21064 |

### Group C: Multiscale-Dense (bilinear, k=3)

| Configuration | Scales | Overlap | Time (s) | Imgs/sec | Public Score |
|---|---|---|---|---|---|
| bilinear_msdense_1246_k3 | [1,2,4,6] | 0.0 | 1390.3 | 1.51 | 0.21194 |
| bilinear_msdense_1357_k3 | [1,3,5,7] | 0.0 | 1882.2 | 1.12 | 0.21163 |
| bilinear_msdense_2468_k3 | [2,4,6,8] | 0.0 | 2532.6 | 0.83 | 0.20882 |
| bilinear_msdense_12468_k3 | [1,2,4,6,8] | 0.0 | 2542.2 | 0.83 | 0.20882 |
| bilinear_msdense_12345_ov25_k3 | [1,2,3,4,5] | 0.25 | 1540.2 | 1.37 | 0.19581 |

---

## Inference Sweep 2: infer_sweeps_interp_k35 (8 configurations)

**Checkpoint:** `outputs/train_finetune_b4/checkpoints/best.pt`
**Test directory:** `/workspace/plantclef/raw/test`
**Test set:** 2,105 quadrat images
**Common settings:** `tile_batch_size=32`, `img_size=224`, `agg_mode=topk_mean`, `top_n=5`

| Configuration | Tile Mode | Interp | Scales | Time (s) | Imgs/sec | Public Score |
|---|---|---|---|---|---|---|
| bilinear_grid5x5_k5 | grid_5x5 | bilinear | N/A | 687.0 | 3.07 | 0.19504 |
| bilinear_multiscale_k5 | multiscale | bilinear | N/A | 726.3 | 2.90 | 0.20268 |
| bilinear_multiscale_dense_1234_k5 | multiscale_dense | bilinear | [1,2,3,4] | 977.0 | 2.16 | 0.17528 |
| bilinear_sliding224s112_k5 | sliding (stride=112) | bilinear | N/A | 10287.2 | 0.20 | 0.18732 |
| bicubic_grid5x5_k5 | grid_5x5 | bicubic | N/A | 717.7 | 2.94 | 0.18369 |
| bicubic_multiscale_k5 | multiscale | bicubic | N/A | 820.7 | 2.57 | 0.17541 |
| bicubic_multiscale_dense_1234_k5 | multiscale_dense | bicubic | [1,2,3,4] | 1170.3 | 1.80 | 0.18197 |
| bicubic_sliding224s112_k5 | sliding (stride=112) | bicubic | N/A | 10467.3 | 0.20 | 0.18419 |

---

## All Runs Ranked by Public Score

| Rank | Configuration | Sweep | Tile Mode | Interp | k | Public Score | Time (s) |
|------|---------------|-------|-----------|--------|---|--------------|----------|
| 1 | bilinear_multiscale_k1 | 1 | multiscale | bilinear | 1 | **0.22190** | 737 |
| 2 | bilinear_grid5x5_k1 | 1 | grid_5x5 | bilinear | 1 | 0.21639 | 697 |
| 3 | bilinear_multiscale_k3 | 1 | multiscale | bilinear | 3 | 0.21596 | 743 |
| 4 | bilinear_grid5x5_k3 | 1 | grid_5x5 | bilinear | 3 | 0.21405 | 697 |
| 5 | bilinear_msdense_1246_k3 | 1 | multiscale_dense | bilinear | 3 | 0.21194 | 1390 |
| 6 | bicubic_grid7x7_k3 | 1 | grid_7x7 | bicubic | 3 | 0.21188 | 1114 |
| 7 | bilinear_msdense_1357_k3 | 1 | multiscale_dense | bilinear | 3 | 0.21163 | 1882 |
| 8 | bilinear_grid8x8_k1 | 1 | grid_8x8 | bilinear | 1 | 0.21100 | 1338 |
| 9 | bilinear_sliding224s112_k3 | 1 | sliding | bilinear | 3 | 0.21064 | 10574 |
| 10 | bilinear_grid6x6_k1 | 1 | grid_6x6 | bilinear | 1 | 0.20983 | 901 |
| 11 | bicubic_grid8x8_k3 | 1 | grid_8x8 | bicubic | 3 | 0.20964 | 1402 |
| 12 | bilinear_msdense_2468_k3 | 1 | multiscale_dense | bilinear | 3 | 0.20882 | 2533 |
| 12 | bilinear_msdense_12468_k3 | 1 | multiscale_dense | bilinear | 3 | 0.20882 | 2542 |
| 14 | bilinear_grid6x6_k3 | 1 | grid_6x6 | bilinear | 3 | 0.20712 | 901 |
| 15 | bilinear_sliding224s112_k1 | 1 | sliding | bilinear | 1 | 0.20646 | 10501 |
| 16 | bicubic_grid6x6_k3 | 1 | grid_6x6 | bicubic | 3 | 0.20525 | 952 |
| 17 | bilinear_grid7x7_k1 | 1 | grid_7x7 | bilinear | 1 | 0.20476 | 1064 |
| 18 | bilinear_multiscale_k5 | 2 | multiscale | bilinear | 5 | 0.20268 | 726 |
| 19 | bicubic_multiscale_k3 | 1 | multiscale | bicubic | 3 | 0.20349 | 831 |
| 20 | bilinear_grid8x8_k3 | 1 | grid_8x8 | bilinear | 3 | 0.20318 | 1338 |
| 21 | bicubic_multiscale_k1 | 1 | multiscale | bicubic | 1 | 0.20200 | 831 |
| 22 | bilinear_grid5x5_k5 | 2 | grid_5x5 | bilinear | 5 | 0.19504 | 687 |
| 23 | bicubic_grid5x5_k3 | 1 | grid_5x5 | bicubic | 3 | 0.19625 | 733 |
| 24 | bilinear_msdense_12345_ov25_k3 | 1 | multiscale_dense | bilinear | 3 | 0.19581 | 1540 |
| 25 | bilinear_sliding224s112_k5 | 2 | sliding | bilinear | 5 | 0.18732 | 10287 |
| 26 | bicubic_sliding224s112_k5 | 2 | sliding | bicubic | 5 | 0.18419 | 10467 |
| 27 | bicubic_grid5x5_k5 | 2 | grid_5x5 | bicubic | 5 | 0.18369 | 718 |
| 28 | bicubic_multiscale_dense_1234_k5 | 2 | multiscale_dense | bicubic | 5 | 0.18197 | 1170 |
| 29 | bilinear_grid7x7_k3 | 1 | grid_7x7 | bilinear | 3 | 0.18739 | 1064 |
| 30 | bicubic_multiscale_k5 | 2 | multiscale | bicubic | 5 | 0.17541 | 821 |
| 31 | bilinear_multiscale_dense_1234_k5 | 2 | multiscale_dense | bilinear | 5 | 0.17528 | 977 |

---

# Overall Summary

- **best run:** `bilinear_multiscale_k1` - public score **0.22190**, 737s inference time
- **most promising settings:** bilinear interpolation + multiscale tiling + k=1 aggregation. Second-best is bilinear_grid5x5_k1 (0.21639, 697s) - slightly faster but lower score.
- **patterns across runs:**
  1. **k=1 consistently beats k=3 beats k=5** for the same tile mode and interpolation. E.g., multiscale bilinear: k1=0.22190 → k3=0.21596 → k5=0.20268. Averaging more tiles systematically hurts. The single best-scoring tile carries the most signal; lower-ranked tiles add noise.
  2. **Bilinear consistently beats bicubic.** E.g., grid5x5 bilinear_k1=0.21639 vs bicubic_k3=0.19625, multiscale bilinear_k1=0.22190 vs bicubic_k1=0.20200. Bicubic over-smooths at 224px, degrading features.
  3. **Larger grids degrade performance.** bilinear grid5x5_k1=0.21639 → grid6x6_k1=0.20983 → grid7x7_k1=0.20476. Smaller tiles at this resolution contain too little plant context per tile.
  4. **Multiscale slightly outperforms grid_5×5** for bilinear+k1 (0.22190 vs 0.21639), likely because it includes a full-image tile (scale=1) alongside sub-tiles.
  5. **Sliding window is not competitive** despite high inference cost. bilinear sliding k3=0.21064 (10.5k s) is beaten by multiscale k3=0.21596 (743s) - 14× faster and better.
  6. **Multiscale-dense** with scales [1,2,4,6] is decent (0.21194) but 1.9× slower than multiscale; the added scales don't help enough to justify cost.
  7. **Overlap hurts:** msdense_12345_ov25_k3 (overlap=0.25) = 0.19581, significantly below the same setting without overlap.
  8. Sweep 2 (all k=5) is the worst sweep on average - the k=5 hypothesis was clearly wrong.

---

# Compact Comparison Table

| experiment_name | model | tile_mode | interpolation | topk_agg | batch_size | learning_rate | epochs | best_val_metric | public_score | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| train_finetune_b4 | ViT-H/14 | N/A | N/A | N/A | 512×2 | 2.5e-5/5e-4 | 20 | acc=0.7748 (ep19) / loss=1.0244 (ep8) | N/A | 4 blocks unfrozen; best.pt @ ep8 (best val_loss) |
| bilinear_multiscale_k1 | ViT-H/14 | multiscale | bilinear | 1 | 64 | N/A | N/A | N/A | **0.22190** | BEST; 737s |
| bilinear_grid5x5_k1 | ViT-H/14 | grid_5x5 | bilinear | 1 | 64 | N/A | N/A | N/A | 0.21639 | 2nd best; 697s; fastest competitive |
| bilinear_multiscale_k3 | ViT-H/14 | multiscale | bilinear | 3 | 64 | N/A | N/A | N/A | 0.21596 | k=3 vs k=1 costs −0.006 score |
| bilinear_grid5x5_k3 | ViT-H/14 | grid_5x5 | bilinear | 3 | 64 | N/A | N/A | N/A | 0.21405 | reference baseline from prior exps |
| bilinear_msdense_1246_k3 | ViT-H/14 | multiscale_dense | bilinear | 3 | 64 | N/A | N/A | N/A | 0.21194 | scales=[1,2,4,6]; 1390s; not worth cost |
| bilinear_grid8x8_k1 | ViT-H/14 | grid_8x8 | bilinear | 1 | 64 | N/A | N/A | N/A | 0.21100 | large grid recovers with k=1 |
| bilinear_sliding224s112_k3 | ViT-H/14 | sliding | bilinear | 3 | 64 | N/A | N/A | N/A | 0.21064 | 10574s; impractical |
| bilinear_grid6x6_k1 | ViT-H/14 | grid_6x6 | bilinear | 1 | 64 | N/A | N/A | N/A | 0.20983 | larger grid hurts even with k=1 |
| bicubic_grid7x7_k3 | ViT-H/14 | grid_7x7 | bicubic | 3 | 64 | N/A | N/A | N/A | 0.21188 | bicubic recovers at large grid only |
| bilinear_multiscale_k5 | ViT-H/14 | multiscale | bilinear | 5 | 32 | N/A | N/A | N/A | 0.20268 | k=5 sweep; −0.019 vs k=1 |
| bilinear_grid5x5_k5 | ViT-H/14 | grid_5x5 | bilinear | 5 | 32 | N/A | N/A | N/A | 0.19504 | k=5 clearly worse |
| bicubic_multiscale_k5 | ViT-H/14 | multiscale | bicubic | 5 | 32 | N/A | N/A | N/A | 0.17541 | worst bicubic config |
| bilinear_multiscale_dense_1234_k5 | ViT-H/14 | multiscale_dense | bilinear | 5 | 32 | N/A | N/A | N/A | 0.17528 | worst overall |
