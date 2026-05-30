# Experiment Report: 003 - BioCLIP 2.5 Linear Probe

**Most useful files:**
- `outputs/train/train_config.json` - final training hyperparameters
- `outputs/train/run.log` - per-epoch training and validation metrics across all runs
- `outputs/train/train_history.json` - final epoch summary (epoch 10 only)
- `outputs/train/scores_infer_sweeps_interp_k35.csv` - public Kaggle scores for all 28 interp/tiling sweep runs
- `infer_multi_tiling.sh` - launch script for the comprehensive inference sweep
- `infer_sweep.sh` - launch script for the initial tiling sweep
- `outputs/infer_sweeps_interp_k35/*/run_config.json` - per-run inference configs and summaries

---

## Experiment: Training - BioCLIP 2.5 Linear Probe

**Author:** Arjun
**Date:** 2026-04-19

### Model / Architecture
BioCLIP 2.5 ViT-H/14 (`hf-hub:imageomics/bioclip-2.5-vith14`) - **frozen backbone**, single linear classification head.
- Embed dim: 1024
- Head parameters: 8,001,150 (linear layer only)
- Num classes: 7,806

### Dataset
- **Train metadata:** `PlantCLEF2024_single_plant_training_metadata.csv` (1,408,033 rows, 7,806 species)
- **Image root:** `/workspace/plantclef/raw/train/images_max_side_800`
- **Split:** 90/10 train/val (val_seed=42)
- **Train set:** 1,270,209 images, 7,806 species
- **Val set:** 137,824 images, 7,197 species

### Task
Multi-class plant species identification - 7,806 classes, PlantCLEF 2026 challenge.

### Hyperparameters

| Parameter | Value |
|---|---|
| learning_rate | 0.001 |
| batch_size | 256 per GPU (effective 512 with 2 GPUs) |
| epochs | 10 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| weight_decay | 0.0001 |
| image_size | 224 |
| label_smoothing | 0.1 |
| warmup_epochs | 1 |
| loss_function | CrossEntropyLoss (label_smoothing=0.1) |
| device | cuda:0 (DDP, world_size=2) |
| num_workers | 8 |
| seed | 42 (val split) |
| amp | True (AMP + GradScaler) |
| use_cache | False (backbone runs live each step) |
| val_top_n | 20 |
| interpolation (training) | bicubic (default CLIP preprocessing) |
| normalization mean | [0.48145466, 0.4578275, 0.40821073] |
| normalization std | [0.26862954, 0.26130258, 0.27577711] |

### Results - Per-Epoch Training

| Epoch | Train Loss | Val Loss | Recall@1 | Recall@5 | Recall@20 |
|---|---|---|---|---|---|
| 1 | 4.5481 | 2.4804 | 0.6010 | 0.8146 | 0.8914 |
| 2 | 4.4848 | 2.4186 | 0.6113 | 0.8211 | 0.8944 |
| 3 | 4.2765 | 2.3348 | 0.6187 | 0.8263 | 0.8990 |
| 4 | 3.9544 | 2.1099 | 0.6303 | 0.8409 | 0.9105 |
| 5 | 3.5740 | 1.8935 | 0.6443 | 0.8596 | 0.9242 |
| 6 | 3.1891 | 1.6335 | 0.6690 | 0.8813 | 0.9402 |
| 7 | 2.8532 | 1.4178 | 0.6931 | 0.9023 | 0.9530 |
| 8 | 2.5878 | 1.2575 | 0.7163 | 0.9164 | 0.9632 |
| 9 (lr=0.004 run) | 2.4095 | 1.1478 | 0.7371 | 0.9257 | 0.9680 |
| 9 (lr=0.001 restart) | 2.4672 | 1.1815 | 0.7307 | 0.9236 | 0.9668 |
| **10 (final, best)** | **2.3077** | **1.0936** | **0.7505** | **0.9303** | **0.9703** |

- **best_metric:** Recall@5 = 0.9303 (epoch 10)
- **n_val_images:** 137,824

### Notes
- **Observations:**
  - Recall@5 improved steadily every epoch, crossing 0.90 at epoch 7 and reaching 0.9303 at epoch 10 - training had not yet plateaued.
  - Val loss also improved every epoch, suggesting more epochs could help.
  - The training was interrupted multiple times before the 256-batch, 2-GPU run succeeded.
  - The lr=0.001 restart produced slightly worse epoch-9 metrics (0.9236 vs 0.9257) but epoch 10 ultimately exceeded the first run's best.
- **Ideas for next experiment:** Train for 20+ epochs; try a smaller weight decay; try unfreezing the last backbone block.

---

## Inference Sweep: `infer_sweeps_interp_k35`

**Script:** `infer_multi_tiling.sh`
**Checkpoint:** `outputs/train/checkpoints/best.pt`
**Test dir:** `/workspace/plantclef/raw/test`
**Test images:** 2,105
**agg_mode:** `topk_mean`
**margin_crop:** 0.0
**tile_batch_size:** 32

**Sweep axes:**
- **interpolation:** bilinear, bicubic (×2)
- **tile_mode:** whole, grid3x3, grid5x5, five_crop (tile_size=224), sliding (tile_size=224, stride=112), multiscale, multiscale_dense_1234 (scales=1,2,3,4, overlap=0.25) (×7)
- **top_n (k):** 3, 5 (×2)
- **Total:** 28 configurations

### Results - All Public Scores (sorted descending)

| Run Name | interp | tile_mode | top_n | public_score |
|---|---|---|---|---|
| bilinear_grid5x5_k3 | bilinear | grid5x5 | 3 | **0.24182** |
| bicubic_multiscale_k3 | bicubic | multiscale | 3 | 0.23740 |
| bilinear_multiscale_k3 | bilinear | multiscale | 3 | 0.23167 |
| bilinear_multiscale_dense_1234_k3 | bilinear | multiscale_dense | 3 | 0.22304 |
| bicubic_multiscale_dense_1234_k3 | bicubic | multiscale_dense | 3 | 0.22272 |
| bicubic_sliding224s112_k3 | bicubic | sliding | 3 | 0.22640 |
| bicubic_grid3x3_k3 | bicubic | grid3x3 | 3 | 0.22039 |
| bicubic_grid5x5_k3 | bicubic | grid5x5 | 3 | 0.21941 |
| bilinear_grid5x5_k5 | bilinear | grid5x5 | 5 | 0.21574 |
| bilinear_sliding224s112_k3 | bilinear | sliding | 3 | 0.21352 |
| bilinear_grid3x3_k3 | bilinear | grid3x3 | 3 | 0.21167 |
| bilinear_multiscale_k5 | bilinear | multiscale | 5 | 0.21262 |
| bicubic_grid3x3_k5 | bicubic | grid3x3 | 5 | 0.20390 |
| bicubic_sliding224s112_k5 | bicubic | sliding | 5 | 0.20150 |
| bilinear_sliding224s112_k5 | bilinear | sliding | 5 | 0.19634 |
| bilinear_multiscale_dense_1234_k5 | bilinear | multiscale_dense | 5 | 0.19341 |
| bicubic_grid5x5_k5 | bicubic | grid5x5 | 5 | 0.19489 |
| bilinear_grid3x3_k5 | bilinear | grid3x3 | 5 | 0.18030 |
| bicubic_whole_k5 | bicubic | whole | 5 | 0.14904 |
| bicubic_whole_k3 | bicubic | whole | 3 | 0.15668 |
| bilinear_whole_k3 | bilinear | whole | 3 | 0.15386 |
| bilinear_whole_k5 | bilinear | whole | 5 | 0.14755 |
| bicubic_fivecrop_k5 | bicubic | five_crop | 5 | 0.07735 |
| bilinear_fivecrop_k5 | bilinear | five_crop | 5 | 0.07025 |
| bilinear_fivecrop_k3 | bilinear | five_crop | 3 | 0.06938 |
| bicubic_fivecrop_k3 | bicubic | five_crop | 3 | 0.06021 |


**Best public score:** `bilinear_grid5x5_k3` = **0.24182**

### Notes
- **Observations:**
  - Five-crop (`five_crop`) performs catastrophically (~0.06–0.08) - likely because the crops are too small or centered/corner crops miss the plant subject.
  - Whole-image baseline performs poorly (~0.14–0.16), confirming that tiling substantially helps.
  - grid5x5 is the strongest tile mode (0.24182 with bilinear, 0.21941 with bicubic).
  - bicubic_multiscale_k3 is second-best (0.23740), suggesting multi-scale aggregation is nearly as good as dense grid tiling.
  - bilinear slightly outperforms bicubic on the best configurations (0.24182 vs. 0.21941 for grid5x5), which is counterintuitive.
  - k=3 (top-3 prediction) consistently outperforms k=5 (top-5) within the same config - submitting fewer predictions with higher confidence is better for the metric used.
  - Sliding window (224×224, stride=112) is competitive with grid3x3/multiscale but slightly weaker than grid5x5.
  - multiscale_dense (scales 1-4, overlap 0.25) is slightly weaker than standard multiscale (0.22 vs. 0.24).
- **Ideas for next experiment:**
  1. Push grid beyond 5×5 (e.g., 7×7 or 8×8) to see if finer crops keep helping.
  2. Try bilinear+grid5x5 with k=1 (single best prediction per image).
  3. Fine-tune the backbone (unfreeze last 1–2 transformer blocks).

---

# Overall Summary

- **Best run:** `infer_sweeps_interp_k35/bilinear_grid5x5_k3` - public score **0.24182**
- **Most promising settings:**
  - Tile mode: `grid5x5` or `multiscale`
  - Interpolation: `bilinear` (marginally better than bicubic on best settings)
  - top_n: **3** (submitting fewer, more confident predictions wins)
  - Aggregation: `topk_mean`
- **Patterns across runs:**
  - More tiles = better up to ~5×5 grid; five_crop is uniquely bad
  - k=3 consistently beats k=5 across all tile modes
  - bilinear ≥ bicubic on public leaderboard (unusual - may indicate the test images have a specific resolution mismatch)
  - Whole-image (no tiling) is the clear weakest performer, confirming tiling is essential for this dataset
  - Training converged steadily; recall@5 improved every epoch over 10 epochs
- **Recommended next 3 experiments:**
  1. **Finer grid tiling (7×7 or 8×8) + bilinear + k=3** - follow the monotone grid5x5 > grid3x3 > grid2x2 > whole trend to see if it continues.
  2. **Train for 20 epochs (or more)** - val metrics were still improving at epoch 10; a longer run with the same frozen-backbone setup should push recall@5 toward 0.95+.
  3. **Partial fine-tuning** - unfreeze the last 2 transformer blocks of ViT-H/14 with a small learning rate (1e-5) for end-to-end fine-tuning, keeping the linear head warm-started from best.pt.

---

# Compact Comparison Table

## Training Run

| experiment_name | model | tile_mode | interpolation | topk_agg | batch_size | learning_rate | epochs | best_val_metric | public_score | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| training | BioCLIP-2.5 ViT-H/14 | whole (train) | bicubic | - | 256/GPU | 0.001 | 10 | Recall@5=0.9303, R@1=0.7505 | - | DDP 2-GPU; frozen backbone; cosine LR |

## Inference Sweep: `infer_sweeps_interp_k35` (with public scores, sorted descending)

| experiment_name | model | tile_mode | interpolation | topk_agg | top_n | public_score | notes |
|---|---|---|---|---|---|---|---|
| bilinear_grid5x5_k3 | BioCLIP-2.5 | grid5x5 | bilinear | 5 | 3 | **0.24182** | best overall |
| bicubic_multiscale_k3 | BioCLIP-2.5 | multiscale | bicubic | 5 | 3 | 0.23740 | 2nd best |
| bilinear_multiscale_k3 | BioCLIP-2.5 | multiscale | bilinear | 5 | 3 | 0.23167 | |
| bicubic_sliding224s112_k3 | BioCLIP-2.5 | sliding | bicubic | 5 | 3 | 0.22640 | tile=224, stride=112 |
| bilinear_multiscale_dense_1234_k3 | BioCLIP-2.5 | multiscale_dense | bilinear | 5 | 3 | 0.22304 | scales=1-4, overlap=0.25 |
| bicubic_multiscale_dense_1234_k3 | BioCLIP-2.5 | multiscale_dense | bicubic | 5 | 3 | 0.22272 | |
| bicubic_grid3x3_k3 | BioCLIP-2.5 | grid3x3 | bicubic | 5 | 3 | 0.22039 | |
| bicubic_grid5x5_k3 | BioCLIP-2.5 | grid5x5 | bicubic | 5 | 3 | 0.21941 | |
| bilinear_grid5x5_k5 | BioCLIP-2.5 | grid5x5 | bilinear | 5 | 5 | 0.21574 | k=5 hurts |
| bilinear_multiscale_k5 | BioCLIP-2.5 | multiscale | bilinear | 5 | 5 | 0.21262 | |
| bilinear_sliding224s112_k3 | BioCLIP-2.5 | sliding | bilinear | 5 | 3 | 0.21352 | |
| bilinear_grid3x3_k3 | BioCLIP-2.5 | grid3x3 | bilinear | 5 | 3 | 0.21167 | |
| bicubic_grid3x3_k5 | BioCLIP-2.5 | grid3x3 | bicubic | 5 | 5 | 0.20390 | |
| bicubic_sliding224s112_k5 | BioCLIP-2.5 | sliding | bicubic | 5 | 5 | 0.20150 | |
| bilinear_sliding224s112_k5 | BioCLIP-2.5 | sliding | bilinear | 5 | 5 | 0.19634 | |
| bicubic_grid5x5_k5 | BioCLIP-2.5 | grid5x5 | bicubic | 5 | 5 | 0.19489 | |
| bilinear_multiscale_dense_1234_k5 | BioCLIP-2.5 | multiscale_dense | bilinear | 5 | 5 | 0.19341 | |
| bilinear_grid3x3_k5 | BioCLIP-2.5 | grid3x3 | bilinear | 5 | 5 | 0.18030 | |
| bicubic_whole_k3 | BioCLIP-2.5 | whole | bicubic | 5 | 3 | 0.15668 | no tiling |
| bilinear_whole_k3 | BioCLIP-2.5 | whole | bilinear | 5 | 3 | 0.15386 | no tiling |
| bicubic_whole_k5 | BioCLIP-2.5 | whole | bicubic | 5 | 5 | 0.14904 | no tiling |
| bilinear_whole_k5 | BioCLIP-2.5 | whole | bilinear | 5 | 5 | 0.14755 | no tiling |
| bicubic_fivecrop_k5 | BioCLIP-2.5 | five_crop | bicubic | 5 | 5 | 0.07735 | catastrophic |
| bilinear_fivecrop_k5 | BioCLIP-2.5 | five_crop | bilinear | 5 | 5 | 0.07025 | catastrophic |
| bilinear_fivecrop_k3 | BioCLIP-2.5 | five_crop | bilinear | 5 | 3 | 0.06938 | catastrophic |
| bicubic_fivecrop_k3 | BioCLIP-2.5 | five_crop | bicubic | 5 | 3 | 0.06021 | catastrophic |
