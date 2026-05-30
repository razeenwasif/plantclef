# Experiment Report: 010 — BioCLIP 2.5 End-to-End Fine-Tuning + Multitask Taxonomy

**Experiment folder:** `src_experiments/010_bioclip25_end_to_end_finetune_multitask/`  
**Date written:** 2026-04-29
**Author**: Arjun

---

## Most Useful Files

| File | Purpose |
|---|---|
| [README.md](README.md) | Architecture, dataset, augmentation, CLI flags — very thorough |
| [train.py](train.py) | Main training script (multi-GPU torchrun, AMP, warmup+cosine) |
| [model.py](model.py) | `BioCLIP25MultiTask`: backbone + SharedMLP + 5 classification heads |
| [infer_tiles.py](infer_tiles.py) | Standard tile inference (fixed top-k) |
| [infer_tiles_adaptive.py](infer_tiles_adaptive.py) | Adaptive inference (gap / prob-threshold / relative-threshold) |
| [scripts/train_head_only.sh](scripts/train_head_only.sh) | Stage 1 training script |
| [scripts/train_last_blocks.sh](scripts/train_last_blocks.sh) | Stage 2: unfreeze 4 blocks |
| [scripts/train_last_blocks_8.sh](scripts/train_last_blocks_8.sh) | Stage 2b: unfreeze 8 blocks |
| [scripts/train_last_blocks_12.sh](scripts/train_last_blocks_12.sh) | Stage 2c: unfreeze 12 blocks |
| [scripts/train_full_finetune.sh](scripts/train_full_finetune.sh) | Stage 3: full backbone |
| [scripts/infer_tiles.sh](scripts/infer_tiles.sh) | Parallel tile sweep script (two GPUs) |
| [scripts/infer_best_adaptive.sh](scripts/infer_best_adaptive.sh) | Adaptive inference sweep |
| [outputs/head_only/metrics.csv](outputs/head_only/metrics.csv) | Head-only training metrics |
| [outputs/last_blocks/metrics.csv](outputs/last_blocks/metrics.csv) | last-4-blocks metrics |
| [outputs/last_blocks_8/metrics.csv](outputs/last_blocks_8/metrics.csv) | last-8-blocks metrics |
| [outputs/last_blocks_12/metrics.csv](outputs/last_blocks_12/metrics.csv) | last-12-blocks metrics |
| [outputs/full_finetune/metrics.csv](outputs/full_finetune/metrics.csv) | Full finetune metrics |
| [scores.csv](scores.csv) | Kaggle public scores for head_only and last_blocks sweeps |
| [scores_fixed_k.csv](scores_fixed_k.csv) | Kaggle scores for last_blocks_8 fixed-k sweeps |
| [scores_apdative_last_8.csv](scores_apdative_last_8.csv) | Kaggle scores for last_blocks_8 adaptive sweeps (**best overall**) |
| [scores_2.csv](scores_2.csv) | Kaggle scores for last_blocks_12 adaptive sweeps (only 3 runs) |

---

## High-Level Summary

This experiment folder tests staged end-to-end fine-tuning of BioCLIP 2.5 (ViT-H/14) for PlantCLEF 2026 species prediction. The model uses a shared MLP head on top of the CLIP visual encoder and auxiliary taxonomy losses (genus, family, order, class) to regularise training.

The key question was: **how many transformer blocks should be unfrozen, and how should tiled inference be done?** Five training stages were run (head-only, last-4, last-8, last-12, full), followed by extensive tiling + aggregation sweeps on the test set and direct Kaggle submissions. 127 unique submissions were scored. No W&B logging was used; all metrics come from CSV files and run logs.

---

## Dataset and Task

**Competition:** PlantCLEF 2026 (Kaggle) — multi-label species presence prediction in quadrat images.  
**Training data:** 1,408,033 images across 7,806 species from `PlantCLEF2024_single_plant_training_metadata.csv`.  
**Val split:** Stratified 10% per species; species with < 5 images are kept only in train. Final val set: **137,824 images** covering 7,197 species (some rare species have no val images).  
**Test set:** 2,105 quadrat images (from `/workspace/plantclef/raw/test`).

### Taxonomy coverage

| Level | Unique classes | Coverage |
|---|---|---|
| species | 7,806 | 100% |
| genus | 1,446 | 100% |
| family | 181 | 100% |
| order | 61 | 99.9% |
| class | 6 | 99.9% |

---

## Model / Architecture

**Backbone:** BioCLIP 2.5 ViT-H/14 (`hf-hub:imageomics/bioclip-2.5-vith14`), 32 transformer blocks, 1024-dim embedding.

```
Input image (224×224)
       │
BioCLIP 2.5 ViT-H/14 backbone   ← frozen by default
       │  (1024-dim embedding)
       ▼
LayerNorm → Linear(1024→1024) → GELU → Dropout(0.2)   ← shared MLP
       │
   ┌───┴───┬───────┬────────┬──────────┐
species  genus  family   order    class    ← linear heads
(7806)  (1446)  (181)    (61)      (6)
```

**Multi-task loss weights:**
```
loss = species_CE + 0.30×genus_CE + 0.15×family_CE + 0.05×order_CE + 0.02×class_CE
```
Missing taxonomy labels (NaN → -1) are masked out, so the auxiliary losses only apply to labelled samples.

---

## Training Experiments

Five training stages were run in sequence (each resuming from the previous best checkpoint). All used:
- `--use-taxonomy-heads` (always on)
- `--precision bf16`
- `--weight-decay 1e-4`
- `--label-smoothing 0.1`
- `--warmup-epochs 1` with cosine decay
- 2 GPUs with `torchrun --nproc_per_node=2`

### Stage 0: Smoke Test

A 1-epoch, 200-sample sanity check. Not evaluated on Kaggle.  
Output: [outputs/smoke_test/](outputs/smoke_test/)

---

### Stage 1: Head-Only Training (`head_only`)

| Parameter | Value |
|---|---|
| Frozen/Unfrozen | Backbone frozen, head + MLP only |
| Params trained | ~10.8M (head + shared MLP) |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 (not used) |
| batch_size | 512 per GPU × 2 GPUs = 1024 effective |
| grad_accum | 2 (effective batch = 2048) |
| epochs | 10 |
| epoch time | ~1497s / epoch (~25 min) |
| total train time | ~306 min |
| output | [outputs/head_only/](outputs/head_only/) |
| best checkpoint | [outputs/head_only/checkpoints/best.pt](outputs/head_only/checkpoints/best.pt) |
| config | [outputs/head_only/train_config.json](outputs/head_only/train_config.json) |

**Per-epoch val metrics** ([outputs/head_only/metrics.csv](outputs/head_only/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 9.133 | 2.504 | 0.5655 | 0.8223 | 0.7924 | 0.8850 |
| 3 | 3.949 | 1.348 | 0.7004 | 0.9081 | 0.8671 | 0.9231 |
| 5 | 3.689 | 1.244 | 0.7183 | 0.9172 | 0.8751 | 0.9277 |
| 7 | 3.593 | 1.211 | 0.7254 | 0.9204 | 0.8773 | 0.9292 |
| 9 | 3.556 | 1.199 | 0.7277 | 0.9214 | 0.8780 | 0.9295 |
| **10** | **3.550** | **1.199** | **0.7279** | **0.9214** | **0.8780** | **0.9294** |

Best val top-5: **0.9214** at epoch 10. Best checkpoint is `epoch_010` (last epoch also happened to be best).  
Training loss had not fully converged — still slowly decreasing at epoch 10. Could have benefited from more epochs.

---

### Stage 2a: Unfreeze Last 4 Blocks (`last_blocks`)

| Parameter | Value |
|---|---|
| Frozen/Unfrozen | Last 4 transformer blocks + ln_post/proj unfrozen |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 |
| batch_size | 64 per GPU × 2 = 128 |
| grad_accum | 4 (effective batch = 512) |
| epochs | 5 |
| epoch time | ~2068s / epoch (~34 min) |
| resume from | [outputs/head_only/checkpoints/best.pt](outputs/head_only/checkpoints/best.pt) |
| output | [outputs/last_blocks/](outputs/last_blocks/) |
| best checkpoint | [outputs/last_blocks/checkpoints/best.pt](outputs/last_blocks/checkpoints/best.pt) |

**Per-epoch val metrics** ([outputs/last_blocks/metrics.csv](outputs/last_blocks/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 3.543 | 1.204 | 0.7227 | 0.9213 | 0.8793 | 0.9330 |
| 3 | 3.342 | 1.117 | 0.7420 | 0.9300 | 0.8864 | 0.9374 |
| **5** | **3.236** | **1.093** | **0.7475** | **0.9323** | **0.8882** | **0.9385** |

Best val top-5: **0.9323** (+0.0109 vs head_only).

---

### Stage 2b: Unfreeze Last 8 Blocks (`last_blocks_8`)

| Parameter | Value |
|---|---|
| Frozen/Unfrozen | Last 8 transformer blocks unfrozen |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 |
| batch_size | 64 per GPU × 2 = 128 |
| grad_accum | 4 (effective batch = 512) |
| epochs | 5 |
| epoch time | ~2490s / epoch (~41 min) |
| resume from | [outputs/last_blocks/checkpoints/best.pt](outputs/last_blocks/checkpoints/best.pt) |
| output | [outputs/last_blocks_8/](outputs/last_blocks_8/) |
| best checkpoint | [outputs/last_blocks_8/checkpoints/best.pt](outputs/last_blocks_8/checkpoints/best.pt) |

**Per-epoch val metrics** ([outputs/last_blocks_8/metrics.csv](outputs/last_blocks_8/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 3.245 | 1.112 | 0.7418 | 0.9303 | 0.8875 | 0.9387 |
| 3 | 3.140 | 1.062 | 0.7525 | 0.9350 | 0.8919 | 0.9421 |
| **5** | **3.054** | **1.045** | **0.7569** | **0.9364** | **0.8934** | **0.9429** |

Best val top-5: **0.9364** (+0.0041 vs last_blocks_4).

---

### Stage 2c: Unfreeze Last 12 Blocks (`last_blocks_12`)

| Parameter | Value |
|---|---|
| Frozen/Unfrozen | Last 12 transformer blocks unfrozen |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 |
| batch_size | 64 per GPU × 2 = 128 |
| grad_accum | 4 (effective batch = 512) |
| epochs | 5 |
| epoch time | ~2880s / epoch (~48 min) |
| resume from | [outputs/last_blocks_8/checkpoints/best.pt](outputs/last_blocks_8/checkpoints/best.pt) |
| output | [outputs/last_blocks_12/](outputs/last_blocks_12/) |
| best checkpoint | [outputs/last_blocks_12/checkpoints/best.pt](outputs/last_blocks_12/checkpoints/best.pt) |

**Per-epoch val metrics** ([outputs/last_blocks_12/metrics.csv](outputs/last_blocks_12/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 3.064 | 1.063 | 0.7516 | 0.9350 | 0.8929 | 0.9432 |
| 3 | 2.984 | 1.023 | 0.7602 | 0.9387 | 0.8967 | 0.9462 |
| **5** | **2.905** | **1.010** | **0.7641** | **0.9400** | **0.8978** | **0.9464** |

Best val top-5: **0.9400** (+0.0036 vs last_blocks_8). Still improving at epoch 5 — more epochs might help.

---

### Stage 3: Full Fine-Tune (`full_finetune`)

| Parameter | Value |
|---|---|
| Frozen/Unfrozen | Entire model (all 32 blocks) |
| head_lr | 5e-4 (from config) |
| backbone_lr | 5e-5 (from config) |
| batch_size | 32 per GPU × 2 = 64 |
| grad_accum | 8 (effective batch = 512) |
| epochs | 10 |
| epoch time | ~9590s / epoch (~160 min) |
| resume from | [outputs/last_blocks/checkpoints/best.pt](outputs/last_blocks/checkpoints/best.pt) (**note: resumed from last_4 checkpoint, not last_12**) |
| output | [outputs/full_finetune/](outputs/full_finetune/) |
| best checkpoint | [outputs/full_finetune/checkpoints/best.pt](outputs/full_finetune/checkpoints/best.pt) |
| total train time | ~1657 min (~27.6 hours) |

**Per-epoch val metrics** ([outputs/full_finetune/metrics.csv](outputs/full_finetune/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 3.305 | 1.348 | 0.6831 | 0.9043 | 0.8433 | 0.9114 |
| 3 | 3.118 | 1.203 | 0.7154 | 0.9171 | 0.8597 | 0.9234 |
| 5 | 2.676 | 1.092 | 0.7448 | 0.9297 | 0.8802 | 0.9368 |
| **7** | **2.300** | **1.067** | **0.7611** | **0.9339** | **0.8893** | **0.9438** |
| 8 | 2.165 | 1.081 | 0.7638 | 0.9332 | 0.8920 | 0.9455 |
| 9 | 2.078 | 1.095 | 0.7660 | 0.9335 | 0.8936 | 0.9464 |
| 10 | 2.041 | 1.102 | 0.7660 | 0.9331 | 0.8937 | 0.9465 |

Best val top-5: **0.9339** at epoch 7 — **lower than last_blocks_12 (0.9400)**.  
Val loss starts increasing after epoch 7, while train loss keeps decreasing → **clear overfitting signal**.  
The full_finetune config also had a higher learning rate (backbone_lr=5e-5 vs 1e-6 for block-unfreezing stages), which may be why it initially drops in performance and then rebounds but overshoots.

**Important:** The full_finetune resumed from `last_blocks/checkpoints/best.pt` (4-block stage), not from the best checkpoint `last_blocks_12`. Despite 27 hours of training it never matched last_blocks_12's validation performance.

---

### Training Summary Table

| Run | Blocks Unfrozen | Epochs | Best Val Top-1 | Best Val Top-5 | Val Loss (best) | Best Ckpt |
|---|---|---|---|---|---|---|
| smoke_test | 0 (head) | 1 (200 samples) | — | — | — | — |
| head_only | 0 (head only) | 10 | 0.7279 | 0.9214 | 1.1985 | epoch_010 |
| last_blocks (4) | 4 | 5 | 0.7475 | 0.9323 | 1.0929 | epoch_005 |
| last_blocks_8 | 8 | 5 | 0.7569 | 0.9364 | 1.0449 | epoch_005 |
| last_blocks_12 | 12 | 5 | 0.7641 | 0.9400 | 1.0101 | epoch_005 |
| full_finetune | 32 (all) | 10 | 0.7660 | 0.9339 | 1.0673 | epoch_007 |

The best validation checkpoint across all runs is **last_blocks_12 epoch_005** (top5=0.9400).  
The full_finetune checkpoint is **not** the best for inference despite having the most trainable parameters.

---

## Inference and Tiling Experiments

Inference was run in four families, using `infer_tiles.py` (fixed top-k) or `infer_tiles_adaptive.py` (adaptive selection). All used tile_size=448 unless otherwise noted.

### Tiling Modes Used

| Mode | Tiles Generated | Notes |
|---|---|---|
| `whole` | 1 | Only used in smoke test |
| `five_crop` | 5 | Only tested on head_only (one submission) |
| `grid_2x2` | 4 | Only tested on head_only |
| `grid_3x3` | 9 | Only tested on head_only |
| `grid_4x4` | 16 | Main mode — used across all checkpoints |
| `grid_5x5` | 25 | Tested on last_blocks_8 |
| `grid_6x6` | 36 | Tested on last_blocks_8 |
| `multiscale` | 21 | whole + 2×2 + 4×4, tested on head_only |
| `sliding ts224` | varies | Tested on last_blocks (4 blocks) |
| `sliding ts448` | varies | Tested on last_blocks_8 |
| `sliding ts672` | varies | Tested on last_blocks_8 |

### Aggregation Modes Used

| Mode | Description | Notes |
|---|---|---|
| `max` | Element-wise max over tile logits | Good for detecting rare species |
| `mean` | Simple mean of tile logits | Consistently weakest |
| `softmax_mean` | Mean in probability space | Consistently best |

### Selection Modes (Adaptive Inference)

| Mode | Description |
|---|---|
| `fixed_topk` | Always return top-k species (k=2,3,4) |
| `gap` | Include up to k species while consecutive score gap > ratio |
| `prob_threshold` | Include species with probability > threshold |
| `relative_threshold` | Include species with score > max_score × threshold |

---

### Inference Family 1: Head-Only Sweeps (`tile_sweeps/head_only/`)

**Checkpoint:** [outputs/head_only/checkpoints/best.pt](outputs/head_only/checkpoints/best.pt)  
**Config ref:** [outputs/tile_sweeps/head_only/grid_4x4_ts448_ov0p0/softmax_mean_top5/run_config.json](outputs/tile_sweeps/head_only/grid_4x4_ts448_ov0p0/softmax_mean_top5/run_config.json)  
**Precision:** fp32  
**Tile modes swept:** whole, five_crop, grid_2x2, grid_3x3, grid_4x4, multiscale  
**Overlaps:** 0.0, 0.25, 0.5, 0.75  
**Aggregations:** max, mean, softmax_mean  
**Top-k:** 1, 2, 3, 4, 5  
**Total submissions:** 56

Selected scores (from [scores.csv](scores.csv)):

| Tile Mode | Overlap | Agg | Top-k | Public Score |
|---|---|---|---|---|
| grid_4x4 | 0.0 | softmax_mean | 4 | **0.36197** |
| grid_4x4 | 0.0 | softmax_mean | 3 | 0.35931 |
| grid_4x4 | 0.0 | softmax_mean | 2 | 0.35038 |
| grid_4x4 | 0.0 | max | 3 | 0.35516 |
| grid_4x4 | 0.25 | softmax_mean | 2 | 0.34673 |
| multiscale | 0.25 | softmax_mean | 2 | 0.32978 |
| grid_3x3 | 0.75 | softmax_mean | 2 | 0.28597 |
| five_crop | 0.0 | max | 1 | 0.21709 |
| grid_3x3 | 0.75 | mean | 1 | 0.20681 |

**Observation:** grid_4x4 with zero overlap + softmax_mean is best. Overlap consistently hurts. Mean aggregation is worst.

---

### Inference Family 2: Last-4-Blocks Sweeps (`tile_sweeps/last_blocks/`)

**Checkpoint:** [outputs/last_blocks/checkpoints/best.pt](outputs/last_blocks/checkpoints/best.pt)  
**Tile modes swept:** grid_4x4 (overlaps 0.0, 0.25, 0.5, 0.75), sliding_ts224 (overlaps 0.0, 0.25, 0.5, 0.75), sliding_ts448 (overlaps 0.0, 0.25, 0.5)  
**Aggregations:** max, softmax_mean  
**Top-k:** 2, 3, 4  
**Total submissions:** 23 (subset scored in [scores.csv](scores.csv))

Selected scores:

| Tile Mode | Overlap | Agg | Top-k | Public Score |
|---|---|---|---|---|
| grid_4x4 | 0.0 | softmax_mean | 3 | **0.38333** |
| grid_4x4 | 0.0 | softmax_mean | 2 | 0.37313 |
| grid_4x4 | 0.0 | softmax_mean | 4 | 0.37174 |
| grid_4x4 | 0.0 | max | 3 | 0.36632 |
| grid_4x4 | 0.0 | max | 2 | 0.36186 |
| sliding_ts224 | 0.25 | softmax_mean | 4 | 0.33946 |
| sliding_ts224 | 0.0 | max | 4 | 0.33304 |
| sliding_ts224 | 0.0 | max | 2 | 0.27291 |

---

### Inference Family 3: Last-8-Blocks Fixed-K Sweeps (`tile_sweeps/last_blocks_8/`)

**Checkpoint:** [outputs/last_blocks_8/checkpoints/best.pt](outputs/last_blocks_8/checkpoints/best.pt)  
**Tile modes swept:** grid_4x4 (ov 0.0, 0.25), grid_5x5 (ov 0.0, 0.25), grid_6x6 (ov 0.0, 0.25), sliding_ts448 (ov 0.0), sliding_ts672 (ov 0.0, 0.25)  
**Aggregations:** max, softmax_mean  
**Top-k:** 2, 3, 4  
**Total submissions:** 32 (scored in [scores_fixed_k.csv](scores_fixed_k.csv))

Selected scores:

| Tile Mode | Overlap | Agg | Top-k | Public Score |
|---|---|---|---|---|
| grid_4x4 | 0.0 | softmax_mean | 3 | **0.39004** |
| grid_4x4 | 0.0 | softmax_mean | 2 | 0.37971 |
| grid_4x4 | 0.0 | softmax_mean | 4 | 0.37312 |
| grid_5x5 | 0.0 | softmax_mean | 3 | 0.37809 |
| grid_6x6 | 0.0 | softmax_mean | 4 | 0.36222 |
| sliding_ts672 | 0.0 | softmax_mean | 3 | 0.36157 |
| grid_4x4 | 0.25 | softmax_mean | 3 | 0.35880 |
| grid_5x5 | 0.25 | softmax_mean | 2 | 0.35296 |
| grid_4x4 | 0.25 | max | 2 | 0.32551 |

---

### Inference Family 4: Last-8-Blocks Adaptive Sweeps (`last_blocks_8_tile_sweep/`)

**Checkpoint:** [outputs/last_blocks_8/checkpoints/best.pt](outputs/last_blocks_8/checkpoints/best.pt)  
**Tile mode:** grid_4x4, tile_size=448, overlap=0.0  
**Aggregation:** softmax_mean only  
**Selection modes:** fixed_topk (k=2,3,4), gap (0.4, 0.5, 0.6), prob_threshold (0.02, 0.03, 0.05), relative_threshold (0.15, 0.20, 0.25, 0.30)  
**max_k:** 5  
**Precision:** bf16  
**Scored in:** [scores_apdative_last_8.csv](scores_apdative_last_8.csv)

Selected scores (all 13 runs scored):

| Selection Mode | Param | Public Score |
|---|---|---|
| gap | 0.5 | **0.39140** ← best overall |
| fixed_topk | 3 | 0.38960 |
| gap | 0.6 | 0.38428 |
| prob_threshold | 0.02 | 0.38017 |
| relative_threshold | 0.30 | 0.37989 |
| gap | 0.4 | 0.37983 |
| prob_threshold | 0.03 | 0.37953 |
| relative_threshold | 0.15 | 0.37919 |
| relative_threshold | 0.25 | 0.37802 |
| fixed_topk | 2 | 0.37816 |
| relative_threshold | 0.20 | 0.37730 |
| fixed_topk | 4 | 0.37298 |
| prob_threshold | 0.05 | 0.36843 |

---

### Inference Family 5: Last-12-Blocks Adaptive Sweeps (`last_blocks_12_tile_sweep/`, `last_blocks_12_tile_sweep_maxk10/`)

**Checkpoint:** [outputs/last_blocks_12/checkpoints/best.pt](outputs/last_blocks_12/checkpoints/best.pt)  
**Tile mode:** grid_4x4, tile_size=448, overlap=0.0  
**Aggregation:** softmax_mean only  
**Scored in:** [scores_2.csv](scores_2.csv) (only 3 of 26 runs were submitted to Kaggle)

| Selection Mode | max_k | Public Score |
|---|---|---|
| gap | 0.6 | **0.37479** |
| gap | 0.5 | 0.36878 |
| fixed_topk | 2 | 0.36544 |

**Note:** Despite last_blocks_12 having the best validation top-5 (0.9400) of all checkpoints, it only scored 0.37479 on the public leaderboard — **lower than last_blocks_8's best of 0.39140**. This is surprising and warrants investigation. Only 3 configurations were submitted, so the full adaptive sweep was never scored on Kaggle.

---

### Inference Family 6: Full-Finetune Sweeps (`tile_sweeps/full_finetune/`)

**Checkpoint:** [outputs/full_finetune/checkpoints/best.pt](outputs/full_finetune/checkpoints/best.pt)  
**Tile modes generated:** grid_4x4 (ov 0.0, 0.25, 0.5), grid_5x5 (ov 0.0, 0.25, 0.5)  
**Status:** Submission CSVs generated but **not submitted to Kaggle** — no public scores available.

---

## Kaggle Submission Results

All 127 scored submissions ranked.

### Top 20 Runs

| Rank | Checkpoint | Tile Mode | Overlap | Agg | Selection | Public Score | Source |
|---|---|---|---|---|---|---|---|
| 1 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | gap 0.5 | **0.39140** | scores_apdative_last_8.csv |
| 2 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | top3 | 0.39004 | scores_fixed_k.csv |
| 3 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | top3 (adaptive) | 0.38960 | scores_apdative_last_8.csv |
| 4 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | gap 0.6 | 0.38428 | scores_apdative_last_8.csv |
| 5 | last_blocks (4) | grid_4x4 | 0.0 | softmax_mean | top3 | 0.38333 | scores.csv |
| 6 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | probT 0.02 | 0.38017 | scores_apdative_last_8.csv |
| 7 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | relT 0.30 | 0.37989 | scores_apdative_last_8.csv |
| 8 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | gap 0.4 | 0.37983 | scores_apdative_last_8.csv |
| 9 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | top2 | 0.37971 | scores_fixed_k.csv |
| 10 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | probT 0.03 | 0.37953 | scores_apdative_last_8.csv |
| 11 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | relT 0.15 | 0.37919 | scores_apdative_last_8.csv |
| 12 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | top2 (adaptive) | 0.37816 | scores_apdative_last_8.csv |
| 13 | last_blocks_8 | grid_5x5 | 0.0 | softmax_mean | top3 | 0.37809 | scores_fixed_k.csv |
| 14 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | relT 0.25 | 0.37802 | scores_apdative_last_8.csv |
| 15 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | relT 0.20 | 0.37730 | scores_apdative_last_8.csv |
| 16 | last_blocks_12 | grid_4x4 | 0.0 | softmax_mean | gap 0.6 | 0.37479 | scores_2.csv |
| 17 | last_blocks (4) | grid_4x4 | 0.0 | softmax_mean | top2 | 0.37313 | scores.csv |
| 18 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | top4 | 0.37312 | scores_fixed_k.csv |
| 19 | last_blocks_8 | grid_4x4 | 0.0 | softmax_mean | top4 (adaptive) | 0.37298 | scores_apdative_last_8.csv |
| 20 | last_blocks (4) | grid_4x4 | 0.0 | softmax_mean | top4 | 0.37174 | scores.csv |

### Bottom 10 Runs

| Rank | Checkpoint | Tile Mode | Overlap | Agg | Selection | Public Score |
|---|---|---|---|---|---|---|
| 118 | head_only | grid_4x4 | 0.5 | mean | top4 | 0.22654 |
| 119 | head_only | grid_3x3 | 0.75 | mean | top3 | 0.22509 |
| 120 | head_only | grid_3x3 | 0.75 | mean | top2 | 0.22347 |
| 121 | head_only | five_crop | 0.0 | max | top1 | 0.21709 |
| 122 | head_only | grid_4x4 | 0.75 | mean | top4 | 0.21237 |
| 123 | head_only | grid_3x3 | 0.75 | mean | top4 | 0.21124 |
| 124 | head_only | grid_4x4 | 0.5 | mean | top5 | 0.20931 |
| 125 | head_only | grid_3x3 | 0.75 | mean | top1 | 0.20681 |
| 126 | head_only | grid_3x3 | 0.75 | mean | top5 | 0.20153 |
| 127 | head_only | grid_4x4 | 0.75 | mean | top5 | 0.20017 |

### Score Distribution by Checkpoint Family

| Checkpoint | Submissions Scored | Best Score | Avg Score | Worst Score |
|---|---|---|---|---|
| head_only | 56 | 0.36197 | 0.27938 | 0.20017 |
| last_blocks (4 blocks) | 23 | 0.38333 | 0.33397 | 0.27291 |
| last_blocks_8 | 45 | **0.39140** | **0.35937** | 0.30289 |
| last_blocks_12 | 3 | 0.37479 | 0.36967 | 0.36544 |
| full_finetune | 0 | — | — | — |

---

## Best Runs

1. **last_blocks_8 + grid_4x4_ov0.0 + softmax_mean + gap0.5 → 0.39140**  
   The single best submission. Grid 4×4 tiling with zero overlap, softmax probability aggregation, and gap-based adaptive k selection (ratio=0.5). Uses the best training checkpoint by inference performance, not by validation loss.

2. **last_blocks_8 + grid_4x4_ov0.0 + softmax_mean + top3 → 0.39004**  
   Very close to #1. Fixed k=3 without adaptive selection. Suggests that for most images, top3 is the right answer.

3. **last_blocks_8 adaptive + softmax_mean + top3 → 0.38960**  
   Same run as #2 but using the adaptive inference script. Near-identical.

4. **last_blocks_8 + grid_4x4_ov0.0 + softmax_mean + gap0.6 → 0.38428**  
   Slightly more conservative gap threshold (0.6 = larger jump required to include next species = fewer predictions).

5. **last_blocks (4) + grid_4x4_ov0.0 + softmax_mean + top3 → 0.38333**  
   Competitive with last_blocks_8. This model has 4 fewer unfrozen blocks but was still close.

**Common pattern across top runs:**
- Checkpoint: last_blocks_8 (not last_blocks_12, not full_finetune)
- Tile mode: grid_4x4
- Overlap: 0.0
- Aggregation: softmax_mean
- Top-k: 3 or gap~0.5

---

## Failed / Weak Runs

1. **All head_only + mean aggregation runs (scores 0.20–0.26)**  
   Mean aggregation in logit space dilutes the signal from informative tiles. Weak tiles (e.g., sky/ground tiles) pull the mean down aggressively. Mean in probability space (softmax_mean) avoids this.

2. **head_only + any overlap > 0.25 runs (scores 0.20–0.30)**  
   High overlap with head_only creates many redundant near-duplicate tiles. Since the backbone was frozen, the model likely couldn't handle highly overlapping crops well. With unfrozen blocks, this issue disappears.

3. **head_only + five_crop + max + top1 → 0.21709**  
   Single-species prediction with max aggregation over only 5 crops. Far too conservative and misses the multi-label nature of the task.

4. **sliding_ts224 runs for last_blocks (all below 0.34)**  
   Small 224px tiles from an 800px image lose context. The model was trained on 224px images (center-cropped from 256px resize), so 224px tiles should theoretically work, but the very small tiles from an 800px wide image miss too much context per tile.

5. **last_blocks_12 submissions (best only 0.37479)**  
   Counterintuitively, this checkpoint with the best validation score (top5=0.9400) scored lower than last_blocks_8 on Kaggle (best 0.39140). Only 3 configurations were submitted so this may not represent its true potential. Possible explanations: (a) only limited sweep was done, (b) the 12-block model may be slightly overfit to training distribution vs the test quadrat domain shift.

---

## Main Findings

### 1. Staged unfreezing clearly helps — up to last_blocks_8

Each stage of block unfreezing improved validation top-5: head_only (0.9214) → last_4 (0.9323) → last_8 (0.9364) → last_12 (0.9400). The improvement is consistent but diminishing. The Kaggle public score followed the same trend **up to last_blocks_8** (best 0.39140), but last_blocks_12 did not improve on Kaggle despite better val metrics.

### 2. Full finetune did not help — and was started from the wrong checkpoint

The full_finetune run started from `last_blocks/checkpoints/best.pt` (only 4 blocks unfrozen), not from `last_blocks_12`. Using a high learning rate (backbone_lr=5e-5) caused early epochs to degrade (val top5 dropped to 0.9043 at epoch 1 from a starting point of ~0.9323). The model recovered but peaked at 0.9339 (epoch 7), which is worse than last_blocks_12. Val loss was clearly increasing after epoch 7, indicating overfitting. **If full finetune is rerun, it should start from last_blocks_12 and use a lower backbone LR (1e-7 or 5e-7).**

### 3. Aggregation mode matters more than tiling strategy

`softmax_mean` beats `max` beats `mean` across all tile modes and checkpoints consistently. The effect is large: for head_only + grid_4x4_ov0.5, switching from mean to softmax_mean improves score from ~0.258 to ~0.340. This is because `mean` in logit space averages over uninformative tiles (background, blur), while `softmax_mean` (= mean in probability space) averages predictions that are already calibrated.

### 4. Zero overlap is best; more overlap hurts

For grid_4x4 with zero overlap, tiles are non-redundant and each covers 25% of the image. With overlap=0.25 or higher, tiles become correlated and the model processes duplicate patches. This probably adds noise to the aggregated prediction. The best scores always come from overlap=0.0.

### 5. Optimal top-k is around 3 (or gap~0.5)

Across all checkpoints, top-k=3 tends to score highest. top-k=2 underpredicts (missing species), top-k=4 or 5 overpredicts (too many false positives). Gap-based selection at ratio=0.5 adaptively selects top-3 for most images but adjusts when the score distribution warrants it — hence its marginal improvement over fixed top-3.

### 6. Multitask taxonomy loss appears to have helped regularisation

Val loss at the end of each stage kept decreasing with more blocks unfrozen without clear overfitting (except full_finetune). Given that only the species head is used during inference, the taxonomy heads likely improved the shared MLP representation quality by forcing the model to predict hierarchically consistent features. This is hard to isolate without an ablation.

### 7. Species-level overfitting vs quadrat domain gap

The training images are single-plant images, while the test images are full quadrat scenes. The best inference approach (grid_4x4 = 16 tiles of 448px) decomposes the quadrat into crops that more closely resemble training images. The fact that more tiles (grid_5x5 or grid_6x6) did not help suggests 4×4 already captures sufficient overlap with the training distribution.

---

## Next Experiments

1. **Re-run full_finetune from last_blocks_12** with backbone_lr=5e-7 or 1e-7. The current run started from a weaker checkpoint and used too high a backbone LR, causing early degradation.

2. **Submit last_blocks_12 with the full adaptive sweep** (all 13 selection modes). Only 3 of 26 inference runs were submitted. The checkpoint has the best val score and may outperform last_blocks_8 with the right inference config.

3. **Try adaptive top-k with max_k > 5** for last_blocks_12 (the `last_blocks_12_tile_sweep_maxk10` directory exists with max_k=10 but was not submitted). Already generated; just needs submission.

4. **Ensemble last_blocks_8 and last_blocks_12** logits (both saved as `.npz` in the `logits/` subdirectories). Averaging their probabilities before selecting top-k may recover from the domain gap seen in last_blocks_12.

5. **Calibrate thresholds per species frequency.** Rare species (few training examples) are likely underpredicted. Apply prior-adjusted logits: `logit_adj = logit - log(prior_frequency)`. This could benefit from the species frequency distribution in the training metadata.

6. **Sliding window with larger tiles (ts896 or ts1120).** The test images are ~800×600. Sliding with a tile that covers ~50% of the image per tile and no overlap could give better context than 16 small tiles.

7. **Add test-time augmentation (TTA).** The current inference uses a single orientation per tile. Horizontal flip + original = 2× the predictions for essentially free compute, possibly improving robustness.

8. **Pseudo-labeling for test quadrats.** Use confident predictions from the best model to augment training with unlabeled test data. Given 2105 test images, even weak supervision could help.

9. **Investigate why last_blocks_12 underperforms on Kaggle vs validation.** Check if the val split overlaps with test quadrat locations or conditions. If the val split is from the same single-plant images as train, the Kaggle quadrat images represent a harder domain shift that may penalise over-specialized backbone features.

10. **Train without taxonomy heads as ablation.** Add `--no-taxonomy-heads` flag and train an equivalent model to isolate the taxonomy loss contribution to the final Kaggle score.

