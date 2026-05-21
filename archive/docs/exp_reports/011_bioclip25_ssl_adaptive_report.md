# Experiment Report: 011 — BioCLIP 2.5 SSL Domain Adaptation + Adaptive Inference

**Experiment folder:** `src_experiments/011_bioclip25_ssl_adaptive/`  
**Date written:** 2026-04-29  
**Author:** Arjun  
**Builds on:** Experiment 010 (`010_bioclip25_end_to_end_finetune_multitask`)

---

## Most Useful Files

| File | Purpose |
|---|---|
| [README.md](README.md) | Full architecture, SSL pipeline, adaptive inference modes |
| [train_ssl.py](train_ssl.py) | SimSiam SSL pre-training script |
| [train.py](train.py) | Supervised fine-tuning (extended with `--ssl-backbone-checkpoint`) |
| [model.py](model.py) | `BioCLIP25SSL` with projector/predictor; `load_ssl_backbone()` |
| [infer_tiles.py](infer_tiles.py) | Extended with 4 adaptive selection modes |
| [scripts/train_ssl.sh](scripts/train_ssl.sh) | Runs SSL pre-training (20 epochs, pseudo-quadrats) |
| [scripts/train_from_ssl_stage1_stage2.sh](scripts/train_from_ssl_stage1_stage2.sh) | Runs Stage 1 + Stage 2 supervised from SSL backbone |
| [scripts/infer_best_adaptive.sh](scripts/infer_best_adaptive.sh) | Adaptive inference sweep |
| [outputs/ssl_bioclip25_simsiam_e20/ssl_metrics.csv](outputs/ssl_bioclip25_simsiam_e20/ssl_metrics.csv) | SSL training loss + z_std per epoch |
| [outputs/ssl_bioclip25_simsiam_e20/ssl_config.json](outputs/ssl_bioclip25_simsiam_e20/ssl_config.json) | SSL hyperparameters |
| [outputs/ssl_head_only/metrics.csv](outputs/ssl_head_only/metrics.csv) | Supervised Stage 1 training metrics |
| [outputs/ssl_last_blocks/metrics.csv](outputs/ssl_last_blocks/metrics.csv) | Supervised Stage 2 training metrics |
| [scores_011_exp_best_infer.csv](scores_011_exp_best_infer.csv) | All 23 Kaggle public scores |

---

## High-Level Summary

This experiment adds two orthogonal improvements on top of experiment 010:

1. **SimSiam SSL pre-training** on 212,762 unlabeled pseudo-quadrat images (LUCAS survey) to adapt the BioCLIP 2.5 backbone to the PlantCLEF test image distribution before supervised fine-tuning.
2. **Adaptive inference** with variable-length species selection (gap detection, probability threshold, relative threshold) as a cleaner replacement for fixed top-k.

The core hypothesis is that the training images (single-plant photos) differ substantially from the test quadrat images (wide-angle field photos), and SSL domain adaptation on the test distribution should close this gap before supervised training.

**Key finding: SSL pre-training did not help and likely hurt.** The SSL-warm-started head-only checkpoint achieved Kaggle score 0.31523 (best) vs 0.36197 for the equivalent 010 head-only checkpoint — a drop of ~0.047. The last-blocks checkpoint also scored lower (0.37715 vs 0.39140 for 010's best). Only 23 total submissions were made.

---

## Dataset and Task

Same as Experiment 010:
- **Competition:** PlantCLEF 2026 — multi-label species prediction in quadrat images
- **Labeled training data:** 1,408,033 images, 7,806 species
- **Val split:** 137,824 images (stratified 10%)
- **Test set:** 2,105 quadrat images

**Additional SSL data:**

| Path | Images | Source |
|---|---|---|
| `/workspace/plantclef/raw/pseudo_quadrats` | 212,762 | Unlabeled LUCAS field survey images |

---

## Model / Architecture

### Supervised model (same as 010)

```
Input image (224×224)
       │
BioCLIP 2.5 ViT-H/14 backbone   (embed_dim=1024)
       │
LayerNorm → Linear(1024→1024) → GELU → Dropout(0.2)   ← shared MLP
       │
   ┌───┴───┬───────┬────────┬──────────┐
species  genus  family   order    class
(7806)  (1446)  (181)    (61)      (6)
```

Note: README.md says `1280-dim embedding`, but the actual run log reports `embed_dim=1024`, consistent with experiment 010. The 1280 likely refers to ViT-H's native patch dimension before projection.

### SSL model (new in 011)

```
Unlabeled quadrat image
       │
  ┌────┴────┐
 aug₁     aug₂    ← two independent strong augmentations
  │         │
BioCLIP 2.5 ViT-H/14 backbone  (last 4 blocks + ln_post unfrozen)
  │         │
projector  projector   ← 3-layer MLP: 1024 → 2048 → 2048 → 256 (BN, no-affine last layer)
  │         │
predictor  predictor   ← 2-layer MLP: 256 → 512 → 256 (BN)
 (p₁,z₁)  (p₂,z₂)
       │
SimSiam loss = −cos(p₁, sg(z₂)) − cos(p₂, sg(z₁))   ← stop-gradient on z
```

SimSiam prevents collapse without a momentum encoder. Collapse is monitored via `z_std` (std of projection vectors across the batch).

### SSL → Supervised warm-start

After SSL, `backbone.pt` is loaded into `BioCLIP25MultiTask` with `strict=False`. The projector and predictor are discarded. The supervised head is always randomly initialized.

---

## Training Pipeline

### Step 1: SSL Pre-training

**Config:** [outputs/ssl_bioclip25_simsiam_e20/ssl_config.json](outputs/ssl_bioclip25_simsiam_e20/ssl_config.json)

| Parameter | Value |
|---|---|
| Data | `/workspace/plantclef/raw/pseudo_quadrats` (212,762 images) |
| Backbone | BioCLIP 2.5 ViT-H/14 |
| Unfrozen blocks | Last 4 of 32 + ln_post/proj |
| Backbone params trained | 80,023,040 |
| Projector | 1024 → 2048 → 2048 → 256 (BN) |
| Predictor | 256 → 512 → 256 (BN) |
| Loss | SimSiam (negative cosine similarity) |
| Epochs | 20 |
| Warmup epochs | 2 |
| Batch size | 128 (single GPU) |
| Backbone LR | 1e-6 |
| Head (proj/pred) LR | 1e-4 |
| Weight decay | 0.05 |
| Precision | bf16 |
| Batches/epoch | 1,662 |
| Total steps | 33,240 |
| Epoch time | ~1,261s (~21 min) |
| Total training time | ~423.6 min (~7.1 hours) |
| Output | [outputs/ssl_bioclip25_simsiam_e20/](outputs/ssl_bioclip25_simsiam_e20/) |
| Backbone checkpoint | [outputs/ssl_bioclip25_simsiam_e20/checkpoints/backbone.pt](outputs/ssl_bioclip25_simsiam_e20/checkpoints/backbone.pt) |

**Per-epoch SSL metrics** ([outputs/ssl_bioclip25_simsiam_e20/ssl_metrics.csv](outputs/ssl_bioclip25_simsiam_e20/ssl_metrics.csv)):

| Epoch | SSL Loss | z_std | LR (head) |
|---|---|---|---|
| 1 | -0.6190 | 1.0039 | 5e-05 (warmup) |
| 2 | -0.8655 | 1.0039 | 1e-04 (peak) |
| 5 | -0.8768 | 1.0039 | 9.33e-05 |
| 10 | -0.8995 | 1.0039 | 5.87e-05 |
| 15 | -0.9735 | 1.0039 | 1.79e-05 |
| 20 | **-0.9761** | 1.0039 | 1e-06 |

**Notes:**
- z_std was stable at ~1.004 throughout — no collapse at any point.
- SSL loss improved steadily from -0.619 to -0.976 and appeared to plateau around epochs 16–17 before cosine decay brought the LR to minimum.
- The SSL loss range is [-1, 0] by definition (negative cosine); -0.976 means the predicted and target projections were ~97.6% aligned, indicating good representation learning.
- Best SSL loss: **-0.9761** (epoch 20, also the final epoch).

---

### Step 2: Supervised Head-Only from SSL Backbone (`ssl_head_only`)

**Config:** [outputs/ssl_head_only/train_config.json](outputs/ssl_head_only/train_config.json)

| Parameter | Value |
|---|---|
| SSL backbone | [outputs/ssl_bioclip25_simsiam_e20/checkpoints/backbone.pt](outputs/ssl_bioclip25_simsiam_e20/checkpoints/backbone.pt) |
| Frozen/Unfrozen | Backbone frozen, head + MLP only |
| head_lr | 1e-4 |
| batch_size | 512 per GPU × 2 GPUs = 1024 |
| grad_accum | 2 (effective batch = 2048) |
| epochs | 10 |
| epoch time | ~1507s (~25 min) |
| total train time | ~309.5 min |
| output | [outputs/ssl_head_only/](outputs/ssl_head_only/) |
| best checkpoint | [outputs/ssl_head_only/checkpoints/best.pt](outputs/ssl_head_only/checkpoints/best.pt) |

**Per-epoch val metrics** ([outputs/ssl_head_only/metrics.csv](outputs/ssl_head_only/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 10.093 | 4.118 | 0.3786 | 0.6338 | 0.5908 | 0.7471 |
| 3 | 4.767 | 1.569 | 0.6584 | 0.8800 | 0.8175 | 0.8687 |
| 5 | 4.256 | 1.422 | 0.6903 | 0.8996 | 0.8389 | 0.8850 |
| 7 | 4.085 | 1.370 | 0.7015 | 0.9053 | 0.8461 | 0.8909 |
| 9 | 4.024 | 1.355 | 0.7045 | 0.9072 | 0.8477 | 0.8922 |
| **10** | **4.014** | **1.354** | **0.7044** | **0.9073** | **0.8480** | **0.8922** |

**Best val top-5: 0.9073** — significantly lower than 010's equivalent head_only (0.9214). This is the main warning sign.

**Comparison with 010 head_only:**

| Epoch 1 | 010 head_only | 011 ssl_head_only |
|---|---|---|
| Train loss | 9.133 | 10.093 |
| Val loss | 2.504 | 4.118 |
| Top-5 acc | 0.8223 | 0.6338 |

The SSL backbone starts from a much worse supervised baseline. The representation geometry from SimSiam appears to be less linearly separable for species classification when the backbone is frozen.

---

### Step 3: Supervised Last-8-Blocks from SSL Head (`ssl_last_blocks`)

**Config:** [outputs/ssl_last_blocks/train_config.json](outputs/ssl_last_blocks/train_config.json)

| Parameter | Value |
|---|---|
| SSL backbone reload | None (ssl_backbone_checkpoint=null) |
| Resume from | [outputs/ssl_head_only/checkpoints/best.pt](outputs/ssl_head_only/checkpoints/best.pt) |
| Frozen/Unfrozen | Last **8** transformer blocks unfrozen |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 |
| batch_size | 64 per GPU × 2 = 128 |
| grad_accum | 4 (effective batch = 512) |
| epochs | **10** (vs 5 in 010) |
| epoch time | ~2512s (~42 min) |
| total train time | ~484.2 min (~8.1 hours) |
| output | [outputs/ssl_last_blocks/](outputs/ssl_last_blocks/) |
| best checkpoint | [outputs/ssl_last_blocks/checkpoints/best.pt](outputs/ssl_last_blocks/checkpoints/best.pt) |


**Per-epoch val metrics** ([outputs/ssl_last_blocks/metrics.csv](outputs/ssl_last_blocks/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 3.719 | 1.231 | 0.7167 | 0.9180 | 0.8734 | 0.9295 |
| 3 | 3.303 | 1.113 | 0.7418 | 0.9303 | 0.8866 | 0.9389 |
| 5 | 3.147 | 1.067 | 0.7524 | 0.9348 | 0.8915 | 0.9419 |
| 7 | 3.062 | 1.041 | 0.7569 | 0.9373 | 0.8935 | 0.9434 |
| 9 | 3.022 | 1.034 | 0.7587 | 0.9379 | 0.8940 | 0.9438 |
| **10** | **3.013** | **1.032** | **0.7590** | **0.9380** | **0.8941** | **0.9439** |

Best val top-5: **0.9380** — only slightly lower than 010's last_blocks_12 (0.9400), and a +0.0016 improvement over 010's last_blocks_8 (0.9364). However, 011 ran 10 epochs vs 010's 5 epochs for the 8-block stage, so this modest improvement may just reflect the extra training time.

---

## Training Summary — 011 vs 010 Comparison

| Run | Exp | Blocks Unfrozen | Epochs | Best Val Top-5 | Δ vs 010 equivalent |
|---|---|---|---|---|---|
| head_only (010) | 010 | 0 | 10 | 0.9214 | — |
| ssl_head_only (011) | 011 | 0 | 10 | **0.9073** | **−0.0141** ← SSL hurt |
| last_blocks_8 (010) | 010 | 8 | 5 | 0.9364 | — |
| ssl_last_blocks (011) | 011 | 8 | 10 | **0.9380** | **+0.0016** ← marginal gain |

---

## Inference

All inference used `grid_4x4`, `tile_size=448`, `overlap=0.0`, `agg=softmax_mean`, `precision=bf16` (adaptive runs) or `fp32` (fixed-k runs), `min_k=2`, `max_k=5`.

### Inference Family 1: ssl_head_only Fixed-K Sweep (`tile_sweeps/head_only/`)

**Checkpoint:** [outputs/ssl_head_only/checkpoints/best.pt](outputs/ssl_head_only/checkpoints/best.pt)  
**Script:** [scripts/infer_tiles.sh](scripts/infer_tiles.sh)  
**Config ref:** [outputs/tile_sweeps/head_only/grid_4x4_ts448_ov0p0/softmax_mean_top3/run_config.json](outputs/tile_sweeps/head_only/grid_4x4_ts448_ov0p0/softmax_mean_top3/run_config.json)

| Agg | Selection | Public Score |
|---|---|---|
| max | top2 | **0.30004** |
| softmax_mean | top4 | 0.29046 |
| softmax_mean | top3 | 0.29024 |
| softmax_mean | top2 | 0.28359 |
| max | top3 | 0.27961 |
| max | top4 | 0.27004 |

Best: **0.30004** (max + top2). Note: `max` beats `softmax_mean` here, which is unusual. This may indicate the SSL backbone's probability outputs are less well-calibrated.

---

### Inference Family 2: ssl_head_only Adaptive Sweep (`ssl_adaptive_infer_head_only/`)

**Checkpoint:** [outputs/ssl_head_only/checkpoints/best.pt](outputs/ssl_head_only/checkpoints/best.pt)  
**Script:** [scripts/infer_best_adaptive.sh](scripts/infer_best_adaptive.sh)  
**Config ref:** [outputs/ssl_adaptive_infer_head_only/softmax_mean_gap0.5/run_config.json](outputs/ssl_adaptive_infer_head_only/softmax_mean_gap0.5/run_config.json)

| Selection Mode | Param | Public Score |
|---|---|---|
| gap | 0.6 | **0.31523** |
| gap | 0.5 | 0.30836 |
| gap | 0.4 | 0.30420 |
| prob_threshold | 0.05 | 0.29349 |
| softmax_mean | top4 | 0.28498 |
| prob_threshold | 0.03 | 0.28880 |
| relative_threshold | 0.25 | 0.28737 |
| relative_threshold | 0.30 | 0.28867 |
| relative_threshold | 0.15 | 0.28721 |
| relative_threshold | 0.20 | 0.28721 |
| softmax_mean | top3 | 0.28359 |
| softmax_mean | top2 | 0.28274 |
| prob_threshold | 0.02 | 0.27906 |

Best: **0.31523** (gap0.6). The gap selection consistently outperforms fixed-k and threshold methods for this checkpoint.

---

### Inference Family 3: ssl_last_blocks Adaptive Sweep (`ssl_last_blocks_adaptive_infer/`)

**Checkpoint:** [outputs/ssl_last_blocks/checkpoints/best.pt](outputs/ssl_last_blocks/checkpoints/best.pt)  
**Config ref:** [outputs/ssl_last_blocks_adaptive_infer/softmax_mean_gap0.6/run_config.json](outputs/ssl_last_blocks_adaptive_infer/softmax_mean_gap0.6/run_config.json)

Only 3 of 13 adaptive configurations were submitted to Kaggle:

| Selection Mode | Param | Public Score |
|---|---|---|
| gap | 0.6 | **0.37715** |
| fixed_topk | top3 | 0.36864 |
| gap | 0.5 | 0.36818 |

Best: **0.37715** (gap0.6). The remaining 10 configurations (probT, relT, other gaps, top2/top4) have submission CSVs but were not submitted to Kaggle.

---

## Kaggle Submission Results — All 23 Runs

### Complete Ranked Table

| Rank | Checkpoint | Sweep | Selection | Public Score |
|---|---|---|---|---|
| 1 | ssl_last_blocks | adaptive | gap 0.6 | **0.37715** |
| 2 | ssl_last_blocks | adaptive | fixed top3 | 0.36864 |
| 3 | ssl_last_blocks | adaptive | gap 0.5 | 0.36818 |
| 4 | ssl_head_only | adaptive | gap 0.6 | 0.31523 |
| 5 | ssl_head_only | adaptive | gap 0.5 | 0.30836 |
| 6 | ssl_head_only | fixed-k | max top2 | 0.30004 |
| 7 | ssl_head_only | adaptive | gap 0.4 | 0.30420 |
| 8 | ssl_head_only | adaptive | prob T 0.05 | 0.29349 |
| 9 | ssl_head_only | adaptive | fixed top4 | 0.28498 |
| 10 | ssl_head_only | adaptive | rel T 0.30 | 0.28867 |
| 11 | ssl_head_only | adaptive | prob T 0.03 | 0.28880 |
| 12 | ssl_head_only | adaptive | rel T 0.25 | 0.28737 |
| 13 | ssl_head_only | adaptive | rel T 0.15 | 0.28721 |
| 14 | ssl_head_only | adaptive | rel T 0.20 | 0.28721 |
| 15 | ssl_head_only | adaptive | fixed top3 | 0.28359 |
| 16 | ssl_head_only | fixed-k | softmax_mean top4 | 0.29046 |
| 17 | ssl_head_only | fixed-k | softmax_mean top3 | 0.29024 |
| 18 | ssl_head_only | fixed-k | softmax_mean top2 | 0.28359 |
| 19 | ssl_head_only | adaptive | fixed top2 | 0.28274 |
| 20 | ssl_head_only | fixed-k | max top3 | 0.27961 |
| 21 | ssl_head_only | adaptive | prob T 0.02 | 0.27906 |
| 22 | ssl_head_only | fixed-k | max top4 | 0.27004 |

### Comparison with Experiment 010 Best Scores

| Metric | 010 Best | 011 Best | Delta |
|---|---|---|---|
| Head-only best Kaggle | 0.36197 | 0.31523 | **−0.047** |
| Last-blocks best Kaggle | 0.39140 | 0.37715 | **−0.014** |
| Best val top-5 (head-only) | 0.9214 | 0.9073 | −0.014 |
| Best val top-5 (last-blocks) | 0.9364 | 0.9380 | +0.002 |

The SSL pre-training strategy did not improve performance at the Kaggle level.

---

## Main Findings

### 1. SSL pre-training hurt the head-only stage significantly

The ssl_head_only checkpoint's best Kaggle score was 0.31523, compared to 0.36197 for the equivalent 010 head_only. This is a large drop (~0.047). The val top-5 also dropped from 0.9214 to 0.9073.

The root cause is visible in epoch 1 of ssl_head_only training: val top-5 was only 0.6338 (vs 0.8223 for 010 head_only). The SSL-adapted backbone changed the representation space so that a freshly initialized linear classification head could not fit it easily. The backbone was in a "contrastive representation" regime rather than a "classification-ready" regime.

Over 10 epochs the gap narrowed (final top-5 = 0.9073 vs 0.9214) but never closed. This is consistent with findings in the SSL literature: SimSiam representations need some supervised fine-tuning of the encoder itself (i.e., unfreezing blocks) to become competitive with purely supervised training, even when the backbone is domain-adapted.

### 2. SSL warm-start plus block unfreezing partially recovered

After unfreezing 8 blocks in ssl_last_blocks (10 epochs), validation top-5 reached 0.9380, which is marginally better than 010's 5-epoch last_blocks_8 (0.9364). The extra 5 epochs likely account for most of this difference rather than the SSL pre-adaptation itself. The Kaggle score (0.37715) remained lower than 010's equivalent (0.39140), suggesting the SSL backbone change persisted in some domain-relevant features.

### 3. SimSiam did not collapse — but may have over-regularised the backbone

The z_std was stable at ~1.004 throughout all 20 epochs (healthy range: > 0.1). Loss decreased smoothly from -0.619 to -0.976. The SSL run was technically successful: representations learned, collapse was avoided. But a z_std of 1.004 (close to the maximum possible for unit-normalised vectors) suggests the projections were already near-orthogonal — the backbone may not have been changed as much as intended by the very low backbone_lr (1e-6). Most of the adaptation likely happened in the projector/predictor (lr=1e-4).

### 4. Only 20 epochs of SSL on 212K images may not be enough

SimSiam and other contrastive methods typically need much more data or more epochs to materially shift the backbone. 20 epochs × 212K images = ~4.25M image passes. For a ViT-H with mostly frozen blocks, the backbone adaptation was limited to only 4 of 32 blocks at 1e-6 learning rate. The effective backbone update was very small, making the SSL pre-training a minimal perturbation to the original BioCLIP 2.5 weights.

### 5. gap selection is robust across both checkpoints

Both the ssl_head_only and ssl_last_blocks checkpoints perform best with `gap` selection. The gap ratio 0.5–0.6 range gives the best Kaggle scores consistently. This validates the design choice: adaptive selection is better than fixed top-k because different quadrats have different species densities.

### 6. ssl_last_blocks had limited Kaggle evaluation

Only 3 of 13 adaptive configurations were submitted for ssl_last_blocks. The full sweep might reveal better scores, but even so, the best possible combination is unlikely to exceed 010's last_blocks_8 best of 0.39140 given that the validated configurations are already lower.

### 7. Script label inconsistency

The script `train_from_ssl_stage1_stage2.sh` has a comment "Last 4 Blocks" for Stage 2 but actually runs `--unfreeze-last-n-blocks 8`. This discrepancy is harmless (the actual run used 8 blocks as confirmed by both the log and train_config.json) but could cause confusion when re-running.

---

## Failed / Weak Runs

1. **ssl_head_only fixed-k + max aggregation (0.27–0.30)**  
   Worst runs overall for this experiment. Max aggregation with the SSL backbone is particularly weak, scoring even lower than softmax_mean. The SSL backbone's softmax distribution appears more informative than the raw logit maxima.

2. **ssl_head_only + prob_threshold 0.02 → 0.27906**  
   The strictest probability threshold produces the lowest adaptive scores. The SSL-adapted model's probabilities are lower overall (the head was harder to train), so aggressive thresholding cuts off valid species.

3. **All ssl_head_only runs vs 010 head_only runs**  
   Every single submission from ssl_head_only (0.27–0.31) scores below every 010 head_only submission that used grid_4x4 + softmax_mean (0.34–0.36). The SSL backbone change harmed classification performance too severely to recover with a frozen backbone.

---

## Next Experiments

1. **SSL with more backbone blocks unfrozen (8 or 12).**  The current SSL only unfroze 4 blocks. Using 8–12 blocks in the SSL stage would give the backbone more capacity to adapt to pseudo-quadrat statistics. Trade-off: larger LR batch or more warmup may be needed to avoid SimSiam collapse.

2. **Longer SSL training (50–100 epochs) or larger batch.**  20 epochs is likely too short for a contrastive method to materially change a pre-trained ViT-H. Try 50+ epochs or doubling batch size to 256.

3. **Try MoCo v3 or DINO instead of SimSiam.**  DINO and DINOv2 are significantly better domain adaptation methods for ViTs than SimSiam. The momentum encoder in MoCo provides more stable training. DINO's self-distillation with multi-crop is particularly well-suited to quadrat images.

4. **Use the SSL backbone as a feature extractor for re-ranking**, not as a replacement for the supervised backbone. The SSL logits could be used as a second signal for ensemble voting rather than replacing the supervised model.

5. **SSL on the training plant images alongside pseudo-quadrats.**  SimSiam over a mixture of labeled training images and unlabeled quadrats (ignoring labels for SSL, using labels for supervised loss) could adapt the backbone without losing classification structure.

6. **Submit the full ssl_last_blocks adaptive sweep to Kaggle.**  The remaining 10 configurations (probT, relT, gap0.4, top2, top4) have submission CSVs ready at `outputs/ssl_last_blocks_adaptive_infer/` but were not submitted. This is the cheapest next step.

7. **Do a controlled ablation: train experiment 010 style (no SSL) from the ssl_last_blocks checkpoint.**  Specifically: take ssl_last_blocks/checkpoints/best.pt and continue with more block unfreezing (12 blocks, then full finetune). This isolates whether the SSL initialization helps or hurts for deeper unfreeze stages.

---

## Notes on Missing Data

- **10 of 13 ssl_last_blocks inference configurations were not submitted.** Submission CSVs exist at `outputs/ssl_last_blocks_adaptive_infer/` for all modes; only 3 were scored.
- **No fixed-k tile sweep for ssl_last_blocks** (unlike 010 which swept grid sizes, overlaps, and aggregations). Only adaptive inference was run on the ssl_last_blocks checkpoint.
- **No deeper unfreezing stages for 011.** Unlike 010 which tested last_blocks_4 → last_blocks_8 → last_blocks_12 → full, experiment 011 stopped at 8 blocks unfrozen.
- **SSL dataset size not confirmed from a second source.** 212,762 images is taken from the run log; the actual folder was not independently counted.
