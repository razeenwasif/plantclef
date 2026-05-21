# Experiment Report: 009 — BioCLIP-2.5 ViT-H/14 PlantNet-Style Full Fine-Tune (FAILED)

**Most useful files for this report:**
- `README.md` — recipe (mirrors 008's), why this exists, hardware budget
- `REMOTE_CONTEXT.md` — handoff brief, healthy-trajectory expectations, monitoring commands
- `bioclip_model.py` — `BioCLIP25SinglePlantClassifier` (open_clip ViT-H/14 trunk + LayerNorm + Linear head) and CLIP-stat transforms
- `train_phase_a.py` — training script (mirrors 008's; head-only warmup → unfreeze → OneCycleLR + CE + label-smooth + logit-adjust)
- `dump_test_probs.py` — multi-scale tile-inference dump (mirrors 008's)
- `dump_grid4x4.py` — grid_NxN tile-inference dump matching Arjun's 010 winning recipe (grid_4x4, ov=0, softmax_mean, top-3)
- `npz_to_submission.py` — npz → PlantCLEF submission CSV (top-K via `argsort(-probs)` stable)
- Reuses `008_dinov3_plantnet_finetune/single_plant_dataset.py` and `004_bioclip_few_shot/dataset.py:load_species_ids` via `sys.path` shim

---

## Run: bioclip25_plantnet_v1

**Experiment ID:** 009
**Date:** Training launched 2026-04-26 02:20 UTC; entire run verdicted DEAD by 2026-04-27.

### Why this experiment exists

The 008 PhaseA × 004-frozen-BioCLIP-2.5 RRF ensemble cracked **0.34642** on Kaggle (α=0.70 top-3, +0.016 over Arjun's solo BioCLIP best of 0.33). The fusion gain came from two models making *different* errors. The frozen prototype-matcher leg was the bottleneck — replacing it with a *fine-tuned* BioCLIP-2.5 trained on the same 1.4M PC24 corpus that 008 used should turn the ensemble into "two strong species classifiers" instead of "one strong + one frozen prototype matcher", lifting the ceiling toward 0.36-0.38+.

### Model / Architecture

`BioCLIP25SinglePlantClassifier`:
- **Trunk:** BioCLIP-2.5 ViT-H/14 (`hf-hub:imageomics/bioclip-2.5-vith14`) loaded via `open_clip` — visual tower only
- **Head:** `LayerNorm(1280) → Linear(1280, 7806)`
- **Backbone params:** ~632M (~2× DINOv3-L's 300M)
- **Patch:** 14 (so img_size must be a multiple of 14)
- **Normalization:** BioCLIP/CLIP stats `(0.481, 0.458, 0.408) / (0.269, 0.261, 0.276)` — explicitly NOT ImageNet (REMOTE_CONTEXT: "Wrong stats here silently kill convergence")

### Dataset

- **Train CSV:** `/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv` — **1,381,785** PC24 single-plant rows, 7806 species (same as 008)
- **Train images:** `/workspace/plantclef/raw/train/images_max_side_800/`
- **Val:** 1% holdout (~13,800 single-plant images), 7806-way top-1
- **Species CSV:** `002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv`
- **Test set:** 2,105 quadrat images at `/workspace/plantclef/processed/test_images_jpeg85_max800/` (preprocessed by 008's `preprocess_test_quadrats.py`)
- **Pre-trained backbone cache:** `~/.cache/huggingface/hub/models--imageomics--bioclip-2.5-vith14/`

### Task

Full Phase-A fine-tune of BioCLIP-2.5 on PC24 single-plants (7806-way softmax CE), aimed at producing a stronger ensemble peer for fusion with 008 PhaseA.

### Training Recipe

- **Stage 0 (warmup, ep0):** freeze trunk, train `LayerNorm + Linear` head only for 1 epoch.
- **Stage 1 (main, ep1-8):** unfreeze trunk, two LR groups, OneCycleLR, AdamW.
- **Loss:** CE + `label_smoothing=0.1` + per-class logit adjustment (`τ · log(class_freq/total)`).
- **Aug:** RandomResizedCrop + HFlip + ColorJitter + RandomErasing.

### Hyperparameters / Run Config

| Parameter | Value |
|---|---|
| backbone | BioCLIP-2.5 ViT-H/14 (~632M) |
| img_size | 224 (multiple of 14) |
| batch_size (per-GPU) | 24 |
| accum | 1 |
| world_size | 2 (DDP, `torchrun --nproc_per_node=2`) → effective batch 48 |
| precision | bf16 |
| grad_checkpoint | True (~25% step slowdown, ~30% VRAM saving) |
| epochs | 8 main + 1 head-only warmup |
| lr-backbone | 5e-5 (main: max_lr 5e-5) |
| lr-head | 1e-3 (main: max_lr 1e-4 — divided by 10 to avoid overpowering 632M trunk) |
| weight_decay | 0.05 |
| optimizer | AdamW(β=(0.9,0.95)) |
| schedule | OneCycleLR |
| label_smoothing | 0.1 |
| val_frac | 0.01 |
| num_workers | 8 |
| ETA | ~30 min warmup + 8 × ~2.5 h ≈ **20 h total** |
| Process | `nohup setsid torchrun ... &` (NOT in tmux — find via `pgrep -af train_phase_a.py`) |
| Output dir | `/workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1/` |
| Training log | `/workspace/working/logs/bioclip25_p1.log` |
| Per-GPU VRAM | ~18-22 GB (with grad checkpointing) |

### Expected (REMOTE_CONTEXT) vs Realised Validation Trajectory

Expected val_top1 (1% single-plant holdout):
- ep1 ≥ 0.30 → ep3 ≥ 0.45 → ep5 ≥ 0.52 → ep8 ≥ 0.55-0.60 (target; PlantNet DINOv2 hit 0.76 with 75 ep)

### Inference Recipes Tried Post-Training

1. **Multi-scale tiled** via `dump_test_probs.py`:
   ```
   --tile-sizes 224 336 --tile-overlap 112 --hflip-tta --whole-image --bf16 --batch-size 32
   ```
   then npz → top-K submission.
2. **Grid_4x4 (Arjun's 010 winning recipe)** via `dump_grid4x4.py`:
   ```
   --grid-n 4 --img-size 224 --batch-size 64 --bf16
   ```
   Aggregation: `probs_mean` (= softmax_mean), top-3.

The grid_4x4 + softmax_mean + top-3 recipe was tried for 009 specifically because that exact recipe produced 010's team-best 0.38333 — see `dump_grid4x4.py` docstring.

### Submitted Checkpoints

Two checkpoints made it to Kaggle, both bad:

| Checkpoint | Submission | Kaggle ref | F1 |
|---|---|---|---|
| `phase_a_ep6.pth` | `submission_009_ep6_grid4x4_softmax_mean_top3.csv` | 52080441 | **0.20777** |
| `phase_a_best.pth` (likely ep1, file-size matches `last.pth`) | `submission_009_best_grid4x4_softmax_mean_top3.csv` | 52080949 | **0.20407** |

### Verdict

**ENTIRE RUN IS BAD — 009 PlantNet recipe DEAD.** Both checkpoints land in the *same* ~0.205 regime, so this is not "ep6 forgot what ep1 had" — the training recipe itself is broken end-to-end.

### Likely Root Cause (per the team memory)

> *"head-only ep1 warmup with random-init head + PlantNet data for ONE epoch is nowhere near converged on 7806-class CE; subsequent unfreeze with AdamW lr-bb=5e-5 OneCycleLR over a bad head poisons the backbone. Also possible the val-metric-tracker is selecting on the wrong metric (val acc on PlantNet ≠ PlantCLEF leaderboard)."*

In other words: 1 head-only warmup epoch on a 632M trunk is not enough to bring the random-init Linear head close to convergence on 7806-way CE before unfreezing. The full FT then drags the trunk away from BioCLIP-2.5's Tree-of-Life prior — exactly the "destroy the taxonomy prior" failure mode. Compare 010 (last_blocks fine-tune, only last 4 transformer blocks + ln_post + proj unfrozen, head warmup absorbed via 5 training epochs) → **0.38333**.

### Trajectory Map (across the BioCLIP-2.5 fine-tune family)

- 006 BioCLIP-2.5 frozen (Arjun): **0.33**
- 010 head-only fine-tune: **~0.33** (extrapolated)
- 010 last_blocks ep4 (unfreeze_n=4): **0.38333** ← team best
- 009 full FT ep6: **0.20777** ← collapsed
- 009 full FT best.pth: **0.20407** ← collapsed

→ "There's a sweet spot around 'last 4 blocks unfrozen' — going deeper destroys the taxonomy prior. **Full FT is dead.**"

### Status

**Retired.** Don't run more inference variants on 009. Use 010 last_blocks fine-tune as the anchor. Next moves: 010 sweeps over `unfreeze_n ∈ {3, 5, 6}`, different epoch counts, mosaics / multi-label aug. Even fusing 009 with 010 is unlikely to help since 009's signal is essentially a noisier subset of 010's.

### Key Takeaways

1. **Single-epoch head warmup is insufficient on a random-init head over 7806 classes** when the trunk is 632M params. The unfreeze step then poisons the pretrained representation.
2. **There is a sweet spot in unfreeze depth.** Last-4-blocks (010) > full FT (009). BioCLIP-2.5's Tree-of-Life prior is fragile; over-fine-tuning destroys it.
3. **Recipe parity does not imply outcome parity.** 009 deliberately mirrored 008 to make the ensemble "apples-to-apples" — but 008's DINOv3 trunk had no plant prior to lose, while BioCLIP-2.5 did, so the same recipe destroyed one model and not the other.
4. **`submission_*.csv` naming convention** (used by `scripts/submit_predictions_kaggle.sh`) — submissions matching `submission_009_*` were submitted from `/workspace/working/PlantCLEF2026/outputs/` after running `dump_grid4x4.py` + `npz_to_submission.py`.
