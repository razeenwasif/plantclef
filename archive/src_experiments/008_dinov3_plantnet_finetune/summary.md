# Experiment Report: 008 — DINOv3-L PlantNet-Style Full Fine-Tune (Phase A) + Collage LoRA (Phase B)

**Most useful files for this report:**
- `README.md` — recipe, hardware budget, full multi-scale + ExG + BMA pipeline
- `REMOTE_CONTEXT.md` — handoff brief for the v2 retrain (root-cause of the 0.0002 v1 collapse + healthy v2 trajectory)
- `model.py` — `DINOv3SinglePlantClassifier` (CLS + LayerNorm + `Linear(1024, 7806)`); re-exports 005's `DINOv3MultiLabel` for Phase B
- `single_plant_dataset.py` — CSV-backed `(image, class_idx)` dataset
- `train_phase_a.py` — head-only warmup → full FT, OneCycleLR, CE + logit-adjust, bf16, two LR groups, DDP-aware
- `train_phase_b.py` — loads Phase-A backbone into `DINOv3MultiLabel`, applies LoRA, ASL on 50k labelled collages
- `dump_test_probs.py` / `dump_test_probs_phase_a.py` — tiled multi-label inference → fp16 probs (max/mean/noisy_or). Multi-scale, hflip-TTA, ExG vegetation filter
- `preprocess_test_quadrats.py` — Lanczos resize + JPEG-85 recompression to match PC24 byte distribution (TheHeartOfNoise 2025 winner recipe)
- `vegetation_filter.py` — Excess-Green per-tile plant-coverage filter
- `enhance_probs.py` — post-hoc logit adjustment + BMA + dynamic-threshold search
- `fuse_phase_a_bioclip.py` — late RRF/direct fusion of PhaseA probs × BioCLIP scores
- `outputs/fuse/`, `outputs/fusion_lb/`, `outputs/fusion_lb_direct/` — fusion sweep submissions
- `outputs/submission_phase_a_*top{2,3,4}_max.csv` — 008 PhaseA-only submissions (multi-scale, overlap-112, 3-scale)
- `outputs/submission_pbv3_*.csv` — Phase B v3 (LUCAS pseudo-labels) submissions
- `outputs/submission_q70/q75_top{2,3,4}_max.csv` — quantile-thresholded PhaseA submissions

---

## Run: dinov3_plantnet_v2_retrain (Phase A) + Phase B variants

**Experiment ID:** 008
**Date:** April 2026 (v1 launched ~04-15, v2 retrain launched 2026-04-22 20:20 UTC)

### Why this experiment exists

005 DINOv3 zero-shot scored **0.13** on Kaggle (well below 006 BioCLIP-2.5 at 0.33). The diagnosis: **DINOv3-L's LVD-1689M pretraining has no plant-specific signal**. PlantNet released DINOv2-B after weeks of GPU time fine-tuning on the 1.4M PC24 corpus and hit 76.16% PC24 top-1; 007 PlantNet-DINOv2 + collage LoRA used that backbone and scored **0.17476**. 008 reproduces the PlantNet recipe but on the strictly-more-expressive DINOv3-L.

### Model / Architecture

**Phase A** — `DINOv3SinglePlantClassifier`:
- Backbone: DINOv3 ViT-L/16 (1024-d), 4 register tokens
- Head: `LayerNorm(1024) → Linear(1024, 7806)` softmax (single-label)

**Phase B** — re-exports 005's `DINOv3MultiLabel`:
- Same backbone, weights loaded from Phase A's best checkpoint
- Multi-label head (CLS + GeM + 2-layer MLP, sigmoid on 7806 classes)
- LoRA r=32, α=64, dropout=0.05 on the backbone

### Dataset

- **Phase A train:** `train_metadata_cleaned_verified_stratified.csv` — **1,381,785** PC24 single-plant rows, 7806 species
- **Phase A images:** `/workspace/plantclef/raw/train/images_max_side_800/`
- **Phase A val:** 1% holdout (~13,800 single-plant images), 7806-way top-1
- **Phase B train (v1/v2):** `synthetic_collages.csv` — 50k labelled K-plant mosaics
- **Phase B v3 train:** LUCAS pseudo-labels (Phase A self-distillation, val_f1@0.5 = 0.6114 in-domain — failed to transfer)
- **Test set:** 2,105 quadrat images, preprocessed via `preprocess_test_quadrats.py` to `test_images_jpeg85_max800/`

### Phase A — Training Recipe

PlantNet-style: `head-only warmup → unfreeze full backbone → OneCycleLR + AdamW + CE + logit adjustment`. v2 retrain ran on **2× RTX 5090 DDP** in tmux session `phase_a_v2`, output to `models/dinov3_plantnet_v2_retrain/`.

### Hyperparameters / Run Config (Phase A v2 retrain)

| Parameter | Value |
|---|---|
| backbone | DINOv3 ViT-L/16 (300M params) |
| img_size | 224 (multiple of 16) |
| batch_size (per-GPU) | 48 |
| accum | 1 |
| world_size | 2 (DDP) → effective batch 96 |
| precision | bf16 |
| epochs | 12 main + 2 warmup |
| lr-backbone | 5e-5 (main: max_lr 5e-5) |
| lr-head | 1e-3 (main: max_lr 1e-4 — divided by 10 to avoid overpowering backbone) |
| optimizer | AdamW(β=(0.9,0.95), wd=0.05) |
| schedule | OneCycleLR |
| loss | CE + logit adjustment (`τ · log(class_freq/total)`) |
| augmentation | RandomResizedCrop + HFlip + ColorJitter + RandomErasing |
| num_workers | 10 |
| ETA | ~48 min/epoch on 2× 5090 → ~11 h total |

Per-GPU VRAM @ batch 48, 224 px, bf16: ~15 GB of 32 GB.

### Why v1 failed (the 0.0002 leaderboard score)

v1 ran one useful epoch before the LR schedule destabilised the pretrain (ep1 val 0.7342 → ep4 0.6823). Phase B then over-fitted to collage visuals, producing:

| submission | τ | config | Kaggle F1 |
|------------|---|--------|-----------|
| `submission_v1_tau0.25` | 0.25 | multi-scale + hflip + ExG + BMA + logit-adj | **0.00026** |
| `submission_v1_tau0.2803` | 0.2803 | same (dynamic-τ) | 0.00015 |
| `submission_v1_tau0.30` | 0.30 | same | 0.00016 |
| `submission_v3_NOTILE_tau0.30` | 0.30 | full-image CenterCrop(448), no enhance | **0.00255** |

Diagnosis (REMOTE_CONTEXT.md §"Root-cause"): species_ids order, weight loading, val_f1, and tiling pipeline all verified correct. Actual cause was **collage → real-quadrat domain shift** with a backbone that hadn't internalised plant-ID priors. The fix was the v2 retrain with gentler LRs and 12 main epochs.

### Phase A v2 — Expected vs Realised Trajectory

Expected (REMOTE_CONTEXT.md):
- ep2: ≥ 0.45 → ep4: ≥ 0.55 → ep6: ≥ 0.60 → ep8: ≥ 0.65 → ep12: ≥ 0.70

### Phase B Variants

1. **Phase B v1 (LoRA on collages):** broken end-to-end (0.0002).
2. **Phase B v2 LUCAS collages:** **0.036** Kaggle (failed).
3. **Phase B v3 LUCAS pseudo-labels:** in-domain `val_f1@0.5 = 0.6114` — **failed to transfer**, best Kaggle **0.227**. Self-distillation via Phase A pseudo-labels is a dead end on this domain shift.

### Phase A Tiled Inference (Best Standalone 008 Result)

Recipe: `--tile-sizes 336 448 560` (multi-scale) or `--tile-sizes 448` with `--tile-overlap 112`, `--hflip-tta`, `--whole-image`, ExG `--min-vegetation-frac 0.15`. Aggregation: `probs_max` top-K via `npz_to_submission.py`.

- **`submission_phase_a_multiscale_top3_max.csv`** — best single-config PhaseA Kaggle: **0.305** (top-3 max aggregation)
- Multi-scale + hflip-TTA + whole-image was the best PhaseA-only configuration found.

### Fusion Variants (output dirs)

**`outputs/fuse/`** — late RRF fusion of PhaseA × Arjun's frozen BioCLIP-2.5 (006):

| α (PhaseA weight) | top-K | result |
|---|---|---|
| 0.30 | 3 | submitted |
| 0.50 | 3 / 4 | submitted |
| **0.70** | **3** | **0.34642** ← prior team best (+0.016 over Arjun's 0.33 solo) |
| 0.65 | 3 | submitted |
| 0.75 | 3 | submitted |
| 0.85 / 0.90 / 0.95 | 3 | submitted |

**`outputs/fusion_lb/`** — PhaseA × 010 head-only BioCLIP-2.5 RRF (k=60), grid_4x4: α=0.10/0.20/0.30 top-3 swept; peaked at **α=0.65 → 0.34671** (PRIOR best, beaten by 010 last_blocks alone).

**`outputs/fusion_lb_direct/`** — direct prob-mix at α=0.20 top-3 → **0.36410** (worse than RRF — PhaseA's confident-but-wrong tiles win the argmax).

### Quantile-threshold submissions

`submission_q70_top3_max.csv`, `submission_q75_top{2,3,4}_max.csv` — selected by per-row quantile rather than fixed threshold.

### Results Summary (Kaggle Public F1)

| Submission family | Best Kaggle | Notes |
|---|---|---|
| PhaseA tile inference (multi-scale + hflip-TTA + whole-image, top-3 max) | **0.305** | Best 008-only score |
| PhaseA × frozen BioCLIP-2.5 RRF (α=0.70, top-3, RRF-k=60) | **0.34642** | Prior team best, beaten by 010 |
| PhaseA × 010 head-only RRF (α=0.65, top-3) | **0.34671** | Now beaten by 010 last_blocks alone (0.38333) |
| PhaseA × 010 last_blocks RRF (any α) | ≤ 0.37464 | All BELOW 010-alone 0.38333 |
| PhaseA × 010 last_blocks DIRECT prob-mix (α=0.20) | 0.36410 | Worse than RRF |
| Phase B v3 LUCAS pseudo-labels (best) | 0.227 | Failed to transfer |
| Phase B v2 LUCAS collages | 0.036 | Failed |
| Phase B v1 (collage LoRA) | 0.00255 / 0.00026 | Broken — Phase A only ran 1 useful epoch |

### Status / Verdict

**Phase A retired from the ensemble.** Per the team-best memo: "DINOv3 / PhaseA RETIRED — fusion hurts at every α, every fusion mechanism." Pure 010-last_blocks (no fusion) outscores every PhaseA-fusion configuration. The 008 PhaseA dump stays on disk for sanity, but no new fusions to try.

### Key Takeaways

1. **Tile + post-hoc logit adjustment + BMA hurt this task.** v1's full enhance pipeline scored 13× *worse* than no-tile CenterCrop(448).
2. **JPEG byte distribution matters.** `preprocess_test_quadrats.py` (Lanczos + JPEG-85) is part of the PlantNet winner recipe — the model has internalised PC24's compression artifacts.
3. **In-domain `val_f1` is not a Kaggle predictor under domain shift.** Phase B v3's 0.6114 val_f1 collapsed to 0.227 on the leaderboard.
4. **Fusion ceiling is bounded by the weaker leg's noise.** A 0.305 PhaseA leg cannot lift a 0.38333 010 leg — RRF and direct mix both regress.
5. The `Infastructure/` directory misspelling is intentional — paths in the README and REMOTE_CONTEXT depend on it.
