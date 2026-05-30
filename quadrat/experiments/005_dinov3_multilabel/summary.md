# Experiment Report: 005 — DINOv3 ViT-L/16 Multi-Label Two-Phase Training

**Most useful files for this report:**
- `README.md` — top-level recipe, architecture diagram, critical implementation notes
- `model.py` — `DINOv3MultiLabel` (CLS + GeM-pooled patch tokens, 2048-d feature, multi-label classifier head, `apply_lora()`)
- `mosaic_dataset.py` — synthetic K-plant mosaics (K ∈ {1..5}, dist `[0.30, 0.30, 0.20, 0.12, 0.08]`) for Phase 2
- `extract_features.py` — Phase 1 feature cache (one-time fp16 dump of CLS+GeM features)
- `train.py` — two-phase trainer (`--phase 1` linear probe / `--phase 2` LoRA + ASL on mosaics)
- `run_inference.py` — tiled multi-label submission, reuses `004/tiling.py` + `004/aggregation.py`
- `run_validation.py` — local mosaic-based macro-F1 sweep over `(agg_mode, threshold)`
- `experiments/logit_adjustment_collapse.md` — full post-mortem of the Phase 2 sigmoid-collapse bug
- `diagnose_predictions.py`, `diagnose_recall.py` — diagnostic scripts kept after the collapse fix

---

## Run: dinov3_v1 (single-backbone DINOv3 multi-label)

**Experiment ID:** 005
**Date:** April 2026

### Model / Architecture

- **Backbone:** DINOv3 ViT-L/16 (`dinov3_vitl16` via torch.hub or `vit_large_patch16_dinov3.lvd1689m` via timm), 1024-d embedding, **4 register tokens** (slice `[:, 1+4:, :]` to recover patches)
- **Feature path:** `CLS (1024) || GeM-pooled patches (1024)` → concat → 2048-d
- **Classifier head:** `LayerNorm → Linear(2048, 1024) → GELU → Dropout(0.2) → Linear(1024, 7806)` → sigmoid (multi-label)
- **Pretraining:** Generic LVD-1689M SSL (Gram-anchored DINOv3, Aug 2025) — *no plant-specific signal*. This is the failure mode 008/009 try to fix by adding a PC24 fine-tune stage in front.
- **Normalization:** ImageNet `(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)` — explicitly NOT CLIP stats

### Dataset

- **Phase 1 source:** `train_metadata_cleaned_verified_stratified.csv` (1.38M PC24 single-plant rows, 7806 species) — features cached to `phase1_feature_cache.pt`
- **Phase 2 source:** `MosaicDataset` — synthetic K-plant collages built on the fly from the same single-plant corpus
- **Test set:** 2,105 quadrat images at `/workspace/plantclef/kaggle_uploads/test/images`
- **Species vocabulary:** 7,806 plant species (canonical 004 ordering via `dataset.load_species_ids`)

### Task

Single-backbone, single-resolution (DINOv3 ViT-L/16) baseline for multi-label species ID on quadrats. Demonstrates that DINOv3's Gram-anchored patch tokens make the heavyweight `src/` triple-ensemble unnecessary in principle — though in practice the lack of a plant prior caps this run far below BioCLIP-2.5.

### Training Recipe

**Phase 1 — head-only linear probe on cached features.**
- Loss: `LogitAdjustmentLoss` (softmax CE + `τ · log(class_freq/total)` shift). Logit-adjustment is *correct* here because the loss is softmax CE.
- Trains only the classifier head on cached `(CLS+GeM, class_idx)` pairs. ~30 min on a single GPU.
- Result: `val_top1 = 11.97%`, `val_top5 = 26.13%` on a held-out single-plant split — healthy.

**Phase 2 — LoRA fine-tune on synthetic mosaics.**
- LoRA targets: `["qkv", "proj", "fc1", "fc2"]` (verify via `list_lora_target_candidates(model)`).
- Loss: `AsymmetricLoss(γ_pos=1, γ_neg=4, clip=0.05, eps=1e-8)`, with `logit_adjustments=None` after the bug fix.
- Mosaics: K ∈ {1..5} sampled from `[0.30, 0.30, 0.20, 0.12, 0.08]`, K-hot multi-label targets.
- 5 epochs × 12,500 steps, effective batch 512, bf16.

### Hyperparameters / Run Config

| Parameter | Value |
|---|---|
| backbone | DINOv3 ViT-L/16 (1024-d) |
| feature dim | 2048 (CLS + GeM patches) |
| n_classes | 7806 |
| patch size | 16 (so canvas/tile must be multiples of 16) |
| default tile size | 384 (= 24 × 16) |
| Phase 1 loss | LogitAdjustmentLoss (CE + log-prior) |
| Phase 2 loss | AsymmetricLoss (γ_pos=1, γ_neg=4, clip=0.05) |
| Phase 2 LoRA | targets `qkv/proj/fc1/fc2` |
| Phase 2 epochs | 5 × 12,500 steps (1M samples/epoch, eff. batch 512) |
| Mosaic K dist | `[0.30, 0.30, 0.20, 0.12, 0.08]` for K=1..5 |
| Inference agg | `noisy_or` (logits, not probs — converts back via 004's aggregation) |
| Default threshold | 0.3 |

### Critical Bug Found and Fixed: Logit-Adjustment Collapse

`experiments/logit_adjustment_collapse.md` documents the full post-mortem.

- **Symptom:** `run_validation.py` returned **bit-identical** macro-F1 (`0.008085503482025221`) across 4 agg_modes × 6 thresholds = 24 configurations.
- **Diagnosis:** Per-class sigmoid output saturated near 1.0 for every class on every input — `min=0.9458, median=0.9995, max=1.0000` across 7806 classes. Model collapsed to "predict everything as positive."
- **Root cause:** The `τ · log π_c ≈ −9` adjustment was added to *sigmoid* logits. This trick is derived for *softmax* CE (where the normaliser couples classes); on independent-Bernoulli ASL, the loss-minimum is to push every raw logit up by +9, killing discrimination. Average loss (3.58) looked fine because ASL's `(1-p_t)^4` focal weight + `clip=0.05` make the saturated regime approximately free.
- **Fix (one line in `train.py:349-355`):** set `logit_adjustments=None`. ASL's `γ_neg=4` already handles multi-label imbalance.
- **Verification post-fix:** loss starts at ~107 (meaningful gradient), decays smoothly to ~5 over the first epoch — the healthy multi-label trajectory.

### Diagnostic Numbers (broken, K-only ranking, no threshold, 200 mosaics, avg truth size 2.25)

| K | recall@K | macro-F1 (top-K) | chance recall@K |
|---|---|---|---|
| 1 | 1.87% | 0.022 | 0.026% |
| 3 | 2.04% | 0.014 | 0.077% |
| 5 | 2.71% | 0.014 | 0.128% |
| 10 | 4.58% | 0.013 | 0.256% |
| 20 | 5.08% | 0.008 | 0.513% |
| 50 | 6.75% | 0.004 | 0.640% |

→ ~70× above chance @ K=1 dropping to ~10× @ K=50: weak ranking signal exists but is too noisy to submit.

### Results

- **Public F1 score (Kaggle leaderboard):** **0.13** (RETIRED — significantly behind BioCLIP-2.5 baselines)
- **Phase 1 val_top1:** 11.97% (single-plant holdout — healthy)
- **Phase 2 final loss:** ~5 (after the collapse fix)
- The 0.13 number is what motivated experiments 008 (DINOv3 + PC24 fine-tune) and 009 (BioCLIP-2.5 + PC24 fine-tune) — the gap to 006 (Arjun's frozen BioCLIP-2.5 at 0.33) was attributed to the missing **plant prior** in DINOv3's pretraining, not the architecture.

### Outputs

- `outputs/{run_slug}/submission.csv` — PlantCLEF format `(quadrat_id, species_ids)`, `csv.QUOTE_ALL`
- `outputs/{run_slug}/predictions_scored.csv` — per-quadrat, per-rank species + score
- `outputs/{run_slug}/run_config.json` and `summary.json` — provenance
- `outputs/validation/validation_*.json` — macro-F1 sweep results

### Status / Verdict

**Retired.** 0.13 is the floor of the single-backbone-without-plant-prior path. Successor experiments:
- **008** — replaces Phase 1 with full DINOv3-L fine-tune on 1.4M PC24 single-plants (PlantNet recipe).
- **009** — same recipe but BioCLIP-2.5 ViT-H/14 backbone.
- **010** — last-4-blocks fine-tune of BioCLIP-2.5 → 0.38333 (current team best, beats every fusion that includes 005/008 PhaseA).

### Key Takeaways

1. **Logit-adjustment formulas are loss-specific** — softmax-CE log-prior shift does not transfer to sigmoid multi-label.
2. **Absolute loss values are not diagnostics in multi-label regimes** — focal weight + clip can hide total collapse.
3. **Identical metrics across unrelated hyperparameters is the loudest possible signal that something is ignoring its inputs.** Bit-equal F1 across 24 configs was the smoking gun.
4. The `noisy_or` aggregation in `004/aggregation.py` re-sigmoids internally, so `run_inference.py` and `run_validation.py` convert sigmoid probs back to logits before calling it.
