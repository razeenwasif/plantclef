# 005 — DINOv3 multi-label PlantCLEF 2026

A focused single-backbone pipeline built on Meta's **DINOv3 ViT-L/16** (Aug 2025).
DINOv3's *Gram anchoring* fixes the noisy-patch-token problem that motivated the
heavyweight DINOv2 + ConvNeXt + BioCLIP triple-ensemble in `src/`. With clean
dense features at any spatial location, a single backbone + tiled multi-label
inference is enough.

## What this directory contains

| File | Purpose |
|---|---|
| `model.py` | DINOv3 backbone wrapper with CLS + GeM-pooled patch tokens (slicing past 1 CLS + 4 register tokens), 2048-d feature, multi-label classifier head, `apply_lora()` for PEFT. |
| `mosaic_dataset.py` | Synthetic K-plant mosaic dataset for Phase 2; K ∈ {1..5} sampled from `[0.30, 0.30, 0.20, 0.12, 0.08]`. Also includes `SinglePlantDataset` for Phase 1. |
| `extract_features.py` | Phase 1 feature cache — runs DINOv3 once over single-plant data, stores fp16 (CLS+GeM) features + class labels. |
| `train.py` | Two-phase trainer. `--phase 1` = head-only linear probe on cached features. `--phase 2` = LoRA + ASL on synthetic mosaics. |
| `run_inference.py` | Tiled multi-label submission. Reuses 004's `tiling.py` and `aggregation.py`. |
| `run_validation.py` | Local mosaic-based macro-F1 sweep over `(agg_mode, threshold)` to pick a leaderboard config without paying the Kaggle round-trip. |

## Quickstart

```bash
cd src_experiments/005_dinov3_multilabel

# 0) (one-time) accept the DINOv3 license and authenticate
huggingface-cli login

# 1) cache features (~few hours on a single GPU; one-time per backbone+resolution)
python extract_features.py \
    --metadata-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
    --image-root   /workspace/plantclef/raw/train/images_max_side_800 \
    --species-csv  /workspace/plantclef/processed/species_lookup_with_gbif_cleaned_names.csv \
    --output       ../../models/dinov3_v1/phase1_feature_cache.pt

# 2) Phase 1: head-only linear probe (~30 min)
python train.py --phase 1 \
    --feature-cache ../../models/dinov3_v1/phase1_feature_cache.pt \
    --output-dir    ../../models/dinov3_v1

# 3) Phase 2: LoRA on synthetic mosaics
python train.py --phase 2 \
    --metadata-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
    --image-root   /workspace/plantclef/raw/train/images_max_side_800 \
    --species-csv  /workspace/plantclef/processed/species_lookup_with_gbif_cleaned_names.csv \
    --p1-head      ../../models/dinov3_v1/phase1_head.pth \
    --output-dir   ../../models/dinov3_v1 \
    --bf16

# 4) (optional) sweep agg-mode + threshold on a mosaic val set
python run_validation.py \
    --checkpoint   ../../models/dinov3_v1/phase2_lora.pth \
    --metadata-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
    --image-root   /workspace/plantclef/raw/train/images_max_side_800 \
    --species-csv  /workspace/plantclef/processed/species_lookup_with_gbif_cleaned_names.csv \
    --bf16

# 5) tiled inference -> submission.csv
python run_inference.py \
    --checkpoint  ../../models/dinov3_v1/phase2_lora.pth \
    --images-root /workspace/plantclef/kaggle_uploads/test/images \
    --agg-mode    noisy_or \
    --threshold   0.3 \
    --bf16

# 6) submit
bash ../../scripts/submit_predictions_kaggle.sh \
    -i src_experiments/005_dinov3_multilabel/outputs \
    -p 'submission_*.csv'
```

## Architecture

```
input image (3, H, W)  H, W ∈ multiples of 16
       │
       ▼
DINOv3 ViT-L/16  ─────►  tokens (B, 1 + 4 + n_patch, 1024)
                              │     │     │
                              │     │     └──► patches[:, 5:, :] ──► GeM pool ──► 1024
                              │     └──► 4 register tokens (skip)
                              └──► CLS [:, 0, :] ──► 1024
                                                          │
                                                          ▼
                                            concat ──► (B, 2048)
                                                          │
                                                          ▼
                            LayerNorm → Linear(2048, 1024) → GELU → Dropout(0.2)
                                                          │
                                                          ▼
                                                  Linear(1024, 7806)
                                                          │
                                                          ▼
                                                    sigmoid (multi-label)
```

## Critical implementation notes

1. **Register-token slicing.** DINOv3 has **4** register tokens. `forward_features` returns `(B, 1 + 4 + n_patch, D)`. Slice `[:, 1+4:, :]` *before* GeM pooling. Forgetting this is the most common DINOv3 bug.
2. **Patch size 16 ≠ DINOv2's 14.** All resolutions (canvas, tile) **must be multiples of 16**. The default 384 = 24 × 16 is fine.
3. **ImageNet normalisation, not CLIP.** DINOv3 was pretrained with ImageNet stats `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`. The `src/data/dataloader.py` DALI pipeline uses CLIP stats — do **not** reuse it without overriding the normalisation.
4. **HuggingFace gating.** First load needs `huggingface-cli login` after accepting the DINOv3 license on the model page.
5. **LoRA target verification.** Before kicking off Phase 2, dump the backbone module names to confirm `["qkv", "proj", "fc1", "fc2"]` matches DINOv3's timm naming:
   ```python
   from model import DINOv3MultiLabel, list_lora_target_candidates
   m = DINOv3MultiLabel(n_classes=7806, pretrained=False)
   print(list_lora_target_candidates(m, sample_n=40))
   ```
6. **`noisy_or` aggregation operates on logits, not probabilities.** `run_inference.py` and `run_validation.py` convert sigmoid probs back to logits before applying `noisy_or` (since the implementation in 004's `aggregation.py` re-sigmoids internally).

## Why this design

- **Single backbone, not ensemble.** DINOv3's Gram-anchored patch tokens give us BioCLIP-quality CLS embeddings *and* better dense features than DINOv2+ConvNeXt fusion. Ensembling can be revisited after a single-backbone baseline lands.
- **Two-phase training.** Phase 1 gives the head a strong starting point cheaply (cached features, ~30 min). Phase 2 with synthetic mosaics is the one that actually shapes the model for the multi-species test distribution.
- **Synthetic mosaics, not real quadrat labels.** The training set is *single-plant* images; the test set is *multi-species quadrats*. Mosaics bridge this domain shift while letting us use the labelled single-plant data we already have.
- **Reuse 004 utilities.** Tiling, aggregation, support-bank patterns are already tested and abstract over the model. Don't reimplement.

## Outputs

- `outputs/{run_slug}/submission.csv` — PlantCLEF format `(quadrat_id, species_ids)` with `csv.QUOTE_ALL`.
- `outputs/{run_slug}/predictions_scored.csv` — per-quadrat, per-rank species + score.
- `outputs/{run_slug}/run_config.json` and `summary.json` — provenance.
- `outputs/validation/validation_*.json` — macro-F1 sweep results.
