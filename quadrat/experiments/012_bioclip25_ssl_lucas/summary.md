# Experiment Report: 012 — BioCLIP-2.5 SSL Continual Pretraining on Geometry-Corrected LUCAS

**Most useful files for this report:**
- `README.md` — three-step pipeline (geometry-correct LUCAS → DINO-style SSL → 006 supervised fine-tune), measured test↔LUCAS gap, anchor at 0.38333
- `prepare_lucas.py` — center-crop + resize ≤800px LUCAS pseudo-quadrats to match test geometry (aspect 1.34 → 1.0)
- `ssl_train.py` — DINO-style SSL trainer (2 globals + 6 locals at 224 px, EMA teacher cosine 0.996→1.0, partial unfreeze of last 4 blocks + ln_post + proj, bf16 + grad-checkpoint)
- `materialize_ssl_for_006.py` — bridge SSL backbone state_dict → 006-format starter ckpt (`epoch=-1`, embedded `idx_to_species`)
- `dump_test_probs.py` — 006-class tile-inference dump that mirrors 010's npz schema (probs_max/mean/noisy_or over grid_4x4 tiles)
- `outputs/ssl_bioclip25_backbone_ep5.pt` — 5-epoch SSL backbone (visual encoder state_dict only)
- `outputs/finetune/checkpoints/best.pt` — supervised-fine-tune ckpt (006 plain Linear head, val r@1=0.7726)
- `outputs/submission_012_grid4x4_softmaxmean_top3.csv` — Kaggle submission (ref 52124292, public F1 = 0.30261)

---

## Run: 012 SSL-warmstart + 006 plain-Linear fine-tune

**Experiment ID:** 012
**Date:** April 2026 (SSL launched 2026-04-26, supervised fine-tune finished 2026-04-28)

### Why this experiment exists

010 BioCLIP-2.5 last_blocks ALONE is the team best (**0.38333**), and 010's full FT (009) collapses to 0.20777 — there's a sweet spot at "last 4 blocks unfrozen" and going deeper destroys the Tree-of-Life prior. The next lever to pull is **prior alignment**: BioCLIP-2.5 was pretrained on single-organism web imagery, but the test set is 50×50 cm vegetation quadrats. EDA quantified the gap precisely:

| Set | Max-side | Aspect | Mpx |
|---|---|---|---|
| Test (N=2,105) | 800 px | 1.00 (std 0.10) | 0.60 |
| LUCAS raw (N=212K) | 1740 × 1299 | 1.34 | 2.33 |

008 Phase B v3 self-distilled on **raw** LUCAS and crashed to 0.227 — the geometry mismatch made the teacher's pseudo-labels noise. 012 attacks this with two fixes: (1) geometry-correct LUCAS to match the test distribution, and (2) DINO-style SSL — *not* pseudo-label distillation — so the encoder shifts toward quadrat-shaped imagery without committing to the teacher's wrong labels. Then hand off to the team-best 010 last_blocks supervised recipe.

### Model / Architecture

**SSL phase** — DINO-style multicrop self-distillation:
- Backbone: BioCLIP-2.5 ViT-H/14 (`hf-hub:imageomics/bioclip-2.5-vith14`), 32 transformer blocks, embed_dim=1024, 4 register tokens
- Partial unfreeze: **last 4 transformer blocks + ln_post + proj** (matches 010's `unfreeze_n=4`); lower 28 blocks frozen to preserve Tree-of-Life prior
- Student/teacher heads: 3-layer MLP → 2048-d projector with weight-normed prototype layer
- EMA teacher with cosine momentum 0.996 → 1.0; centering on teacher logits

**Supervised phase** — `BioCLIP25LinearProbe` (006 architecture):
- Backbone: SSL-warm-started BioCLIP-2.5 ViT-H/14
- Head: plain `Linear(1024, 7804)` (no SharedMLP, no aux heads)

### Dataset

- **SSL train:** `/workspace/plantclef/processed/lucas_aspect_corrected/` — 212K geometry-corrected LUCAS pseudo-quadrats (center-cropped to aspect 1.0, resized ≤800 px, JPEG q95)
- **Supervised train:** `train_metadata_cleaned_verified_stratified.csv` — 1.38M PC24 single-plant rows, 7,804 species
- **Supervised images:** `/workspace/plantclef/raw/train/images_max_side_800/`
- **Supervised val:** 5% holdout (~69K rows), 7,804-way top-1 / top-5
- **Test set:** 2,105 quadrat images at `/workspace/plantclef/kaggle_uploads/test/images`

### SSL Recipe

| Parameter | Value |
|---|---|
| backbone | BioCLIP-2.5 ViT-H/14 (32 blocks, 1024-d) |
| crop strategy | 2 globals + 6 locals, **all at 224 px** (avoids re-interpolating pretrained pos-embed grid) |
| crop scales | global `(0.5, 1.0)` / local `(0.05, 0.5)` |
| projection head | 3-layer MLP → 2048-d → weight-normed prototype |
| teacher EMA | cosine momentum 0.996 → 1.0 |
| centering | EMA buffer on teacher logits |
| unfreeze | last 4 blocks + ln_post + proj |
| optimizer | AdamW, lr=5e-5 (linearly scaled), wd=0.04 |
| precision | bf16 + gradient checkpointing on unfrozen blocks |
| epochs | 5 |
| batch_size | 16 × grad_accum 4 = effective 64 |
| hardware | 1× RTX 5090 |

### Supervised Recipe

| Parameter | Value |
|---|---|
| recipe | 006 plain Linear (= team-best 010 ablation) |
| epochs | 5 |
| batch_size | 128 |
| precision | bf16 |
| lr-head | 1e-4 |
| lr-backbone | 1e-6 (head_lr × 0.01) |
| optimizer | AdamW(β=(0.9,0.999), wd=0.05) |
| schedule | cosine with warmup |
| loss | CE on species (single-label) |
| augmentation | RandomResizedCrop(224) + HFlip + ColorJitter |
| unfreeze | last 4 blocks + ln_post + proj |
| val r@1 | **0.7726** |
| train time | ~5h 45m on 1× RTX 5090 |

### Inference Recipe (= team-best 010 recipe, byte-for-byte)

```bash
python src_experiments/012_bioclip25_ssl_lucas/dump_test_probs.py \
    --checkpoint src_experiments/012_bioclip25_ssl_lucas/outputs/finetune/checkpoints/best.pt \
    --tile-mode grid_4x4 --tile-size 448 --overlap 0.0 --img-size 224 \
    --batch-size 64 --precision bf16
```

Aggregation: softmax_mean over the 16 grid_4x4 tiles → top-3 (= 010's 0.38333 recipe).

### Results

| Submission | Aggregation | Top-K | Kaggle Public F1 | Ref |
|---|---|---|---|---|
| `submission_012_grid4x4_softmaxmean_top3.csv` | softmax_mean | 3 | **0.30261** | 52124292 |

**Anchor comparison:** −0.0807 below team-best **0.38333** (010 last_blocks ALONE, same inference recipe).

### Status / Verdict

**Architectural gap dominates the SSL gain.** Despite a strong supervised val (r@1=0.7726) and the team's geometry-correction pipeline working as designed, 012's plain `Linear(1024 → 7804)` head underperforms 010's `LayerNorm → Linear(1024→1024) → GELU → Dropout(0.2) → species_head` SharedMLP by ~6 F1 points on quadrats. SSL improved single-plant retrieval but didn't transfer to multi-species quadrats with the wrong head shape.

**Don't:** rerun 012 inference variants (α-sweep, overlap, hflip-TTA) — the head architecture is the bottleneck, not the SSL warmstart.

**Next:** 013 = same SSL backbone + 010's `BioCLIP25MultiTask` shape (SharedMLP + species_head + genus/family/order/class aux heads). User explicitly requested taxonomic heads.

### Key Takeaways

1. **SSL warmstart is real but masked.** The supervised val r@1 of 0.7726 confirms SSL produced a usable backbone — but the metric that matters (Kaggle F1) was bottlenecked by an architectural ablation, not the SSL feature quality.
2. **Geometry correction is the right preprocessing.** Aspect 1.34 → 1.0 + 800-px resize avoids the 008 Phase B v3 collapse pattern (raw LUCAS at 2.33 Mpx → 0.227).
3. **Don't compare across head shapes.** The 012 → 010 leaderboard delta is **architecture**, not data or pretraining; isolating SSL value requires holding the head shape fixed (→ 013).
4. **Native-resolution multicrop matters for ViT-H/14.** Forcing all crops to 224 px (the encoder's native input) avoided re-interpolating the pretrained position-embedding grid — a class of bug that has historically destabilised ViT SSL runs.
5. **Partial unfreeze sweet spot survives the new prior.** The last-4-blocks recipe that was optimal for supervised 010 also turned out to be the right discipline for SSL — reinforces the "lower blocks carry the Tree-of-Life prior" thesis.
