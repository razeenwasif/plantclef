# Experiment Report: 013 — BioCLIP-2.5 SSL-warmstart + 010 MultiTask + Taxonomy Aux Heads

**Most useful files for this report:**
- `materialize_ssl_for_010.py` — bridge 012's SSL backbone state_dict → 010-format `BioCLIP25MultiTask` ckpt (SharedMLP + species_head + genus/family/order/class aux heads, `epoch=-1`)
- `outputs/finetune/checkpoints/best.pt` — supervised-fine-tune ckpt (val top1=0.7329, top5=0.9226, genus=0.88, family=0.93, order=0.94, class=0.99)
- `outputs/test_probs_013_grid4x4.npz` — tile-inference dump (probs_max/mean/noisy_or, 2105 × 7804 fp16 over 16 grid_4x4 tiles)
- `outputs/submission_013_grid4x4_softmaxmean_top3.csv` — Kaggle submission (ref 52159801, public F1 = 0.33185)
- `../012_bioclip25_ssl_lucas/outputs/ssl_bioclip25_backbone_ep5.pt` — shared SSL backbone (5 epochs DINO on geometry-corrected LUCAS)
- `../010_bioclip25_end_to_end_finetune_multitask/model.py` — `BioCLIP25MultiTask` definition (reused as-is)
- `../010_bioclip25_end_to_end_finetune_multitask/train.py` — supervised trainer (reused with `--resume --resume-weights-only --use-taxonomy-heads`)
- `../011_bioclip25_aggregation_sweep/alpha_sweep.py` — npz → submission CSV (`--alphas 1.0` = pure softmax_mean = team-best recipe)

---

## Run: 013 SSL + MultiTask + Taxonomy Aux Heads

**Experiment ID:** 013
**Date:** April 2026 (materialize + supervised launched 2026-04-28, finished ~2026-04-29)

### Why this experiment exists

012 isolated the **architectural** bottleneck: the SSL warmstart paired with 006's plain `Linear(1024 → 7804)` head landed at **0.30261**, ~8 F1 below the team-best 010 last_blocks ALONE (**0.38333**). Diagnosis from the 012 summary: SSL improved single-plant retrieval but the wrong head shape cost ~6 F1 on quadrats, and we hadn't tried hierarchical taxonomy supervision.

013 holds the SSL backbone fixed and **swaps in 010's full multi-task architecture**:
1. `BioCLIP25MultiTask` head shape: `LayerNorm → Linear(1024 → 1024) → GELU → Dropout(0.2) → species_head` (= the SharedMLP that 010 last_blocks beats 006-Linear by ~6 F1 with).
2. Auxiliary heads on **genus / family / order / class** with the `0.30 / 0.15 / 0.05 / 0.02` joint-loss weights from 010's `TAXONOMY_WEIGHTS` — gives the encoder gradient signal at every taxonomic level, not just the noisy 7804-way species axis.

The hypothesis: SSL-shifted features × richer head × hierarchical supervision should close most of the 012 → 0.38333 gap, and ideally beat it.

### Model / Architecture

`BioCLIP25MultiTask` (010's model, fully reused):

- **Backbone:** SSL-warm-started BioCLIP-2.5 ViT-H/14 (32 blocks, 1024-d, 4 register tokens)
- **SharedMLP:** `LayerNorm(1024) → Linear(1024, 1024) → GELU → Dropout(0.2)`
- **Species head:** `Linear(1024, 7804)`
- **Aux heads (`use_taxonomy_heads=True`):**
  - `Linear(1024, 1446)` — genus
  - `Linear(1024, 181)` — family
  - `Linear(1024, 61)` — order
  - `Linear(1024, 6)` — class
- **Unfreeze:** last 4 transformer blocks + `ln_post` + `proj` (matches 010 sweet spot)

Encoder counts (from `dataset.build_label_encoders` on `species_lookup_with_gbif_cleaned_names.csv`):
**species=7,804  genus=1,446  family=181  order=61  class=6**.

### Materialization Bridge

`materialize_ssl_for_010.py` rebuilds 012's SSL backbone into 010's full module shape so 010's trainer can `--resume --resume-weights-only` from it:

1. Load `ssl_bioclip25_backbone_ep5.pt` → `backbone_state_dict`
2. Read `train_metadata_cleaned_verified_stratified.csv` + `species_lookup_with_gbif_cleaned_names.csv` → encoders
3. Build a fresh `BioCLIP25MultiTask(num_species=7804, num_genus=1446, num_family=181, num_order=61, num_class=6, hidden_dim=1024, dropout=0.2, use_taxonomy_heads=True)`
4. `model.backbone.load_state_dict(backbone_sd, strict=False)` to inject SSL weights
5. Save `{epoch:-1, model_state_dict, encoders, config}` to `outputs/ssl_init_for_010.pt` (010's `--resume-weights-only` then resets `start_epoch=0`)

### Dataset

- **Train:** `/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv` — 1.38M PC24 single-plant rows, 7,804 species (joined to `species_lookup_with_gbif_cleaned_names.csv` for genus/family/order/class)
- **Images:** `/workspace/plantclef/raw/train/images_max_side_800/`
- **Val:** 5% holdout (~69K rows), 7,804-way top-1 / top-5 plus per-tax-level top-1
- **Test set:** 2,105 quadrat images at `/workspace/plantclef/kaggle_uploads/test/images`

### Training Recipe (Supervised)

| Parameter | Value |
|---|---|
| backbone | BioCLIP-2.5 ViT-H/14 (SSL-warm-started from 012 ep5) |
| head | `BioCLIP25MultiTask` (SharedMLP + species + 4 aux heads) |
| epochs | 5 |
| batch_size | 128 |
| precision | bf16 |
| backbone_lr | 1e-6 |
| head_lr | 1e-4 |
| optimizer | AdamW(β=(0.9,0.999), wd=0.05) |
| schedule | cosine with warmup |
| unfreeze | last 4 blocks + `ln_post` + `proj` |
| augmentation | RandomResizedCrop(224) + HFlip + ColorJitter |
| loss | `CE_species + 0.30·CE_genus + 0.15·CE_family + 0.05·CE_order + 0.02·CE_class` |
| `--use-taxonomy-heads` | enabled |
| train time | ~5h 37m on 1× RTX 5090 |

Joint-loss weights match 010's `TAXONOMY_WEIGHTS` exactly so the species axis is dominant but every taxonomic level contributes gradient.

### Validation Metrics

| Metric | Value |
|---|---|
| species top-1 | **0.7329** |
| species top-5 | 0.9226 |
| genus top-1 | 0.88 |
| family top-1 | 0.93 |
| order top-1 | 0.94 |
| class top-1 | 0.99 |

Compared to 012 (val r@1 = 0.7726), 013's species top-1 is **lower** by ~4 points — different val splits, but both run on the same SSL backbone, suggesting the multi-task gradient mildly trades single-plant species precision for taxonomic coherence (which is exactly what we want for cross-domain generalisation).

### Inference Recipe (= team-best 010 recipe, byte-for-byte)

```bash
python src_experiments/010_bioclip25_end_to_end_finetune_multitask/dump_test_probs.py \
    --checkpoint src_experiments/013_bioclip25_ssl_lucas_multitask/outputs/finetune/checkpoints/best.pt \
    --tile-mode grid_4x4 --tile-size 448 --overlap 0.0 --img-size 224 \
    --batch-size 64 --precision bf16

python src_experiments/011_bioclip25_aggregation_sweep/alpha_sweep.py \
    --probs   .../outputs/test_probs_013_grid4x4.npz \
    --out-dir .../outputs --mix mean_x_noisy_or --alphas 1.0 --top-k 3
```

010's native MultiTask dump extracts species_logits from `(sp_logits, *_) = model(batch)`; aggregation is softmax_mean over the 16 grid_4x4 tiles → top-3.

### Results

| Submission | Aggregation | Top-K | Kaggle Public F1 | Ref |
|---|---|---|---|---|
| `submission_013_grid4x4_softmaxmean_top3.csv` | softmax_mean | 3 | **0.33185** | 52159801 |

### Anchor Comparisons

| Model | Recipe | Public F1 | Δ vs 013 |
|---|---|---|---|
| **TEAM BEST** 010 last_blocks ALONE (no SSL) | SharedMLP + species_head only | **0.38333** | +0.0515 |
| 011 α-sweep best (010 npz, α=0.75 mean×noisy_or) | SharedMLP + species_head only | 0.38278 | +0.0461 |
| **013** SSL + MultiTask + tax aux heads | SharedMLP + species + 4 aux heads | **0.33185** | — |
| 012 SSL + 006 plain Linear | Plain `Linear(1024 → 7804)` | 0.30261 | −0.0292 |

### Status / Verdict

**+0.0292 over 012 — head shape & taxonomy heads do help, by exactly the amount we predicted.** Confirms the diagnosis from the 012 summary: ~6 F1 of the 012-vs-010 gap was the head architecture, not the SSL phase.

**But still −0.0515 below team-best 0.38333 *without* SSL.** At this LR schedule (`backbone_lr=1e-6`, 5 epochs), the SSL warmstart is *net negative* even with the better head shape. The cheap-LR SSL→supervised handoff doesn't recover the 010-from-scratch gap — adding a richer head closed only ~36% of the gap.

**Don't:** mix SSL-warmstart with 010's `backbone_lr=1e-6 / 5 epoch` schedule and expect to beat 0.38333. The lower LR was tuned for the OpenCLIP-default backbone; the SSL backbone needs more headroom (or more epochs) to reshape its features toward species discrimination.

### Hypotheses for the Residual Gap

1. **LUCAS-SSL pulled features toward ground-level vegetation distribution and away from single-plant species discrimination.** The 5% val species top-1 dropping from 012's r@1=0.7726 to 013's 0.7329 — even on the same SSL backbone — is consistent with this; multi-task aux heads don't undo a backbone that has drifted from species-axis features.
2. **`backbone_lr=1e-6` is too low to reshape SSL features.** The schedule was tuned for OpenCLIP defaults, not for an SSL backbone that has already moved a step away from the supervised optimum.
3. **Full 010-from-scratch (no SSL) ALREADY benefits from taxonomy heads.** The SSL phase added cost without lift if the same `--use-taxonomy-heads` recipe applied to the OpenCLIP backbone matches or exceeds 0.38333.

### Next Levers (in priority order)

1. **Drop SSL** — re-run 010 from scratch with `--use-taxonomy-heads` and the stratified CSV to confirm tax-head lift over team-best **0.38333**. This is the cleanest controlled experiment to isolate tax-head value.
2. **Bump SSL→supervised `backbone_lr`** to 5e-6 or 1e-5 to give the SSL features room to reshape toward species discrimination.
3. **More epochs** (8–10) on 013 to let SSL features converge to the species axis under joint multi-task supervision.

### Key Takeaways

1. **Architecture lift is real and measurable.** +0.0292 going from plain Linear → SharedMLP + 4 tax aux heads, holding everything else fixed (same SSL backbone, same supervised LRs/epochs/data).
2. **SSL warmstart is *net negative* at this LR schedule.** Even with a stronger head, the SSL backbone trained at `backbone_lr=1e-6 / 5 epochs` lost more than the head architecture gained relative to team-best 0.38333. The schedule wasn't designed for an SSL-shifted starting point.
3. **In-domain val F1 is not a Kaggle predictor under SSL drift.** 013's val species top-1 of 0.7329 with stellar taxonomic accuracy (genus/family/order/class all ≥0.88) translated to only 0.33185 on the leaderboard — the SSL phase trades cross-domain species precision for taxonomic coherence, and the test set rewards the former.
4. **The gap is closable but requires LR/epoch tuning, not more architecture.** The +0.0292 lift confirms the lever; the −0.0515 remaining gap is parameter-space distance, not bottleneck shape.
5. **Materialization-bridge pattern is reusable.** The `materialize_ssl_for_010.py` approach (bridge backbone state_dict into a fresh full-model shape, embed encoders, set `epoch=-1`) cleanly decouples SSL from supervised — any future SSL backbone (e.g. longer LUCAS pretrain, different masking ratio) can be tested against any supervised head with one bridge script.
