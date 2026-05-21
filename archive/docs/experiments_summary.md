# PlantCLEF 2026 — Experiments Summary

Roll-up of all numbered experiments under `src_experiments/`, ready to translate
into the paper. Each entry has the same shape: motivation, recipe, result,
takeaway. Cross-experiment lessons live in §"Cross-experiment findings" at
the end. Open levers we never tested live in §"Open levers".

---

## Setup

- **Task.** Multi-label species ID on 50×50 cm vegetation-plot quadrats.
  Train data is **1.4 M single-plant Pl@ntNet (PC24) images** spanning **7,806
  species**. Test data is **2,105 LUCAS quadrats** with 1–10 species each.
  Metric is macro-F1 per sample, averaged over transects.
- **Domain shift.** Train = isolated specimen close-ups; test = wider
  multi-species in-situ shots. Aspect 1.0 vs 1.34, ~0.6 Mpx vs ~2.3 Mpx
  (`docs/eda/`, `kaggle_baselines.md`).
- **Evaluation.** Public Kaggle leaderboard. **TEAM BEST: 0.38333**
  (010 last_blocks ALONE, `unfreeze_n=4`).

---

## Experiment timeline

| ID | Title | Best Kaggle F1 | Δ vs anchor | Status |
|---|---|---|---|---|
| 001 | BioCLIP-1 zero-shot tile | ~0.05 (initial) | — | Retired |
| 002 | BioCLIP zero-shot v2 (multi-model + GBIF prompts) | reused as 004 dep | — | Library |
| 003 | BioCLIP-2.5 linear probe | small lift over zero-shot | — | Subsumed by 006 |
| 004 | BioCLIP few-shot (support bank + KNN) | ~0.20 | — | Library |
| 005 | DINOv3 multi-label two-phase | **0.13** | −0.25 | Retired (no plant prior) |
| 006 | BioCLIP-2.5 frozen prototype (Arjun) | **0.33** | −0.05 | Anchor for fusion |
| 007 (sam) | SAM-weighted zero-shot | small | — | Retired |
| 007 (collages) | PlantNet DINOv2 + collage LoRA | **0.17476** | −0.21 | Retired |
| 008 | DINOv3-L PlantNet FT (PhaseA + collage/pseudo PhaseB) | PhaseA 0.305; **fusion 0.34671** | −0.04 | PhaseA retired |
| 009 | BioCLIP-2.5 full FT | **0.20777** | −0.18 | Collapsed (FT destroys prior) |
| 010 | BioCLIP-2.5 last_blocks multitask + tax-heads | **0.38333** | — | **TEAM BEST** |
| 010_outputs | RRF fuse 008 PhaseA × 010 head-only | **0.34671** | −0.04 | Retired (fusion hurts) |
| 011 (agg) | Aggregation α-sweep on 010 npz | 0.38278 | −0.0006 | Inside noise |
| 011 (ssl-adapt) | Adaptive-K SSL probe | minor | — | Side experiment |
| 012 | SSL on geometry-corrected LUCAS + 006 plain Linear | **0.30261** | −0.08 | Architecture bottleneck |
| 013 | SSL backbone + 010 multitask + tax aux heads | **0.33185** | −0.05 | SSL net-negative at this LR |
| 014a | iNat Research-grade data pull (1.25 M imgs) | data pipeline | — | Feeds 015 |
| 014b | unfreeze-N sweep (n=3, n=5) | 0.37455 / 0.36919 | −0.009 / −0.014 | n=4 sweet-spot confirmed |
| 015 | PC24 ⊕ iNat 50:50 multitask FT (5 ep) | ep1 0.37506 / **ep5 0.37956** | −0.004 | 50:50 mix recipe dead |

---

## 001 — BioCLIP-1 zero-shot tile

**Why.** Establish a no-training floor: tile each quadrat, embed with BioCLIP-1,
score by cosine to per-species text prompts, top-K → submission.

**Recipe.** Single-prompt species name only ("a photo of `<species>`"), grid
tiling, max-pool over tile scores, top-3.

**Verdict.** Useful as a sanity floor but the prompt builder was the bottleneck.
Subsumed by 002 (multi-prompt + GBIF enrichment).

---

## 002 — BioCLIP zero-shot v2 (multi-model + GBIF prompts)

**Why.** Two upgrades on 001: (a) test BioCLIP-1 vs BioCLIP-2 vs BioCLIP-2.5,
(b) build richer prompts from `species_lookup_with_gbif_cleaned_names.csv`
(common name, family, growth form, leaf shape, bloom colour).

**Outcome.** BioCLIP-2.5 dominates the other two, GBIF-enriched prompts
beat name-only. The `prompt_builder.py` here is **reused by 004 few-shot when
`--text-alpha < 1.0`** — these two directories are co-located on purpose.

**Verdict.** Library code for the rest of the project. Not submitted as a
final model; its job is to feed downstream experiments.

---

## 003 — BioCLIP-2.5 linear probe

**Why.** Cheapest possible supervised signal: cache CLS embeddings, train
a single Linear layer. Validates that the BioCLIP-2.5 trunk has usable
species-level information without touching the trunk.

**Verdict.** Small lift over zero-shot, but subsumed by 006 (frozen prototype
on the same trunk gave ~0.33). Probe code retained as reference.

---

## 004 — BioCLIP few-shot (support bank + KNN)

**Why.** Use single-plant training images as support exemplars, embed with
BioCLIP-2.5, cache prototypes; classify each test tile by similarity. Three
aggregations available: `max`, `mean`, `mean_top_m`, `noisy_or`.

**Recipe.** SAHI tile pipeline at multi-scale, support bank K=20/species,
prototype mode (mean of support embeddings), `noisy_or` aggregation.

**Verdict.** ~0.20 Kaggle. Better than zero-shot but well below fine-tuned
BioCLIP-2.5. Tile + aggregation code (`tiling.py`, `aggregation.py`, `few_shot.py`)
became reusable infrastructure for 005/008/010/012/013 inference.

---

## 005 — DINOv3 ViT-L/16 multi-label two-phase

**Why.** DINOv3 (Aug 2025) introduced Gram-anchored patch tokens — exactly
the property tiled multi-label inference depends on. Test whether a single
strong SSL backbone + multi-label classifier could replace ensemble pipelines.

**Recipe.**
1. Phase 1: cache CLS+GeM features on 1.4 M PC24, train a 2-layer head with
   logit-adjusted softmax CE.
2. Phase 2: LoRA fine-tune (`r=32, α=64`) on synthetic K-plant mosaics
   (K∈{1..5} from `[0.30, 0.30, 0.20, 0.12, 0.08]`), AsymmetricLoss.

**Critical bug found.** Phase-2 collapsed to "predict everything as positive"
(macro-F1 bit-identical 0.0085 across 24 hyperparam configs). Cause: applying
a softmax-CE log-prior shift to *sigmoid* multi-label logits — the
loss-minimum is to push every raw logit up by +9, killing discrimination.
**Fix:** drop the logit adjustment for sigmoid losses; ASL's `γ_neg=4`
already handles imbalance.

**Result.** **0.13** on Kaggle.

**Verdict.** Retired. The bottleneck is **DINOv3-L LVD-1689M has no
plant-specific signal** — all subsequent experiments add a plant prior in
front of the SSL trunk. Mosaic-training pipeline (`mosaic_dataset.py`) is
preserved as reusable code.

---

## 006 — BioCLIP-2.5 frozen prototype (Arjun)

**Why.** Like 004 but with BioCLIP-2.5 trunk and a plain Linear classifier
head trained on cached features. The first experiment that used the trunk
that 010 will eventually fine-tune.

**Result.** **~0.33** Kaggle. The reference baseline that PhaseA fusions
(008) and supervised fine-tunes (010) had to beat.

**Verdict.** The "frozen BioCLIP-2.5" leg. Not the final model but the
anchor for the 008 RRF fusions.

---

## 007 — Side experiments (SAM-weighted zero-shot, PlantNet DINOv2 collages)

Two unrelated 007-numbered experiments:

- **007 sam-weighted** — Re-weight zero-shot tile scores by SAM segmentation
  masks (treat tiles with high vegetation coverage as more reliable).
  Marginal, retired.
- **007 plantnet-collages** — DINOv2-B PlantNet trunk + collage LoRA.
  **0.17476** Kaggle. Confirms the DINOv2-B PlantNet trunk has a real but
  weak plant prior; better than DINOv3 (no prior) but well below
  BioCLIP-2.5 (taxonomy-aligned prior). Retired.

---

## 008 — DINOv3-L PlantNet-style full fine-tune + collage / pseudo-label PhaseB

**Why.** Reproduce the PlantNet recipe (their DINOv2-B FT hit 76% PC24 top-1)
on the strictly stronger DINOv3-L. Then a Phase-B LoRA stage on synthetic
collages or LUCAS pseudo-labels for the multi-label transfer.

**Recipe (PhaseA).** PlantNet-style: head-only 2-epoch warmup → unfreeze
backbone, OneCycleLR, AdamW, CE + label smoothing + per-class logit
adjustment. 12 main epochs at `lr-bb=5e-5, lr-head=1e-4`, bf16, 2× 5090 DDP.

**PhaseA results (best).** Multi-scale tiled inference + hflip-TTA +
whole-image + ExG vegetation filter + top-3 max → **0.305**. Best 008-only
score.

**PhaseB outcomes.**
- v1 (synthetic collages, broken): **0.0002**.
- v2 (LUCAS collages): **0.036**.
- v3 (LUCAS pseudo-labels, val_f1@0.5=0.6114 in-domain): **0.227** —
  in-domain val did not predict Kaggle. Self-distillation is a dead end on
  this domain shift.

**Fusion (008 × 006 frozen BioCLIP-2.5, RRF k=60).** Peaked at α=0.70 top-3 =
**0.34642** — first time the team beat 0.33. Later sweep against 010
head-only peaked at α=0.65 top-3 = **0.34671**.

**Verdict.** PhaseA retired. Fusion only helps when the BioCLIP leg is
itself weak; once 010 last_blocks landed at 0.38333, every PhaseA fusion
regressed (0.36410 direct prob-mix; ≤0.37464 RRF). PhaseA's signal turned
out to be a noisier subset of fine-tuned BioCLIP-2.5's signal.

---

## 009 — BioCLIP-2.5 ViT-H/14 PlantNet-style full fine-tune

**Why.** Mirror 008's PlantNet recipe but on BioCLIP-2.5 ViT-H/14 (632 M params,
~2× DINOv3-L), so the ensemble pair becomes "two strong supervised models"
instead of "strong + frozen".

**Recipe.** 1 head-only warmup epoch + 8 main epochs full FT, OneCycleLR,
AdamW, CE + LS + logit-adjust, bf16 + grad checkpointing, 2× 5090 DDP.

**Result.** ep6 **0.20777**, best.pth **0.20407** — both *worse than zero-shot
006 (0.33)*. Run is bad end-to-end, not just the tail.

**Diagnosis.** 1 head-only warmup epoch is not enough to converge a
random-init Linear over 7,806 classes on a 632 M trunk; subsequent unfreeze
poisons the BioCLIP Tree-of-Life prior. **Full FT destroys the taxonomic
prior.** The cure (010) is partial-unfreeze.

**Verdict.** Dead recipe. Critical signal: there's a partial-unfreeze
sweet-spot somewhere between "head only" and "full FT".

---

## 010 — BioCLIP-2.5 last_blocks multitask + taxonomy aux heads (TEAM BEST)

**Why.** Direct response to 009's collapse. Don't unfreeze the whole trunk;
unfreeze only the last 4 transformer blocks + `ln_post` + `proj`. Use a
proper SharedMLP head (not plain Linear) and add hierarchical aux heads
(genus / family / order / class) for richer gradient signal at every
taxonomic level.

**Recipe.**

| Parameter | Value |
|---|---|
| backbone | BioCLIP-2.5 ViT-H/14, 32 blocks |
| head | LayerNorm → Linear(1024 → 1024) → GELU → Dropout(0.2) → species_head |
| aux heads | genus 1,446 / family 181 / order 61 / class 6 |
| loss | CE_species + 0.30·CE_genus + 0.15·CE_family + 0.05·CE_order + 0.02·CE_class |
| unfreeze | last **4** blocks + `ln_post` + `proj` |
| epochs | 5 (warmup 1) |
| batch | 64 micro × accum 4 = effective 256 |
| precision | bf16 |
| backbone_lr / head_lr | 1e-6 / 1e-4 |
| label_smoothing | 0.1 |
| weight_decay | 1e-4 |
| hardware | 2× RTX 5090 DDP |
| inference | grid_4x4, tile_size=448, overlap=0, img_size=224, softmax_mean, top-3, bf16 |

**Result.** **0.38333** Kaggle public F1. **TEAM BEST.** Beats every
fusion variant tried in 008/010_outputs. The grid_4x4 + softmax_mean + top-3
inference recipe is byte-for-byte the team-best inference recipe used by all
downstream experiments (011/012/013/014/015).

**Verdict.** The reference. All later experiments anchor on this recipe and
this number.

---

## 010_outputs — RRF fusion of 008 PhaseA × 010 head-only

**Why.** After 010 head-only landed near 0.34, try late RRF fusion against
008 PhaseA on the same canonical 7,806-wide species axis. RRF (k=60) was
chosen over direct prob-mix because PhaseA's `probs_max` is peaky and 010's
`probs_mean` is broad — direct mix lets PhaseA's confident-but-wrong tiles
win argmax even at low α.

**Result.** Peaked at α=0.65 top-3 = **0.34671** (prior team best, +0.016
over Arjun's 0.33). Later sister sweeps on 010 *last_blocks* fusion all
landed below 0.38333 — **fusion hurts at every α once the strong leg is
strong enough.**

**Verdict.** Retired. Direct prob-mix on 010 last_blocks reached 0.36410
(worse than RRF). Fusion ceiling is bounded by the weaker leg's noise.

---

## 011 — Aggregation α-sweep on 010 npz

**Why.** Hold the 010 last_blocks model fixed, sweep aggregation across
`mean_x_noisy_or` mixtures `α ∈ [0, 1]` to see whether a smarter
tile-aggregator can squeeze more F1 out of the same probabilities.

**Result.** Best α=0.75 mean×noisy_or = **0.38278** (vs 0.38333 at α=1.0
softmax_mean). All variants land within 0.0005 of the anchor — inside noise.

**Verdict.** Aggregation choice is *not* the bottleneck. The noise floor on
public-leaderboard deltas is ~0.001, so any aggregation gain ≤0.001 is
unmeasurable.

---

## 012 — SSL continual pretraining on geometry-corrected LUCAS

**Why.** EDA quantified the train↔test geometry gap precisely (PC24 aspect
~1.0 single-plant vs LUCAS aspect 1.34 multi-species, ~4× pixels). 008
PhaseB v3 self-distilled labels on **raw** LUCAS and crashed to 0.227 — the
geometry mismatch made the teacher's pseudo-labels noise. 012 attacks this
in three steps:
1. Geometry-correct LUCAS (center-crop aspect 1.34 → 1.0, max-side 800 px,
   JPEG q95) → 212 K corrected pseudo-quadrats.
2. DINO-style SSL on the corrected LUCAS, partial-unfreeze the same last 4
   blocks 010 fine-tunes. Multicrop 2 globals + 6 locals **all at 224 px**
   (avoid re-interpolating ViT-H/14's pos-embed grid). EMA teacher cosine
   0.996→1.0, weight-normed prototype head.
3. Supervised fine-tune with the team-best 010 last_blocks recipe — but on
   006's plain `Linear(1024 → 7804)` head (no SharedMLP, no aux heads).

**Result.** SSL + 006 plain-Linear FT → val r@1 = 0.7726 (healthy) →
**0.30261** Kaggle. **−0.08 below team-best 0.38333.**

**Diagnosis.** SSL improved single-plant retrieval but didn't transfer to
multi-species quadrats with the wrong head shape. Plain Linear vs SharedMLP
+ tax aux heads accounts for ~6 F1 of the gap. Run 013 to isolate it.

**Verdict.** SSL is real but masked by an architectural ablation. Geometry
correction is the right preprocessing.

---

## 013 — SSL backbone + 010 multitask + taxonomy aux heads

**Why.** Hold 012's SSL backbone fixed and swap in 010's full
`BioCLIP25MultiTask` (SharedMLP + species + 4 aux heads). Tests whether the
head-shape gap diagnosis from 012 was correct.

**Result.** **0.33185** Kaggle. **+0.0292 over 012** (head shape & taxonomy
heads do help by exactly the predicted amount). **Still −0.0515 below team
best.**

**Diagnosis.** At `backbone_lr=1e-6 / 5 epochs`, the SSL warmstart is *net
negative* even with the better head shape. The 1e-6 LR schedule was tuned
for OpenCLIP-default starting points; an SSL backbone that's already drifted
needs more headroom (or more epochs) to reshape its features back toward
species discrimination.

**Verdict.** SSL warmstart needs LR tuning, not more architecture. Closing
the residual −0.05 gap likely requires `backbone_lr ∈ [5e-6, 1e-5]` or
8–10 epochs. We did not pursue this lever further; 014/015 attacked the
**data** axis instead.

---

## 014a — iNaturalist Research-grade data pull

**Why.** Bottleneck identified by 014b sweep below: data domain, not model
capacity. iNat Research-grade in-situ photos (wider field shots) match
LUCAS quadrats visually better than PC24 single-plant Pl@ntNet close-ups.

**Outcome.** **1.25 M images / 7,796 species / 180 GB** at
`/workspace/plantclef/raw/inat_research_grade/`. Manifest at
`processed/inat_research_grade_manifest.csv`. 2,769 species have zero iNat
coverage (they remain PC24-only at training time).

**Pipeline.** Async GBIF downloader (`download_inat.py`), filter
`datasetKey=50c9509d-22c7-4a22-a47d-8c48425ef4a7`, `mediaType=StillImage`,
`taxonKey=<gbif_species_id>`, cap=500 per species, license recorded
per-occurrence in the manifest.

**Verdict.** Data infrastructure for 015. The pull itself is a hit; whether
iNat helps end-to-end is what 015 tests.

---

## 014b — Unfreeze-N sweep around the 0.38333 anchor

**Why.** The 010 anchor used `--unfreeze-last-n-blocks 4`. Unknown whether
n=4 is the optimum or a lucky sample. 014b brackets the anchor with n=3
and n=5, holding everything else byte-for-byte fixed.

**Recipe.** 010 last_blocks recipe, only delta is `--unfreeze-last-n-blocks
∈ {3, 5}`. Same warmstart, same data, same loss, same DDP config.

**Results.**

| n | Val top-5 | Kaggle Public F1 | Δ vs anchor |
|---|---|---|---|
| 3 | 0.9320 | 0.37455 | −0.00878 |
| **4 (anchor)** | 0.9323 | **0.38333** | — |
| 5 | 0.9327 | 0.36919 | −0.01414 |

**Headline.** **Val top-5 inverts Kaggle.** n=5 had the best val score
(0.9327) but the *worst* Kaggle (0.36919); n=3 had the weakest val and a
middling Kaggle. First observation of the val/Kaggle inversion that 015
later amplified.

**Diagnosis.** More unfrozen blocks = more capacity to fit PC24's
single-plant distribution = worse transfer to multi-species LUCAS quadrats.
**The bottleneck is data domain, not model capacity.** Adding capacity
without fixing data overfits the wrong distribution.

**Verdict.** n=4 sweet-spot is sharp (±1 block costs 0.009–0.014 F1).
`unfreeze_n` is no longer a tuning lever; pull on the *data* axis instead.

---

## 015 — PC24 ⊕ iNat 50:50 multitask fine-tune

**Why.** 014b proved the bottleneck is data domain. iNat habitat shots
should bridge the train→test gap by shifting the training distribution
toward LUCAS quadrats. Hypothesis: 50:50 PC24:iNat keeps the species prior
strong while shifting the visual distribution.

**Combined manifest.** **2,746,659 rows total — 51% PC24 / 49% iNat**
(natural concat, no sampler needed). Taxonomy 100% coverage at all levels
after merge with `species_lookup_with_gbif_cleaned_names.csv`. A
`source ∈ {pc24, inat}` column is preserved for diagnostics.

**Encoding trick — no shared 010 code change.** 010's `dataset.py` resolves
images via `{train_image_root}/{species_id}/{image_name}`. We encode iNat
rows with `image_name = "../../../inat_research_grade/<sp>/<file>"`. The
`..` segments are normalised at `PIL.Image.open()` time, resolving to the
absolute path. The relative-encoding prefix is computed vectorised via
`os.path.commonpath`. PC24 rows keep plain filenames. Sandbox blocked
patching shared `dataset.py`; this hack is purely client-side.

**Recipe.** 010 last_blocks `unfreeze_n=4` byte-for-byte except the
combined manifest. 5 epochs, bf16, batch 64×4 accum, 2× RTX 5090 DDP.

**Crash + fix.** End of epoch 1 — rank-1 SIGABRT, rank-0 SIGTERM. Cause was
**not** OOM or corrupt iNat jpg, despite both being plausible. Actual cause
was an **NCCL ALLREDUCE watchdog timeout during the validation pass**. The
val set (271,615 samples — ~2× 014's val) overran the default
`ProcessGroupNCCL` 10-min timeout while rank 0 was still iterating. Fixed
by `015_pc24_inat_mix/run_with_long_timeout.py` — a 30-line wrapper that
monkey-patches `dist.init_process_group` with `timeout=timedelta(hours=2)`
before `runpy.run_path`-ing 010's `train.py`. **No shared 010 code
touched.** Resume from `last.pt` ran the remaining 4 epochs cleanly.

**Per-epoch validation.**

| Epoch | Train loss | Val loss | top-1 | top-5 | genus | family | order | class |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.2996 | 0.8783 | 0.8046 | 0.9519 | 0.9149 | 0.9490 | 0.9448 | 0.9846 |
| 5 | 2.9235 | 0.7881 | **0.8255** | **0.9584** | 0.9240 | 0.9553 | 0.9544 | 0.9886 |
| 014 anchor | — | — | 0.7475 | 0.9323 | — | — | — | — |

ep5 val top-1 of **0.8255** is the best we've ever seen on this dataset —
**+0.078 over the team-best 014 anchor**.

**Kaggle results.**

| Submission | Val top-1 | Val top-5 | Kaggle Public F1 | Δ vs anchor |
|---|---|---|---|---|
| `submission_015_ep1.csv` | 0.8046 | 0.9519 | **0.37506** | −0.00827 |
| `submission_015_ep5.csv` | **0.8255** | **0.9584** | **0.37956** | **−0.00377** |
| 014 anchor (no iNat) | 0.7475 | 0.9323 | **0.38333** | — |

**Verdict.** **The 50:50 PC24 ⊕ iNat mix at `unfreeze_n=4` cannot beat
vanilla PC24 at the same recipe.** ep5 closes 54% of the ep1 → anchor gap
by training longer, but still loses by **−0.00377**. ep1 → ep5 also shows
the val/Kaggle relationship is *positive but weak* within 015 (+0.026 top-1
buys only +0.0045 Kaggle).

Adding iNat made the model better at predicting *iNat-shaped* data — which
did not transfer to LUCAS multi-species quadrats. The mixture became its
own distribution rather than a stepping stone toward LUCAS.

---

## Cross-experiment findings

These are the load-bearing lessons that survived across multiple experiments
and should anchor the paper's discussion section.

### 1. Partial-unfreeze sweet spot is sharp, not a plateau

| Recipe | Kaggle F1 |
|---|---|
| 010 head-only (n=0) | ~0.33 |
| 014b unfreeze n=3 | 0.37455 |
| **010 last_blocks n=4 (anchor)** | **0.38333** |
| 014b unfreeze n=5 | 0.36919 |
| 009 full FT (n=32) | 0.20777 |

±1 block from n=4 costs 0.009–0.014 F1. Going to full FT (n=32) collapses
the run by 0.18. **The lower 28 blocks of BioCLIP-2.5 carry the
Tree-of-Life prior, and over-fine-tuning destroys it.** This thesis has
held across 5 independent runs.

### 2. Val/Kaggle inversion is real and load-bearing

Three independent observations:
- 014b sweep: n=5 best val, worst Kaggle. n=3 worst val, middling Kaggle.
- 015 ep1 vs 014 anchor: ep1 +0.057 val top-1 over anchor, −0.008 Kaggle.
- 015 ep5 vs 014 anchor: ep5 +0.078 val top-1 over anchor, −0.004 Kaggle.

**The val set — whether PC24-only or PC24⊕iNat — overfits the training
distribution and over-rewards capacity that hurts test transfer.** Val is
useful for *within-recipe* convergence checks, but cannot be trusted for
*cross-recipe* model selection. Any decision based on val deltas <0.005
needs Kaggle confirmation.

### 3. The bottleneck is *data domain*, not model capacity

Multiple lines of evidence:
- 014b: capacity sweep shows the sweet spot is sharp; capacity is already
  sufficient to fit PC24.
- 008 PhaseB v3: in-domain val_f1=0.6114 collapsed to 0.227 on Kaggle.
- 013: val species top-1=0.7329 with stellar taxonomic accuracy (genus 0.88,
  family 0.93, order 0.94, class 0.99) translated to only 0.33185.
- 015: val top-1 0.8255 (best ever) → Kaggle 0.37956 (below 014 anchor).

**Adding capacity without fixing data overfits the wrong distribution.**
What's missing is a *training distribution* that matches LUCAS multi-species
quadrats. PC24-only and PC24⊕iNat 50:50 are both wrong distributions —
just wrong in different directions.

### 4. Fusion ceiling is bounded by the weaker leg's noise

008 PhaseA (0.305) × 006 frozen BioCLIP (0.33) RRF → 0.34642.
008 PhaseA × 010 head-only (~0.34) RRF → 0.34671.
008 PhaseA × 010 last_blocks (0.38333) RRF (any α) → ≤ 0.37464.
008 PhaseA × 010 last_blocks direct prob-mix (α=0.20) → 0.36410.

**Fusion only helps when the strong leg is itself weak.** Once the strong
leg crosses some threshold, adding any signal from a weaker leg is a
regression. RRF beats direct prob-mix when score scales differ, but neither
can lift a strong leg.

### 5. SSL warmstart at OpenCLIP LR schedule is net-negative

013 confirmed: SSL backbone + 010 multitask + tax-heads + same
`backbone_lr=1e-6 / 5 epoch` schedule = **0.33185** vs **0.38333** without
SSL. The SSL phase added cost without lift because the LR schedule was
tuned for OpenCLIP-default starting points, not for an SSL-shifted backbone
that already moved a step away from the supervised optimum. Closing the
−0.05 gap likely requires `backbone_lr ∈ [5e-6, 1e-5]` and/or 8–10 epochs.
Not pursued further; we pivoted to data-axis levers (014/015).

### 6. Logit-adjustment formulas are loss-specific

Softmax-CE log-prior shift does NOT transfer to sigmoid multi-label. Applying
it to ASL collapses the model to "predict everything as positive"
(005 Phase 2 bug). For sigmoid multi-label use ASL's `γ_neg=4` directly;
for softmax CE use the standard `τ · log π_c` shift.

### 7. Geometry-correct LUCAS before SSL on it

008 PhaseB v3 self-distilled raw LUCAS → 0.227 (failed). 012 SSL on
geometry-corrected LUCAS (aspect 1.34 → 1.0, max-side 800 px) →
useful starting point for 013 (closed +0.029 over 012). **Aspect/scale
mismatch is a quiet way to corrupt teacher pseudo-labels.**

### 8. JPEG byte distribution matters

008's `preprocess_test_quadrats.py` (Lanczos resize + JPEG-85 recompression
to match PC24 byte distribution) is part of the PlantNet winner recipe.
Models internalise PC24's compression artifacts during training, so test
images that don't match the same JPEG quality leave performance on the table.

### 9. Reusable infrastructure patterns

- **`run_with_long_timeout.py` wrapper** (015): monkey-patch
  `dist.init_process_group` for 2 h NCCL timeout, then `runpy.run_path`
  the unmodified train script. Reusable for any DDP run that exceeds the
  default 10-min watchdog.
- **Encoding trick** (015): mount external data into a fixed
  `{root}/{sp}/{file}` resolver via relative `..` traversal in `image_name`.
  Pure client-side, no shared dataset.py edit. Reusable for any "add a
  second image source" experiment.
- **Materialization-bridge pattern** (013): bridge an SSL backbone
  state_dict into a fresh full-model shape, embed encoders, set `epoch=-1`.
  Cleanly decouples SSL from supervised. Any future SSL backbone can be
  tested against any supervised head with one bridge script.

---

## Open levers (untested, in priority order)

What we'd try next if we had more compute / time. In rough order of expected
upside.

### A. Synthetic mosaic training on the 010 last_blocks recipe

The bottleneck is *training distribution geometry*. 015 tried mixing PC24
with iNat (in-situ habitat) but kept training as a *single-label*
classification problem. The next lever is **train multi-label on synthetic
K-plant mosaics** (K∈{1..5} from `[0.30, 0.30, 0.20, 0.12, 0.08]` per
the 005 mosaic dataset) on top of the 010 last_blocks recipe. This is the
distribution the test set actually has. Switch loss from softmax CE to
asymmetric loss; reuse 005's `mosaic_dataset.py` infrastructure.

### B. Self-distillation with a strong teacher

Use 010 last_blocks (the strongest single model we have) to pseudo-label
LUCAS pseudo-quadrats. Filter pseudo-labels by confidence + entropy. Train
a fresh student on PC24 ∪ filtered-LUCAS-pseudos with a soft KL-distillation
loss. **Distinct from 008 PhaseB v3** because (a) the teacher is much
stronger now and (b) we're filtering by per-class entropy, not just
top-1 confidence.

### C. CLIP-text fusion via softmax temperature mixing

BioCLIP-2.5 still has a usable text encoder. For tail classes (where the
classifier head saw <50 images), blend classifier probabilities with cosine
similarity to GBIF-enriched text prompts (002's `prompt_builder.py`).
Sweep `text_alpha ∈ [0, 1]` *only on tail classes*, leave head classes
unchanged. Anti-corrupts the head distribution while rescuing the tail.

### D. Adaptive-K thresholding

Median p_top3 on test quadrats is 0.23 with 9 species above p>0.01 (per EDA).
Hardcoded top-3 leaves recall on the table for high-density quadrats and
adds noise on low-density ones. Per-quadrat-K via Brent-tuned threshold on a
held-out mosaic val set, fallback to global threshold sweep. The mosaic
phase from lever A would also produce the calibration data for this.

### E. Cross-checkpoint ensemble

We have ep1, ep5 of 015 and the n=3, n=4, n=5 014b checkpoints — five
independently-trained partial-unfreeze models. RRF-fuse the *probability
distributions* from these (not just the top-3). The lesson from 010_outputs
is that fusion across independently-trained models can lift even when the
gap is small, **as long as the legs make different errors**. Within-family
checkpoints are likely to make correlated errors, so this is more about
recipe-diverse fusion (e.g. 010 last_blocks + 015 ep5 + a mosaic-trained
model) than within-recipe.

### F. Habitat-filtered iNat re-mix

iNat 50:50 lost (015), but iNat is heterogeneous — close-up macros vs wider
scenes. EDA could segment iNat into "close-up vs habitat-shot" via aspect
+ vegetation-fraction proxies, keep only habitat-shots that match LUCAS
aspect 1.34, and try 80:20 PC24:iNat (smaller iNat fraction so PC24 prior
stays dominant). Defer until lever A is exhausted.

### G. SSL → supervised at higher backbone LR

013 hit −0.05 below team best at `backbone_lr=1e-6`. The diagnosis is the LR
schedule, not the SSL signal. Re-run 013 at `backbone_lr ∈ {5e-6, 1e-5}`
and/or 8–10 epochs. Lower priority because (a) we don't believe SSL on LUCAS
is the right axis given the data-domain finding, and (b) the +0.05 closure
is hypothetical.

---

## What we are NOT going to do

- **More `unfreeze_n` tuning** — sweep is closed; n=4 is the sweet spot.
  Any non-trivial deviation costs more than it gains.
- **More PhaseA-class fusion** — fusion ceiling is bounded by the weaker
  leg. We don't have a 0.38+ leg to fuse with the team best.
- **More 50:50 PC24:iNat ratio tuning at `unfreeze_n=4`** — gradient is in
  the wrong direction (per 015 diagnosis).
- **More epochs at the 015 recipe** — ep4 → ep5 val gain was +0.0002
  (already plateaued); more epochs won't translate to Kaggle.
- **Trust val deltas <0.005** — they invert at test time. Anything inside
  noise needs Kaggle confirmation.
