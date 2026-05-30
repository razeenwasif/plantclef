# Experiment Report: i002 — BioCLIP 2.5 with Per-Species Image Capping (PlantCLEF 2026)

**Experiment folder:** `src_experiments/i002_bioclip25_cap_image/`
**Date written:** 2026-05-13
**Author:** Arjun

---

## 1. Goal

Re-run the BioCLIP 2.5 multi-task fine-tuning pipeline from
[010_bioclip25_end_to_end_finetune_multitask](../010_bioclip25_end_to_end_finetune_multitask/EXPERIMENT_REPORT.md)
on the **new, enlarged training metadata** produced by
[i001_data_download](../i001_data_download/), and add the **infrastructure
required to cap the per-species image count** so future runs can balance the
long-tail training distribution without rewriting the data pipeline.

In practice this experiment did three things:

1. Built the dataset-loading and capping infrastructure
   ([data/metadata_utils.py](data/metadata_utils.py),
   [scripts/verify_dataset_cap.py](scripts/verify_dataset_cap.py),
   capping-aware training flags in [train.py](train.py)).
2. Trained three staged fine-tuning runs on the full (un-capped) 2.65 M-image
   metadata: head-only, last-4-blocks, last-8-blocks.
3. Performed an extensive inference and Kaggle submission sweep, including a
   **logit-adjustment (long-tail calibration)** sweep on the last-4-blocks
   checkpoint.

The cap-at-1000 training scripts ([train_cap1000_head_only.sh](scripts/train_cap1000_head_only.sh),
[train_cap1000_last4_taxonomy.sh](scripts/train_cap1000_last4_taxonomy.sh))
are present but were **not executed** in this experiment — no
`outputs/cap1000_*` directories exist. They are kept as a ready-to-run starting
point for the follow-up experiment
[i003_bioclip25_cap_image_extra500](../i003_bioclip25_cap_image_extra500/).

---

## 2. Motivation

The PlantCLEF 2024 single-plant training set is heavily long-tailed: a small
minority of species have several hundred images and a few exceed a thousand,
while a long tail has very few. From the training-distribution log printed in
[outputs/head_only/run.log](outputs/head_only/run.log):

```
Median per species : 225
Mean per species   : 306
Max  per species   : 1191
Bin counts (species per image-count bin):
  [       1]:   145 species
  [       2]:   109 species
  [     3-5]:   445 species
  [    6-10]:   349 species
  [   11-20]:   445 species
  [   21-50]:   698 species
  [  51-100]:   595 species
  [ 101-250]: 1,358 species
  [ 251-500]: 1,387 species
  [501-1000]: 2,263 species
  [   >1000]:    12 species
```

A frequency-balanced sample (cap per species, optional WeightedRandomSampler)
is the standard remedy. This experiment puts the plumbing in place but defers
the actual cap-vs-no-cap comparison to subsequent experiments.

Two secondary motivations:

- **Larger training pool.** The new metadata CSV from i001
  ([metadata_filled_genus_family.csv](/root/workspace/PlantCLEF2026/src_experiments/i001_data_download/data/training_usage/metadata_filled_genus_family.csv))
  has **2,653,781 rows** vs the 1,408,033 used in experiment 010, so even
  without any cap the model sees ~1.7× more data.
- **Long-tail calibration at inference time.** A `--logit-adj-tau` flag was
  added to [infer_tiles_adaptive.py](infer_tiles_adaptive.py) so we can
  subtract `tau · log(prior[c])` from the logits before selection — a much
  cheaper alternative to retraining with logit-adjustment loss.

---

## 3. Relationship to Previous BioCLIP Experiments

This experiment builds directly on
[010_bioclip25_end_to_end_finetune_multitask](../010_bioclip25_end_to_end_finetune_multitask/EXPERIMENT_REPORT.md).
The training script, model, tile-inference scripts, and aggregation/selection
logic are nearly identical; only the data layer and a few CLI flags changed.

Concrete deltas from 010 → i002:

| Area | 010 | i002 |
|---|---|---|
| Metadata source | `PlantCLEF2024_single_plant_training_metadata.csv` + GBIF lookup | `metadata_filled_genus_family.csv` from i001 (genus/family pre-filled) |
| Rows in metadata | 1,408,033 | 2,653,781 |
| Taxonomy heads | species + genus + family + order + class (5) | species + genus + family (3); `order`/`class` columns are not in the new CSV |
| Loss weights | `1·sp + 0.30·gen + 0.15·fam + 0.05·ord + 0.02·cls` | `1·sp + 0.30·gen + 0.15·fam` |
| Image path resolution | Always `{root}/{species_id}/{image_name}` | Uses `image_path` column when present, fallback to `{root}/{species_id}/{image_name}` |
| Per-species capping | not implemented | implemented via `--max-images-per-species` (unused in actual runs) |
| Total-rows cap | not implemented | implemented via `--max-train-rows` (used only by smoke test) |
| Weighted sampler | not implemented | implemented via `--use-sample-weights` + `sample_weight` column (unused in actual runs) |
| Logit-adjustment inference | not implemented | added via `--logit-adj-tau` in adaptive inference |
| Training stages run | head-only, last-4, last-8, last-12, full-finetune | head-only, last-4, last-8 |
| Epochs per stage | 10 / 5 / 5 / 5 / 10 | 10 / 10 / 10 |

The follow-up experiment
[i003_bioclip25_cap_image_extra500](../i003_bioclip25_cap_image_extra500/) is
the natural sibling that actually uses the capping infrastructure.

---

## 4. Directory Overview

```
i002_bioclip25_cap_image/
├── README.md                       # NOTE: README is a verbatim copy of 010's README
├── EXPERIMENT_REPORT.md            # NOTE: also a verbatim copy of 010's report — does NOT describe i002
├── data/
│   ├── __init__.py
│   └── metadata_utils.py           # Reusable CSV loader + capping + weighted sampler
├── dataset.py                      # MultiTaskDataset, taxonomy merge, val split (species+genus+family only)
├── model.py                        # BioCLIP25MultiTask (species+genus+family heads)
├── transforms.py                   # Train/val transforms (unchanged from 010)
├── utils.py                        # AMP, scheduler, multi-task loss (sp + 0.3·gen + 0.15·fam)
├── train.py                        # Main training script; new --max-* and --use-sample-weights flags
├── validate.py                     # Standalone evaluation
├── infer_tiles.py                  # Fixed top-k tile inference
├── infer_tiles_adaptive.py         # Adaptive selection + --logit-adj-tau
├── scripts/
│   ├── smoke_test.sh
│   ├── train_head_only.sh          # Stage 1: full data, frozen backbone
│   ├── train_last_blocks.sh        # Stage 2a: full data, unfreeze last 4 blocks
│   ├── train_last_blocks_8.sh      # Stage 2b: full data, unfreeze last 8 blocks
│   ├── train_last_blocks_12.sh     # Defined but not run
│   ├── train_full_finetune.sh      # Defined but not run
│   ├── train_cap1000_head_only.sh  # Cap=1000 variant — defined but not run
│   ├── train_cap1000_last4_taxonomy.sh  # Cap=1000 + WeightedRandomSampler — defined but not run
│   ├── verify_dataset_cap.py       # Standalone CSV verification + cap dry-run
│   ├── infer_tiles.sh              # Parallel sweep (refers to 010 paths — see §13)
│   └── infer_best_adaptive.sh      # Default adaptive-sweep launcher
├── outputs/
│   ├── smoke_test/, smoke_test2/   # 200-sample sanity checks
│   ├── head_only/                  # Stage 1 training run + checkpoints
│   ├── last_blocks/                # Stage 2a (last 4 blocks) training run
│   ├── last_blocks_8/              # Stage 2b (last 8 blocks) training run
│   ├── head_only_infer/            # head_only adaptive inference, grid_4x4 ov=0.0
│   ├── head_only_infer_0.25_ol/    # head_only adaptive inference, ov=0.25
│   ├── head_only_infer_grid_max/   # head_only with agg=max
│   ├── last_blocks_infer/          # last_blocks/best.pt, agg=max
│   ├── last_blocks_infer_epoch_5/  # last_blocks/epoch_005.pt, agg=max
│   ├── last_blocks_infer_softmax_mean/         # last_blocks/best.pt, softmax_mean (best family overall)
│   ├── last_blocks_infer_softmax_mean_epoch_5/ # last_blocks/epoch_005.pt, softmax_mean
│   ├── last_blocks_8_infer/        # last_blocks_8/best.pt
│   ├── last_blocks_8_infer_e5/     # last_blocks_8/epoch_005.pt
│   └── la_tau{0.25,0.5,0.75}/      # Logit-adjustment sweep on last_blocks/best.pt
├── scores.csv                      # Kaggle scores for last_blocks_8 adaptive sweep
├── scores_head_only.csv            # Kaggle scores for head_only sweep (3 variants)
├── scores_last_blocks.csv          # Kaggle scores for last_blocks softmax_mean (both checkpoints)
├── scores_last_blocks_max.csv      # Kaggle scores for last_blocks max-agg
└── scores_logit_adjust.csv         # Kaggle scores for la_tau{0.25,0.5,0.75}
```

> **Heads-up:** [README.md](README.md) and [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)
> in this folder are byte-for-byte copies of the corresponding files in
> experiment 010; they do **not** describe i002. This `report.md` is the
> authoritative document for i002.

---

## 5. Dataset Construction

### 5.1 Source

The metadata CSV
[`/root/workspace/PlantCLEF2026/src_experiments/i001_data_download/data/training_usage/metadata_filled_genus_family.csv`](../i001_data_download/data/training_usage/metadata_filled_genus_family.csv)
is produced by experiment i001. From the training log:

```
Delimiter: ','
Rows: 2,653,781   unique species: 7,806
Columns: ['image_path', 'image_name', 'species_id', 'gbif_species_id',
          'scientific_name', 'license', 'source', 'genus', 'family',
          'url', 'gbif_occurrence_id']
genus  : 2,653,781/2,653,781 (100.0%)
family : 2,653,781/2,653,781 (100.0%)
order  : column not found
class  : column not found
```

Compared to 010, this CSV:

- Already includes an `image_path` column (no `{root}/{species_id}/{image_name}` reconstruction needed in the happy path).
- Has `genus` and `family` filled for **every** row (vs the GBIF-merged version in 010).
- Has no `order`/`class` columns — so the i002 model drops the order and class heads (see [model.py](model.py)).

Loading and normalisation are handled by [data/metadata_utils.py](data/metadata_utils.py):
delimiter sniffing, quote stripping, `species_id` normalisation (`"1234.0" → "1234"`),
and coverage reporting.

### 5.2 Path Resolution

[`resolve_image_paths`](dataset.py#L254) prefers the `image_path` column when
present and non-empty, and falls back to `{train_image_root}/{species_id}/{image_name}`
otherwise. The training log shows that in this run **all 2,653,781 rows
resolved via the `image_path` column** — no fallback was needed.

### 5.3 Stratified Validation Split

`build_val_split` ([dataset.py](dataset.py)) performs a stratified-by-species
10% split with `val_seed=42`; species with fewer than 5 images stay in train.
Resulting sizes (from [outputs/head_only/run.log](outputs/head_only/run.log)):

| Split | Rows | Unique species |
|---|---|---|
| train | 2,391,457 | 7,806 |
| val   |   262,324 | 7,309 |

The 497 species missing from val (7,806 − 7,309) are the long-tail species
with < 5 images.

### 5.4 Path Verification

[`scripts/verify_dataset_cap.py`](scripts/verify_dataset_cap.py) is a
standalone script that loads the CSV, applies the cap, samples N rows
(default 100) and checks that each `image_path` exists on disk. This is
**implemented but no recorded standalone log was found** in the outputs;
in practice the same checks are exercised by the training-time
`resolved_path.notna()` filter.

---

## 6. Per-Species Image Capping Strategy

### 6.1 Implementation

[`apply_max_images_per_species_cap`](data/metadata_utils.py#L141):

```python
def apply_max_images_per_species_cap(df, max_per_species, seed=42):
    if max_per_species <= 0:
        return df                       # cap disabled
    rng = np.random.RandomState(seed)
    parts = []
    for _, grp in df.groupby("species_id", sort=False):
        if len(grp) <= max_per_species:
            parts.append(grp)           # no-op for already-small species
        else:
            parts.append(grp.sample(n=max_per_species, random_state=rng))
    return pd.concat(parts).reset_index(drop=True)
```

Key properties:

- **Stable seed → reproducible cap.** Driven by `--cap-seed` (default 42).
- **Asymmetric**: only species above the cap are sub-sampled; rare species are untouched.
- **Applied to the training split only** (after `build_val_split`). The validation set is never capped, so val metrics remain comparable across cap values.
- **Composable with `--max-train-rows`** for a global cap after per-species capping.

The companion [`apply_max_train_rows_cap`](data/metadata_utils.py#L167)
performs an additional uniform downsample to a total row count (used by the
smoke test, with `max_train_rows=200`).

### 6.2 What Was Actually Run

| Run | `max_images_per_species` (from `train_config.json`) | `max_train_rows` |
|---|---|---|
| head_only      | 0 | 0 |
| last_blocks    | 0 | 0 |
| last_blocks_8  | 0 | 0 |
| smoke_test     | 0 | 200 |
| smoke_test2    | 0 | 200 |

**None of the production training runs used the per-species cap.** The
scripts [train_cap1000_head_only.sh](scripts/train_cap1000_head_only.sh) and
[train_cap1000_last4_taxonomy.sh](scripts/train_cap1000_last4_taxonomy.sh)
that *do* set `--max-images-per-species 1000` are present but were not
executed: there are no corresponding `outputs/cap1000_*` directories.

In other words: the **capping pipeline is implemented and tested by the
smoke test**, but the comparison "cap vs no cap" has not been run inside
this experiment folder. Class-distribution changes caused by capping are
therefore not measured here.

---

## 7. Metadata and Splits

### 7.1 Encoders

Built by `build_label_encoders` ([dataset.py](dataset.py)) and saved as JSON
under `outputs/{run}/encoders/`:

| Encoder | # classes (from `train_config.json`) |
|---|---|
| `idx_to_species.json` / `species_to_idx.json` | 7,806 |
| `idx_to_genus.json`   / `genus_to_idx.json`   | 1,446 |
| `idx_to_family.json`  / `family_to_idx.json`  | 181   |

These are stable across runs as long as the upstream metadata CSV does not
change.

### 7.2 Sample Weights

The metadata CSV may contain a `sample_weight` column (it would be used by
`build_weighted_sampler`). The training logs for the three production runs
show `use_sample_weights: False`. The
[train_cap1000_last4_taxonomy.sh](scripts/train_cap1000_last4_taxonomy.sh)
script switches it on, but again that script was not run here.

### 7.3 Duplicate / Missing Handling

- `species_id` is normalised to a clean string via `.str.split(".").str[0].str.strip()` to merge variants like `"1234.0"` and `"1234"`.
- Rows whose `resolved_path` ends up `None` are dropped before training (see [train.py](train.py)).
- Missing taxonomy labels (`NaN`) are encoded as `-1` and masked out of the auxiliary cross-entropy loss in [`compute_multitask_loss`](utils.py).

---

## 8. Model and Training Pipeline

### 8.1 Architecture

`BioCLIP25MultiTask` ([model.py](model.py)) is the same architecture as in 010
but with only two auxiliary heads:

```
Input image (224×224)
       │
BioCLIP 2.5 ViT-H/14 backbone   (hf-hub:imageomics/bioclip-2.5-vith14)
       │  (1024-dim embedding)
       ▼
LayerNorm → Linear(1024→1024) → GELU → Dropout(0.2)   ← shared MLP
       │
   ┌───┴───┬───────┐
species  genus  family
(7806)  (1446)  (181)
```

Multi-task loss ([utils.py](utils.py) `TAXONOMY_WEIGHTS`):

```
loss = species_CE  +  0.30 · genus_CE  +  0.15 · family_CE
```

Missing taxonomy labels are masked out per-sample.

### 8.2 Common Training Settings

All three production runs share:

- `--precision bf16` (no `GradScaler`)
- `--weight-decay 1e-4`
- `--label-smoothing 0.1`
- `--warmup-epochs 1` followed by cosine decay
- `--use-taxonomy-heads`
- `--val-fraction 0.1`, `--val-seed 42`
- 2 GPUs via `torchrun --standalone --nproc_per_node=2` (`world_size=2` in each `train_config.json`)

### 8.3 Stage 1: Head-Only Training (`outputs/head_only/`)

Launched by [scripts/train_head_only.sh](scripts/train_head_only.sh).

| Parameter | Value |
|---|---|
| Frozen / Unfrozen | Backbone fully frozen; head + shared MLP only |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 (unused) |
| batch_size | 512 per GPU × 2 = 1024 |
| grad_accum | 2 (effective batch = 2048) |
| epochs | 10 |
| epoch time | ~2,748 s (~46 min) |
| total_steps | 11,680 (warmup 1,168) |
| trainable params | 12,823,769 (logged in `run.log`) |
| best ckpt | [outputs/head_only/checkpoints/best.pt](outputs/head_only/checkpoints/best.pt) |
| config | [outputs/head_only/train_config.json](outputs/head_only/train_config.json) |

**Per-epoch val metrics** ([outputs/head_only/metrics.csv](outputs/head_only/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 7.412 | 1.3553 | 0.7312 | 0.9096 | 0.8811 | 0.9341 |
| 3 | 3.326 | 0.9557 | 0.7951 | 0.9437 | 0.9108 | 0.9494 |
| 5 | 3.165 | 0.9028 | 0.8053 | 0.9483 | 0.9159 | 0.9536 |
| 7 | 3.100 | 0.8811 | 0.8101 | 0.9500 | 0.9175 | 0.9552 |
| 9 | 3.071 | 0.8748 | 0.8114 | 0.9504 | 0.9180 | 0.9555 |
| **10** | **3.068** | **0.8739** | **0.8117** | **0.9505** | **0.9181** | **0.9555** |

Best top-5 = **0.9505** at epoch 10 (best.pt). Note that epoch 1 of i002's
head-only is already higher than the **final** epoch-10 of 010's head-only
(0.9214 → 0.9505), reflecting the larger and cleaner training set.
Two duplicate epoch-1 rows are present in `metrics.csv` from a re-run
that re-validated the same epoch_001 checkpoint with slightly different
shuffle seeds before continuing — they are not used downstream.

### 8.4 Stage 2a: Unfreeze Last 4 Blocks (`outputs/last_blocks/`)

Launched by [scripts/train_last_blocks.sh](scripts/train_last_blocks.sh),
resuming weights-only from `outputs/head_only/checkpoints/best.pt`.

| Parameter | Value |
|---|---|
| Frozen / Unfrozen | Last 4 transformer blocks + `ln_post`/`proj` unfrozen |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 |
| batch_size | 128 per GPU × 2 = 256 |
| grad_accum | 4 (effective batch = 1024) |
| epochs | 10 |
| epoch time | ~3,560 s (~59 min) |
| total_steps | 23,350 (warmup 2,335) |
| best ckpt | [outputs/last_blocks/checkpoints/best.pt](outputs/last_blocks/checkpoints/best.pt) |
| config | [outputs/last_blocks/train_config.json](outputs/last_blocks/train_config.json) |

**Per-epoch val metrics** ([outputs/last_blocks/metrics.csv](outputs/last_blocks/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 3.052 | 0.8786 | 0.8098 | 0.9505 | 0.9186 | 0.9569 |
| 3 | 2.927 | 0.8337 | 0.8169 | 0.9547 | 0.9232 | 0.9596 |
| 5 | 2.842 | 0.8097 | 0.8224 | 0.9567 | 0.9254 | 0.9609 |
| 7 | 2.794 | 0.7942 | 0.8260 | 0.9579 | 0.9268 | 0.9615 |
| 9 | 2.769 | 0.7894 | 0.8266 | 0.9583 | 0.9271 | 0.9618 |
| **10** | **2.765** | **0.7892** | **0.8266** | **0.9583** | **0.9270** | **0.9618** |

Best val top-5 = **0.9583** at epoch 10. Both `best.pt` and `epoch_005.pt`
are used downstream (see §9.3).

### 8.5 Stage 2b: Unfreeze Last 8 Blocks (`outputs/last_blocks_8/`)

Launched by [scripts/train_last_blocks_8.sh](scripts/train_last_blocks_8.sh),
resuming weights-only from `outputs/last_blocks/checkpoints/best.pt`.

| Parameter | Value |
|---|---|
| Frozen / Unfrozen | Last 8 transformer blocks unfrozen |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 |
| batch_size | 128 per GPU × 2 = 256 |
| grad_accum | 4 |
| epochs | 10 |
| epoch time | ~4,300 s (~72 min) |
| best ckpt | [outputs/last_blocks_8/checkpoints/best.pt](outputs/last_blocks_8/checkpoints/best.pt) |
| config | [outputs/last_blocks_8/train_config.json](outputs/last_blocks_8/train_config.json) |

**Per-epoch val metrics** ([outputs/last_blocks_8/metrics.csv](outputs/last_blocks_8/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 2.776 | 0.7978 | 0.8238 | 0.9576 | 0.9261 | 0.9616 |
| 3 | 2.730 | 0.7813 | 0.8272 | 0.9591 | 0.9282 | 0.9629 |
| 5 | 2.675 | 0.7664 | 0.8308 | 0.9604 | 0.9297 | 0.9639 |
| 7 | 2.636 | 0.7561 | 0.8331 | 0.9613 | 0.9309 | 0.9643 |
| 9 | 2.616 | 0.7517 | 0.8335 | 0.9615 | 0.9312 | 0.9645 |
| **10** | **2.611** | **0.7516** | **0.8336** | **0.9615** | **0.9312** | **0.9646** |

Best val top-5 = **0.9615** at epoch 10.

### 8.6 Stages Not Run

`scripts/train_last_blocks_12.sh` and `scripts/train_full_finetune.sh` are
both present and point at the expected resume checkpoints, but no
`outputs/last_blocks_12/` or `outputs/full_finetune/` directory exists.
These stages were left to a follow-up run and are out of scope for i002.

### 8.7 Cross-Stage Summary

| Run | Blocks Unfrozen | Epochs | Best Val Top-1 | Best Val Top-5 | Val Loss (best) | Best Ckpt |
|---|---|---|---|---|---|---|
| smoke_test  | 0  | 1 (200 samples) | 0.0000 | 0.0000 | 9.02 | epoch_001 |
| smoke_test2 | 0  | 1 (200 samples) | 0.0000 | 0.0000 | 8.99 | epoch_001 |
| head_only      | 0  | 10 | 0.8117 | 0.9505 | 0.8739 | epoch_010 |
| last_blocks    | 4  | 10 | 0.8266 | 0.9583 | 0.7892 | epoch_010 |
| last_blocks_8  | 8  | 10 | 0.8336 | 0.9615 | 0.7516 | epoch_010 |

For context, the equivalent stages from 010 (different metadata, 5-block-stage
epochs only) scored at best Top-5 = 0.9214 / 0.9323 / 0.9364 — i002 is
consistently +0.025 to +0.029 higher, attributable to (a) the larger
training pool, (b) longer schedules (10 vs 5 epochs in the block-unfreezing
stages), and (c) the cleaner pre-filled taxonomy in the i001 CSV.

---

## 9. Inference and Submission Pipeline

All inference uses [`infer_tiles_adaptive.py`](infer_tiles_adaptive.py), which
in one pass over the test set produces submissions for several aggregation +
selection-mode combinations. Test set: **2,105 quadrat images** at
`/workspace/plantclef/raw/test` (from `summary.json` of each run).

### 9.1 Tiling and Aggregation

All scored runs in i002 use the same tile geometry that 010 found best:

- `tile_mode = grid_4x4`
- `tile_size = 448`
- `overlap = 0.0` (one exception: `head_only_infer_0.25_ol` uses `overlap = 0.25`)

Aggregation modes used:

- `softmax_mean` — dominant in i002; used in 10 of 12 inference directories.
- `max` — tested only for head_only (`head_only_infer_grid_max/`) and the two `last_blocks_infer*` (non-`softmax_mean`) directories.

### 9.2 Selection Modes

Implemented in [infer_tiles_adaptive.py](infer_tiles_adaptive.py):

| Mode | CLI arg | Values swept |
|---|---|---|
| `fixed_topk` | `--top-ks` | 1, 2, 3, 4, 5 |
| `prob_threshold` | `--prob-thresholds` | 0.02, 0.03, 0.05 (logit-adjust runs add 0.07) |
| `relative_threshold` | `--relative-thresholds` | 0.15, 0.20, 0.25, 0.30 |
| `gap` | `--gap-ratios` | 0.40, 0.50, 0.60 |
| min_k / max_k bounds | `--min-k` / `--max-k` | (1, 10) most runs; (2, 10) for last_blocks_8 |

### 9.3 Inference Families Run

| Inference dir | Source ckpt | Agg | Overlap | logit_adj_tau | Source script |
|---|---|---|---|---|---|
| [outputs/head_only_infer/](outputs/head_only_infer/) | `head_only/best.pt` | softmax_mean | 0.0 | 0 | scripts/infer_best_adaptive.sh |
| [outputs/head_only_infer_0.25_ol/](outputs/head_only_infer_0.25_ol/) | `head_only/best.pt` | softmax_mean | 0.25 | 0 | manual variant |
| [outputs/head_only_infer_grid_max/](outputs/head_only_infer_grid_max/) | `head_only/best.pt` | max | 0.0 | 0 | manual variant |
| [outputs/last_blocks_infer/](outputs/last_blocks_infer/) | `last_blocks/best.pt` | max | 0.0 | 0 | manual variant |
| [outputs/last_blocks_infer_epoch_5/](outputs/last_blocks_infer_epoch_5/) | `last_blocks/epoch_005.pt` | max | 0.0 | 0 | manual variant |
| [outputs/last_blocks_infer_softmax_mean/](outputs/last_blocks_infer_softmax_mean/) | `last_blocks/best.pt` | softmax_mean | 0.0 | 0 | manual variant |
| [outputs/last_blocks_infer_softmax_mean_epoch_5/](outputs/last_blocks_infer_softmax_mean_epoch_5/) | `last_blocks/epoch_005.pt` | softmax_mean | 0.0 | 0 | manual variant |
| [outputs/last_blocks_8_infer/](outputs/last_blocks_8_infer/) | `last_blocks_8/best.pt` | softmax_mean | 0.0 | 0 | infer_best_adaptive.sh |
| [outputs/last_blocks_8_infer_e5/](outputs/last_blocks_8_infer_e5/) | `last_blocks_8/epoch_005.pt` | softmax_mean | 0.0 | 0 | manual variant |
| [outputs/la_tau0.25/](outputs/la_tau0.25/) | `last_blocks/best.pt` | softmax_mean | 0.0 | **0.25** | manual variant |
| [outputs/la_tau0.5/](outputs/la_tau0.5/) | `last_blocks/best.pt` | softmax_mean | 0.0 | **0.50** | manual variant |
| [outputs/la_tau0.75/](outputs/la_tau0.75/) | `last_blocks/best.pt` | softmax_mean | 0.0 | **0.75** | manual variant |

Each directory contains one sub-folder per `{agg}_{selection_key}` combo,
each holding `submission.csv`, `predictions_scored.csv`, `run_config.json`,
and `summary.json`. Inference throughput logged in `summary.json` is roughly
5–6 images/sec on grid_4x4 with `bf16`.

### 9.4 Logit Adjustment

[`build_logit_adjustment`](infer_tiles_adaptive.py#L476) computes
Laplace-smoothed log-priors over the full metadata CSV and returns
`tau · log(prior + eps)`. At inference, this vector is subtracted from the
aggregated per-image logit before selection
([infer_tiles_adaptive.py:800-801](infer_tiles_adaptive.py)):

```python
if logit_adjustment is not None:
    agg_logit = agg_logit.to(device) - logit_adjustment
```

The training CSV used to build the priors is the same i001
`metadata_filled_genus_family.csv`. Sweeping `tau ∈ {0.25, 0.5, 0.75}` on
`last_blocks/best.pt` produced the `la_tau*/` directories.

### 9.5 Saved Logits

When `--save-logits` is set (default in [infer_best_adaptive.sh](scripts/infer_best_adaptive.sh)),
the per-image aggregated logits are saved to
`{output_dir}/logits/softmax_mean_logits.pt` for later ensembling. Saved
in: `head_only_infer/`, `head_only_infer_0.25_ol/`, `head_only_infer_grid_max/`,
`last_blocks_infer/`, `last_blocks_infer_epoch_5/`,
`last_blocks_infer_softmax_mean/`, `last_blocks_infer_softmax_mean_epoch_5/`,
`last_blocks_8_infer/`, `last_blocks_8_infer_e5/`. The `la_tau*/` runs were
launched with `--save-logits` disabled.

---

## 10. Output Files

### 10.1 Training Outputs

Each `outputs/{run}/` directory contains:

```
checkpoints/
  best.pt            # top-5-acc-best checkpoint
  last.pt            # latest (for resuming)
  epoch_NNN.pt       # per-epoch checkpoints, every --save-every epochs
encoders/
  idx_to_species.json, species_to_idx.json,
  idx_to_genus.json,   genus_to_idx.json,
  idx_to_family.json,  family_to_idx.json
metrics.csv          # per-epoch row, appended each epoch
metrics.json         # full {"history": [...]} dump
train_config.json    # all CLI args + derived metadata
run.log              # full training log
```

Checkpoint sizes (approx): head_only 4.10 GB, last_blocks 4.74 GB,
last_blocks_8 5.37 GB — the increase tracks the additional unfrozen
parameters that get an optimizer state.

### 10.2 Inference Outputs

Each `outputs/{infer_dir}/{agg}_{selection_key}/` contains:

```
submission.csv          # PlantCLEF Kaggle format (quadrat_id, species_ids)
predictions_scored.csv  # per-prediction rows with scores
run_config.json         # full inference config used
summary.json            # n_images, n_errors, total_secs, throughput
```

All scored submissions span **2,105 rows + header** (verified via `wc -l`).

### 10.3 Scores

The five score CSVs are Kaggle public-leaderboard exports. Totals (counting
"COMPLETE" lines only):

| File | Rows | Notes |
|---|---|---|
| [scores.csv](scores.csv) | 30 | last_blocks_8 adaptive sweep (best.pt + epoch_005.pt). One `submit_failed`. |
| [scores_head_only.csv](scores_head_only.csv) | 41 | head_only across three inference dirs. |
| [scores_last_blocks.csv](scores_last_blocks.csv) | 30 | last_blocks softmax_mean (both best.pt and epoch_005.pt). |
| [scores_last_blocks_max.csv](scores_last_blocks_max.csv) | 30 | last_blocks max-agg (both checkpoints). |
| [scores_logit_adjust.csv](scores_logit_adjust.csv) | 32 | la_tau{0.25, 0.5, 0.75} sweep. |
| **Total scored submissions** | **162** | One additional submission status is `submit_failed`. |

---

## 11. Results and Observations

### 11.1 Top 15 Public Leaderboard Submissions

Sorted from all five score CSVs.

| Rank | Public Score | Source dir | Selection |
|---|---|---|---|
| 1  | **0.41165** | last_blocks_infer_softmax_mean | softmax_mean_probT0.05 |
| 2  | 0.40590 | la_tau0.25 | softmax_mean_probT0.03 |
| 3  | 0.40486 | last_blocks_8_infer_e5 | softmax_mean_top3 |
| 4  | 0.40451 | last_blocks_infer_softmax_mean | softmax_mean_top3 |
| 5  | 0.40363 | last_blocks_infer_softmax_mean | softmax_mean_relT0.3 |
| 6  | 0.40163 | last_blocks_infer_softmax_mean_epoch_5 | softmax_mean_probT0.05 |
| 7  | 0.40147 | last_blocks_infer_softmax_mean_epoch_5 | softmax_mean_top3 |
| 8  | 0.40022 | last_blocks_infer_softmax_mean | softmax_mean_top1 / softmax_mean_top2 (tie) |
| 9  | 0.39989 | last_blocks_infer_softmax_mean | softmax_mean_relT0.15 |
| 10 | 0.39983 | last_blocks_infer_softmax_mean | softmax_mean_relT0.2 |
| 11 | 0.39979 | last_blocks_infer_softmax_mean | softmax_mean_relT0.25 |
| 12 | 0.39878 | la_tau0.25 | softmax_mean_top2 |
| 13 | 0.39813 | last_blocks_8_infer | softmax_mean_probT0.03 |
| 14 | 0.39768 | last_blocks_infer_softmax_mean | softmax_mean_probT0.03 |
| 15 | 0.39731 | la_tau0.25 | softmax_mean_top3 |

(Ties from `top1`/`top2` arise because the selection in this script enforces
`min_k = 2`: a "top-1" request is bumped up to 2 species.)

### 11.2 Best Score per Inference Dir

| Inference dir | Best public score |
|---|---|
| head_only_infer                       | 0.39416 (softmax_mean_top4) |
| head_only_infer_0.25_ol               | 0.37134 |
| head_only_infer_grid_max              | 0.36325 |
| last_blocks_infer (max)               | 0.37509 |
| last_blocks_infer_epoch_5 (max)       | 0.37745 |
| **last_blocks_infer_softmax_mean**    | **0.41165** ← best in i002 |
| last_blocks_infer_softmax_mean_epoch_5| 0.40163 |
| last_blocks_8_infer                   | 0.39813 |
| last_blocks_8_infer_e5                | 0.40486 |
| la_tau0.25                            | 0.40590 |
| la_tau0.5                             | 0.38908 |
| la_tau0.75                            | 0.34620 |

### 11.3 Observations

- **Best i002 score (0.41165) exceeds best 010 score (0.39140).** The
  improvement comes from inference on the **`last_blocks` (4-block)**
  checkpoint with `softmax_mean + probT 0.05`, not from the deeper
  `last_blocks_8` checkpoint. The same checkpoint with `top3` scored 0.40451.
  The single submission whose checkpoint trained for more epochs
  (`last_blocks_8`, last_blocks_8_infer_e5/top3 → 0.40486) is competitive but
  not better.

- **The mismatch between validation and Kaggle score persists.** Val top-5
  is monotone in unfrozen blocks (0.9505 → 0.9583 → 0.9615), but the best
  Kaggle public score is on the **middle** stage (last_blocks/4-block). This
  echoes the 010 finding that last_blocks_12 had the best val score but a
  lower Kaggle score than last_blocks_8.

- **softmax_mean clearly beats max.** For `last_blocks/best.pt` the best max
  score is 0.37509 (last_blocks_infer) vs 0.41165 for softmax_mean —
  about +4 absolute points, consistent with 010.

- **Logit adjustment helps mildly at low tau.** `la_tau0.25` reached 0.40590
  (probT 0.03), narrowly below the best uncalibrated run (0.41165) but
  ahead of any individual top-k variant from the same checkpoint without
  adjustment (`last_blocks_infer_softmax_mean/softmax_mean_probT0.03 = 0.39768`
  → +0.00822 from adjustment at tau=0.25). Higher tau (0.5, 0.75) consistently
  underperforms: `la_tau0.75` peaks at 0.34620.

- **Overlap=0.25 hurts head_only.** `head_only_infer_0.25_ol` peaks at
  0.37134 vs `head_only_infer` at 0.39416. Same finding as in 010.

- **Best selection mode is configuration-dependent.** The all-stages winner
  is `prob_threshold 0.05`, but `top3` and `relT 0.3` are within 0.005 of it.
  No single selection rule dominates across checkpoints.

### 11.4 Bottom Submissions

The weakest scores all come from the high-tau logit-adjustment runs
(`la_tau0.75` top1/relT0.15/relT0.2 land near 0.29) and from `head_only`
with `max` aggregation at overlap=0.25 (`head_only_infer_grid_max/max_relT0.15 = 0.32668`).

---

## 12. Reproducibility

### 12.1 Training

```bash
cd src_experiments/i002_bioclip25_cap_image

# Stage 1: head-only (10 epochs, frozen backbone)
bash scripts/train_head_only.sh 2

# Stage 2a: last 4 blocks (10 epochs, resume weights from stage 1 best.pt)
bash scripts/train_last_blocks.sh outputs/head_only/checkpoints/best.pt 2

# Stage 2b: last 8 blocks (10 epochs, resume weights from stage 2a best.pt)
bash scripts/train_last_blocks_8.sh outputs/last_blocks/checkpoints/best.pt 2
```

Each script auto-detects 2 GPUs from its first argument and forwards the
arguments to `torchrun --standalone --nproc_per_node=N train.py …`. The
metadata CSV path is hardcoded inside the scripts to the i001 output.

### 12.2 Dataset Cap Verification

```bash
python scripts/verify_dataset_cap.py \
  --metadata-csv /root/workspace/PlantCLEF2026/src_experiments/i001_data_download/data/training_usage/metadata_filled_genus_family.csv \
  --max-images-per-species 1000 \
  --cap-seed 42 \
  --sample-check 200
```

### 12.3 Inference

The default sweep ([scripts/infer_best_adaptive.sh](scripts/infer_best_adaptive.sh))
runs the full adaptive sweep on one checkpoint and dumps logits:

```bash
# Adaptive sweep on last_blocks_8 best checkpoint
bash scripts/infer_best_adaptive.sh \
  outputs/last_blocks_8/checkpoints/best.pt \
  /workspace/plantclef/raw/test \
  outputs/last_blocks_8_infer
```

The logit-adjustment sweeps were launched as the same `infer_tiles_adaptive.py`
invocation with `--logit-adj-tau {0.25, 0.5, 0.75}` and
`--metadata-csv {i001 CSV}` added. The `la_tau*` `run_config.json` files
record the exact arguments per run.

### 12.4 Seeds and Determinism

- `val_seed = 42` is fixed for the train/val split.
- `cap_seed = 42` is fixed for `apply_max_images_per_species_cap` and `apply_max_train_rows_cap`.
- Random-augmentation seeds in the training DataLoader are *not* explicitly fixed; minor run-to-run variance in train loss is expected (cf. the two near-identical epoch-1 rows in `outputs/head_only/metrics.csv`).

---

## 13. Limitations and Notes

- **The folder name promises "cap image" but no production run was capped.**
  All training data shown in §8 used the full 2.39 M-image post-split training
  set. Capping is implemented, tested by smoke runs, and ready to use — but
  cap-vs-no-cap comparison was deferred to
  [i003_bioclip25_cap_image_extra500](../i003_bioclip25_cap_image_extra500/).

- **README.md and EXPERIMENT_REPORT.md in this folder are stale copies of 010's files.** They describe experiment 010's last_blocks_12 / full_finetune
  stages, the 5-head taxonomy loss, and 010's Kaggle results — none of which
  apply here. This `report.md` supersedes them for i002.

- **No order/class taxonomy heads.** The i001 metadata CSV does not carry
  `order`/`class` columns. The i002 model has only species + genus + family
  heads, vs the 5-head 010 model. This is a substantive architectural
  difference that may explain part of the val-vs-Kaggle gap.

- **`scripts/infer_tiles.sh` is stale.** It points to the experiment 010
  directory (`cd /root/workspace/PlantCLEF2026/src_experiments/010_bioclip25_end_to_end_finetune_multitask`)
  and writes to that experiment's outputs. It was *not* used for i002's
  inference; the [infer_best_adaptive.sh](scripts/infer_best_adaptive.sh)
  script and one-off manual invocations of `infer_tiles_adaptive.py`
  produced the `outputs/*_infer*` directories.

- **No standalone `validate.py` runs recorded.** The `validate.py` script is
  present but no `outputs/*/eval/` directories were produced; val metrics
  in `metrics.csv` come from the in-training validation loop.

- **WeightedRandomSampler is implemented but unused.** No production run set
  `--use-sample-weights`; this is queued for i003.

- **Last-blocks-12 and full-finetune scripts exist but were not executed.**
  The follow-up direction stated at the end of the 010 report (re-run
  full-finetune from last-blocks-12) is not addressed here.

- **The "submit_failed" entry in [scores.csv](scores.csv)** corresponds to
  `last_blocks_8_infer/softmax_mean_probT0.05` — its Kaggle submission did
  not complete. The on-disk CSV is intact and could be resubmitted.

- **Two duplicate epoch-1 rows in [outputs/head_only/metrics.csv](outputs/head_only/metrics.csv)** reflect a re-validation
  after a transient restart; they did not perturb the cosine schedule, and
  the second-epoch onwards rows are continuous in `train_loss` and `head_lr`.

- **Statistics not measured.** Class-distribution change under capping (since
  no capped run was completed), per-species precision/recall, per-quadrat
  error analysis, ensemble of saved `softmax_mean_logits.pt` files.

---

## 14. Summary

i002 served two purposes: **(a) port the BioCLIP 2.5 multi-task pipeline to
the new i001 metadata** (which produced a stronger model just by adding
data), and **(b) add the dataset-capping and logit-adjustment infrastructure**
needed to actually study long-tail balancing in later experiments.

What was implemented:

- Re-usable `data/metadata_utils.py` with per-species cap, total-rows cap,
  weighted sampler, and distribution printing.
- `--max-images-per-species`, `--max-train-rows`, `--use-sample-weights`,
  `--cap-seed` flags in `train.py`.
- `--logit-adj-tau` and `--metadata-csv` flags in `infer_tiles_adaptive.py`,
  with `build_logit_adjustment` (Laplace-smoothed log-prior).
- A standalone `scripts/verify_dataset_cap.py` dry-run tool.

What was run:

- Three staged training runs on the **full** 2.39 M training split:
  `head_only` (top-5 = 0.9505), `last_blocks` (4 blocks, 0.9583),
  `last_blocks_8` (8 blocks, 0.9615). Smoke tests passed.
- 162 Kaggle submissions across 12 inference directories, including a
  logit-adjustment sweep at `tau ∈ {0.25, 0.5, 0.75}`.

What was produced:

- Three full training-output trees with per-epoch checkpoints (10 epochs each).
- Saved aggregated logits for nine of the twelve inference dirs (potential
  for ensembling).
- Five Kaggle score CSVs; **best public score 0.41165** on
  `last_blocks_infer_softmax_mean / softmax_mean_probT0.05` (4-block
  checkpoint, full data, `softmax_mean` + `prob_threshold 0.05`,
  no logit adjustment).

What is planned but not completed:

- Cap-vs-no-cap comparison (deferred to i003).
- WeightedRandomSampler runs (deferred to i003).
- Last-12-blocks and full-finetune stages.
- Standalone `validate.py` runs and per-class diagnostics.
- Logit ensembling across the saved `.pt` files.
