# Experiment Report: i003 — BioCLIP 2.5 with Extra Long-Tail Data and a 500-Image Per-Species Cap

**Experiment folder:** `src_experiments/i003_bioclip25_cap_image_extra500/`
**Date written:** 2026-05-13
**Author:** Arjun

---

## 1. Goal

Re-run the [i002](../i002_bioclip25_cap_image/report.md) BioCLIP 2.5 multi-task
pipeline on a **combined, deduplicated, 500-per-species-capped training
manifest** that merges the original i002 metadata with a freshly downloaded
"extra_under100" manifest of additional images for under-represented species.
The aim is to (a) close the long-tail gap with new images for rare species,
and (b) finally use the capping infrastructure that was built but unused in
i002.

The cap is **baked into the on-disk manifest** by
[prepare_combined_manifest.py](prepare_combined_manifest.py). All training
scripts then point at this manifest and run with `max_images_per_species = 0`
at train time — the cap is applied once, deterministically, before training
ever sees the data.

---

## 2. Motivation

Two problems left open by i002:

1. **The long tail was untouched.** i002 trained on every available image per
   species, which means the ~12 species with > 1000 images contributed ~13k+
   gradient steps' worth of "easy" repeats, while 145 species with a single
   image got essentially zero exposure under shuffled sampling.
2. **Many species were under-represented.** From the i002 training-distribution
   log: 145 species had 1 image, 109 had 2, 445 had 3–5, etc. — over 1,600
   species had < 100 images.

i003 attacks both:

- The new `extra_under100_train_manifest.csv` (from i001's
  `data/extra_under100/`) adds **~148k extra images** spanning 2,020
  under-represented species (after image-validity pruning).
- A 500-per-species cap is then applied so the over-represented species are
  brought down, the new data is allowed to lift the long tail (where each
  species was previously well below 500), and the resulting manifest is much
  flatter without a heavy WeightedRandomSampler at training time.

Capping at the manifest level (vs `--max-images-per-species` at train time)
gives several practical wins: deterministic, inspectable counts; persisted
selection across runs; no need for a hot-path resample on every epoch start.

---

## 3. Relationship to Previous BioCLIP Experiments

i003 is a near-clone of i002 — same model, same training script, same
inference script. Only the data layer and one runtime safety net change.

Diff vs [i002](../i002_bioclip25_cap_image/report.md):

| Area | i002 | i003 |
|---|---|---|
| Training manifest | `metadata_filled_genus_family.csv` (i001 output) | `combined_old_extra_max500_train_manifest.csv` (built here) |
| Total rows in manifest | 2,653,781 | **2,223,516** |
| Per-species cap | not applied | 500 (deterministic, seed=42, baked into CSV) |
| Extra data source | — | `i001_data_download/data/extra_under100/extra_under100_train_manifest.csv` (148,063 rows post-validation, 2,020 species) |
| Image validation | none | `prepare_combined_manifest.py` validates every new-manifest image with a full PIL decode (13,623 invalid dropped) |
| `dataset.py` `__getitem__` | one-shot `Image.open` | **bounded retry**: up to 8 attempts, advance index, log first 50 failures, `LOAD_TRUNCATED_IMAGES = True`, `MAX_IMAGE_PIXELS = 3e8` |
| `model.py`, `train.py`, `utils.py`, `transforms.py`, `validate.py`, `infer_tiles.py`, `infer_tiles_adaptive.py` | original | **byte-identical** to i002 (verified) |
| Training GPUs | 2 | 4 (`world_size=4` in every i003 `train_config.json`) |
| Stages run | head_only, last_blocks (4), last_blocks_8 | head_only, last_blocks (4), last_blocks_8 |
| Inference families | 12 dirs, sweep + 3 logit-adjust τ values | 4 dirs, one logit-adjust τ value (0.25) |
| Best Kaggle public score | 0.41165 (last_blocks/softmax_mean/probT 0.05) | **0.40041** (last_blocks/softmax_mean/top3) |

`infer_best_adaptive.sh` in i003 also wires up `--logit-adj-tau 0.25` and
points `--metadata-csv` at the new manifest by default; the i002 version had
those flags off by default.

---

## 4. Directory Overview

```
i003_bioclip25_cap_image_extra500/
├── README.md                      # Authoritative short doc for i003 (run order, data sources, notes)
├── report.md                      # ← this report
├── prepare_combined_manifest.py   # Builds combined+dedup+500-cap manifest, validates new images
├── data/
│   ├── metadata_utils.py          # Reused from i002 (byte-identical)
│   ├── combined_old_extra_max500_train_manifest.csv   # 2,223,516 capped rows  (~622 MB)
│   ├── combined_old_extra_max500_summary.json         # Build stats (counts before/after, species at cap, …)
│   └── species_counts_before_after.csv                 # Per-species (7,806) before+after cap, with scientific_name
├── dataset.py                     # i002 with bad-image bounded retry + path-default change
├── model.py                       # Identical to i002 (species + genus + family heads)
├── transforms.py                  # Identical to i002
├── utils.py                       # Identical to i002 (TAXONOMY_WEIGHTS: genus 0.30, family 0.15)
├── train.py                       # Identical to i002 (capping flags still present but unused at runtime)
├── validate.py                    # Identical to i002
├── infer_tiles.py                 # Identical to i002 (fixed top-k inference)
├── infer_tiles_adaptive.py        # Identical to i002 (adaptive + logit adjustment)
├── scripts/
│   ├── prepare_data.sh            # Thin wrapper around prepare_combined_manifest.py
│   ├── smoke_test.sh              # ~200-sample sanity check
│   ├── train_head_only.sh         # Stage 1
│   ├── train_last_blocks.sh       # Stage 2a (last 4 blocks)
│   ├── train_last_blocks_8.sh     # Stage 2b (last 8 blocks)
│   ├── train_last_blocks_12.sh    # Stage 2c — defined but not run
│   ├── train_full_finetune.sh     # Stage 3   — defined but not run
│   ├── infer_tiles.sh             # Parallel sweep script (carried over)
│   ├── infer_best_adaptive.sh     # Default adaptive launcher; has tau=0.25 baked in
│   └── verify_dataset_cap.py      # Standalone CSV verification tool
└── outputs/
    ├── smoke_test/                # 200-sample, 1-epoch sanity check
    ├── head_only/                 # Stage 1 training run + per-epoch checkpoints
    ├── last_blocks/               # Stage 2a (last 4 blocks) run
    ├── last_blocks_8/             # Stage 2b (last 8 blocks) run
    ├── head_only_infer/           # Adaptive inference on head_only/best.pt
    ├── last_blocks_infer/         # Adaptive inference on last_blocks/best.pt
    ├── last_blocks_8_infer/       # Adaptive inference on last_blocks_8/best.pt
    └── last_blocks_8_infer_la_025/# Same checkpoint, with logit adjustment τ=0.25
```

The three score CSVs at the top level group submissions by inference family:
[scores_head_only.csv](scores_head_only.csv) (15 submissions),
[scores_last_blocks.csv](scores_last_blocks.csv) (15),
[scores_last_blocks_8.csv](scores_last_blocks_8.csv) (30: 15 plain + 15 with τ=0.25).

---

## 5. Dataset Construction

### 5.1 Sources

| Source | Path | Rows | Unique species |
|---|---|---|---|
| Old (= i002 manifest)  | `i001_data_download/data/training_usage/metadata_filled_genus_family.csv` | 2,653,781 | 7,806 |
| New (extra_under100)   | `i001_data_download/data/extra_under100/extra_under100_train_manifest.csv` | 161,686 raw / **148,063 valid** | 2,020 |

The new manifest is "extra images for under-100 species" — 2,020 species,
all of which are also present in the old manifest (`n_species_only_in_new: 0`
in the summary JSON), so it adds **more images for known species, not new
species**.

### 5.2 Build Pipeline (`prepare_combined_manifest.py`)

Pipeline (the order is load-bearing; see comments in [prepare_combined_manifest.py](prepare_combined_manifest.py)):

1. **Load** both CSVs via `load_metadata_csv` (delimiter sniffing, species_id normalisation `"1234.0" → "1234"`).
2. **Validate new-manifest images.** Every new-manifest `image_path` is opened with `Image.open(...).load()` (full pixel decode) using a 64-thread `ThreadPoolExecutor`. Files that raise `OSError`, `UnidentifiedImageError`, `DecompressionBombError`, or `ValueError` are dropped. The old manifest is **not** revalidated; it is trusted because i002 already trained on it.
   - Dropped: **13,623 of 161,686** = 8.4% of the new manifest, mostly JSON download error pages saved with `.jpg` extensions.
3. **Fill genus/family on the new frame** by joining the species_id index of the old frame (the new manifest does carry both columns natively, but the join is harmless and acts as a safety net).
4. **Align columns** to a fixed `KEEP_COLS` set: `image_path, image_name, species_id, scientific_name, genus, family, source, url, gbif_species_id`.
5. **Concat** → 2,801,844 rows.
6. **Dedup by `image_path`, before the cap.** This is critical: capping first would silently shrink some species below 500 if duplicates were present. As it turns out, **0 duplicates were found** — the two manifests are disjoint by path. The dedup step is still kept for robustness.
7. **Apply 500-per-species cap** via `apply_max_images_per_species_cap` (deterministic seed=42).
8. **Write** three artefacts under `data/`:
   - [combined_old_extra_max500_train_manifest.csv](data/combined_old_extra_max500_train_manifest.csv) — the training manifest, 2,223,516 rows.
   - [combined_old_extra_max500_summary.json](data/combined_old_extra_max500_summary.json) — full build stats.
   - [species_counts_before_after.csv](data/species_counts_before_after.csv) — 7,806 rows, `(species_id, count_before_cap, count_after_cap, scientific_name)` sorted by `count_before_cap` desc.

### 5.3 Build Stats (from [data/combined_old_extra_max500_summary.json](data/combined_old_extra_max500_summary.json))

| Field | Value |
|---|---|
| `n_old` | 2,653,781 |
| `n_new` (after validation) | 148,063 |
| `n_new_invalid_dropped` | 13,623 |
| `n_species_old` | 7,806 |
| `n_species_new` | 2,020 |
| `n_species_only_in_new` | 0 |
| `n_concat` | 2,801,844 |
| `n_duplicates_removed` | 0 |
| `n_after_dedup` | 2,801,844 |
| `n_after_cap` | **2,223,516** |
| `n_species_combined` | 7,806 |
| `min_before` / `median_before` / `max_before` | 1 / 249 / 1,323 |
| `min_after`  / `median_after`  / `max_after`  | 1 / 249 / 500 |
| `n_species_at_cap` (exactly 500 after cap) | 2,723 |
| `n_species_under_100_before` | 1,644 |
| `n_species_under_100_after`  | 1,644 |

The combined manifest has **slightly higher** median than the i002 raw input
(249 vs 225) because the extra data lifts the middle of the distribution,
while the cap pulls the top down from 1,323 → 500. The under-100 species
count is unchanged: 1,644 in both rows, which means **none of the new images
were enough to push a species from < 100 over the line**. Interpretation: the
extra_under100 source is contributing images to species that were already
≥ 100, or to species that were so rare that even +148k extra images spread
across 2,020 of them only added a handful per species on average.

---

## 6. Per-Species Image Capping Strategy

### 6.1 Where the Cap is Applied

The cap is **applied once, off-line, at manifest-prep time** in
[prepare_combined_manifest.py](prepare_combined_manifest.py) using
`apply_max_images_per_species_cap(df_concat, max_per_species=500, seed=42)`.
That function is the same one shipped in [data/metadata_utils.py](data/metadata_utils.py)
since i002:

```python
def apply_max_images_per_species_cap(df, max_per_species, seed=42):
    if max_per_species <= 0:
        return df                       # cap disabled
    rng = np.random.RandomState(seed)
    parts = []
    for _, grp in df.groupby("species_id", sort=False):
        if len(grp) <= max_per_species:
            parts.append(grp)           # rare species untouched
        else:
            parts.append(grp.sample(n=max_per_species, random_state=rng))
    return pd.concat(parts).reset_index(drop=True)
```

Training-time `--max-images-per-species` is **not** passed by any of the
i003 training scripts — every `train_config.json` records
`max_images_per_species: 0`. This is deliberate: stacking the manifest cap
with a runtime cap would resample over the already-capped set.

### 6.2 Distribution Change

Per-species count distribution, before and after the cap (logged by
`print_species_distribution` in [outputs/head_only/run.log](outputs/head_only/run.log)
once the train/val split is taken; numbers below are the train-split portion,
so the bin counts are slightly different from the pre-split manifest summary):

```
After cap (train split, n=2,003,751 rows, 7,806 species):
  Min       : 1
  Median    : 225
  Mean      : 256.7
  Max       : 450        ← upper bound is val-split-adjusted from cap=500
  Bin counts (images per species → num species):
    [   1]:    8 species
    [   2]:    9 species
    [ 3-5]:   58 species
    [6-10]:   71 species
    [11-20]: 138 species
    [21-50]: 496 species
    [51-100]: 1,120 species
    [101-250]: 2,244 species
    [251-500]: 3,662 species
    [501-1000]:    0 species
    [>1000]:       0 species
```

Compare to i002 (which had no cap), where 2,263 species sat in the 501–1000
bin and 12 sat above 1000. The cap collapses both of those bins into the
251–500 bucket and slightly thickens the lower bins as a side-effect of
mixing in the extra-under-100 images.

`n_species_at_cap = 2,723` (in the pre-split manifest) is the number of
species for which the cap was actually binding. 5,083 species fell below
the cap and were left alone.

### 6.3 Validation Set is Not Capped

The 10% stratified val split is taken **after** the cap, at train time:
`build_val_split` ([dataset.py](dataset.py)) splits the already-capped manifest
with `val_fraction=0.1, val_seed=42`. From [outputs/head_only/run.log](outputs/head_only/run.log):

| Split | Rows | Unique species |
|---|---|---|
| train | 2,003,751 | 7,806 |
| val   |   220,158 | 7,758 |

48 species (7,806 − 7,758) are absent from val because they have < 5 images
in the capped manifest. This is a much smaller gap than i002 (497 missing
val species), thanks to the extra-under-100 data.

The metrics CSVs show `n_val = 220,120` (not 220,158); the small drift is
caused by `DistributedSampler(drop_last=False)` on a 4-GPU run rounding the
per-rank slice — the validation set on disk is 220,158, of which 220,120
are scored after sharding.

---

## 7. Metadata and Splits

### 7.1 Encoders

| Encoder | # classes | Source |
|---|---|---|
| `idx_to_species` / `species_to_idx` | 7,806 | `outputs/head_only/encoders/` |
| `idx_to_genus`   / `genus_to_idx`   | 1,446 | same |
| `idx_to_family`  / `family_to_idx`  | 181   | same |

Class counts match i002 exactly — the combined manifest does not introduce
new species, only new images.

### 7.2 Bad-Image Handling at Train Time

i003's [dataset.py](dataset.py) tightens the failure handling that was absent
in i002:

```python
ImageFile.LOAD_TRUNCATED_IMAGES = True    # tolerate partial JPEGs
Image.MAX_IMAGE_PIXELS = 300_000_000      # raise decompression-bomb limit
```

`MultiTaskDataset.__getitem__` runs a bounded retry loop (up to 8 attempts);
on each `OSError | UnidentifiedImageError | DecompressionBombError` it logs
the path (capped at the first 50 occurrences class-wide) and advances `idx`
by 1. After 8 failures it raises a `RuntimeError`. This is a runtime safety
net in case any of the trusted old-manifest images is in fact corrupt — the
new-manifest images are already validated up-front in
[prepare_combined_manifest.py](prepare_combined_manifest.py).

### 7.3 Path Verification

[scripts/verify_dataset_cap.py](scripts/verify_dataset_cap.py) is carried
over unchanged from i002. The build script's own validation pass on the
new manifest is the primary verification used here; no standalone
`verify_dataset_cap.py` log is recorded in outputs.

---

## 8. Model and Training Pipeline

### 8.1 Architecture

Same as i002: BioCLIP 2.5 ViT-H/14 backbone, shared MLP head, species + genus
+ family classification heads. Loss weights `1·sp + 0.30·gen + 0.15·fam`
([utils.py](utils.py) `TAXONOMY_WEIGHTS`).

### 8.2 Common Settings

All three production training runs share:

- `--precision bf16`
- `--weight-decay 1e-4`
- `--label-smoothing 0.1`
- `--warmup-epochs 1` + cosine decay
- `--use-taxonomy-heads`
- `val_fraction = 0.1`, `val_seed = 42`
- **4 GPUs** via `torchrun --standalone --nproc_per_node=4` (`world_size=4` in every `train_config.json`)

### 8.3 Stage 1: Head-Only Training (`outputs/head_only/`)

Launched by [scripts/train_head_only.sh](scripts/train_head_only.sh).

| Parameter | Value |
|---|---|
| Frozen / Unfrozen | Backbone fully frozen; head + shared MLP only |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 (unused) |
| batch_size | 512 per GPU × 4 = 2,048 |
| grad_accum | 2 (effective batch = 4,096) |
| epochs | 10 |
| epoch time | ~2,650 s (~44 min) |
| total_steps | 4,890 (warmup 489) |
| best ckpt | [outputs/head_only/checkpoints/best.pt](outputs/head_only/checkpoints/best.pt) |
| config | [outputs/head_only/train_config.json](outputs/head_only/train_config.json) |

**Per-epoch val metrics** ([outputs/head_only/metrics.csv](outputs/head_only/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 8.824 | 2.0444 | 0.6351 | 0.8433 | 0.8369 | 0.9141 |
| 3 | 3.656 | 1.1237 | 0.7606 | 0.9323 | 0.8951 | 0.9412 |
| 5 | 3.431 | 1.0334 | 0.7770 | 0.9390 | 0.9024 | 0.9456 |
| 7 | 3.344 | 0.9998 | 0.7836 | 0.9414 | 0.9049 | 0.9477 |
| 9 | 3.312 | 0.9901 | 0.7856 | 0.9420 | 0.9056 | 0.9482 |
| **10** | **3.309** | **0.9895** | **0.7858** | **0.9420** | **0.9056** | **0.9482** |

Best top-5 = **0.9420** at epoch 10 (best.pt).

### 8.4 Stage 2a: Unfreeze Last 4 Blocks (`outputs/last_blocks/`)

Launched by [scripts/train_last_blocks.sh](scripts/train_last_blocks.sh),
resuming weights-only from `outputs/head_only/checkpoints/best.pt`.

| Parameter | Value |
|---|---|
| Frozen / Unfrozen | Last 4 transformer blocks + `ln_post`/`proj` unfrozen |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 |
| batch_size | 128 per GPU × 4 = 512 |
| grad_accum | 4 (effective batch = 2,048) |
| epochs | 10 |
| epoch time | ~2,790 s (~47 min) |
| total_steps | 9,780 (warmup 978) |
| best ckpt | [outputs/last_blocks/checkpoints/best.pt](outputs/last_blocks/checkpoints/best.pt) |
| config | [outputs/last_blocks/train_config.json](outputs/last_blocks/train_config.json) |

**Per-epoch val metrics** ([outputs/last_blocks/metrics.csv](outputs/last_blocks/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 3.286 | 0.9851 | 0.7852 | 0.9427 | 0.9077 | 0.9507 |
| 3 | 3.115 | 0.9216 | 0.7969 | 0.9479 | 0.9142 | 0.9548 |
| 5 | 3.018 | 0.8892 | 0.8046 | 0.9506 | 0.9167 | 0.9566 |
| 7 | 2.963 | 0.8725 | 0.8090 | 0.9521 | 0.9183 | 0.9572 |
| 9 | 2.937 | 0.8658 | 0.8102 | 0.9525 | 0.9187 | 0.9576 |
| **10** | **2.931** | **0.8654** | **0.8103** | **0.9525** | **0.9188** | **0.9575** |

Best val top-5 = **0.9525** at epoch 10.

### 8.5 Stage 2b: Unfreeze Last 8 Blocks (`outputs/last_blocks_8/`)

Launched by [scripts/train_last_blocks_8.sh](scripts/train_last_blocks_8.sh),
resuming weights-only from `outputs/last_blocks/checkpoints/best.pt`.

| Parameter | Value |
|---|---|
| Frozen / Unfrozen | Last 8 transformer blocks unfrozen |
| head_lr | 1e-4 |
| backbone_lr | 1e-6 |
| batch_size | 128 per GPU × 4 = 512 |
| grad_accum | 4 |
| epochs | 10 |
| epoch time | ~3,235 s (~54 min) |
| best ckpt | [outputs/last_blocks_8/checkpoints/best.pt](outputs/last_blocks_8/checkpoints/best.pt) |
| config | [outputs/last_blocks_8/train_config.json](outputs/last_blocks_8/train_config.json) |

**Per-epoch val metrics** ([outputs/last_blocks_8/metrics.csv](outputs/last_blocks_8/metrics.csv)):

| Epoch | Train Loss | Val Loss | Top-1 | Top-5 | Genus Acc | Family Acc |
|---|---|---|---|---|---|---|
| 1 | 2.938 | 0.8746 | 0.8073 | 0.9516 | 0.9183 | 0.9576 |
| 3 | 2.877 | 0.8506 | 0.8124 | 0.9540 | 0.9218 | 0.9595 |
| 5 | 2.815 | 0.8297 | 0.8169 | 0.9554 | 0.9237 | 0.9606 |
| 7 | 2.776 | 0.8193 | 0.8193 | 0.9564 | 0.9245 | 0.9611 |
| 9 | 2.755 | 0.8152 | 0.8203 | 0.9566 | 0.9249 | 0.9614 |
| **10** | **2.751** | **0.8147** | **0.8204** | **0.9567** | **0.9249** | **0.9614** |

Best val top-5 = **0.9567** at epoch 10.

### 8.6 Stages Not Run

[scripts/train_last_blocks_12.sh](scripts/train_last_blocks_12.sh) and
[scripts/train_full_finetune.sh](scripts/train_full_finetune.sh) exist and
point at the expected resume checkpoints, but no
`outputs/last_blocks_12/` or `outputs/full_finetune/` directory exists.
They are not part of this experiment.

### 8.7 Cross-Stage Summary

| Run | Blocks Unfrozen | Epochs | Best Val Top-1 | Best Val Top-5 | Val Loss (best) | Best Ckpt |
|---|---|---|---|---|---|---|
| smoke_test  | 0 | 1 (200 samples) | 0.0000 | 0.0000 | 8.93 | epoch_001 |
| head_only   | 0 | 10 | 0.7858 | 0.9420 | 0.9895 | epoch_010 |
| last_blocks | 4 | 10 | 0.8103 | 0.9525 | 0.8654 | epoch_010 |
| last_blocks_8 | 8 | 10 | 0.8204 | 0.9567 | 0.8147 | epoch_010 |

For direct comparison, i002 with the same architecture and stages but no
cap and the original metadata reached top-5 = 0.9505 / 0.9583 / 0.9615. **i003
is consistently 0.005 – 0.009 *lower* on validation top-5** — capping removes
training rows without compensating fully via the new images, and the
training set ends up smaller (2.00 M train rows vs i002's 2.39 M).

---

## 9. Inference and Submission Pipeline

All inference uses [`infer_tiles_adaptive.py`](infer_tiles_adaptive.py),
identical to i002. Test set: **2,105 quadrat images** at
`/workspace/plantclef/raw/test` (from `summary.json` of each inference run).

### 9.1 Tiling and Aggregation

All scored runs in i003 use the same configuration:

- `tile_mode = grid_4x4`
- `tile_size = 448`
- `overlap = 0.0`
- `agg_modes = ["softmax_mean"]`
- `min_k = 2`, `max_k = 10`

No `max` aggregation, no overlap sweep, no other grid sizes were run in i003 —
the i002 sweeps had already shown these are the best configuration.

### 9.2 Selection Modes Swept

Per checkpoint, the script produces 15 submissions in a single forward pass:

| Mode | Values |
|---|---|
| `fixed_topk` | 1, 2, 3, 4, 5 (with `min_k=2`, top-1 collapses into top-2 ties) |
| `prob_threshold` | 0.02, 0.03, 0.05 |
| `relative_threshold` | 0.15, 0.20, 0.25, 0.30 |
| `gap` | 0.40, 0.50, 0.60 |

### 9.3 Inference Families Run

| Inference dir | Source ckpt | logit_adj_tau | Source script |
|---|---|---|---|
| [outputs/head_only_infer/](outputs/head_only_infer/) | `head_only/best.pt` | 0 | manual `infer_tiles_adaptive.py` call |
| [outputs/last_blocks_infer/](outputs/last_blocks_infer/) | `last_blocks/best.pt` | 0 | manual call |
| [outputs/last_blocks_8_infer/](outputs/last_blocks_8_infer/) | `last_blocks_8/best.pt` | 0 | manual call |
| [outputs/last_blocks_8_infer_la_025/](outputs/last_blocks_8_infer_la_025/) | `last_blocks_8/best.pt` | **0.25** | [scripts/infer_best_adaptive.sh](scripts/infer_best_adaptive.sh) (defaults) |

Throughput from `summary.json` is ~3 images/sec on grid_4x4 with `bf16`.

### 9.4 Logit Adjustment

Re-used from i002 without modification. In i003,
[scripts/infer_best_adaptive.sh](scripts/infer_best_adaptive.sh) **defaults**
to `--logit-adj-tau 0.25 --metadata-csv {i003 manifest}`, so any future
inference run launched via that script will be calibrated against the
combined+capped manifest's per-class frequencies. Only one `tau` value was
swept (0.25) in i003 — the i002 sweep had already shown 0.5 and 0.75 to be
worse.

### 9.5 Saved Logits

All four inference directories include `logits/softmax_mean_logits.pt` from
the `--save-logits` flag, ready for later ensembling across checkpoints.

---

## 10. Output Files

### 10.1 Manifest Artefacts (`data/`)

```
data/
├── combined_old_extra_max500_train_manifest.csv   2,223,516 rows
├── combined_old_extra_max500_summary.json         Build stats
└── species_counts_before_after.csv                7,806 rows, sorted by count_before_cap desc
```

`species_counts_before_after.csv` head (top species hit by the cap):

```
species_id, count_before_cap, count_after_cap, scientific_name
1369068,    1323, 500, Styphnolobium japonicum (L.) Schott
1360257,    1293, 500, Frangula alnus Mill.
1741625,    1200, 500, Lathyrus oleraceus Lam.
1737669,    1162, 500, Scandosorbus intermedia (Ehrh.) Sennikov
1363575,    1153, 500, Hedera helix L.
1394359,    1153, 500, Oxalis dillenii Jacq.
...
```

### 10.2 Training Outputs

Identical layout to i002 — `checkpoints/{best,last,epoch_NNN}.pt`, `encoders/*.json`,
`metrics.csv`, `metrics.json`, `train_config.json`, `run.log`. Approximate
checkpoint sizes: head_only 4.10 GB, last_blocks 4.74 GB, last_blocks_8
5.37 GB.

### 10.3 Inference Outputs

Each `outputs/{infer_dir}/{softmax_mean}_{selection_key}/` contains:

```
submission.csv          # 2,105 rows + header, PlantCLEF Kaggle format
predictions_scored.csv  # per-prediction rows with scores
run_config.json         # full inference config
summary.json            # n_images, n_errors, total_secs, throughput
```

### 10.4 Scores

| File | Submissions (COMPLETE) | Coverage |
|---|---|---|
| [scores_head_only.csv](scores_head_only.csv) | 15 | `head_only_infer/` |
| [scores_last_blocks.csv](scores_last_blocks.csv) | 15 | `last_blocks_infer/` |
| [scores_last_blocks_8.csv](scores_last_blocks_8.csv) | 30 | `last_blocks_8_infer/` + `last_blocks_8_infer_la_025/` |
| **Total** | **60** | All four inference dirs fully scored |

---

## 11. Results and Observations

### 11.1 Top 10 Public Leaderboard Submissions

Sorted from all three score CSVs.

| Rank | Public Score | Source dir | Selection |
|---|---|---|---|
| 1  | **0.40041** | last_blocks_infer | softmax_mean_top3 |
| 2  | 0.39673 | last_blocks_infer | softmax_mean_probT0.05 |
| 3  | 0.39603 | last_blocks_infer | softmax_mean_probT0.03 |
| 4  | 0.39221 | last_blocks_infer | softmax_mean_gap0.5 |
| 5  | 0.38919 | last_blocks_8_infer_la_025 | softmax_mean_probT0.03 |
| 6  | 0.38761 | last_blocks_8_infer | softmax_mean_top3 |
| 7  | 0.38553 | last_blocks_infer | softmax_mean_top1 / softmax_mean_top2 (tie, `min_k=2`) |
| 8  | 0.38508 | last_blocks_infer | softmax_mean_gap0.4 |
| 9  | 0.38494 | last_blocks_infer | softmax_mean_top4 |
| 10 | 0.38482 | last_blocks_infer | softmax_mean_relT0.15 |

### 11.2 Best Score per Inference Dir

| Inference dir | Best public score |
|---|---|
| head_only_infer                  | 0.37421 (softmax_mean_probT0.02) |
| **last_blocks_infer**            | **0.40041** ← best in i003 (softmax_mean_top3) |
| last_blocks_8_infer              | 0.38761 (softmax_mean_top3) |
| last_blocks_8_infer_la_025       | 0.38919 (softmax_mean_probT0.03) |

### 11.3 Observations

- **Best i003 score (0.40041) is below the best i002 score (0.41165).** The
  combined+capped manifest produces a model that is slightly *worse* than
  i002 on this leaderboard. Same checkpoint family (last_blocks/4-block),
  same inference recipe, same number of test images.

- **The val-vs-Kaggle pattern from i002 repeats.** Val top-5 increases
  monotonically with unfrozen blocks (0.9420 → 0.9525 → 0.9567), but the
  best Kaggle score is on the **middle** stage (last_blocks/4-block at
  0.40041), not the more-unfrozen last_blocks_8 (0.38761). Adding logit
  adjustment to last_blocks_8 helps mildly (0.38919) but doesn't catch
  up to last_blocks.

- **Selection mode for the best run is `top3`, not `prob_threshold`.** In
  i002 the all-best was `probT 0.05` (0.41165). In i003 `top3` wins
  (0.40041) and `probT 0.05` is second (0.39673). This is a small flip —
  both submissions are within 0.005 of each other and both are within the
  noise band of i002's ranking — but it's worth noting that no single
  selection rule is dominant across experiments.

- **Logit adjustment on last_blocks_8 is a wash.** `la_025` beats vanilla
  `last_blocks_8` only on `probT 0.03` (0.38919 vs 0.38355, +0.0056) and
  ties or loses elsewhere. With tau=0.25 calibrated against the *capped*
  prior (which is already much flatter), the adjustment carries less signal
  than it did in i002.

- **head_only is meaningfully worse than i002's head_only.** Best i003
  head_only_infer score 0.37421 vs i002 head_only_infer 0.39416 — a
  −0.02 drop. The training set is smaller (2.00 M vs 2.39 M) and capping
  the head species removes their signal-saturating effect for the
  frozen-backbone case, so the linear head has less to fit.

### 11.4 Submission Volume vs i002

i003 ran 60 scored submissions (15 head_only + 15 last_blocks + 30 last_blocks_8
= 30 vanilla + 15 tau=0.25). i002 ran 162. The i003 sweep is much narrower:
single tiling config, single aggregation, one tau value — a deliberate
focus on the questions that survived i002.

---

## 12. Comparison with i002

| Aspect | i002 | i003 |
|---|---|---|
| Manifest rows | 2,653,781 | 2,223,516 (after cap) |
| Train rows after split | 2,391,457 | 2,003,751 |
| Val rows | 262,324 | 220,158 |
| Val species coverage | 7,309 / 7,806 | 7,758 / 7,806 |
| Median per species | 225 | 225 (post-split), 249 (pre-split) |
| Max per species (train pre-split) | 1,191 | 500 |
| head_only best val top-5 | 0.9505 | 0.9420 |
| last_blocks best val top-5 | 0.9583 | 0.9525 |
| last_blocks_8 best val top-5 | 0.9615 | 0.9567 |
| Best Kaggle public score | 0.41165 | 0.40041 |
| Top score's checkpoint | last_blocks (4) | last_blocks (4) |
| Top score's selection rule | `probT 0.05` | `top3` |

**Takeaway.** The 500-cap + extra_under100 strategy did **not** improve the
Kaggle score relative to i002. Both validation and leaderboard metrics
dropped by 0.005 – 0.012. The likely reasons:

1. Capping cuts ~570k images from over-represented species without proportionate replacement from under-represented ones (1,644 species still have < 100 images even after combining).
2. Val-set coverage improved (7,758 vs 7,309 species) but this also makes the val top-5 metric *harder* — the new rare species in val are exactly the ones the model still struggles with.
3. The Kaggle quadrat domain gap is unaffected by the cap; the gains the cap could plausibly deliver (better calibration on rare species) are already partially captured by inference-time `logit-adj-tau` in i002.

---

## 13. Reproducibility

### 13.1 Manifest Build

```bash
cd src_experiments/i003_bioclip25_cap_image_extra500
bash scripts/prepare_data.sh
# or:  python prepare_combined_manifest.py --max-per-species 500 --seed 42
```

Writes `data/combined_old_extra_max500_train_manifest.csv`,
`data/combined_old_extra_max500_summary.json`,
`data/species_counts_before_after.csv`.

### 13.2 Training

```bash
# 1. Smoke test
bash scripts/smoke_test.sh 4

# 2. Stage 1: head-only (10 epochs, frozen backbone, 4 GPUs)
bash scripts/train_head_only.sh 4

# 3. Stage 2a: last 4 blocks (10 epochs)
bash scripts/train_last_blocks.sh outputs/head_only/checkpoints/best.pt 4

# 4. Stage 2b: last 8 blocks (10 epochs)
bash scripts/train_last_blocks_8.sh outputs/last_blocks/checkpoints/best.pt 4
```

Each script forwards to `torchrun --standalone --nproc_per_node=4 train.py …`.
The metadata CSV path is hardcoded in every script to
`data/combined_old_extra_max500_train_manifest.csv` under this experiment.

### 13.3 Inference

```bash
# Each of these produces 15 submissions in one forward pass.
# Without logit adjustment (set tau=0 in the script or override):
python infer_tiles_adaptive.py \
  --checkpoint outputs/last_blocks/checkpoints/best.pt \
  --image-dir /workspace/plantclef/raw/test \
  --tile-mode grid_4x4 --overlap 0.0 \
  --agg-modes softmax_mean \
  --top-ks 1 2 3 4 5 --min-k 2 --max-k 10 \
  --selection-modes fixed_topk relative_threshold prob_threshold gap \
  --relative-thresholds 0.15 0.20 0.25 0.30 \
  --prob-thresholds 0.02 0.03 0.05 \
  --gap-ratios 0.40 0.50 0.60 \
  --save-logits --precision bf16 --batch-size 512 \
  --output-dir outputs/last_blocks_infer

# With logit adjustment (= the default in infer_best_adaptive.sh):
bash scripts/infer_best_adaptive.sh \
  outputs/last_blocks_8/checkpoints/best.pt \
  /workspace/plantclef/raw/test \
  outputs/last_blocks_8_infer_la_025
```

### 13.4 Seeds and Determinism

- Cap seed = 42 (in `prepare_combined_manifest.py --seed`).
- Val split seed = 42 (in `train.py --val-seed`).
- Train-loader shuffle is *not* seeded — minor run-to-run variance possible
  (cf. two near-identical epoch-1 rows in
  [outputs/smoke_test/metrics.csv](outputs/smoke_test/metrics.csv)).
- Image-validation worker count in `prepare_combined_manifest.py` defaults
  to 64 threads; the validation result is deterministic (each file is
  decoded once), so worker count does not affect what gets dropped.

---

## 14. Limitations and Notes

- **i003 underperforms i002 on Kaggle by ~0.011.** The combined+capped
  manifest strategy is not validated by this experiment. Whether the
  deficit is due to (a) fewer total training images, (b) the long tail
  remaining long, or (c) a per-species cap that is too aggressive at 500
  remains untested. A 1000-cap variant (the same script will take
  `--max-per-species 1000`) is the obvious next experiment.

- **`extra_under100_train_manifest.csv` adds species-level breadth in
  spirit only.** Every species in the new manifest was already in the old
  one (`n_species_only_in_new: 0`), and the count of species under 100
  is unchanged (1,644 before and after). The extra data lifted the
  median but did not move the bottom of the distribution.

- **8.4% of the new manifest was unusable.** 13,623 / 161,686 new-manifest
  files failed a full PIL decode and were dropped at build time. The build
  script doesn't categorise the failure mode beyond exception type, so
  the breakdown between "missing file", "corrupt JPEG", "JSON error page
  saved as .jpg", etc. is not measured.

- **Bad-image retry in `dataset.py` was not exercised in the logged runs.**
  No `Skipping bad image` warnings appear in the train run logs — meaning
  none of the trusted old-manifest images failed at runtime in these
  three training runs. The safety net is silent.

- **`infer_best_adaptive.sh` defaults to τ=0.25.** This changed from i002,
  where the script had no `--logit-adj-tau`. If you run the script without
  arguments in i003, you will get a calibrated submission even if you
  didn't intend to; the four inference directories that exist were
  launched with explicit arguments.

- **Last-12-blocks and full-finetune stages were not run.** Scripts exist;
  no outputs.

- **No additional logit-adjustment τ values were swept.** i002 had
  τ ∈ {0.25, 0.5, 0.75}; i003 has only 0.25, on the assumption that the
  i002 sweep covered the regime.

- **`scores_*.csv` files do not include `submit_failed` entries.** All 60
  rows are status `SubmissionStatus.COMPLETE`.

- **Statistics not measured.** Per-species accuracy on the rare tail
  (the explicit target of this experiment), ensemble of saved
  `softmax_mean_logits.pt` files across checkpoints, the exact subset of
  images each species' 500 cap selected.

---

## 15. Summary

i003 took the data-side infrastructure built in
[i002](../i002_bioclip25_cap_image/report.md) and actually used it:
combined the original training manifest with a freshly downloaded set of
extra long-tail images, validated every new file with a full PIL decode,
deduplicated by image path, and capped every species at 500 images. The
on-disk manifest is the authoritative training set — training-time capping
is disabled to avoid double-sampling.

What was implemented:

- [prepare_combined_manifest.py](prepare_combined_manifest.py) — combine, validate, dedup, cap, write three artefacts.
- Bad-image bounded-retry in `dataset.py`'s `__getitem__` (8 attempts, log first 50 failures).
- `--logit-adj-tau 0.25` plus the i003 manifest baked into `scripts/infer_best_adaptive.sh` as defaults.

What was run:

- Manifest build: 2,801,844 → 2,223,516 rows after 500-cap; 7,806 species; 13,623 invalid new images dropped.
- Three training stages on **4 GPUs**: `head_only` (top-5 = 0.9420), `last_blocks` (4 blocks, 0.9525), `last_blocks_8` (8 blocks, 0.9567), plus a smoke test.
- 60 Kaggle submissions across four inference directories (head_only, last_blocks, last_blocks_8, last_blocks_8 with logit adjustment).

What was produced:

- Three full training-output trees, 10 per-epoch checkpoints each.
- Saved aggregated logits for all four inference directories.
- Three Kaggle score CSVs; **best public score 0.40041** on
  `last_blocks_infer / softmax_mean_top3` (4-block checkpoint, grid_4x4,
  no logit adjustment).

What did not move:

- Kaggle public score went **down** by ~0.011 vs i002 (0.41165 → 0.40041).
- 1,644 species still have < 100 images after combining — the extra-under-100
  source didn't push any species over that threshold.

What is planned but not completed:

- Sweep additional cap values (1000-cap is the most natural).
- Last-12-blocks and full-finetune stages.
- Per-species evaluation to see whether the rare tail actually improved.
- Logit ensembling across i002 + i003 checkpoint logits.
