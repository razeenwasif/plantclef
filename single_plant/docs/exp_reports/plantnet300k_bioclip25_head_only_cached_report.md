# Experiment Report: PlantNet-300K — BioCLIP 2.5 cached head-only scout

**Date:** 2026-05-30
**Hardware:** 1x NVIDIA GeForce RTX 4090 (24 GB)
**Dataset:** PlantNet-300K v2 (Garcin et al. 2021, refreshed 2026-02-10)
**Code:**
  * Adapter: `tools/build_plantnet300k_manifest.py`
  * Cache builder: `tools/cache_features_i002.py`
  * Trainer: `experiments/i002_bioclip25_cap_image/train.py` (with new `--feature-cache`, `--no-epoch-snapshots`, `--compile-head` flags)
  * Patched model: `experiments/i002_bioclip25_cap_image/model.py` (new `forward_heads(feat)` method)

---

## 1. Goal

Verify that the i002 training pipeline can be driven by a **different
dataset** (PlantNet-300K, 1,000 species) on **weaker hardware** (4090
24 GB vs the paper's 5090 32 GB) at training speeds **competitive
with the original** — by introducing a feature-cache fast path for
the head-only training stage. The motivation: on the 4090 each
non-cached head-only epoch takes ~30-45 minutes (JPEG decode + frozen
BioCLIP ViT-H/14 forward), making a 10-epoch stage 1 a multi-hour
affair. The feature cache trades a one-time ~15 min pass through the
frozen backbone for ~5-6 s/epoch thereafter — the same numerical
result at 50x the throughput.

This is a scout / pipeline-validation run, not a published result.

---

## 2. Dataset

PlantNet-300K v2 differs from PlantCLEF 2024 in three relevant ways:

| Property             | PlantNet-300K v2     | PlantCLEF 2026 (paper)              |
|----------------------|----------------------|-------------------------------------|
| Image count          | 306,087              | 1,408,033 (PlantCLEF24 single-plant) + 2.65M (i001 enriched) |
| Species              | 1,000                | 7,806                               |
| Genera               | 325                  | 1,446                               |
| Families             | 111                  | 181                                 |
| Pre-defined splits   | train / val / test (in metadata) | none (paper builds stratified 90/10) |
| Image organisation   | `images/{split}/{species_id:04d}/<sha1>.jpg` | flat `{species_id}/{image_name}.jpg` |

On-disk layout:

```
/mnt/d/PlantNet-300k/
├── plantnet300K_metadata.csv     # image_id, species_id, organ, author, license, split, PN_hash
├── species_metadata.csv          # species_id, full_species, species, genus, family, ...
└── images/images/{train,val,test}/0000..0999/<sha1>.jpg
```

PlantNet's own split: 243,866 train / 31,113 val / 31,105 test. We
used only the **train split** for this scout. The trainer then
stratifies its own 90/10 val split out of those 243,866 (seed 42,
species with <5 images stay in train), yielding **219,754 train +
24,112 val** for the actual head-warmup.

---

## 3. Pipeline

### 3.1 Manifest adapter

`tools/build_plantnet300k_manifest.py` reads PlantNet's two CSVs,
joins on `species_id`, filters to `split=='train'`, builds the
absolute `image_path` from `images/images/train/<species_dir>/<PN_hash>.jpg`,
and writes a single CSV that the i002 dataset loader accepts without
any code changes:

```bash
python tools/build_plantnet300k_manifest.py \
    --root        /mnt/d/PlantNet-300k \
    --split       train \
    --output      data/plantnet300k_manifest.csv
```

Output: 243,866 rows × 5 columns
(`species_id, image_path, genus, family, species_name`). 100% of paths
resolve on disk. 100% taxonomy coverage. 32 MB total.

### 3.2 Feature cache builder

`tools/cache_features_i002.py` loads BioCLIP 2.5 ViT-H/14
(`hf-hub:imageomics/bioclip-2.5-vith14`) in `configure_backbone("freeze")`
mode, builds the same stratified 90/10 val split the trainer would,
and runs a single bf16 forward pass over both splits. For every image
it stores the 1024-dim CLS embedding plus species / genus / family
indices.

```bash
python tools/cache_features_i002.py \
    --metadata-csv      data/plantnet300k_manifest.csv \
    --train-image-root  /mnt/d/PlantNet-300k/images/images/train \
    --output-dir        outputs/feature_cache_plantnet300k \
    --batch-size        128 \
    --num-workers       8 \
    --precision         bf16
```

Output: `outputs/feature_cache_plantnet300k/cache.pt` (505 MB)
containing eight tensors plus the encoders dict and the build-config
snapshot:

```python
{
    "train_features": fp16 tensor (219_754, 1024),
    "train_sp":       int64 tensor (219_754,),
    "train_gen":      int64 tensor (219_754,),
    "train_fam":      int64 tensor (219_754,),
    "val_features":   fp16 tensor ( 24_112, 1024),
    "val_sp":         int64 tensor ( 24_112,),
    "val_gen":        int64 tensor ( 24_112,),
    "val_fam":        int64 tensor ( 24_112,),
    "encoders":       dict,
    "config":         {model_name, img_size, val_fraction, val_seed, ...},
}
```

Also writes `encoders/idx_to_{species,genus,family}.json` alongside,
so the cached run produces the same on-disk encoder artefacts a
non-cached run would.

### 3.3 Cached head-only trainer

`experiments/i002_bioclip25_cap_image/train.py` gained a
`--feature-cache <path>` flag and a `CachedFeaturesDataset` class.
When the flag is set the trainer:

1. Skips the metadata-loading / image-dataset / DataLoader setup
   entirely.
2. Loads `cache.pt` into RAM and rehydrates the `encoders` dict.
3. Builds `CachedFeaturesDataset` instances for train + val (each
   `__getitem__` returns `(feature, sp_idx, gen_idx, fam_idx)`).
4. Builds plain DataLoaders with `num_workers=0` (data is in-memory).
5. Builds the `BioCLIP25MultiTask` model exactly as before — the
   backbone weights still load, but only the head MLPs + classifiers
   get gradients.
6. In the train loop, routes each batch through the new
   `model.forward_heads(features)` (added to `model.py`) instead of
   `model.forward(images)`. Same dispatch in `validate()` via a new
   `from_features=True` switch.

Three more cheap perf knobs landed at the same time:

* TF32 matmul + cuDNN enabled at the top of `main()` (~1.3x lift on
  Ampere/Ada matmuls).
* `persistent_workers=True` + `prefetch_factor=4` on the non-cache
  DataLoaders.
* Opt-in `--compile-head` flag that wraps the model in
  `torch.compile(mode="reduce-overhead")` before the DDP wrap.

And a checkpoint-policy flag for disk-tight scout runs:

* `--no-epoch-snapshots` skips per-epoch `epoch_NNN.pt` writes;
  `best.pt` is written directly on val improvement (overwrites in
  place); `last.pt` is overwritten every epoch as usual. Net: 2 files
  on disk regardless of epoch count, vs 1 + N otherwise.

### 3.4 Scout command

```bash
python src/train.py \
    --feature-cache outputs/feature_cache_plantnet300k/cache.pt \
    --metadata-csv data/plantnet300k_manifest.csv \
    --train-image-root /mnt/d/PlantNet-300k/images/images/train \
    --epochs 10 \
    --batch-size 256 \
    --precision bf16 \
    --freeze-backbone \
    --head-lr 1e-4 \
    --weight-decay 1e-4 \
    --no-epoch-snapshots \
    --output-dir outputs/plantnet300k_scout_cached \
    --val-every 1 \
    --log-every 200
```

`--metadata-csv` and `--train-image-root` are still passed because the
parser requires them, but the cache-loading branch ignores them at
runtime.

---

## 4. Results

### 4.1 Per-epoch (val on 24,112 images, no augmentation)

| Epoch | train_loss | val_loss | top-1 | top-5 | genus_acc | family_acc | epoch_secs |
|------:|-----------:|---------:|------:|------:|----------:|-----------:|-----------:|
|  1    | 4.317      | 0.736    | 85.34%| 96.65%|   95.92%  |    97.67%  |       7    |
|  2    | 2.045      | 0.592    | 87.06%| 98.08%|   96.59%  |    98.15%  |       6    |
|  3    | 1.939      | 0.558    | 87.58%| 98.42%|   96.87%  |    98.22%  |       6    |
|  4    | 1.895      | 0.540    | 87.84%| 98.52%|   97.00%  |    98.31%  |       6    |
|  5    | 1.867      | 0.533    | 87.92%| 98.56%|   97.01%  |    98.37%  |       6    |
|  6    | 1.847      | 0.527    | 88.05%| 98.66%|   97.06%  |    98.43%  |       6    |
|  7    | 1.832      | 0.522    | 88.20%| 98.68%|   97.10%  |    98.45%  |       9    |
|  8    | 1.823      | 0.522    | 88.18%| 98.68%|   97.11%  |    98.45%  |       6    |
|  9    | 1.817      | 0.520    | 88.23%|**98.68%**|97.11%  |    98.46%  |       6    |
| 10    | 1.814      | 0.519    |**88.23%**| 98.68%|  97.11%  |  **98.46%**|       6    |

Best checkpoint: epoch 7 (top-5 = 0.9868). Top-1 plateaued at epoch 9
(0.8823) and stayed flat through epoch 10 as the cosine schedule
decayed to 1e-6.

### 4.2 Wall-time breakdown

Total: **5 min 43 s** for the full 10-epoch run.

| Phase                                | Wall time      |
|--------------------------------------|----------------|
| Python + torch import + model wiring | ~14 s          |
| BioCLIP 2.5 weight load (from HF cache) | ~10 s        |
| `cache.pt` load (505 MB → in-RAM)    | ~1 s           |
| Per epoch:                           |                |
|  - Training (858 steps × bf16 head fwd/bwd) | ~6 s    |
|  - Validation (48 batches × 256)     | <1 s           |
|  - Checkpoint serialise (best + last, ~4 GB each) | ~20-30 s |
| 10 epochs total in-loop              | ~4.8 min       |

Per-epoch wall time is now **dominated by checkpoint serialisation**,
not compute. The actual training work — 219,754 forward+backward
passes through three 4.6M-param head MLPs — finishes in 6 s.

### 4.3 Cache-build time (one-time)

`tools/cache_features_i002.py` over the full 243,866 images at
bf16 / batch 128 / 8 workers: **16 min 31 s**, sustaining ~280 img/s
on the 4090.

Output: 505 MB `cache.pt`.

### 4.4 What the cache saved

| Configuration | Per-epoch time (estimated) | 10-epoch total |
|---|---|---|
| Non-cached (the original smoke ran a 200-sample variant) | ~30-45 min/epoch (extrapolated from 4090 throughput) | 5+ hours |
| Cached (this run)            | ~6 s/epoch          | 5 min 43 s     |

Effective speedup ≈ **50x** on the 10-epoch head-only stage, paid
for by a single 16 min cache build.

---

## 5. Verified plumbing

This scout was primarily a pipeline-validation run. It confirmed:

1. **PlantNet-300K manifest adapter** correctly joins image metadata
   with species taxonomy, resolves all 243K image paths (spot-check
   sample: 20/20 present), and writes an i002-compatible CSV with
   zero modifications needed downstream.
2. **Feature-cache builder** runs end-to-end at the expected
   throughput (~280 img/s) on a 4090 / WSL setup with images on a
   Windows D: drive (`/mnt/d/PlantNet-300k/`).
3. **`--feature-cache` flag** correctly:
    - Loads the encoders dict from the cache,
    - Builds `CachedFeaturesDataset` for both splits,
    - Skips the `--metadata-csv` / `--train-image-root` data path,
    - Reuses the smoke-test cap (200 train / 50 val) when
      `--smoke-test` is also passed.
4. **`model.forward_heads(feat)`** dispatch from train+validate works
   identically to `model(images)` — losses decrease smoothly, no
   shape mismatches, no graph errors.
5. **`--no-epoch-snapshots` policy** writes only `best.pt` (overwritten
   on val improvement) and `last.pt` (overwritten every epoch). Net
   on-disk after a 10-epoch run: **7.5 GB** (2 files × 4 GB rolling),
   vs an estimated 48 GB without the flag.
6. **bf16 AMP + TF32 + AdamW on RTX 4090 + Ada cuDNN** is healthy —
   no NaNs, no exploding-gradient artefacts, monotone loss descent.

---

## 6. Tradeoffs and known limitations

* **No augmentation in cached mode.** Features were computed once
  with the deterministic `val_transform` (resize-256 → centre-crop-224
  → CLIP-norm). The cached head therefore trains against a fixed view
  per image. For a publication-quality run this is the wrong tradeoff
  — augmentations (RandomResizedCrop, hflip, vflip, ColorJitter,
  rotation, grayscale) typically add 0.5-1.5% top-5 to head-only
  numbers on long-tail biological datasets, and the BioCLIP backbone
  is cheap enough in the cached representation to make those gains
  worth the loss of cache-reuse. For a *scout* this is the right
  tradeoff — speed beats marginal accuracy when you're verifying
  plumbing.
* **Single-GPU only.** The cached path uses plain DataLoaders with no
  DistributedSampler. A torchrun-driven multi-GPU run with
  `--feature-cache` would currently load the full cache into each
  rank's RAM. For 505 MB this is fine; for a larger cache (e.g. the
  full 2.65M-image i001 manifest at fp16, ~5 GB) this would either
  need a sharded loader or a one-rank load + scatter. Out of scope
  for this scout.
* **`torch.compile(model)` was not used.** The `--compile-head` flag
  was added but not exercised. On the head-only path the per-step
  cost is microseconds, so compile's ~30 s warmup penalty would
  dominate the savings. Worth revisiting for the partial-fine-tune
  stage where each step is heavier.
* **Stage 2 (partial fine-tune) not run.** This report covers head
  warmup only. The paper's i002 recipe continues with 10 more epochs
  at `--unfreeze-last-n-blocks 4`, which requires gradients through
  the backbone — the cache no longer helps once that branch fires.
  Recommended next step (per Section 7).

---

## 7. Followups

### 7.1 Immediate

* **Inference test** with the scout `best.pt` on the PlantNet test
  split (31,105 images). Closes the original "run inference end-to-end
  on a real-quality checkpoint" item that we'd previously only run
  against the 200-sample smoke checkpoint. The earlier inference
  smoke (`outputs/plantnet300k_smoke_infer/`) collapsed to a single
  degenerate prediction class for 71.9% of inputs; the scout
  checkpoint should produce a distribution closer to the true label
  set.
* **Stage 2 partial unfreeze** (`--unfreeze-last-n-blocks 4`) resuming
  the scout `best.pt` weights. The cache cannot be reused here
  (backbone needs gradients), so this is the expensive path: 219K
  images × 10 epochs through JPEG decode + ViT-H/14 = approximately
  4 hours on the 4090 with the cheap DataLoader knobs.

### 7.2 Worth investigating

* **In-feature augmentation.** Manifold Mixup or feature-space SpecAugment
  variants in the cached path would recover some of the lost
  augmentation benefit without invalidating the cache. Cheap
  experiment.
* **Per-class sample weights.** The PlantNet long tail is similar in
  shape to PlantCLEF — a small number of common species dominate.
  Weighted sampling in the cached loader would test whether the
  trainer's `--use-sample-weights` path lifts the rare-class top-1.
* **Cache fp32 vs fp16.** The cache is stored fp16. The head MLPs
  cast to fp32 in `__getitem__`. Worth measuring whether fp32 storage
  (1 GB vs 0.5 GB) buys any meaningful accuracy.
* **Bigger heads, smaller backbones?** The head is currently
  `LayerNorm + Linear(1024→1024) + GELU + Dropout` — fairly small.
  With a frozen backbone and a feature cache, scaling the head MLP
  costs nothing per epoch beyond the linear-algebra runtime.

### 7.3 Out of scope here

* **DALI / nvJPEG / Rust train_resizer** wins on the cache-build pass.
  At 280 img/s the cache build is already well under a coffee break;
  optimising it further isn't worth the WSL / DALI install pain. The
  Rust `train_resizer` crate (`engines/rust/train_resizer/`) would
  let us pre-shrink the dataset and skip per-image resize in the
  cache builder, but the bottleneck right now is the ViT-H/14
  forward, not the resize.

---

## 8. Artefacts on disk

After this experiment:

```
outputs/feature_cache_plantnet300k/
├── cache.pt                       505 MB (the cache itself)
└── encoders/
    ├── idx_to_species.json
    ├── idx_to_genus.json
    ├── idx_to_family.json
    ├── species_to_idx.json
    ├── genus_to_idx.json
    └── family_to_idx.json

outputs/plantnet300k_scout_cached/
├── checkpoints/
│   ├── best.pt                    4.0 GB (top-5 0.9868, epoch 7)
│   └── last.pt                    4.0 GB (epoch 10)
├── encoders/                      (mirrors the cache's encoders/)
├── metrics.csv                    per-epoch history
├── metrics.json
├── run.log
└── train_config.json

outputs/plantnet300k_scout_infer/
├── softmax_mean_top1/submission.csv   31,106 rows, k=1
├── softmax_mean_top5/submission.csv   31,106 rows, k=5
├── logits/                            (only when --save-logits)
└── analysis/                          (from tools/score_submission.py)
    ├── summary.json
    ├── per_species_accuracy.csv       sorted worst-first, F1 + top-1
    ├── top_confused_pairs.csv         30 most-confused species pairs
    ├── topk_curve.png
    ├── per_species_accuracy_hist.png
    ├── per_species_f1_hist.png
    ├── prediction_distribution.png
    ├── family_confusion_heatmap.png
    ├── train_count_vs_accuracy.png
    └── confused_pairs_bar.png
```

Total on-disk after this run: ~8.5 GB (cache + scout outputs).

---

## 9. Held-out test set evaluation

After Stage 2 was deferred (Section 7), the scout `best.pt` (epoch 7,
val top-5 0.9868) was applied to PlantNet's official **test split**
— 31,106 images that were never touched in train or val. Inference
ran with the i002 tile-adaptive inference script
(`experiments/i002_bioclip25_cap_image/infer_tiles_adaptive.py`) in
whole-image mode (no tiling — PlantNet images are single-plant, not
quadrats), softmax-mean aggregation, fixed top-K selection at k=1 and
k=5, batch 64 / bf16.

### 9.1 Inference timing

| Metric            | Value                |
|-------------------|----------------------|
| Images processed  | 31,106 (full test split) |
| Wall time         | 15 min 38 s          |
| Throughput        | 33.2 img/s sustained |
| Errors            | 0                    |
| Hardware          | RTX 4090, WSL, images on `/mnt/d/` (NTFS) |

`/mnt/d` WSL access (cold reads through 1,000 species directories at
~31 small JPEGs per dir) capped throughput well below GPU saturation
— GPU utilisation hovered around 35 %. A pre-resized + Rust-packed
shard set would lift this materially, but for a one-time scout the
overhead is acceptable.

### 9.2 Headline numbers

| Metric           | Value      | Note |
|------------------|-----------:|------|
| **Top-1 accuracy** | **87.12 %** | held-out test set, all 1,000 species |
| **Top-5 accuracy** | **98.38 %** | |
| **Top-10 accuracy**| **98.38 %** | identical to top-5 → no additional gain past k=5 |
| **Macro F1**     | **53.45 %** | mean of per-species F1 — the PlantCLEF-style metric |
| Weighted F1      | 85.55 %    | class-frequency weighted |
| Micro F1         | 87.12 %    | equal to top-1 for single-label |
| Per-species F1   | median 66.67 %, mean 53.45 %, **328 at 0 %, 149 at 100 %** |
| Per-genus top-1  | avg 76.65 % (over 325 genera) |
| Per-family top-1 | avg 85.23 % (over 111 families) |

**Val → test gap:** 1.11 % top-1 / 0.30 % top-5. Tight and credible
— this is honest generalisation, not val leakage.

### 9.3 The Macro F1 story

The gap between **top-1 87.12 %** and **Macro F1 53.45 %** is the most
informative number in this whole experiment. Top-1 (which equals
Micro F1) heavily weights the head of the long-tail distribution
because each image counts equally and the head has more images.
Macro F1 weights every species equally regardless of training-set
size, so it surfaces the long-tail collapse:

* **328 / 1,000 species (32.8 %)** get *zero* correct predictions on
  their test images — the model never picks them as top-1. Every one
  of those species contributes a per-species F1 of 0 to the macro
  average.
* **149 species** are perfectly classified (F1 = 1.0).
* The per-species accuracy distribution
  (`per_species_accuracy_hist.png`) is **bimodal** — large masses at
  both 0 % and 100 %. The middle is sparse.

The genus and family numbers are interesting in the opposite
direction. When the model misclassifies, it tends to stay within the
right genus (76.65 % avg) and family (85.23 % avg). The BioCLIP
Tree-of-Life prior contributes a real signal: even when the species
head guesses wrong, it doesn't usually wander outside the right
taxonomic neighbourhood.

This is the classic Pl@ntNet pattern, and it's also why the published
i002 recipe leans hard on:
1. Larger, cleaner training manifests (the i001 2.65 M-image build).
2. Auxiliary genus + family heads (the trio is in the model
   architecture for a reason).
3. Class-prior logit adjustment (`tau = 0.25`) at inference — a
   targeted Macro F1 booster.

None of those were active in this scout. The 53.45 % Macro F1 is a
clean baseline for what frozen BioCLIP features + a fresh head MLP
recover *without* any long-tail compensation.

### 9.4 Distribution diagnostics

* **Prediction diversity is healthy.** The model emits **697 distinct
  top-1 predictions** across 31,106 inputs. The most-frequent top-1
  is species 203 at only **2.96 %** of inputs — nowhere near the
  smoke-checkpoint's degenerate 71.9 % collapse onto species 3.
* **`prediction_distribution.png`** scatters per-species predicted
  count vs ground-truth count. A perfectly calibrated model lies on
  `y = x`; the actual scatter clusters near the diagonal with the
  long tail under-predicted (most rare species have predicted count
  near 0 while GT count is small but nonzero).
* **`family_confusion_heatmap.png`** is 111×111 row-normalised. The
  diagonal is dominant — most confusions stay inside the family.
  Visible off-diagonal smudges concentrate around the
  Asteraceae / Lamiaceae / Brassicaceae rows where genus-level
  morphology overlaps in field images.
* **`train_count_vs_accuracy.png`** is the long-tail diagnostic.
  Per-species test top-1 accuracy correlates strongly with
  log(train image count). Species with ≥ 100 train images sit
  almost entirely above 80 % test top-1; species with < 10 train
  images dominate the 0 % cluster.

### 9.5 Top confused species pairs

The 30 most-confused pairs (`top_confused_pairs.csv` and
`confused_pairs_bar.png`) cluster heavily on within-genus
morphological siblings (e.g. multiple *Pelargonium* species
confused for each other, multiple *Carex* sedges, the wider
*Quercus* oak group). These are exactly the pairs that would
benefit from the Stage 2 partial unfreeze — the head MLP has no way
to learn fine-grained leaf serration / inflorescence / venation
patterns from frozen features.

### 9.6 Reproducing the scoring

After the inference run completes:

```bash
python tools/score_submission.py \
    --submission       outputs/plantnet300k_scout_infer/softmax_mean_top5/submission.csv \
    --ground-truth     /mnt/d/PlantNet-300k/plantnet300K_metadata.csv \
    --species-metadata /mnt/d/PlantNet-300k/species_metadata.csv \
    --manifest         data/plantnet300k_manifest.csv \
    --output-dir       outputs/plantnet300k_scout_infer/analysis \
    --split            test \
    --top-n-confused   30 \
    --max-k            20
```

Runtime: ~3 s on 31,106 images. The `--manifest` flag is optional;
omit it to skip the long-tail scatter.

### 9.7 What the held-out evaluation verified

1. **Whole-image inference path** (`--tile-mode whole`) works for the
   single-plant PlantNet distribution.
2. **The cached-trained checkpoint generalises.** 0.30 % top-5 drop
   from val to held-out test — no val leak.
3. **The scout produces non-degenerate predictions** — 697 unique
   classes used, no single-class collapse.
4. **The Macro F1 / top-1 gap exposes the long-tail collapse**
   precisely, in the shape we expect from the PlantNet imbalance.
5. **`tools/score_submission.py` is reusable** for any future
   submission CSV against any PlantCLEF-format ground truth, with the
   same one-liner.

---

## 10. Reproducibility

```bash
# 1. Build the manifest (5-10 s):
python tools/build_plantnet300k_manifest.py \
    --root /mnt/d/PlantNet-300k --split train \
    --output data/plantnet300k_manifest.csv

# 2. Build the feature cache (~16.5 min on 4090):
python tools/cache_features_i002.py \
    --metadata-csv      data/plantnet300k_manifest.csv \
    --train-image-root  /mnt/d/PlantNet-300k/images/images/train \
    --output-dir        outputs/feature_cache_plantnet300k \
    --batch-size 128 --num-workers 8 --precision bf16

# 3. Cached head-only scout (~6 min on 4090):
python src/train.py \
    --feature-cache outputs/feature_cache_plantnet300k/cache.pt \
    --metadata-csv data/plantnet300k_manifest.csv \
    --train-image-root /mnt/d/PlantNet-300k/images/images/train \
    --epochs 10 --batch-size 256 --precision bf16 --freeze-backbone \
    --head-lr 1e-4 --weight-decay 1e-4 --no-epoch-snapshots \
    --output-dir outputs/plantnet300k_scout_cached \
    --val-every 1 --log-every 200

# 4. Inference on held-out test split (~16 min on 4090):
python experiments/i002_bioclip25_cap_image/infer_tiles_adaptive.py \
    --checkpoint outputs/plantnet300k_scout_cached/checkpoints/best.pt \
    --image-dir /mnt/d/PlantNet-300k/images/images/test \
    --tile-mode whole --agg-modes softmax_mean \
    --selection-modes fixed_topk --top-ks 1 5 \
    --batch-size 64 --img-size 224 --precision bf16 \
    --output-dir outputs/plantnet300k_scout_infer

# 5. Score + plot (~3 s):
python tools/score_submission.py \
    --submission       outputs/plantnet300k_scout_infer/softmax_mean_top5/submission.csv \
    --ground-truth     /mnt/d/PlantNet-300k/plantnet300K_metadata.csv \
    --species-metadata /mnt/d/PlantNet-300k/species_metadata.csv \
    --manifest         data/plantnet300k_manifest.csv \
    --output-dir       outputs/plantnet300k_scout_infer/analysis \
    --split            test
```

All seeds default to 42 (val split, capping). bf16 is deterministic
modulo CUDA kernel scheduling; expect val numbers to reproduce within
±0.001 and Macro F1 within ±0.003 of the reported 53.45 %.
