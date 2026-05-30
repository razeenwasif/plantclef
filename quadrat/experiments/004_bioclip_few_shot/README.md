# 004 - BioCLIP Tile Few-Shot

Few-shot inference pipeline for PlantCLEF 2026 multi-label quadrat images.

## Concept

Instead of zero-shot text prompts, this experiment **builds a support bank** from the
labeled single-plant training images and classifies test tiles by similarity to
support-set embeddings.

```
Training images (single plant)
       │
       ▼
  build_support_bank.py
  → sample K images per species
  → embed with BioCLIP
  → compute prototypes           ← per-species mean embedding
  → cache to disk (bank.pt)

Test quadrat images
       │
       ▼
  run_inference.py
  → tile each quadrat image
  → embed each tile with BioCLIP
  → compare tiles to bank (prototype or KNN mode)
  → aggregate tile scores → image-level species scores
  → threshold → submission CSV
```

## File structure

```
004_bioclip_few_shot/
├── build_support_bank.py    # Step 1: build + cache support bank
├── run_inference.py         # Step 2: few-shot inference on test images
├── run_validation.py        # Step 3: local validation on train split
├── models.py                # BioCLIP model loader abstraction
├── few_shot.py              # SupportBank class + prototype/knn scoring
├── tiling.py                # Tiling + image encoding utilities
├── aggregation.py           # Multi-label tile-to-image aggregation
├── dataset.py               # Training metadata loading + path resolution
├── README.md
├── cache/                   # Default cache dir (support banks)
└── outputs/                 # Default inference output dir
```

## Requirements

Same dependencies as the other experiments in this repo:

```bash
pip install open_clip_torch torch pillow pandas scikit-learn
```

## Quick start

### 0. Change to the experiment directory

```bash
cd /root/workspace/PlantCLEF2026/src_experiments/004_bioclip_few_shot
```

---

## Step 1 - Build the support bank

Sample K training images per species, embed them, and cache the result.

```bash
# Default: BioCLIP 1, K=5, random sampling
python build_support_bank.py

# BioCLIP 2, K=10
python build_support_bank.py \
    --model-name bioclip-2 \
    --k 10

# BioCLIP 2.5, K=5 (lower batch size for VRAM)
python build_support_bank.py \
    --model-name bioclip-2.5 \
    --k 5 \
    --batch-size 16

# K=20, top_n_per_species sampling (organ-diverse)
python build_support_bank.py \
    --k 20 \
    --sampling-mode top_n_per_species

# Smoke test: 50 species only
python build_support_bank.py \
    --limit-species 50 \
    --k 3

# Custom paths
python build_support_bank.py \
    --train-meta-csv /path/to/metadata.csv \
    --train-image-root /path/to/images \
    --cache-dir /tmp/my_cache
```

**Key CLI options for `build_support_bank.py`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `bioclip` | `bioclip`, `bioclip-2`, `bioclip-2.5`, or full HF-hub path |
| `--k` | `5` | Max support images per species |
| `--sampling-mode` | `random` | `random`, `capped_all`, `top_n_per_species` |
| `--seed` | `42` | Random seed for reproducibility |
| `--cache-dir` | `./cache` | Where to write the bank artifacts |
| `--train-meta-csv` | *(project default)* | Path to `PlantCLEF2024_single_plant_training_metadata.csv` |
| `--train-image-root` | `/workspace/plantclef/raw/train/images_max_side_800` | Training images root |
| `--species-csv` | *(project default)* | `species_lookup_with_gbif_cleaned_names.csv` |
| `--overwrite` | off | Rebuild even if cache already exists |
| `--limit-species` | None | Only process first N species (smoke test) |

**Output artifacts (`./cache/{run_slug}/`):**

| File | Description |
|------|-------------|
| `bank.pt` | `SupportBank` - embeddings + prototypes (load with `SupportBank.load()`) |
| `bank_metadata.json` | Model name, K, seed, build timestamp, stats |
| `manifest.csv` | One row per support image: species_id, image_name, path |

---

## Step 2 - Run few-shot inference on test images

```bash
# Prototype mode (default), BioCLIP 1, K=5 bank
python run_inference.py \
    --bank-dir ./cache/bioclip_k5_random_seed42 \
    --images-root /workspace/plantclef/kaggle_uploads/test/images

# KNN mode, top-20 species per image
python run_inference.py \
    --bank-dir ./cache/bioclip-2_k10_random_seed42 \
    --images-root /path/to/test/images \
    --scoring-mode knn \
    --top-n 20

# With score threshold
python run_inference.py \
    --bank-dir ./cache/bioclip_k5_random_seed42 \
    --images-root /path/to/test/images \
    --threshold 0.15

# Text fusion: 70% image similarity + 30% zero-shot text similarity
python run_inference.py \
    --bank-dir ./cache/bioclip_k5_random_seed42 \
    --images-root /path/to/test/images \
    --text-alpha 0.7

# mean_top_m aggregation (top 5 tiles averaged)
python run_inference.py \
    --bank-dir ./cache/bioclip_k5_random_seed42 \
    --images-root /path/to/test/images \
    --agg-mode mean_top_m \
    --agg-top-m 5

# Smoke test: 5 images only
python run_inference.py \
    --bank-dir ./cache/bioclip_k5_random_seed42 \
    --images-root /path/to/test/images \
    --limit 5
```

**Key CLI options for `run_inference.py`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--bank-dir` | *(required)* | Support bank directory |
| `--images-root` | *(test images path)* | Test quadrat images |
| `--scoring-mode` | `prototype` | `prototype` or `knn` |
| `--agg-mode` | `max` | `max`, `mean`, `mean_top_m`, `noisy_or` |
| `--agg-top-m` | `3` | Tile count for `mean_top_m` |
| `--threshold` | `0.0` | Global score threshold |
| `--top-n` | `20` | Max predictions per image |
| `--tile-size` | `224` | Tile side length in pixels |
| `--tile-overlap` | `112` | Tile overlap in pixels |
| `--text-alpha` | `1.0` | Image/text fusion weight (1.0 = image-only) |
| `--output-dir` | `./outputs` | Output root |
| `--limit` | None | Process only first N images (debug) |

**Output (`./outputs/{run_slug}/`):**

| File | Description |
|------|-------------|
| `submission.csv` | PlantCLEF format: `quadrat_id`, `species_ids` |
| `predictions_scored.csv` | Per-image per-species scored predictions |
| `run_config.json` | All parameters for reproducibility |
| `summary.json` | Timing, counts |

---

## Step 3 - Local validation

Evaluate on a held-out split of the training data.  The split is **within-species**
so support and query images are always disjoint.

```bash
# Basic validation: prototype mode
python run_validation.py \
    --bank-dir ./cache/bioclip_k5_random_seed42 \
    --val-seed 99

# KNN mode with per-species diagnostics
python run_validation.py \
    --bank-dir ./cache/bioclip-2_k10_random_seed42 \
    --scoring-mode knn \
    --per-species-diag

# Fast smoke test: 100 species
python run_validation.py \
    --bank-dir ./cache/bioclip_k5_random_seed42 \
    --limit-species 100 \
    --val-seed 99
```

**Key CLI options for `run_validation.py`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--bank-dir` | *(required)* | Support bank directory |
| `--scoring-mode` | `prototype` | `prototype` or `knn` |
| `--val-seed` | `99` | Seed for query selection (keep different from bank seed) |
| `--max-query-per-species` | `20` | Cap on query images per species |
| `--min-images-for-val` | `2` | Skip species with too few images |
| `--per-species-diag` | off | Write per-species hit rates to CSV |
| `--limit-species` | None | Only evaluate on first N species |

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| `recall_at_1` | Ground-truth species in top-1 predictions |
| `recall_at_5` | Ground-truth species in top-5 predictions |
| `recall_at_10` | Ground-truth species in top-10 predictions |
| `recall_at_20` | Ground-truth species in top-20 predictions |
| `micro_precision` | Precision across all predictions |
| `micro_recall` | Recall across all predictions |
| `micro_f1` | Harmonic mean of micro precision and recall |
| `macro_recall` | Mean per-species recall |

**Output (`./val_outputs/{run_slug}/`):**

| File | Description |
|------|-------------|
| `val_metrics.json` | All scalar metrics |
| `val_per_species.csv` | Per-species hit rate (if `--per-species-diag`) |
| `val_split_manifest.csv` | Which images were support vs. query |
| `val_config.json` | Full parameters |

---

## Step 4 - Export submission CSV

The `submission.csv` produced by `run_inference.py` is already in PlantCLEF format:

```
"quadrat_id","species_ids"
"ABD_00001","[1396710, 1423789, 1200340]"
...
```

Copy or symlink it to your submission directory.

---

## Scoring modes

### Prototype (default, recommended baseline)

```
prototype[species] = mean(support_embeddings[species])  →  L2-normalise
score[tile, species] = dot(tile_emb, prototype[species])
```

Fast: one matrix multiply per image batch.  Well-calibrated when support
images are diverse enough.

### KNN

```
score[tile, species] = sum(dot(tile_emb, support_k)) / n_support[species]
  where sum is over all support images for that species
```

More expressive than prototype: captures multi-modal species appearance.
Scales with `K x n_species` but manageable for K ≤ 20.

---

## Aggregation modes

| Mode | Formula | When to use |
|------|---------|-------------|
| `max` | `max(tile_scores, dim=tiles)` | Rare/small species - best "exists anywhere" signal |
| `mean` | `mean(tile_scores, dim=tiles)` | Species spread across quadrat |
| `mean_top_m` | `mean(top_m tiles)` | Balance: less noise than mean, less extreme than max |
| `noisy_or` | `1 - prod(1 - sigmoid(score))` | Probabilistic "at least one tile contains species" |

---

## Supported models

| Shorthand | HF-hub path | Notes |
|-----------|-------------|-------|
| `bioclip` | `hf-hub:imageomics/bioclip` | Default; fastest |
| `bioclip-2` | `hf-hub:imageomics/bioclip-2` | Stronger features |
| `bioclip-2.5` | `hf-hub:imageomics/bioclip-2.5-vith14` | Largest; needs `--batch-size 16` |

The bank records which model was used.  `run_inference.py` reads the model from
the bank's metadata automatically - you only need `--model-name` to override.

---

## Image path layout (training data)

```
{train_image_root}/
    {species_id}/
        {image_name}
```

Example:
```
/workspace/plantclef/raw/train/images_max_side_800/1396710/59feabe1...jpg
```

This is discovered automatically from the training metadata CSV.

---

## Hyperparameter sweep example

```bash
for K in 5 10 20; do
  python build_support_bank.py --k $K --model-name bioclip
  python run_validation.py \
    --bank-dir ./cache/bioclip_k${K}_random_seed42 \
    --limit-species 200 \
    --val-seed 99
done
```

---

## Notes and assumptions

1. **Path resolution**: Training images are stored as `{root}/{species_id}/{image_name}`.
   This was confirmed from the actual on-disk layout.

2. **Support/query disjointness**: The validation split assigns images to support
   positionally (first K rows per species) to match the bank.  For exact disjointness,
   run `build_support_bank.py` with `--seed 42` and `run_validation.py` with `--val-seed 99`.

3. **Text fusion**: When `--text-alpha < 1.0`, the experiment imports the prompt
   builder from `002_bioclip_tile_zero_shot_v2`.  Both experiments must be co-located.

4. **Species coverage**: The bank only contains species present in both the training
   metadata CSV and the images directory.  Species with no images are silently skipped
   during bank building.

5. **No fine-tuning**: This pipeline is purely embedding-based.  The BioCLIP weights
   are frozen throughout.
