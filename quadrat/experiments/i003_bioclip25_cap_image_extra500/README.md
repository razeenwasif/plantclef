# i003 — BioCLIP 2.5 with extra_under100 data, capped at 500/species

This experiment is identical to **i002_bioclip25_cap_image** in every respect
(model, augmentations, optimizer, training loop, validation split, inference,
checkpoint format) **except for the training manifest**. It trains on a
combined manifest that merges:

1. The original i002 training manifest (~2.65M rows, 7,806 species).
2. A newly downloaded `extra_under100_train_manifest.csv` (~161.7K rows,
   2,022 under-represented species).

The combined manifest is deduplicated by `image_path` and capped at a maximum
of **500 images per species** (deterministic, seed=42). The cap is baked into
the on-disk manifest so the CSV is the authoritative training set.

## Data sources

| Source | Path | Rows | Species |
|---|---|---|---|
| Old (i002) | `i001_data_download/data/training_usage/metadata_filled_genus_family.csv` | 2,653,782 | 7,806 |
| New (extra_under100) | `i001_data_download/data/extra_under100/extra_under100_train_manifest.csv` | 161,686 | 2,022 |
| Combined (i003) | `data/combined_old_extra_max500_train_manifest.csv` | (filled in by `prepare_data.sh`) | — |

## What changed vs i002

1. **New script** `prepare_combined_manifest.py` builds the combined+capped
   training manifest. It reuses `data/metadata_utils.py` from i002 unchanged
   (`load_metadata_csv`, `apply_max_images_per_species_cap`,
   `print_species_distribution`).
2. **`dataset.py`**: `DEFAULT_TRAIN_META_CSV` now points to
   `data/combined_old_extra_max500_train_manifest.csv` under this experiment.
3. **All `scripts/*.sh`** point to the i003 combined manifest. They do **not**
   pass `--max-images-per-species` — the cap is already baked into the manifest,
   so passing the flag would re-sample over an already-capped set.
4. The new `extra_under100` manifest lacks `genus`/`family` columns. They are
   filled at prep time by joining `species_id` against the old manifest. Rows
   with no match keep `NaN`, which `MultiTaskDataset._encode` already handles
   as `-1` and masks during loss computation.

## Run order

```bash
cd /root/workspace/PlantCLEF2026/src_experiments/i003_bioclip25_cap_image_extra500

# 1. Build combined+capped manifest (writes data/*.csv and data/*.json)
bash scripts/prepare_data.sh

# 2. Smoke test: ~200 samples, 1 epoch, frozen backbone, fp16 — verifies pipeline
bash scripts/smoke_test.sh           # 1 GPU
bash scripts/smoke_test.sh 2         # 2 GPUs

# 3. Stage 1: head-only training (frozen backbone), 10 epochs, bf16, batch 512
bash scripts/train_head_only.sh 2

# 4. Stage 2a: unfreeze last 4 blocks, resume from Stage 1
bash scripts/train_last_blocks.sh ./outputs/head_only/checkpoints/best.pt 2

# 5. Optional follow-on stages
bash scripts/train_last_blocks_8.sh  ./outputs/last_blocks/checkpoints/best.pt 2
bash scripts/train_last_blocks_12.sh ./outputs/last_blocks_8/checkpoints/best.pt 2
bash scripts/train_full_finetune.sh  ./outputs/last_blocks_12/checkpoints/best.pt 2
```

Inference scripts (`scripts/infer_tiles.sh`, `scripts/infer_best_adaptive.sh`)
work the same as in i002 — point them at a checkpoint produced above.

## Output locations

- `data/combined_old_extra_max500_train_manifest.csv` — final capped manifest
  (the training set).
- `data/combined_old_extra_max500_summary.json` — counts before/after dedup and
  cap, distribution stats, species-at-cap, species-still-under-100.
- `data/species_counts_before_after.csv` — per-species counts before and after
  the cap, with `scientific_name`.
- `outputs/{run_name}/checkpoints/best.pt` — best checkpoint per stage.
- `outputs/{run_name}/encoders/` — JSON label encoders (species/genus/family).
- `outputs/{run_name}/metrics.csv`, `metrics.json` — per-epoch metrics.
- `outputs/{run_name}/train_config.json` — full hyperparameter snapshot.

## Notes / invariants

- **Dedup before cap.** The prepare script removes duplicate `image_path` rows
  *before* the 500-per-species cap. Capping first and deduping after would
  silently shrink species below 500.
- **Validation split happens at train time.** `train.py` calls `build_val_split`
  on the (already-capped) manifest with `val_fraction=0.1`, `seed=42`, stratified
  by `species_id`. Species with fewer than 5 images stay entirely in train. No
  changes from i002.
- **No `--max-images-per-species` at train time.** The manifest is authoritative.
- **Reproducibility.** Cap seed = 42 (also configurable via `--seed`); val seed =
  42 (configurable via `--val-seed`).
