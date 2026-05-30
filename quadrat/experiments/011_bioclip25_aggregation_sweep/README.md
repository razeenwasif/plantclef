# 011 — BioCLIP-2.5 aggregation α-sweep

Inference-only experiment. **No new training.** Takes the existing 010
last_blocks per-tile probability dump and tests whether mixing `probs_mean`
with `probs_noisy_or` (or `probs_max`) lifts the team's current 0.38333
baseline.

## Why

EDA in `eda/plantclef2026_eda.ipynb` (commit f87e0a7) found:

- Median **# species with p > 0.01 = 9** per quadrat — top-3 truncates
  signal, but the K-sweep memory (`top-2=0.373, top-3=0.383, top-4=0.372`)
  says fixed K is locked in.
- For the 5 spread-of-entropy quadrats inspected: confident species fire
  with `max/mean ≈ 1` (uniformly across all 16 tiles); uncertain species
  hit `max/mean = 5–15×` — i.e. one strong tile + noise everywhere else.
- `softmax_mean` (current team-best aggregation) averages those localized
  strong tiles into oblivion. `probs_noisy_or` (= 1 − ∏(1 − p_tile)) fires
  whenever *any* tile is confident.

The hypothesis: a hybrid `α · mean + (1 − α) · noisy_or` should rescue
the localized-but-strong species without losing the "many-tiles-agree"
signal.

## Inputs

- `/workspace/working/workspace/PlantCLEF2026/src_experiments/010_bioclip25_end_to_end_finetune_multitask/outputs/test_probs_010_last_blocks_grid4x4.npz`
  — already contains `probs_max`, `probs_mean`, `probs_noisy_or` over the
  16 grid_4x4 tiles per quadrat for the team-best 010 last_blocks model.

## Run

```bash
python alpha_sweep.py \
    --probs /workspace/working/workspace/PlantCLEF2026/src_experiments/010_bioclip25_end_to_end_finetune_multitask/outputs/test_probs_010_last_blocks_grid4x4.npz \
    --out-dir outputs \
    --mix mean_x_noisy_or \
    --alphas 0.0 0.25 0.5 0.75 1.0 \
    --top-k 3
```

Produces 5 CSVs:

```
submission_011_aggMxNO_a0p00_top3.csv   # pure probs_noisy_or
submission_011_aggMxNO_a0p25_top3.csv
submission_011_aggMxNO_a0p50_top3.csv
submission_011_aggMxNO_a0p75_top3.csv
submission_011_aggMxNO_a1p00_top3.csv   # pure probs_mean (= 010 baseline)
```

The α=1.0 file should reproduce the 010 last_blocks score (~0.374–0.383
depending on bf16 noise) — it's the canary that the pipeline is correct.

## Submit

```bash
bash scripts/submit_predictions_kaggle.sh \
    -i src_experiments/011_bioclip25_aggregation_sweep/outputs \
    -p 'submission_011_*.csv' \
    -o scores.csv
```

## Anchor

Beat **0.38333** (010 last_blocks pure `probs_mean`, top-3). Any α < 1.0
that beats this validates the localized-tile-rescue hypothesis and a finer
sweep is warranted.
