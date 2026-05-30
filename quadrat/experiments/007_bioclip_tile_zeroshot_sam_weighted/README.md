# Experiment 007 — BioCLIP Zero-Shot with SAM Vegetation-Aware Tile Weighting

## What this experiment does

Extends the 002 BioCLIP tiled zero-shot baseline by adding a **vegetation scoring step**
that estimates how much of each tile contains plant/vegetation content.
Those scores are used to **softly reweight tile logits** before image-level aggregation,
so tiles rich in vegetation contribute more to the final species prediction.

## How it differs from 002

| Feature                    | 002 (baseline)       | 007 (this experiment)         |
|----------------------------|----------------------|-------------------------------|
| Vegetation scoring         | None                 | RGB ExG or SAM masks          |
| Tile aggregation           | Max-pool only        | max / mean / weighted_mean / topk_mean / weighted_topk_mean |
| Tile weighting             | No                   | w = clip(α + β·veg_ratio, min, max) |
| Per-tile metadata CSV      | No                   | Yes                           |
| Visualizations             | None                 | 5 panel types per image + dataset summary |

The old experiment is untouched; this one is self-contained.

## File structure

```
007_bioclip_tile_zeroshot_sam_weighted/
├── run_inference.py     # Main entry point
├── config.py            # Constants and CLI defaults
├── sam_weighting.py     # Vegetation scoring (RGB ExG + SAM)
├── aggregation.py       # All aggregation methods
├── visualization.py     # All visualization code
├── utils.py             # Tiling + BioCLIP encoding (copied from 002)
├── prompt_builder.py    # Species prompt building (copied from 002)
├── download_sam.py      # Helper to fetch SAM checkpoint
└── outputs/
    ├── visualizations/  # Per-image and summary plots
    ├── predictions/     # (used as default --output-dir)
    └── diagnostics/
```

## How weighting works

For each tile *i* in an image:

1. **Vegetation score** `veg_i` = fraction of tile pixels classified as vegetation
   - `rgb` mode: `veg_i = mean(ExG > threshold)` where `ExG = 2G - R - B`
   - `sam` mode: SAM segments the tile; each mask is scored for greenness;
     `veg_i = union-area of green masks / total tile area`

2. **Tile weight**:
   ```
   w_i = clip(alpha + beta * veg_i, w_min, w_max)
   ```
   Default: `alpha=0.5, beta=1.0, w_min=0.1, w_max=2.0`
   → pure-background tile: w=0.5;  mixed tile: w=1.0;  pure-plant tile: w=1.5

3. **Weighted aggregation** (example: `weighted_mean`):
   ```
   image_logits = Σ (w_i / Σw_j) * tile_logits_i
   ```

No hard filtering is applied — every tile still contributes.

## How to run

### Quick smoke test (RGB scoring, 5 images, baseline max-pool)

```bash
cd 007_bioclip_tile_zeroshot_sam_weighted
python run_inference.py \
    --aggregation max \
    --scoring rgb \
    --n-visualize 3 \
    --limit 5
```

### Weighted inference (RGB, no SAM checkpoint needed)

```bash
python run_inference.py \
    --aggregation weighted_mean \
    --scoring rgb \
    --weight-alpha 0.5 \
    --weight-beta 1.0 \
    --n-visualize 10 \
    --limit 50
```

### SAM-weighted inference

First download the SAM checkpoint (~375 MB, one-time):

```bash
python download_sam.py --model-type vit_b
```

Then run:

```bash
python run_inference.py \
    --scoring sam \
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \
    --aggregation weighted_mean \
    --n-visualize 10
```

### Run baseline alongside weighted (saves comparison CSV + comparison plot)

```bash
python run_inference.py \
    --aggregation weighted_mean \
    --also-run-baseline \
    --scoring rgb \
    --n-visualize 10 \
    --limit 100
```

### Ablation sweep

```bash
for agg in max mean weighted_mean topk_mean weighted_topk_mean; do
    python run_inference.py --aggregation $agg --scoring rgb --limit 200
done
```

## CLI reference

| Argument               | Default                    | Description |
|------------------------|----------------------------|-------------|
| `--model-name`         | `hf-hub:imageomics/bioclip` | BioCLIP model |
| `--aggregation`        | `weighted_mean`            | max \| mean \| topk_mean \| weighted_mean \| weighted_topk_mean |
| `--scoring`            | `rgb`                      | rgb (fast, no checkpoint) \| sam |
| `--sam-checkpoint`     | `./checkpoints/sam_vit_b…` | SAM .pth file |
| `--sam-model-type`     | `vit_b`                    | vit_b \| vit_l \| vit_h |
| `--weight-alpha`       | `0.5`                      | Base weight for zero-veg tile |
| `--weight-beta`        | `1.0`                      | Weight slope per unit veg_ratio |
| `--weight-min`         | `0.1`                      | Minimum tile weight |
| `--weight-max`         | `2.0`                      | Maximum tile weight |
| `--exg-threshold`      | `20.0`                     | Raw ExG value for green-pixel detection |
| `--topk-tiles`         | `3`                        | k for topk_mean aggregation |
| `--also-run-baseline`  | off                        | Also compute max-pool baseline per image |
| `--n-visualize`        | `10`                       | Images for detailed visualizations |
| `--no-save-tiles`      | off                        | Skip per-tile metadata CSV |
| `--limit`              | None                       | Process only first N images |

## Outputs

```
outputs/{run_slug}/
    run_config.json              all CLI args + runtime metadata
    prompt_table.csv             per-species prompts
    prompt_summary.json          prompt aggregate stats
    submission.csv               PlantCLEF format: quadrat_id, species_ids
    predictions_topk.csv         per-image top-k predictions
    tile_metadata.csv            per-tile: coords, veg_ratio, weight, top-1 species/score
    comparison_baseline.csv      baseline predictions (if --also-run-baseline)
    summary.json                 timing + mean veg stats
    visualizations/
        {image_id}/
            A_tile_overview.png        tile grid coloured by veg_ratio and weight
            B_veg_heatmap.png          tile thumbnails annotated with veg/weight
            C_sam_tile??.png           SAM mask overlays (sam mode only)
            D_tile_diagnostics.png     per-tile table: veg, weight, top species
            E_prediction_comparison.png  baseline vs weighted top-k
        summary_veg_stats.png    histograms + scatter across all tiles
```

## Known limitations

- **RGB mode** (`--scoring rgb`): ExG is a colour heuristic.  It works well for green
  leafy vegetation but may over-count green rocks, blue sky can undercount pale leaves.
  It is fast and requires no model download.

- **SAM mode** (`--scoring sam`): SAM segments *objects* without semantic understanding.
  The vegetation label is assigned by ExG greenness per mask, not by a plant classifier.
  Dense uniform backgrounds may produce zero masks (falls back to RGB score for that tile).
  SAM adds ~0.5–2s per tile on CPU; use GPU (`--device cuda`) for reasonable speed.

- **No hard filtering**: weights are soft — all tiles always contribute.  This is intentional:
  hard filtering can hurt recall on sparse-veg images.

- **BioCLIP 2.5 + SAM on same GPU**: both models competing for VRAM may require reducing
  `--batch-size` to 8 for BioCLIP 2.5.
