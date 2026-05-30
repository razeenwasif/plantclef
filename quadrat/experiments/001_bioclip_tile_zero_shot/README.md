# BioCLIP Zero-Shot Tile Classification

Zero-shot plant species classification on image tiles using [BioCLIP](https://huggingface.co/imageomics/bioclip),
with SAHI-style max-pool aggregation from tiles to image-level predictions.

## Setup

```bash
pip install open_clip_torch pillow pandas matplotlib
```

## Debug mode - one image

```bash
cd /root/workspace/PlantCLEF2026/src_experiments/bioclip_tile_zero_shot

# First image in test set (default)
python run_one_image.py

# Specific image
python run_one_image.py --image_path /workspace/plantclef/kaggle_uploads/test/images/CBN-PdlC-A1-20130807.jpg

# Custom tiling / top-k
python run_one_image.py --tile_size 336 --stride 168 --top_k 5
```

Outputs (saved to `./outputs/`):

| File | Description |
|------|-------------|
| `<name>_tile_predictions.csv` | One row per tile, top-3 species + scores |
| `<name>_tile_grid.jpg` | Image annotated with tile boxes and top-1 label |
| `<name>_confidence_heatmap.jpg` | Image overlaid with top-1 confidence heatmap |

Also prints the image-level top-K species after SAHI max-pool aggregation.

## Submission mode - all images

```bash
python run_all_images_submission.py

# Custom paths / settings
python run_all_images_submission.py \
    --images_dir /workspace/plantclef/kaggle_uploads/test/images \
    --mapping /workspace/plantclef/raw/models/pretrained_models/species_id_to_name.txt \
    --output ./outputs/submission.csv \
    --tile_size 224 \
    --stride 112 \
    --top_k 5
```

Output CSV format:

```text
"quadrat_id","species_ids"
"CBN-PdlC-A1-20130807","[1395974, 1392611, 1412585, 1646477, 1390677]"
```

## Aggregation

Tile logits are aggregated using **max pooling** (SAHI-style):

```python
image_logits = tile_logits.max(dim=0).values  # (n_species,)
```

This picks the strongest signal across all tiles before ranking, without
diluting confident predictions from partial tiles.

## Key parameters

| Arg | Default | Description |
|-----|---------|-------------|
| `--tile_size` | 224 | Square tile side length (pixels) |
| `--stride` | 112 | Stride between tiles; 112 = 50% overlap |
| `--batch_size` | 64 | Image encoding batch size |
| `--top_k` | 5 | Top species per image in submission |

## Notes

- Text features are encoded once per run and reused for all images.
- Each species uses 3 prompt templates; embeddings are averaged then re-normalized.
- No fine-tuning or classifier head - pure zero-shot cosine similarity.
