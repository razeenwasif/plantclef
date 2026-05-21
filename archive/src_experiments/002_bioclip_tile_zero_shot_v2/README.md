# 003 - BioCLIP Zero-Shot Multi-Model

Zero-shot species classification for PlantCLEF 2026 with support for
**BioCLIP 1, 2, and 2.5** and enriched prompt ensembling from GBIF metadata.

---

## What changed from the old experiment (`bioclip_tile_zero_shot`)

| Feature | Old | New |
|---|---|---|
| Models | BioCLIP 1 only | BioCLIP 1, 2, 2.5 |
| Text prompts | 3 templates × species name | 3-20 prompts/species via 4 template families |
| Prompt ensembling | average 3 templates | avg + renorm over all per-species prompts |
| Species metadata | `species_id_to_name.txt` (name only) | enriched GBIF CSV (common names, family, synonyms) |
| Tokenizer | hardcoded to BioCLIP 1 | model-specific (via `open_clip.get_tokenizer`) |
| Transforms | hardcoded to BioCLIP 1 | model-specific (via `open_clip.create_model_and_transforms`) |
| Output | submission.csv only | submission, top-k CSV, prompt table, config, summary |
| Ablations | none | 4 prompt modes (scientific, scientific_common, scientific_family, all) |
| Cross-model comparison | none | `compare_models.py` |

---

## Files

```
003_bioclip_zero_shot_multimodel/
├── run_inference.py      Main CLI inference script
├── prompt_builder.py     Builds per-species text prompts from the GBIF CSV
├── utils.py              Tiling, encoding, scoring utilities
├── compare_models.py     Compare run summaries across models / prompt modes
└── outputs/              Created at runtime
    └── {model}_{mode}/
        ├── run_config.json
        ├── prompt_table.csv
        ├── prompt_summary.json
        ├── submission.csv
        ├── predictions_topk.csv
        └── summary.json
```

---

## Quick start

```bash
cd /root/workspace/PlantCLEF2026/src_experiments/003_bioclip_zero_shot_multimodel
```

### Smoke test (5 images, BioCLIP 1)

```bash
python run_inference.py \
    --model-name hf-hub:imageomics/bioclip \
    --species-csv ./data/species_lookup_with_gbif_cleaned_names.csv \
    --images-root /path/to/images \
    --prompt-mode scientific \
    --limit 5
```

---

## Running BioCLIP 1

```bash
python run_inference.py \
    --model-name hf-hub:imageomics/bioclip \
    --species-csv ../data/species_lookup_with_gbif_cleaned_names.csv \
    --images-root /path/to/images \
    --prompt-mode all \
    --batch-size 64 \
    --tile-size 224 \
    --tile-overlap 112 \
    --top-k 5 \
    --output-dir outputs/
```

## Running BioCLIP 2

```bash
python run_inference.py \
    --model-name hf-hub:imageomics/bioclip-2 \
    --species-csv ./data/species_lookup_with_gbif_cleaned_names.csv \
    --images-root /path/to/images \
    --prompt-mode all \
    --batch-size 64 \
    --output-dir outputs/
```

## Running BioCLIP 2.5

BioCLIP 2.5 uses ViT-H/14 - use a smaller batch size on typical GPUs.

```bash
python run_inference.py \
    --model-name hf-hub:imageomics/bioclip-2.5-vith14 \
    --species-csv ./data/species_lookup_with_gbif_cleaned_names.csv \
    --images-root /path/to/images \
    --prompt-mode all \
    --batch-size 16 \
    --output-dir outputs/
```

---

## Prompt modes

| Mode | Families | Approx prompts/species |
|---|---|---|
| `scientific` | A only | 3-9 |
| `scientific_common` | A + B + D | 6-18 |
| `scientific_family` | A + C | 5-11 |
| `all` | A + B + C + D | 8-22 |

**Recommended default:** `all` for best accuracy; `scientific` for fastest runs.

Template families:

- **A** - `"a photo of {sci}"`, `"a close-up photo of {sci}"`, `"a wild plant of species {sci}"`
- **B** - `"a photo of {common}"`, `"a close-up photo of {common}"`, `"a wild plant called {common}"`
- **C** - `"a photo of {sci}, a plant in the family {family}"`, (close-up variant)
- **D** - `"a photo of {sci} ({common})"`, (close-up variant)

---

## Comparing runs

```bash
# Auto-discover all runs under outputs/
python compare_models.py --parent-dir outputs/

# Explicit runs + save to CSV
python compare_models.py \
    outputs/bioclip_scientific \
    outputs/bioclip-2_all \
    outputs/bioclip-2-5-vith14_all \
    --csv outputs/comparison.csv
```

---

## CLI reference

```
run_inference.py arguments:

  --model-name          OpenCLIP model id  [default: hf-hub:imageomics/bioclip]
  --species-csv         Enriched GBIF species CSV
  --images-root         Directory with test images (flat, jpg/jpeg/png)
  --output-dir          Root output dir  [default: ./outputs]
  --prompt-mode         scientific | scientific_common | scientific_family | all
  --max-common-names    Extra common names per species  [default: 3]
  --max-synonyms        GBIF synonyms as extra scientific names  [default: 2]
  --tile-size           Tile side length in pixels  [default: 224]
  --tile-overlap        Overlap between tiles  [default: 112]
  --batch-size          Image batch size (model-specific default: 64/16)
  --text-batch-size     Text encoding batch size  [default: 256]
  --device              auto | cuda | cpu | cuda:0  [default: auto]
  --top-k               Top-k predictions per image  [default: 5]
  --limit               Process only first N images (smoke test)
```

---

## VRAM and speed notes

| Model | Param count | Recommended batch | ~VRAM (bs=64) |
|---|---|---|---|
| BioCLIP 1 (ViT-B/16) | ~86M | 64 | ~6 GB |
| BioCLIP 2 (ViT-B/16) | ~86M | 64 | ~6 GB |
| BioCLIP 2.5 (ViT-H/14) | ~632M | 16 | ~12 GB |

Text encoding is done once per run regardless of image count.
With 7806 species × ~18 prompts (`all` mode) ≈ 140k prompts - expect 30-90s for text encoding.

---

## Output file reference

| File | Description |
|---|---|
| `run_config.json` | All CLI args + resolved device/stride |
| `prompt_table.csv` | Per-species: id, canonical name, n_prompts, full prompt list (JSON) |
| `prompt_summary.json` | Aggregate prompt stats |
| `submission.csv` | PlantCLEF format: `quadrat_id`, `species_ids` |
| `predictions_topk.csv` | Per-image top-k: rank, species_id, species_name, logit_score |
| `summary.json` | Timing, counts, model/prompt config |
