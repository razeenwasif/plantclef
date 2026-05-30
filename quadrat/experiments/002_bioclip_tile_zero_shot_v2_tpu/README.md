# 002 - BioCLIP Tile Zero-Shot v2 (TPU/XLA version)

Zero-shot species classification for PlantCLEF 2026 — TPU-compatible port of
`src_experiments/002_bioclip_tile_zero_shot_v2`.

Supports **BioCLIP 1, 2, and 2.5** with SAHI-style tiled inference and prompt
ensembling from GBIF metadata.

---

## What changed from the CUDA version

| Area | CUDA version | TPU version |
|---|---|---|
| Device resolution | `"cuda"` if available, else `"cpu"` | TPU (XLA) > CUDA > CPU via `device_utils.py` |
| `--device` choices | `auto\|cuda\|cpu\|cuda:0` | `auto\|tpu\|cuda\|cpu` |
| XLA graph flushing | n/a | `xm.mark_step()` after each encode batch |
| `tile_top_k` | `.numpy()` (fails on XLA tensors) | `.cpu().numpy()` |
| New file | n/a | `device_utils.py` — device resolution + `mark_step` helper |

All inference logic (tiling, prompt ensembling, scoring, aggregation, output
format) is identical to the CUDA version.

---

## Files

```
002_bioclip_tile_zero_shot_v2/
├── run_inference.py      Main CLI inference script (adapted)
├── device_utils.py       Device resolution + XLA mark_step helper (NEW)
├── utils.py              Tiling, encoding, scoring utilities (adapted)
├── prompt_builder.py     Builds per-species text prompts (unchanged)
├── compare_models.py     Compare run summaries across models (unchanged)
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

## Dependencies

```
torch
open_clip_torch
Pillow
pandas
```

For TPU support:

```
torch-xla   # Google Cloud TPU — follow https://cloud.google.com/tpu/docs/pytorch-xla
```

`torch_xla` is imported lazily inside `device_utils.py`, so the code runs
normally on CPU/CUDA even if `torch_xla` is not installed.

---

## Device selection

| `--device` | Behaviour |
|---|---|
| `auto` | Try TPU (XLA) → CUDA → CPU. **Default.** |
| `tpu` | Require PyTorch/XLA; hard error if not installed or no TPU found. |
| `cuda` | Require CUDA; hard error if no GPU. |
| `cpu` | Always use CPU regardless of available hardware. |

`summary.json` records both `device` (the resolved device string) and
`backend` (`xla`, `cuda`, or `cpu`) for later comparison.

---

## Smoke test (5 images, CPU fallback)

Use this to verify the code runs correctly without requiring a TPU:

```bash
cd /path/to/src_experiments_tpu/002_bioclip_tile_zero_shot_v2

python run_inference.py \
    --model-name hf-hub:imageomics/bioclip \
    --species-csv ./data/species_lookup_with_gbif_cleaned_names.csv \
    --images-root /path/to/test/images \
    --prompt-mode scientific \
    --device cpu \
    --limit 5 \
    --output-dir ./outputs_smoke
```

---

## TPU run (full inference)

```bash
python run_inference.py \
    --model-name hf-hub:imageomics/bioclip \
    --species-csv ./data/species_lookup_with_gbif_cleaned_names.csv \
    --images-root /path/to/test/images \
    --prompt-mode all \
    --device tpu \
    --batch-size 64 \
    --tile-size 224 \
    --tile-overlap 112 \
    --top-k 5 \
    --output-dir ./outputs
```

For BioCLIP 2.5 (ViT-H/14), use a smaller batch size:

```bash
python run_inference.py \
    --model-name hf-hub:imageomics/bioclip-2.5-vith14 \
    --prompt-mode all \
    --device tpu \
    --batch-size 16 \
    --output-dir ./outputs
```

---

## Comparing runs

```bash
# Auto-discover all runs
python compare_models.py --parent-dir outputs/

# Save comparison to CSV
python compare_models.py --parent-dir outputs/ --csv outputs/comparison.csv
```

---

## XLA / TPU execution notes

- **`xm.mark_step()`** is called after each text-encode batch and each
  image-encode batch. This flushes the XLA execution graph and prevents
  unbounded graph growth / OOM.
- **Recompilation**: XLA traces the computation graph on first use.
  If images have consistent dimensions (same tile count per image), subsequent
  images reuse the cached graph. A variable tile count (mixed image sizes) can
  trigger extra recompilations but does not affect correctness.
- **bfloat16**: Not enabled by default (preserves parity with the CUDA
  version). OpenCLIP's `model.to(device)` on TPU will use bfloat16 if the
  device default is bfloat16 — check your XLA runtime configuration.
- **CPU fallback**: The code runs on CPU with no code changes by passing
  `--device cpu`. This is useful for local development and CI.

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
  --device              auto | tpu | cuda | cpu  [default: auto]
  --top-k               Top-k predictions per image  [default: 5]
  --limit               Process only first N images (smoke test)
```

---

## Output file reference

| File | Description |
|---|---|
| `run_config.json` | All CLI args + resolved device, backend, stride |
| `prompt_table.csv` | Per-species: id, canonical name, n_prompts, full prompt list (JSON) |
| `prompt_summary.json` | Aggregate prompt stats |
| `submission.csv` | PlantCLEF format: `quadrat_id`, `species_ids` |
| `predictions_topk.csv` | Per-image top-k: rank, species_id, species_name, logit_score |
| `summary.json` | Timing, counts, model/prompt config, device, backend |
