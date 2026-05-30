# BioCLIP 2.5 Linear Probe Baseline - PlantCLEF 2026

Frozen BioCLIP 2.5 ViT-H/14 backbone + single linear classification head.

## Files

| File | Purpose |
|------|---------|
| `dataset.py`     | Metadata loading, class mapping, train/val split, datasets |
| `model.py`       | `BioCLIP25LinearProbe` - frozen backbone + linear head |
| `transforms.py`  | Training augmentation, val/inference transforms, `InferencePreprocessor` |
| `tiling.py`      | Grid tiling, multi-scale mode, tile encoding, aggregation |
| `utils.py`       | Logging, checkpointing, recall@K metrics, DDP helpers |
| `train.py`       | Training loop (single-GPU and multi-GPU via torchrun) |
| `build_cache.py` | Pre-compute all training embeddings once → fast head-only training |
| `validate.py`    | Standalone validation with configurable tiling/preprocessing ablations |
| `infer.py`       | Test inference → `submission.csv` |

---

## Quick start

All scripts must be run from the experiment directory:

```bash
cd /root/workspace/PlantCLEF2026/src_experiments/003_bioclip25_linear_probe
```

---

## Training

### Single GPU (standard - backbone runs live each step)

```bash
python train.py \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-3 \
    --output-dir ./outputs/train
```

### Multi-GPU (4 GPUs via torchrun)

```bash
torchrun --nproc_per_node=4 train.py \
    --epochs 10 \
    --batch-size 64 \
    --lr 1e-3 \
    --output-dir ./outputs/train
```

Effective batch size = `batch_size x n_gpus`.  Scale LR linearly if needed:
`--lr 4e-3` for 4 GPUs at batch 64 per GPU.

### Fast training with cached embeddings (recommended for iteration speed)

**Step 1: Build the embedding cache once (~30–60 min on a single GPU)**

```bash
python build_cache.py \
    --output-dir ./cache \
    --batch-size 128
```

This produces `cache/train_embeddings.pt`, `cache/val_embeddings.pt`,
`cache/class_mapping.txt`, and `cache/cache_meta.json`.

**Step 2: Train head-only on the cache (very fast - no backbone FLOPs)**

```bash
python train.py \
    --use-cache \
    --cache-dir ./cache \
    --epochs 30 \
    --batch-size 2048 \
    --lr 1e-2 \
    --output-dir ./outputs/train_cached
```

### Useful training flags

```bash
# Cap 50 samples per species for a quick 10-min smoke test
python train.py --max-samples-per-class 50 --epochs 2

# Disable mixed precision (slower, for debugging)
python train.py --no-amp

# Resume interrupted run
python train.py --resume ./outputs/train/checkpoints/latest.pt
```

---

## Validation (whole-image vs. tiled)

### Whole-image baseline (default)

```bash
python validate.py \
    --checkpoint ./outputs/train/checkpoints/best.pt
```

### 2x2 grid tiling

```bash
python validate.py \
    --checkpoint ./outputs/train/checkpoints/best.pt \
    --tile-mode grid_2x2
```

### 4x4 grid tiling

```bash
python validate.py \
    --checkpoint ./outputs/train/checkpoints/best.pt \
    --tile-mode grid_4x4
```

### Multi-scale tiling (whole + 2x2 + 4x4)

```bash
python validate.py \
    --checkpoint ./outputs/train/checkpoints/best.pt \
    --tile-mode multiscale \
    --agg-mode topk_mean \
    --topk-agg 5
```

### Quick smoke test (first 200 val images)

```bash
python validate.py \
    --checkpoint ./outputs/train/checkpoints/best.pt \
    --limit 200
```

---

## Preprocessing / interpolation ablations

All ablations are controlled via flags to `validate.py` or `infer.py`.

### Interpolation: bicubic vs. lanczos

```bash
# Bicubic (default)
python validate.py --checkpoint ... --interp bicubic

# Lanczos
python validate.py --checkpoint ... --interp lanczos
```

### JPEG recompression

```bash
# No recompression (default)
python validate.py --checkpoint ...

# JPEG q=94 (mild)
python validate.py --checkpoint ... --jpeg-quality 94

# JPEG q=85 (moderate)
python validate.py --checkpoint ... --jpeg-quality 85

# JPEG q=85 + 4:2:0 chroma subsampling
python validate.py --checkpoint ... --jpeg-quality 85 --jpeg-subsampling 2
```

### Margin crop

```bash
# 5% border crop on each side before any other transform
python validate.py --checkpoint ... --margin-crop 0.05
```

### Combined ablation: lanczos + JPEG q=94 + 5% crop

```bash
python validate.py \
    --checkpoint ./outputs/train/checkpoints/best.pt \
    --tile-mode multiscale \
    --interp lanczos \
    --jpeg-quality 94 \
    --margin-crop 0.05
```

---

## Test inference → submission

### Whole-image inference

```bash
python infer.py \
    --checkpoint ./outputs/train/checkpoints/best.pt \
    --test-dir /workspace/plantclef/kaggle_uploads/test/images \
    --output-dir ./outputs/infer_whole
```

### 2x2 tiled inference

```bash
python infer.py \
    --checkpoint ./outputs/train/checkpoints/best.pt \
    --tile-mode grid_2x2 \
    --output-dir ./outputs/infer_2x2
```

### Multi-scale + lanczos (strongest setting)

```bash
python infer.py \
    --checkpoint ./outputs/train/checkpoints/best.pt \
    --tile-mode multiscale \
    --agg-mode topk_mean \
    --topk-agg 5 \
    --interp lanczos \
    --output-dir ./outputs/infer_multiscale
```

### Smoke test (5 images)

```bash
python infer.py \
    --checkpoint ./outputs/train/checkpoints/best.pt \
    --limit 5 \
    --output-dir ./outputs/infer_smoke
```

Output: `./outputs/infer/submission.csv` in PlantCLEF format.

---

## Submit to Kaggle

```bash
# From repo root
bash /root/workspace/PlantCLEF2026/scripts/submit_predictions_kaggle.sh \
    /root/workspace/PlantCLEF2026/src_experiments/003_bioclip25_linear_probe/outputs/infer/submission.csv
```

---

## Design notes

**Backbone**: `hf-hub:imageomics/bioclip-2.5-vith14` (ViT-H/14). Embed dim
probed dynamically at model construction - not hard-coded.

**Training objective**: `CrossEntropyLoss(label_smoothing=0.1)` on single-label
integer species indices.  7,806 classes.

**Optimizer**: AdamW on head parameters only, CosineAnnealingLR, warmup not
needed for a linear probe.

**Mixed precision**: AMP with `GradScaler` by default (`--no-amp` to disable).

**DDP**: `find_unused_parameters=False` since all frozen backbone parameters
have `requires_grad=False` - DDP only syncs the head gradients.  Rank-0 does
all checkpointing, logging, and validation.

**Tiling**: Grid-based (exact NxN cells), not sliding-window.  Overlap is
additive margin per side (e.g. `--overlap 0.25` adds 25% of cell size on each
side).

**Aggregation**:
- `max`: element-wise maximum over tiles - retains strongest signal per class
- `topk_mean`: average the k most confident tiles (ranked by their peak logit)

**Preprocessing pipeline order** (inference): margin crop → JPEG recompress → resize → center crop → normalize.

