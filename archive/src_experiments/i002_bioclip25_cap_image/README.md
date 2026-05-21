# Experiment 010 — BioCLIP 2.5 End-to-End Fine-Tuning with Multi-Task Taxonomy Learning

End-to-end supervised fine-tuning of BioCLIP 2.5 (ViT-H/14) for PlantCLEF 2026 species classification.  
Adds a deeper MLP prediction head and auxiliary taxonomy losses (genus, family, order, class) to regularise training and improve generalisation.

---

## Contents

```
010_bioclip25_end_to_end_finetune_multitask/
├── dataset.py        # Metadata + taxonomy merge, label encoders, MultiTaskDataset
├── model.py          # BioCLIP25MultiTask: backbone + SharedMLP + 5 classification heads
├── transforms.py     # Train (strong augmentations) and val transforms
├── utils.py          # AMP context, cosine+warmup scheduler, loss, metrics helpers
├── train.py          # Main training script
├── validate.py       # Standalone evaluation → predictions.csv
├── infer_tiles.py    # Tile-based test inference → submission.csv
└── scripts/
    ├── smoke_test.sh
    ├── train_head_only.sh
    ├── train_last_blocks.sh
    └── train_full_finetune.sh
```

---

## Architecture

```
Input image (224×224)
       │
BioCLIP 2.5 ViT-H/14 backbone   ← frozen by default
       │  (1024-dim embedding)
       ▼
LayerNorm → Linear(1024→1024) → GELU → Dropout(0.2)   ← shared MLP
       │
   ┌───┴───┬───────┬────────┬──────────┐
species  genus  family   order    class    ← linear heads
(7806)  (1446)  (181)    (61)      (6)
```

**Species head** is always active.  
**Taxonomy heads** (genus/family/order/class) are optional and controlled by `--use-taxonomy-heads`.

### Multi-Task Loss

```
loss = species_loss
     + 0.30 × genus_loss    (only for samples with a genus label)
     + 0.15 × family_loss   (only for samples with a family label)
     + 0.05 × order_loss    (only for samples with an order label)
     + 0.02 × class_loss    (only for samples with a class label)
```

Missing labels (NaN in the CSV) are silently skipped — no crash, no bias.

---

## Dataset

| Level   | Coverage | Unique classes |
|---------|----------|----------------|
| species | 100 %    | 7,806          |
| genus   | 100 %    | 1,446          |
| family  | 100 %    | 181            |
| order   | 99.9 %   | 61             |
| class   | 99.9 %   | 6              |

Training images: **1,408,033** across 7,806 species.  
Val split: stratified 10% per species (species with <5 images stay in train).

### Data sources

| File | Role |
|------|------|
| `PlantCLEF2024_single_plant_training_metadata.csv` | Main metadata (image paths, species/genus/family) |
| `species_lookup_with_gbif_cleaned_names.csv` | Taxonomy enrichment (order, class via GBIF) |
| `/workspace/plantclef/raw/train/images_max_side_800/` | Labeled images (max 800px side) |

---

## Augmentation

**Train** (strong, plant-safe):

| Transform | Parameters |
|-----------|-----------|
| RandomResizedCrop | scale=(0.5, 1.0), bicubic |
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip | p=0.2 |
| RandomRotation | ±20° |
| ColorJitter | brightness/contrast/saturation=0.2, hue=0.03 |
| RandomGrayscale | p=0.05 |
| Normalize | OpenAI CLIP stats |

**Val**: Resize shortest side to 256 → CenterCrop 224 → Normalize.

---

## Fine-Tuning Modes

Three staged modes, selected at runtime:

| Flag | Unfrozen params | Recommended LR (backbone / head) |
|------|----------------|----------------------------------|
| `--freeze-backbone` | head only (~10.8 M) | — / 1e-4 |
| `--unfreeze-last-n-blocks N` | last N blocks + ln_post/proj | 1e-6 / 1e-4 |
| `--full-finetune` | entire model | 5e-7 / 5e-5 |

BioCLIP 2.5 ViT-H/14 has 32 transformer blocks total.

---

## Quick Start

### Smoke test (end-to-end sanity check, ~30 s)

```bash
bash scripts/smoke_test.sh
```

Runs 1 epoch on 200 samples, verifies forward/backward passes, checkpoint saving, and metrics logging.

### Staged fine-tuning (recommended)

```bash
# Stage 1: train head only (10 epochs, frozen backbone)
bash scripts/train_head_only.sh

# Stage 2: unfreeze last 4 blocks (5 epochs, resume from stage 1)
bash scripts/train_last_blocks.sh [path/to/stage1/best.pt]

# Stage 3: full backbone fine-tune (3 epochs, very low LR)
bash scripts/train_full_finetune.sh [path/to/stage2/best.pt]
```

### Manual training

```bash
cd 010_bioclip25_end_to_end_finetune_multitask

python train.py \
  --epochs 10 \
  --batch-size 128 \
  --grad-accum-steps 2 \
  --precision bf16 \
  --use-taxonomy-heads \
  --freeze-backbone \
  --head-lr 1e-4 \
  --output-dir ./outputs/run1

# Resume from checkpoint
python train.py --resume ./outputs/run1/checkpoints/last.pt [...]

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 train.py [...]
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--precision` | `fp16` | `fp16` / `bf16` / `fp32` |
| `--grad-accum-steps` | 1 | Effective batch = batch_size × grad_accum_steps |
| `--warmup-epochs` | 1 | Linear LR warmup then cosine decay |
| `--use-taxonomy-heads` | on | Add genus/family/order/class aux losses |
| `--no-taxonomy-heads` | — | Species classification only |
| `--smoke-test` | — | Cap to 200 samples, 1 epoch |
| `--wandb` | — | Enable Weights & Biases logging |

---

## Tile Inference

Run tiled inference on test quadrat images to produce a PlantCLEF submission:

```bash
python infer_tiles.py \
  --checkpoint outputs/train/checkpoints/best.pt \
  --image-dir  /workspace/plantclef/kaggle_uploads/test/images \
  --tile-mode  multiscale \
  --agg-mode   max \
  --top-k      5 \
  --output-dir outputs/tile_inference
```

### Tiling modes

| Mode | # Tiles (800×600) | Description |
|------|----------|-------------|
| `whole` | 1 | Full image as single tile |
| `grid_2x2` | 4 | 2×2 equal grid |
| `grid_3x3` | 9 | 3×3 equal grid |
| `grid_4x4` | 16 | 4×4 equal grid |
| `five_crop` | 5 | Centre + 4 corners |
| `sliding` | varies | Sliding window (--tile-size, --overlap) |
| `multiscale` | 21 | whole + grid_2×2 + grid_4×4 |

### Aggregation modes

| Mode | Strategy |
|------|----------|
| `max` | Element-wise max over tile logits |
| `mean` | Simple mean of tile logits |
| `softmax_mean` | Mean in probability space (log-softmax returned) |

### Debug preview

```bash
python infer_tiles.py --checkpoint ... --limit 5 --save-tile-preview
```

Writes `tile_preview_{image_stem}.png` showing numbered bounding boxes for each tile.

### Outputs

```
outputs/tile_inference/
├── submission.csv            # PlantCLEF format (quadrat_id, species_ids)
├── predictions_scored.csv    # All predictions with scores and ranks
├── run_config.json           # Full config used for this run
├── summary.json              # Timing, error counts, throughput
└── tile_preview_*.png        # (if --save-tile-preview)
```

---

## Standalone Evaluation

Evaluate a checkpoint on the validation split and save per-image predictions:

```bash
python validate.py \
  --checkpoint outputs/train/checkpoints/best.pt \
  --output-dir outputs/eval
```

Outputs `eval_metrics.json` and `predictions.csv` with `true_species`, `pred_species`, `top5_species`, and `confidence`.

---

## Outputs

Each training run produces:

```
outputs/{run_name}/
├── checkpoints/
│   ├── best.pt           # Checkpoint with highest val top-5 accuracy
│   ├── last.pt           # Latest checkpoint (for resuming)
│   └── epoch_NNN.pt      # Per-epoch checkpoints (--save-every)
├── encoders/
│   ├── idx_to_species.json    # Integer index → species_id
│   ├── idx_to_genus.json      # Integer index → genus name
│   ├── idx_to_family.json
│   ├── idx_to_order.json
│   ├── idx_to_class.json
│   └── {level}_to_idx.json   # Reverse lookups
├── metrics.csv           # Per-epoch training metrics (appended each epoch)
├── metrics.json          # Full training history
├── train_config.json     # All hyperparameters used
└── run.log               # Full training log
```

### Checkpoint format

```python
{
    "epoch":                int,
    "model_state_dict":     dict,          # full model weights
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict":    dict,
    "metrics":              dict,          # val metrics at this epoch
    "config":               dict,          # training config
    "idx_to_species":       list[str],     # species class list
    "encoders":             dict,          # all label encoders
}
```

---

## Implementation Notes

- **Backbone frozen layers stay in eval mode** during training (overridden `train()`) to avoid BatchNorm/Dropout side effects in unfrozen parts.
- **GradScaler** is only used with `fp16`; bf16 training is numerically stable without it on Ampere/Ada hardware.
- **Gradient accumulation** interacts correctly with the step-based cosine+warmup scheduler: the scheduler steps once per *optimizer* step, not per mini-batch.
- **Missing taxonomy labels** (species not in the GBIF lookup, or NaN values) are encoded as `-1` and masked out of the auxiliary cross-entropy loss — no `ignore_index` tricks needed, just boolean masking.
- **Val split** is stratified by species with a floor of 5 images; species with fewer images are always kept in train to avoid zero-shot species in val.
