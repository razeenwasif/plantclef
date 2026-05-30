# Experiment 011 — BioCLIP 2.5 SSL Domain Adaptation + Adaptive Inference

Extends experiment 010 with two orthogonal improvements:

1. **SSL pre-training** — SimSiam self-supervised learning on unlabeled pseudo-quadrat images to adapt the BioCLIP 2.5 backbone to the PlantCLEF test distribution before supervised fine-tuning.
2. **Adaptive inference** — variable-length species selection at test time (probability threshold, relative threshold, gap detection) instead of fixed top-k.

Experiment 010 is unchanged. All modifications live entirely within this folder.

---

## Contents

```
011_bioclip25_ssl_adaptive/
├── dataset.py        # (unchanged from 010) Metadata + taxonomy merge, MultiTaskDataset
├── model.py          # + SimSiamProjector, SimSiamPredictor, BioCLIP25SSL, load_ssl_backbone()
├── transforms.py     # + ssl_augmentation(), SSLTwoViewTransform
├── utils.py          # (unchanged from 010) AMP, scheduler, checkpointing, metrics
├── train.py          # + --ssl-backbone-checkpoint warm-start flag
├── train_ssl.py      # NEW — SimSiam SSL pre-training on unlabeled images
├── validate.py       # (unchanged from 010) Standalone evaluation
├── infer_tiles.py    # + adaptive selection modes (prob_threshold, relative, gap)
└── scripts/
    ├── smoke_test.sh                      # Tests SSL + supervised end-to-end
    ├── train_ssl.sh                       # NEW — run SSL pre-training
    ├── train_from_ssl_stage1_stage2.sh    # NEW — two-stage supervised from SSL backbone
    ├── infer_best_adaptive.sh             # NEW — adaptive inference sweep
    ├── train_head_only.sh                 # (from 010) Plain head-only supervised
    ├── train_last_blocks.sh               # (from 010) Last-4-blocks supervised
    └── infer_tiles.sh                     # (from 010) Fixed top-k inference sweep
```

---

## Architecture

### Supervised model (unchanged from 010)

```
Input image (224×224)
       │
BioCLIP 2.5 ViT-H/14 backbone
       │  (1280-dim embedding)
       ▼
LayerNorm → Linear(1280→1024) → GELU → Dropout(0.2)   ← shared MLP
       │
   ┌───┴───┬───────┬────────┬──────────┐
species  genus  family   order    class    ← linear heads
```

### SSL model (new)

```
Unlabeled image
       │
  ┌────┴────┐
  aug₁    aug₂   ← two independent SSL augmentations per image
  │         │
BioCLIP 2.5 ViT-H/14 backbone   ← shared weights, last N blocks unfrozen
  │         │
projector  projector             ← 3-layer MLP: 1280 → 2048 → 2048 → 256 (BN, no-affine last)
  │         │
predictor  predictor             ← 2-layer MLP: 256 → 512 → 256 (BN)
  z₁  p₁   z₂  p₂
       │
  SimSiam loss = −cos(p₁, sg(z₂)) − cos(p₂, sg(z₁))   ← stop-gradient on z
```

### SSL → Supervised warm-start

After SSL pre-training, `backbone.pt` is loaded into `BioCLIP25MultiTask` with `strict=False`.
Only the visual encoder weights transfer; the projector/predictor are discarded.
Missing or unexpected keys are logged as warnings — the supervised head is always randomly initialised.

---

## Training Pipeline

### Option A — SSL warm-start (recommended)

```
Step 1  SSL pre-training         train_ssl.py          → backbone.pt
Step 2  Head-only supervised     train.py (stage 1)    → ssl_head_only/best.pt
Step 3  Last-4-blocks supervised train.py (stage 2)    → ssl_last_blocks/best.pt
Step 4  Adaptive inference       infer_tiles.py        → submission.csv (per mode)
```

One script runs steps 2 + 3:
```bash
bash scripts/train_ssl.sh
bash scripts/train_from_ssl_stage1_stage2.sh
bash scripts/infer_best_adaptive.sh
```

### Option B — Plain supervised (010-style, still works)

```bash
bash scripts/train_head_only.sh
bash scripts/train_last_blocks.sh
```

---

## Unlabeled Data

| Path | Description |
|------|-------------|
| `/workspace/plantclef/raw/pseudo_quadrats/` | Unlabeled field quadrat images (LUCAS surveys) |

The SSL dataset loader searches all subdirectories recursively for `.jpg`, `.jpeg`, `.png` files.
Additional directories can be added via `--image-dirs dir1 dir2 ...`.

---

## Augmentations

### Supervised training (same as 010)

| Transform | Parameters |
|-----------|-----------|
| RandomResizedCrop | scale=(0.5, 1.0), bicubic |
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip | p=0.2 |
| RandomRotation | ±20° |
| ColorJitter | brightness/contrast/saturation=0.2, hue=0.03 |
| RandomGrayscale | p=0.05 |
| Normalize | OpenAI CLIP stats |

### SSL (stronger, two independent views per image)

| Transform | Parameters |
|-----------|-----------|
| RandomResizedCrop | scale=(0.2, 1.0), bicubic |
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip | p=0.5 |
| RandomRotation | ±30° |
| ColorJitter | brightness/contrast/saturation=0.4, hue=0.08 |
| RandomGrayscale | p=0.2 |
| GaussianBlur | kernel≈23px, p=0.3 |
| RandomAutocontrast | p=0.2 (if torchvision ≥ 0.12) |
| Normalize | OpenAI CLIP stats |

Scale starts at 0.2 (vs 0.5 for supervised) to force the model to learn from partial views — critical for quadrat images where any crop may contain a valid plant.

---

## SSL Pre-Training

### Default config

| Hyperparameter | Value |
|----------------|-------|
| Backbone | BioCLIP 2.5 ViT-H/14 |
| Unfrozen blocks | last 4 (of 32) + ln_post/proj |
| Projector | 1280→2048→2048→256 (BN) |
| Predictor | 256→512→256 (BN) |
| Loss | SimSiam: −½(cos(p₁,sg(z₂)) + cos(p₂,sg(z₁))) |
| Epochs | 2 |
| Batch size | 128 |
| Backbone LR | 1e-6 |
| Head LR | 1e-4 |
| Optimizer | AdamW, weight_decay=1e-4 |
| Schedule | cosine decay (no warmup) |
| Precision | bf16 |

### Running

```bash
bash scripts/train_ssl.sh
```

Or directly:

```bash
python train_ssl.py \
  --image-dirs /workspace/plantclef/raw/pseudo_quadrats \
  --epochs 2 \
  --batch-size 128 \
  --num-workers 16 \
  --precision bf16 \
  --unfreeze-last-n-blocks 4 \
  --backbone-lr 1e-6 \
  --head-lr 1e-4 \
  --output-dir outputs/ssl_bioclip25
```

### Outputs

```
outputs/ssl_bioclip25/
├── checkpoints/
│   ├── last.pt        # Full SSL checkpoint (model + optimizer + scheduler)
│   ├── best.pt        # Full checkpoint at epoch with lowest SSL loss
│   └── backbone.pt    # Backbone state dict only → used for supervised warm-start
├── ssl_config.json
├── ssl_metrics.csv
└── ssl_metrics.json
```

`backbone.pt` format:
```python
{
    "backbone_state_dict": dict,   # clip_model.state_dict() (full CLIP model)
    "epoch":               int,
    "ssl_loss":            float,
}
```

### Monitoring collapse

The training loop logs `z_std` (mean std of projections across the batch) each step and epoch.
A healthy run shows `z_std > 0.1`. If `z_std < 0.001`, representations have collapsed — reduce `backbone_lr` or increase batch size.

---

## Supervised Fine-Tuning (SSL warm-start)

### Stage 1 — head only

Load the SSL backbone, freeze it, train only the shared MLP + classification heads:

```bash
python train.py \
  --ssl-backbone-checkpoint outputs/ssl_bioclip25/checkpoints/backbone.pt \
  --freeze-backbone \
  --epochs 10 \
  --batch-size 512 \
  --grad-accum-steps 2 \
  --precision bf16 \
  --use-taxonomy-heads \
  --head-lr 1e-4 \
  --output-dir outputs/ssl_head_only
```

`--ssl-backbone-checkpoint` loads backbone weights with `strict=False` before any other training step. Missing and unexpected keys are logged but do not abort training.

### Stage 2 — last 4 blocks

Resume weights-only from stage 1, unfreeze the last 4 transformer blocks:

```bash
python train.py \
  --resume outputs/ssl_head_only/checkpoints/best.pt \
  --resume-weights-only \
  --unfreeze-last-n-blocks 4 \
  --epochs 5 \
  --batch-size 64 \
  --grad-accum-steps 4 \
  --precision bf16 \
  --head-lr 1e-4 \
  --backbone-lr 1e-6 \
  --output-dir outputs/ssl_last_blocks
```

Both stages in one script:

```bash
bash scripts/train_from_ssl_stage1_stage2.sh [ssl_backbone_ckpt] [n_gpus]
```

---

## Adaptive Inference

`infer_tiles.py` now supports four selection modes that determine how many species to predict per quadrat image. All modes share `--min-k` and `--max-k` bounds.

### Selection modes

| Mode | Key | Strategy |
|------|-----|----------|
| `fixed_topk` | `top{k}` | Classic top-k — always returns exactly k species |
| `prob_threshold` | `probT{t}` | Keep all species with softmax probability ≥ t |
| `relative_threshold` | `relT{t}` | Keep species within t fraction of the top-1 probability |
| `gap` | `gap{r}` | Cut at the first relative drop ≥ r in the sorted probability list |

All adaptive modes clamp k to `[min_k, max_k]`.

#### `relative_threshold` example

With `--relative-thresholds 0.20` and top probs `[0.40, 0.35, 0.18, 0.05]`:
- cutoff = 0.40 × (1 − 0.20) = 0.32
- Select species with prob ≥ 0.32 → top 2

#### `gap` example

With `--gap-ratios 0.50` and top probs `[0.40, 0.35, 0.10, 0.08]`:
- Drop from rank 2 to rank 3: (0.35 − 0.10) / 0.35 = 0.71 ≥ 0.50 → cut here → top 2

### Running adaptive inference

```bash
bash scripts/infer_best_adaptive.sh
```

Or directly:

```bash
python infer_tiles.py \
  --checkpoint outputs/ssl_last_blocks/checkpoints/best.pt \
  --image-dir /workspace/plantclef/raw/test \
  --tile-mode grid_4x4 \
  --overlap 0.0 \
  --agg-modes softmax_mean \
  --top-ks 2 3 4 \
  --selection-modes fixed_topk relative_threshold prob_threshold gap \
  --min-k 2 \
  --max-k 5 \
  --relative-thresholds 0.15 0.20 0.25 0.30 \
  --prob-thresholds 0.02 0.03 0.05 \
  --gap-ratios 0.40 0.50 0.60 \
  --save-logits \
  --precision bf16 \
  --batch-size 512 \
  --output-dir outputs/ssl_adaptive_infer
```

The model forward pass runs **once per image** regardless of how many `(agg, selection)` combinations are evaluated.

### Inference CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--selection-modes` | `fixed_topk` | Which modes to run (space-separated) |
| `--top-ks` | `[5]` | k values for `fixed_topk` mode |
| `--min-k` | 1 | Minimum species per prediction |
| `--max-k` | 10 | Maximum species per prediction |
| `--prob-thresholds` | `[0.03]` | Thresholds for `prob_threshold` |
| `--relative-thresholds` | `[0.20]` | Thresholds for `relative_threshold` |
| `--gap-ratios` | `[0.50]` | Ratios for `gap` mode |

### Outputs

Each `(agg_mode, selection_key)` pair produces its own subdirectory:

```
outputs/ssl_adaptive_infer/
├── softmax_mean_top2/
│   ├── submission.csv
│   ├── predictions_scored.csv
│   ├── run_config.json
│   └── summary.json
├── softmax_mean_top3/
├── softmax_mean_relT0.15/
├── softmax_mean_relT0.20/
├── softmax_mean_probT0.02/
├── softmax_mean_gap0.4/
├── ...
└── logits/
    └── softmax_mean_logits.pt   (only with --save-logits)
```

---

## Tiling modes (unchanged from 010)

| Mode | Tiles | Best for |
|------|-------|----------|
| `whole` | 1 | Baseline / speed |
| `grid_2x2` | 4 | Coarse spatial coverage |
| `grid_3x3` | 9 | Medium coverage |
| `grid_4x4` | 16 | Best known config |
| `five_crop` | 5 | Fixed corners |
| `sliding` | varies | High-overlap sweep |
| `multiscale` | 21 | whole + 2×2 + 4×4 |

**Best known config**: `--tile-mode grid_4x4 --overlap 0.0 --agg-modes softmax_mean`

---

## Smoke Tests

```bash
bash scripts/smoke_test.sh
```

Runs three sequential checks:

1. **SSL pre-training**: 32 images, 1 epoch, batch_size=8, writes `outputs/ssl_smoke/`
2. **Supervised + SSL warm-start**: ~200 samples, 1 epoch, loads `ssl_smoke/checkpoints/backbone.pt`, writes `outputs/smoke_ssl_supervised/`
3. **Plain supervised** (baseline sanity): ~200 samples, 1 epoch, writes `outputs/smoke_test/`

Total runtime: ~3–5 minutes on a single GPU.

---

## Checkpoint Formats

### SSL checkpoint (`last.pt` / `best.pt`)

```python
{
    "epoch":               int,
    "model_state_dict":    dict,   # BioCLIP25SSL: backbone + projector + predictor
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict":    dict,
    "best_loss":           float,
    "history":             list[dict],
    "config":              dict,
}
```

### SSL backbone-only checkpoint (`backbone.pt`)

```python
{
    "backbone_state_dict": dict,   # clip_model.state_dict() only
    "epoch":               int,
    "ssl_loss":            float,  # SSL loss at this epoch
}
```

### Supervised checkpoint (same as 010)

```python
{
    "epoch":                int,
    "model_state_dict":     dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict":    dict,
    "metrics":              dict,
    "config":               dict,
    "idx_to_species":       list[str],
    "encoders":             dict,
}
```

---

## Implementation Notes

- **SSL uses `strict=False`** when loading the backbone into `BioCLIP25MultiTask`. Keys present in one but not the other (e.g. text encoder weights) are logged as warnings and skipped — the supervised model initialises its classification heads fresh regardless.
- **SimSiam stop-gradient** is applied only to the projection `z`, not the prediction `p`. The predictor is what prevents collapse without a momentum encoder (unlike MoCo/BYOL).
- **BatchNorm in the projector** uses `affine=False` on the final layer (standard SimSiam design). The predictor uses normal BN.
- **`z_std` collapse monitoring**: if projections collapse to a single point, `z_std → 0`. The training script warns when `z_std < 0.001`.
- **Backbone stays in eval mode** for frozen layers during both SSL and supervised training (overridden `train()` method) to avoid BatchNorm/Dropout side-effects in partially unfrozen models.
- **Adaptive selection clamps** k to `[min_k, min(max_k, n_species)]` after applying the threshold or gap rule. `fixed_topk` also respects `min_k`/`max_k` for consistency.
- **One forward pass per image**: tile inference computes `tile_logits` once per image, then all `(agg_mode, selection_spec)` combinations derive their predictions from the cached result — no redundant GPU work.

---

## Differences from Experiment 010

| Component | 010 | 011 |
|-----------|-----|-----|
| `model.py` | Supervised only | + SSL model, projector, predictor, `load_ssl_backbone()` |
| `transforms.py` | Supervised only | + `ssl_augmentation()`, `SSLTwoViewTransform` |
| `train.py` | No warm-start | + `--ssl-backbone-checkpoint` |
| `train_ssl.py` | — | New: SimSiam pre-training on unlabeled data |
| `infer_tiles.py` | Fixed top-k | + 4 adaptive selection modes, `SelectionSpec` |
| `scripts/` | 4 scripts | + 3 new scripts (train_ssl, 2-stage, infer_adaptive) |
| `dataset.py` | Unchanged | Unchanged |
| `utils.py` | Unchanged | Unchanged |
| `validate.py` | Unchanged | Unchanged |
