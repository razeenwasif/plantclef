# BioCLIP 2.5 - Partial Fine-Tuning + Extended Grid Tiling

Experiment 006 extends experiment 003 in two ways:

1. **Partial backbone fine-tuning** via `--unfreeze-blocks N`
2. **Extended grid tiling** - `grid_6x6`, `grid_7x7`, `grid_8x8` added

---

## Changes from 003

| File | Change |
|---|---|
| `model.py` | `unfreeze_last_n_blocks(n)` method; `train()` override respects unfrozen blocks |
| `train.py` | `--unfreeze-blocks`, `--backbone-lr-scale`; two-group AdamW optimizer |
| `tiling.py` | `grid_6x6/7x7/8x8` added to `TILING_MODES`; dispatcher uses regex for any `grid_NxN` |
| `infer.py` | Updated `--tile-mode` help text (choices auto-extend from `TILING_MODES`) |

---

## Training

### Linear probe (fully frozen - same as 003)

```bash
cd /root/workspace/PlantCLEF2026/src_experiments/006_bioclip25_finetune

torchrun --nproc_per_node=2 train.py \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3 \
  --resume /root/workspace/PlantCLEF2026/src_experiments/003_bioclip25_linear_probe/outputs/train/checkpoints/best.pt \
  --output-dir ./outputs/train_e20
```

### Partial fine-tuning - last 2 blocks

```bash
torchrun --nproc_per_node=2 train.py \
  --epochs 20 \
  --batch-size 512 \
  --lr 1e-3 \
  --unfreeze-blocks 2 \
  --backbone-lr-scale 0.05 \
  --resume /root/workspace/PlantCLEF2026/src_experiments/003_bioclip25_linear_probe/outputs/train/checkpoints/best.pt \
  --output-dir ./outputs/train_finetune_b2
```

### Partial fine-tuning - last 4 blocks

```bash
torchrun --nproc_per_node=2 train.py \
  --epochs 10 \
  --batch-size 64 \
  --lr 5e-4 \
  --unfreeze-blocks 4 \
  --backbone-lr-scale 0.05 \
  --resume /root/workspace/PlantCLEF2026/src_experiments/003_bioclip25_linear_probe/outputs/train/checkpoints/best.pt \
  --output-dir ./outputs/train_finetune_b4
```

**Notes:**
- ViT-H/14 has 32 transformer blocks total.
- `--backbone-lr-scale 0.05–0.1` is a safe starting range (backbone LR = head LR × scale).
- Reduce `--batch-size` when unfreezing: backbone activations are now retained for backprop.
- Resume from 003's `best.pt` to warm-start the head.

---

## Inference - extended grid tiling

```bash
cd /root/workspace/PlantCLEF2026/src_experiments/006_bioclip25_finetune

CKPT="./outputs/train_finetune_b2/checkpoints/best.pt"  # or 003 best.pt
TEST_DIR="/workspace/plantclef/raw/test"

# grid7x7 + bilinear + k=3  (follow-on from best 003 result: grid5x5 = 0.242)
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --checkpoint "$CKPT" --test-dir "$TEST_DIR" \
  --tile-mode grid_7x7 --interp bilinear --agg-mode topk_mean --top-n 3 \
  --output-dir ./outputs/infer/bilinear_grid7x7_k3 &

# grid8x8 + bilinear + k=3
CUDA_VISIBLE_DEVICES=1 python infer.py \
  --checkpoint "$CKPT" --test-dir "$TEST_DIR" \
  --tile-mode grid_8x8 --interp bilinear --agg-mode topk_mean --top-n 3 \
  --output-dir ./outputs/infer/bilinear_grid8x8_k3 &

wait
```

---

## Design notes

**`unfreeze_last_n_blocks(n)`** unfreezes:
- The last `n` of the 32 ViT-H transformer resblocks (`visual.transformer.resblocks[-n:]`)
- `visual.ln_post` (final layer norm)
- `visual.proj` (projection to embedding space)

Everything else (patch embedding, positional embedding, earlier blocks) stays frozen.

**`train()` override** - after `backbone.eval()`, each module in the backbone that
owns directly-attached trainable parameters is re-set to `train(True)`, so dropout
and layer-norm in the unfrozen blocks use training statistics correctly.

**Two-group optimizer** - AdamW with separate LR groups:
```
group 0 (backbone): lr = args.lr × args.backbone_lr_scale
group 1 (head):     lr = args.lr
```
CosineAnnealingLR scales both groups proportionally from their initial LRs.

**Grid tiling** - the dispatcher now uses a regex `^grid_(\d+)x\1$` so any
`grid_NxN` string works without adding explicit cases. The named entries in
`TILING_MODES` exist only to populate the `--tile-mode` argparse choices.
