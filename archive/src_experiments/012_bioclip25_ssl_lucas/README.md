# 012 — BioCLIP-2.5 SSL continual pretraining on geometry-corrected LUCAS

Adapt the team-best **010 last_blocks** BioCLIP-2.5 backbone to the
quadrat-shaped imagery distribution before running supervised fine-tuning.
**Three-step pipeline.** No new architecture.

## Why

EDA (`eda/plantclef2026_eda.ipynb`, commit f87e0a7) measured the test ↔
LUCAS gap precisely:

| Set | Max-side | Aspect | Mpx |
|---|---|---|---|
| Test (N=2105) | 800 px | 1.00 (std 0.10) | 0.60 |
| LUCAS (N=139) | 1740 × 1299 | 1.34 | 2.33 |

LUCAS is **NOT** a drop-in proxy. 008 Phase B v3 self-distilled labels on
raw LUCAS and crashed to 0.227 — the geometry mismatch made the teacher's
pseudo-labels noise. We attack that here:

1. **Geometry-correct LUCAS** (center-crop to aspect 1.0, resize to 800 px
   max-side) so it visually matches the test distribution.
2. **DINO-style SSL** on the corrected LUCAS, partial-unfreeze the same
   last 4 transformer blocks that 010 fine-tunes. This shifts the encoder's
   representation toward quadrat-shaped imagery *without losing the
   Tree-of-Life prior in the lower blocks*.
3. **Supervised fine-tune** with the team-best 010 last_blocks recipe
   (handed off to `006_bioclip25_finetune/train.py --resume <ssl-init>`).

If SSL helps, the new fine-tuned model should beat the **0.38333** anchor.

## Pipeline

### Step 1: Geometry-correct LUCAS (one-shot, ~30 min on 16 cores)

```bash
python src_experiments/012_bioclip25_ssl_lucas/prepare_lucas.py \
    --in-dir /workspace/plantclef/raw/pseudo_quadrats \
    --out-dir /workspace/plantclef/processed/lucas_aspect_corrected \
    --max-side 800 --workers 16
```

Center-crops each LUCAS image to a square (aspect 1.34 → 1.0), downscales
to ≤800 px on the long side, saves as JPEG quality 95. Resumable
(skips already-written files). ~212 K LUCAS → ~212 K outputs.

### Step 2: DINO-style SSL (5 epochs, ~6-10 hr on a single 5090)

```bash
python src_experiments/012_bioclip25_ssl_lucas/ssl_train.py \
    --data-dir /workspace/plantclef/processed/lucas_aspect_corrected \
    --out-dir src_experiments/012_bioclip25_ssl_lucas/outputs \
    --epochs 5 --batch-size 16 --grad-accum 4 \
    --unfreeze-blocks 4 --num-locals 6
```

Key choices, all justified:

- **Partial unfreeze (last 4 blocks).** Matches 010's `unfreeze_n=4` sweet
  spot. The lower 28 blocks of BioCLIP-2.5 carry its Tree-of-Life prior
  (the reason 010 last_blocks beats 009 full-FT 0.20777). We adapt the
  upper layers only.
- **Multicrop, all at 224 px.** 2 globals + 6 locals, all at the encoder's
  native 224 px input. Avoids interpolating ViT-H/14's pretrained position
  embedding grid. Local vs global is encoded by `RandomResizedCrop` scale
  ranges (`(0.5, 1.0)` vs `(0.05, 0.5)`), not by image size.
- **EMA teacher** with cosine momentum 0.996 → 1.0; centering on teacher
  logits with EMA buffer.
- **bf16** mixed precision; gradient checkpointing on the unfrozen blocks
  to keep VRAM headroom.

Output: `outputs/ssl_bioclip25_backbone_ep{N}.pt` (BioCLIP visual encoder
state_dict only).

### Step 3: Supervised fine-tune via the 010 last_blocks recipe

```bash
# 3a. Bridge the SSL backbone state_dict into a 006-format starter ckpt:
python src_experiments/012_bioclip25_ssl_lucas/materialize_ssl_for_006.py \
    --ssl-ckpt src_experiments/012_bioclip25_ssl_lucas/outputs/ssl_bioclip25_backbone_ep5.pt \
    --train-meta-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
    --out src_experiments/012_bioclip25_ssl_lucas/outputs/ssl_init_for_006.pt

# 3b. Run the team-best 010 last_blocks recipe on top of the SSL-warm-started backbone:
python src_experiments/006_bioclip25_finetune/train.py \
    --resume src_experiments/012_bioclip25_ssl_lucas/outputs/ssl_init_for_006.pt \
    --train-meta-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
    --train-image-root /workspace/plantclef/raw/train/images_max_side_800/ \
    --unfreeze-blocks 4 --epochs 5 --batch-size 128 \
    --lr 1e-3 --backbone-lr-scale 0.1 \
    --output-dir src_experiments/012_bioclip25_ssl_lucas/outputs/finetune
```

The `--resume` ckpt has `epoch=-1`, so 006 starts a fresh 0..4 epoch run
with the SSL-adapted backbone in place of the OpenCLIP defaults.

### Step 4: Inference (uses 010's existing dump script)

```bash
python src_experiments/010_dump_test_probs.py \
    --ckpt src_experiments/012_bioclip25_ssl_lucas/outputs/finetune/checkpoints/best.pt \
    --out src_experiments/012_bioclip25_ssl_lucas/outputs/test_probs_012_grid4x4.npz \
    --tile-mode grid_4x4 --tile-size 448 --overlap 0.0 --img-size 224 \
    --batch-size 64 --precision bf16

# Then convert npz → submission CSV with the exact 010 recipe
# (softmax_mean over the 16 grid_4x4 tiles, top-3).
```

## Submit

```bash
bash scripts/submit_predictions_kaggle.sh \
    -i src_experiments/012_bioclip25_ssl_lucas/outputs \
    -p 'submission_012_*.csv' \
    -o scores.csv
```

## Anchor

Beat **0.38333** (010 last_blocks alone). The hypothesis: SSL on
geometry-corrected LUCAS gives the upper blocks a richer prior over
quadrat-shaped imagery than ImageNet-style augmentations alone, so
supervised fine-tuning starts from a better basin.

If 012 lands at or below 0.383, SSL pretraining isn't the lever and we
should pivot to long-tail loss (ASL + class-frequency logit adjustment).

## Smoke tests before scaling

1. **prepare_lucas** on `--limit 50` first; verify 50 corrected JPEGs at
   ≤800 px and aspect 1.0 land in `out-dir`.
2. **ssl_train** for 1 epoch with `--batch-size 4 --num-locals 2` and a
   100-image subset; verify loss decreases monotonically and a checkpoint
   saves.
3. **materialize_ssl_for_006** against the smoke ckpt; verify the produced
   `model_state_dict` keys include both `backbone.*` and `head.*`, and
   that 006's `--resume` accepts it (run for 1 step, no error).
