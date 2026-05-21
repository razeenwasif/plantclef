# 009 — remote training context (BioCLIP-2.5 PlantNet-style full FT)

Handoff brief for whoever (human or Claude) is watching the pod.
**Last revised 2026-04-26 02:30 UTC** — training kicked off ~02:20 UTC.

## TL;DR — what is running and why

- **Task:** Full fine-tune of BioCLIP-2.5 ViT-H/14 on 1.38 M PC24 single-plant
  images, 7806-way CE + label-smooth=0.1 + logit adjustment. 8 main epochs +
  1 head-only warmup epoch.
- **Why:** The 008 PhaseA × 004-frozen-BioCLIP RRF ensemble cracked
  **0.34642** (new team best, +0.016 over Arjun's 0.33). The fusion gain came
  from two models making *different* errors. The frozen prototype-matcher leg
  is the bottleneck — replacing it with a properly fine-tuned BioCLIP-2.5
  should lift the ensemble ceiling toward 0.36+.
- **Host:** `dinov3_2_5090`. **2× RTX 5090** via DDP (`torchrun
  --nproc_per_node=2`).
- **Process:** `nohup setsid torchrun ... &` — **NOT in tmux.** Find via
  `ps -p 57622` (PID at launch) or `pgrep -af train_phase_a.py`.
- **Config:** `--img-size 224 --batch-size 24 --accum 1 --bf16
  --grad-checkpoint --epochs 8 --warmup-epochs 1 --num-workers 8`. Effective
  batch = 24 × 1 × 2 = 48. Default LR groups in main stage:
  `max_lr=[5e-5_backbone, 1e-4_head]` — head LR divided by 10 to avoid
  overpowering the 632M backbone.
- **ETA:** ~30 min warmup + 8 × ~2.5 h main ≈ **20 h total**.
- **Started:** 2026-04-26 02:20 UTC.
- **Output dir:**
  `/workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1/`

## What healthy training looks like

### Warmup (epoch 0, head-only, backbone frozen)
- Loss starts ~9.0 (≈ ln(7806)), drops to ~3-4 by step 5000, then continues
  toward ~2 by end of epoch.
- Head-only is fast: ~64 ms/step → ~30 min for 28,499 steps.
- VRAM ~3.9 GB per GPU, both at 95-99% util.
- No val reported during warmup (only `phase_a_warmup_ep0.pth` saved).

### Main training (epochs 1-8, backbone + head unfrozen)
- Loss at start of ep1 jumps as backbone unfreezes, then resumes descent.
- Each epoch: ~28,499 steps × ~0.3-0.4 s/step ≈ **2.5 h**.
- VRAM with grad checkpointing: ~18-22 GB per GPU.
- **Expected val_top1 trajectory** (1% holdout, 13.8k single-plant images):
  - ep1: ≥ 0.30 (sanity floor — head warmup gave it a head start)
  - ep3: ≥ 0.45
  - ep5: ≥ 0.52
  - ep8: ≥ 0.55-0.60 (target; PlantNet DINOv2 hit 0.76 with 75 ep)
- A trajectory below these by 2+ consecutive epochs = LR too hot, see red
  flags below.

### Red flags
- **val_top1 decreases across 2 consecutive main epochs** → LR too hot. Stop
  and consider relaunch with `--lr-head 5e-5 --lr-backbone 2e-5` (half).
- **Loss NaN** — unlikely with bf16 + grad clip, but possible. Kill and
  relaunch with halved LRs.
- **VRAM > 28 GB / OOM at start of ep1** — grad checkpointing should keep
  this in bounds. If it OOMs, drop `--batch-size` to 20 and bump `--accum 2`
  to keep effective batch ≈ 48.
- **Epoch duration > 4 h** — dataloader stall. Check `nvidia-smi` util; if
  GPUs idle below 40%, restart with `--num-workers 6` (open_clip's tokenizer
  can fight Python-side decode for CPU).
- **HF rate-limit / 429 from hub during checkpoint load** — only matters at
  start; once the pretrained safetensors are cached the run is offline.

## File locations

| What | Path |
|------|------|
| Code | `/workspace/working/PlantCLEF2026/src_experiments/009_bioclip25_plantnet_finetune/` |
| Train script | `train_phase_a.py` |
| Model wrapper | `bioclip_model.py` (`BioCLIP25SinglePlantClassifier`) |
| Inference adapter (post-train) | `dump_test_probs.py` (already on pod) |
| Training log | `/workspace/working/logs/bioclip25_p1.log` |
| Ckpt dir | `/workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1/` |
| Train CSV | `/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv` (1,381,785 rows) |
| Train images | `/workspace/plantclef/raw/train/images_max_side_800/` |
| Species list | `/workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv` |
| Preprocessed test quadrats | `/workspace/plantclef/processed/test_images_jpeg85_max800/` (2105 jpegs, ready) |
| 008 PhaseA inference npz (for fusion) | `/workspace/working/PlantCLEF2026/src_experiments/008_dinov3_plantnet_finetune/outputs/test_probs_phase_a_multiscale.npz` |
| Pre-trained backbone cache | `~/.cache/huggingface/hub/models--imageomics--bioclip-2.5-vith14/` |

## Monitoring commands

```bash
# Process alive?
ps -p 57622 -o pid,etime,pcpu,pmem,cmd
# (or if PID drifted)
pgrep -af train_phase_a.py

# Tail live log
tail -f /workspace/working/logs/bioclip25_p1.log

# Last 20 lines of progress
tail -20 /workspace/working/logs/bioclip25_p1.log

# GPU watch
watch -n 5 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Saved checkpoints
ls -lh /workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1/

# Val summaries (after first main epoch)
grep "val_top1\|val_loss\|epoch.*done" /workspace/working/logs/bioclip25_p1.log

# Estimate epoch progress
grep -c "step.*loss=" /workspace/working/logs/bioclip25_p1.log
```

## What to do when training finishes

**Do NOT auto-submit anything to Kaggle — wait for human.** Report:

```bash
python3 -c "
import torch, glob
for p in sorted(glob.glob('/workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1/phase_a_ep*.pth')):
    d = torch.load(p, map_location='cpu', weights_only=False)
    print(p.split('/')[-1], 'ep', d.get('epoch'), 'val_top1', round(d.get('val_top1', -1), 4), 'val_loss', round(d.get('val_loss', -1), 3))
"
```

Then ping the user with the best epoch (usually `phase_a_best.pth`).

### Likely next steps the user will decide

1. **Tiled multi-scale inference** on the 2105 test quadrats:
   ```bash
   cd /workspace/working/PlantCLEF2026/src_experiments/009_bioclip25_plantnet_finetune
   python3 dump_test_probs.py \
     --checkpoint /workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1/phase_a_best.pth \
     --images-root /workspace/plantclef/processed/test_images_jpeg85_max800 \
     --output      outputs/test_probs_009_multiscale.npz \
     --tile-sizes 224 336 --tile-overlap 112 \
     --hflip-tta --whole-image --bf16 --batch-size 32
   ```
   ETA ~2-3 h on one 5090. Tile sizes must be multiples of 14 (ViT-H/14).

2. **Fuse with 008 PhaseA** via existing script:
   ```bash
   cd /workspace/working/PlantCLEF2026/src_experiments/008_dinov3_plantnet_finetune
   python3 fuse_phase_a_bioclip.py \
     --phase-a outputs/test_probs_phase_a_multiscale.npz \
     --bioclip ../009_bioclip25_plantnet_finetune/outputs/test_probs_009_multiscale.npz \
     --bioclip-key probs_max \
     --output-dir ../009_bioclip25_plantnet_finetune/outputs \
     --alphas 0.3 0.4 0.5 0.6 0.7 0.8 \
     --top-k 3 4 \
     --tag fuse_pa_009
   ```
   The 008×004-frozen-BioCLIP fusion peaked at α=0.70 top-3. Two strong
   peers may shift this — sweep widely first.

3. **Submit best to Kaggle** via `scripts/submit_predictions_kaggle.sh -i
   src_experiments/009_bioclip25_plantnet_finetune/outputs -p
   'submission_*.csv'`.

## Non-goals

- **Don't kill** the run unless red flags fire. Each lost main epoch costs
  ~2.5 h.
- **Don't auto-submit** to Kaggle — every submission is leaderboard-visible
  and our daily quota matters.
- **Don't modify** `train_phase_a.py`, `bioclip_model.py`, or
  `single_plant_dataset.py` mid-run — DDP rank 1 will silently drift.
- **Don't delete** the 008 PhaseA ckpts or npz — those are inputs to the
  fusion that creates the actual submission.
- **Don't touch** `~/.cache/huggingface/hub/models--imageomics--bioclip-*` —
  re-downloading the 632M safetensors at the start of inference would hurt.
- **Don't add tile sizes that aren't multiples of 14** to the inference
  command — it's a hard error in `dump_test_probs.py`.

## Kaggle context

- **Team best:** **0.34642** (PhaseA × frozen-BioCLIP RRF, α=0.70 top-3,
  RRF-k=60). 008 PhaseA alone 0.305; Arjun's BioCLIP-2.5 alone 0.33.
- **009 target:** beat 0.34642 with the new strong-strong fusion. Stretch
  ≥ 0.38.
- **Deadline:** ~12 days from 2026-04-26 → final window closes ~2026-05-08.

## If the pod reboots mid-run

- `/workspace` is persistent across pod recreations on this template.
- `train_phase_a.py` supports `--resume
  /workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1/phase_a_last.pth`
  — OneCycleLR restarts from step 0 (loss spike expected), still cheaper than
  restart if > 2 main epochs of progress.
- Full relaunch:
  ```bash
  cd /workspace/working/PlantCLEF2026/src_experiments/009_bioclip25_plantnet_finetune
  mkdir -p /workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1 /workspace/working/logs
  nohup setsid torchrun --standalone --nproc_per_node=2 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:29501 \
    train_phase_a.py \
    --train-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
    --images-root /workspace/plantclef/raw/train/images_max_side_800 \
    --species-csv /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
    --img-size 224 --batch-size 24 --accum 1 --bf16 --grad-checkpoint \
    --epochs 8 --warmup-epochs 1 --num-workers 8 \
    --output-dir /workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1 \
    --resume /workspace/working/PlantCLEF2026/models/bioclip25_plantnet_v1/phase_a_last.pth \
    >> /workspace/working/logs/bioclip25_p1.log 2>&1 < /dev/null &
  ```
  Drop `--resume ...` for a clean restart.
