# 008 — remote training context (Phase A v2 retrain)

Handoff brief for whoever (human or Claude) is watching the pod.
**Last revised 2026-04-22 20:20 UTC** — Phase A v2 retrain launched after the
original Phase A → Phase B pipeline scored **0.0002 on Kaggle**, below random.

## TL;DR — what is running and why

`tmux attach -t phase_a_v2` — live output. `Ctrl+b d` to detach.

- **Task:** Full fine-tune of DINOv3-L on 1.38 M PC24 single-plant images,
  7806-way CE + logit adjustment. 12 main epochs + 2 warmup head-only epochs.
  This is a **retrain** because the original Phase A only ran 1 useful epoch
  before its LR schedule destabilised the pretrain (ep1=0.7342 → ep4=0.6823).
- **Host:** `dinov3_2_5090` (213.173.103.203:32630). **2× RTX 5090** via DDP.
- **Launcher:** `torchrun --standalone --nproc_per_node=2 train_phase_a.py …`
  inside tmux session `phase_a_v2`.
- **Config (gentler than v1):** `--img-size 224 --batch-size 48 --accum 1
  --bf16 --epochs 12 --warmup-epochs 2 --num-workers 10`. Default LR groups:
  `--lr-backbone 5e-5 --lr-head 1e-3` (in main stage the script uses
  `max_lr=[5e-5, 1e-4]` — head LR is divided by 10 to avoid overpowering the
  backbone). No manual LR override.
- **ETA:** ~48 min/epoch on 2× 5090 → 14 epochs ≈ **11 h total**.
- **Started:** 2026-04-22 20:20 UTC.
- **Output dir:** `/workspace/working/PlantCLEF2026/models/dinov3_plantnet_v2_retrain/`
  (separate from the v1 dir so the old Phase B ckpt stays intact as a fallback).

## Why the retrain — what went wrong on Kaggle

Previous run's Kaggle scores (all ~random baseline or below):

| submission | τ | config | Kaggle F1 |
|------------|---|--------|-----------|
| `submission_v1_tau0.25` | 0.25 | multi-scale+hflip+ExG+BMA+logit-adj | 0.00026 |
| `submission_v1_tau0.2803` | 0.2803 | same (dynamic-τ) | 0.00015 |
| `submission_v1_tau0.30` | 0.30 | same | 0.00016 |
| `submission_v3_NOTILE_tau0.30` | 0.30 | **full-image CenterCrop(448), no enhance** | 0.00255 |

The no-tile config (matches Phase B training exactly) was 13× better than the
tiled+enhanced config, but still ~70× worse than 007 PlantNet DINOv2 (0.175).

### Root-cause diagnosis (already ruled out)

1. ✅ **species_ids ordering** — ckpt's species_ids list == canonical CSV list.
2. ✅ **Model weights load correctly** — `missing=0 unexpected=0` on load.
3. ✅ **Val_f1 on collages is real** — reran forward pass on 200 held-out
   collages: top-5 recall 84%, top-10 recall 98%. Matches the logged 0.46 F1.
4. ✅ **Tiling pipeline is sane** — tested 5 inference strategies on val
   collages; all work, full-image CenterCrop is best on collages.

### Actual cause

**Collage → real-quadrat domain shift.** Collages are close-up single-plant
crops pasted together; real quadrats are top-down 50×50 cm vegetation photos
with each plant ~30-80 px. 007 had the same shift but scored 0.175 because its
PlantNet DINOv2 backbone had **75 epochs of PC24 fine-tuning** before Phase B.
Our v1 Phase A ran only 1 useful epoch, so the backbone never internalised deep
plant-ID priors. LoRA on Phase B then over-fitted to collage visuals.

## What healthy Phase A v2 looks like

### Warmup (epochs 0-1, head-only, backbone frozen)
- Loss starts ~9.0 (ln(7806)), drops to ~5-6 by end of ep0.
- Head-only is fast, ~25-30 min/epoch.
- No val reported during warmup.

### Main training (epochs 1-12, backbone + head unfrozen)
- Loss at start of ep1 jumps a bit as backbone unfreezes, then resumes descent.
- **Expected val_top1 trajectory:**
  - ep2 (first main epoch): ≥ 0.45
  - ep4: ≥ 0.55
  - ep6: ≥ 0.60
  - ep8: ≥ 0.65
  - ep12: ≥ 0.70 (target; PlantNet DINOv2 hit 0.76 with 75 ep)
- Each main epoch takes ~48 min on 2× 5090 DDP (bf16, bs=48 per GPU, 224 px).
- GPU0/GPU1 should both be at 95-99% util, VRAM ~18-20 GB each.

### Red flags
- **val_top1 decreases across 2 consecutive main epochs** → LR still too hot.
  Stop and consider restart with `--lr-head 5e-4` (half).
- **Loss NaN** — unlikely with bf16 + grad clip, but possible. Kill and
  relaunch with `--lr-backbone 2e-5 --lr-head 5e-4`.
- **Epoch duration > 75 min** — dataloader stall; check `nvidia-smi`, restart
  with `--num-workers 6` if GPU util drops below 40%.

## File locations

| What | Path |
|------|------|
| Code | `/workspace/working/PlantCLEF2026/src_experiments/008_dinov3_plantnet_finetune/` |
| Phase A v2 script | `train_phase_a.py` (unchanged from v1) |
| Phase A v2 log | `/workspace/working/PlantCLEF2026/logs/phase_a_v2_20260422_202042.log` |
| Phase A v2 tmux | `phase_a_v2` |
| Phase A v2 ckpt dir | `/workspace/working/PlantCLEF2026/models/dinov3_plantnet_v2_retrain/` |
| v1 Phase A ckpts (keep as fallback) | `/workspace/working/PlantCLEF2026/models/dinov3_plantnet_v1_ddp/phase_a_*.pth` |
| v1 Phase B best (0.0002 Kaggle) | `/workspace/working/PlantCLEF2026/models/dinov3_plantnet_v1_ddp/phase_b_best.pth` |
| Train CSV | `/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv` (1,381,785 rows) |
| Train images | `/workspace/plantclef/raw/train/images_max_side_800/` |
| Collages CSV | `/workspace/plantclef/processed/synthetic_collages.csv` (50k rows) |
| Collages images | `/workspace/plantclef/processed/collages/` |
| Species list | `/workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv` |
| Preprocessed test quadrats | `/workspace/plantclef/processed/test_images_jpeg85_max800/` (2105 jpegs, ready) |
| Species train counts | `/workspace/plantclef/processed/species_train_counts.csv` |

## Monitoring commands

```bash
# Live output
tmux attach -t phase_a_v2

# Tail without attach
LOG=$(ls -t /workspace/working/PlantCLEF2026/logs/phase_a_v2_*.log | head -1); tail -f "$LOG"

# GPU watch
watch -n 5 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Saved checkpoints
ls -lh /workspace/working/PlantCLEF2026/models/dinov3_plantnet_v2_retrain/

# Val summaries
grep "val_top1\|val_loss" /workspace/working/PlantCLEF2026/logs/phase_a_v2_*.log

# Epoch count
grep -c "^.*\[INFO\].*epoch.*done" /workspace/working/PlantCLEF2026/logs/phase_a_v2_*.log
```

## What to do when Phase A v2 finishes

**Do NOT auto-launch anything — wait for human.** Report:

```bash
python3 -c "
import torch, glob
for p in sorted(glob.glob('/workspace/working/PlantCLEF2026/models/dinov3_plantnet_v2_retrain/phase_a_ep*.pth')):
    d = torch.load(p, map_location='cpu', weights_only=False)
    print(p.split('/')[-1], 'ep', d.get('epoch'), 'val_top1', round(d.get('val_top1', -1), 4), 'val_loss', round(d.get('val_loss', -1), 3))
"
```

Then ping the user with the best epoch. Likely next steps the user will decide:

1. **Phase B v2 (LoRA on collages)** — the main reason we retrained. Use whichever
   Phase A ep has the best val_top1 (usually the last or second-last).
   ```bash
   python3 train_phase_b.py \
     --collage-csv    /workspace/plantclef/processed/synthetic_collages.csv \
     --collages-root  /workspace/plantclef/processed/collages \
     --species-csv    /workspace/working/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
     --phase-a-ckpt   /workspace/working/PlantCLEF2026/models/dinov3_plantnet_v2_retrain/phase_a_best.pth \
     --img-size 448 --batch-size 16 --accum 2 --bf16 \
     --epochs 15 --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
     --lr-backbone 1e-4 --lr-head 2e-4 --num-workers 10 \
     --output-dir /workspace/working/PlantCLEF2026/models/dinov3_plantnet_v2_retrain
   ```
   ETA ~3.5-4h on single 5090.

2. **Inference — NO TILING.** The v1 investigation showed tiling hurts both on
   collages and test images. Use `test_probs_notile.npz`-style full-image
   CenterCrop(448) forward. Skip the enhance pipeline entirely — post-hoc
   logit-adjustment tanked our scores.

## Non-goals

- **Don't** relaunch any v1 training — it's a dead end at this data volume.
- **Don't** delete v1 ckpts (`dinov3_plantnet_v1_ddp/`) — they're the 0.0002
  baseline and the only data we have on the LR failure mode.
- **Don't** touch `outputs/` in 007 — those are Arjun's current best Kaggle
  artefacts (0.175).
- **Don't** modify `train_phase_a.py` mid-run.
- **Don't** run inference with `dump_test_probs.py --tile-sizes …` — multi-scale
  tiling was confirmed to hurt on this task (82% overlap w/ raw probs shuffled by
  logit-adj). Use `test_probs_notile.npz` approach.

## Kaggle context

- **Team best:** 006 BioCLIP-2.5 at **0.33** (Arjun).
- **007 PlantNet DINOv2 on collages:** 0.175 (same collage recipe, different backbone).
- **008 v1 (broken):** 0.0002-0.00255.
- **Phase A v2 + Phase B v2 target:** match-or-beat 007 (≥ 0.175), stretch ≥ 0.22.
- **Deadline:** 16 days from 2026-04-22 → final window closes ~2026-05-08.

## If the pod reboots mid-run

- `/workspace` is persistent across pod recreations on this RunPod template.
- `train_phase_a.py` supports `--resume .../phase_a_last.pth` — OneCycleLR does
  not resume cleanly (schedule restarts from step 0), so expect a loss spike.
  Cheaper than restart if > 4 epochs of progress.
- Full relaunch command: re-run the `torchrun …` line in the `phase_a_v2`
  tmux after pod is back up. Remember to preserve `--output-dir` so the
  existing ep files aren't overwritten (`--resume` handles ordering).
