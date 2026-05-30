# 014 — Unfreeze-N sweep around the 0.38333 anchor

Re-run the team-best 010 last_blocks recipe with `--unfreeze-last-n-blocks` set
to 3 and 5 (anchor was n=4) to find the partial-unfreeze sweet spot. From the
existing trajectory:

| unfreeze_n | Kaggle F1 | Note |
|---|---|---|
| 0 (head_only) | ~0.33 | starts the 2-phase, never measured alone on test |
| **4 (anchor)** | **0.38333** | team-best |
| 32 (full FT, 009) | 0.20777 | collapsed, taxonomy prior destroyed |

Hypothesis: there's a partial-unfreeze sweet spot somewhere; n=4 might be too
shallow OR too deep within the safe zone.

## Recipe (matches `scripts/train_last_blocks.sh` exactly except `--unfreeze-last-n-blocks`)

- `--epochs 5 --warmup-epochs 1`
- `--batch-size 64 --grad-accum-steps 4` → effective batch 256
- `--precision bf16`
- `--use-taxonomy-heads` (matches team-best)
- `--head-lr 1e-4 --backbone-lr 1e-6 --weight-decay 1e-4`
- `--label-smoothing 0.1`
- 2× RTX 5090 DDP via `torchrun --standalone --nproc_per_node=2`
- Resume from `outputs/head_only/checkpoints/best.pt` with `--resume-weights-only`

Output dirs (under
`/workspace/working/workspace/PlantCLEF2026/src_experiments/010_bioclip25_end_to_end_finetune_multitask/`):
- `outputs/last_blocks_unfreeze3/`
- `outputs/last_blocks_unfreeze5/`

`run_sweep.sh` runs them sequentially in one tmux session.

## After training

Use `dump_test_probs.py` from 010 with `--checkpoint outputs/last_blocks_unfreezeN/checkpoints/best.pt`
and the team-best inference recipe (`grid_4x4 tile_size=448 ov=0 img_size=224
softmax_mean top-3`). Submit each via Kaggle.
