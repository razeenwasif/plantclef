# Experiment Report: 014 — Unfreeze-N Sweep around the 0.38333 Anchor

**Most useful files for this report:**
- `README.md` — sweep design, recipe deltas, output dirs
- `run_sweep.sh` — sequential `n=3` then `n=5` train launcher (DDP, 2× RTX 5090)
- `run_inference.sh` — team-best inference recipe applied to each sweep ckpt
- `outputs/unfreeze3/submission_unfreeze3.csv` — Kaggle submission (n=3)
- `outputs/unfreeze5/submission_unfreeze5.csv` — Kaggle submission (n=5)
- `scores.csv` — Kaggle public scores logged by `submit_predictions_kaggle.sh`

---

## Run: 014 unfreeze-N sweep on the 010 last_blocks recipe

**Experiment ID:** 014 (sweep)
**Date:** 2026-05-01 → 2026-05-02

### Why this experiment exists

010 last_blocks ALONE is the team best (**0.38333**), and 009 full-FT collapses
to 0.20777 — confirming a **partial-unfreeze sweet spot** somewhere between
"head only" and "full FT". The anchor ran with `--unfreeze-last-n-blocks 4`. We
don't know whether n=4 is the optimum or a lucky sample on a flat plateau. 014
brackets the anchor with n=3 and n=5, holding everything else fixed.

### Recipe (matches the team-best 010 recipe byte-for-byte except `--unfreeze-last-n-blocks`)

| Parameter | Value |
|---|---|
| backbone | BioCLIP-2.5 ViT-H/14 (32 blocks, 1024-d) |
| head | `BioCLIP25MultiTask` (SharedMLP + species + 4 aux heads) |
| epochs | 5  (warmup 1) |
| batch_size | 64 micro × grad_accum 4 = effective 256 |
| precision | bf16 |
| backbone_lr | 1e-6 |
| head_lr | 1e-4 |
| label_smoothing | 0.1 |
| weight_decay | 1e-4 |
| `--use-taxonomy-heads` | enabled (loss = `CE_species + 0.30·CE_genus + 0.15·CE_family + 0.05·CE_order + 0.02·CE_class`) |
| unfreeze | last **{3, 5}** blocks + `ln_post` + `proj` |
| resume from | `outputs/head_only/checkpoints/best.pt` (`--resume-weights-only`) |
| hardware | 2× RTX 5090 DDP (`torchrun --standalone --nproc_per_node=2`) |
| train_meta_csv | `train_metadata_cleaned_verified_stratified.csv` |

### Inference Recipe (= team-best 010 recipe, byte-for-byte)

```bash
python infer_tiles.py \
    --checkpoint .../outputs/last_blocks_unfreezeN/checkpoints/best.pt \
    --tile-mode grid_4x4 --tile-size 448 --overlap 0.0 --img-size 224 \
    --agg-mode softmax_mean --top-k 3 \
    --batch-size 64 --precision bf16
```

### Results

| Run | Val top-5 | Kaggle Public F1 | Δ vs anchor |
|---|---|---|---|
| n=3 | 0.9320 | 0.37455 | −0.00878 |
| **n=4 (anchor)** | 0.9323 | **0.38333** | — |
| n=5 | 0.9327 | 0.36919 | −0.01414 |

### Status / Verdict

**Both wings underperformed the anchor on Kaggle.** n=4 sweet spot is real and
narrow (±1 block costs 0.009 – 0.014 F1).

The headline finding: **val top-5 correlates *backwards* with Kaggle.** n=5 had
the best val score (0.9327) but the *worst* Kaggle (0.36919); n=3 had the
weakest val and a middling Kaggle. This is the first observation of the
val/Kaggle inversion that 015 later amplified and that is now a load-bearing
lesson on this competition (see master `docs/experiments_summary.md`).

**Reading.** More unfrozen blocks = more capacity to fit PC24's single-plant
distribution = worse transfer to multi-species LUCAS quadrats. The bottleneck
is *data domain* (PC24→test gap), **not model capacity**. Adding capacity
without fixing data overfits the wrong distribution.

**Don't:**
- Keep tuning `unfreeze_n` on PC24 alone — the hill-climb is in val space and
  doesn't transfer.
- Trust val top-5 deltas <0.005 — they invert at test time.

**Do:**
- Treat n=4 as the team-best discipline going forward and pull on the *data*
  axis (014 iNat pull, 015 PC24+iNat mix) instead of the capacity axis.

### Key Takeaways

1. **Partial-unfreeze sweet spot is sharp, not a plateau.** ±1 block from n=4
   costs 0.009 – 0.014 F1 on Kaggle. The "lower blocks carry the Tree-of-Life
   prior" thesis is reinforced — n=5 unfroze enough additional capacity to
   start eroding it.
2. **Val/Kaggle inversion observed for the first time.** Higher val =
   stronger fit to the PC24 single-plant val distribution; the test set
   (LUCAS quadrats) rewards features that *don't* over-fit the single-plant
   axis. This invalidates val as a model-selection signal across this
   distribution shift.
3. **The bottleneck is data domain, not capacity.** Capacity is already
   sufficient to fit PC24; what's missing is *training distribution* that
   matches LUCAS multi-species quadrats. This justifies the 014 iNat data pull
   and the 015 mixing experiment.
4. **Sweep design held everything else fixed.** Same warmstart, same data,
   same loss weights, same DDP config — so the deltas are clean attribution to
   `unfreeze_n`. This is the controlled experiment 013 wasn't.
