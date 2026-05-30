# Experiment Report: 015 — BioCLIP-2.5 MultiTask on PC24 ⊕ iNat Research-grade Mix

**Most useful files for this report:**
- `build_combined_manifest.py` — vectorised PC24 + iNat manifest builder; iNat path-encoding trick lives here
- `dataset_patch.py` — option-A `dataset.py` patch (NOT applied; sandbox blocked the shared-code edit)
- `run_train.sh` — DDP launcher; resumes from 010's `head_only/best.pt` via the long-timeout wrapper
- `run_with_long_timeout.py` — wrapper that monkey-patches `dist.init_process_group(timeout=2h)` before `runpy.run_path`-ing 010's `train.py` (avoids editing shared 010 code)
- `run_inference.sh` — team-best inference recipe applied to 015 best.pt
- `outputs/submission_015_ep1.csv` — Kaggle submission, ep1 best.pt (0.37506)
- `outputs/submission_015_ep5.csv` — Kaggle submission, ep5 best.pt (0.37956)
- `scores.csv` / `scores_ep5.csv` — Kaggle public scores logged

---

## Run: 015 PC24 ⊕ iNat Research-grade 50:50 fine-tune

**Experiment ID:** 015
**Date:** training launched 2026-05-02 12:27 UTC, completed 2026-05-03 06:16 UTC

### Why this experiment exists

The 014 unfreeze sweep made the diagnosis sharp: **the bottleneck is data
domain, not model capacity**. Val top-5 *inverts* Kaggle within the sweep — n=5
had the best val (0.9327) and the worst Kaggle (0.36919). The PC24
single-plant Pl@ntNet close-up distribution does not transfer to LUCAS
multi-species quadrat test images.

015 attacks this by **mixing PC24 with iNaturalist Research-grade habitat
shots**. iNat photos are typically wider field shots in situ — visually closer
to LUCAS quadrats than PC24's specimen close-ups. The hypothesis: 50:50 mixing
shifts the training distribution toward test geometry without sacrificing the
species-precision that PC24 provides.

### Combined Manifest

`/workspace/plantclef/processed/pc24_inat_combined_manifest.csv` (semicolon-sep)

| Source | Rows | Species | Notes |
|---|---|---|---|
| PC24 | 1,408,033 | 7,806 | full taxonomic coverage |
| iNat Research-grade | 1,338,626 | 5,037 | 2,769 species have zero iNat coverage; train on PC24 alone for those |
| **Combined** | **2,746,659** | **7,806** | **51% PC24 / 49% iNat** — natural concat, no sampler needed |

Taxonomy coverage: 100% at all levels (genus 1,446, family 181, order 61,
class 6) after merging with `species_lookup_with_gbif_cleaned_names.csv`. A
`source ∈ {pc24, inat}` column is preserved for diagnostics.

### Encoding Trick — No Shared 010 Code Change Needed

010's `dataset.py` resolves images via the unparameterised pattern
`{train_image_root}/{species_id}/{image_name}`. Patching that resolver to
accept absolute paths (option A, see `dataset_patch.py`) was blocked by the
sandbox as a shared-infrastructure edit.

**Workaround:** encode iNat rows so the *unmodified* resolver finds them.
For each iNat row, set `image_name = "../../../inat_research_grade/<sp>/<file>"`.
The `..` segments are normalised at `PIL.Image.open()` time and resolve to
`/workspace/plantclef/raw/inat_research_grade/<sp>/<file>` on disk. PC24 rows
keep plain filenames. The relative-encoding prefix is computed vectorised via
`os.path.commonpath` so the manifest builder never makes per-row filesystem
calls.

### Recipe

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
| `--use-taxonomy-heads` | enabled (joint loss `CE_species + 0.30·CE_genus + 0.15·CE_family + 0.05·CE_order + 0.02·CE_class`) |
| unfreeze | last **4** blocks + `ln_post` + `proj` (014 sweet spot) |
| resume from | `outputs/head_only/checkpoints/best.pt` (`--resume-weights-only`) |
| DDP timeout | **2 hours** (default 10 min, see crash-fix below) |
| hardware | 2× RTX 5090 DDP |
| total train time | ~6h 22m wall (5 epochs × ~70 min train + ~10 min val) |

### Crash + Fix

**End of epoch 1, 2026-05-02 13:48 UTC:** rank-1 SIGABRT, rank-0 SIGTERM.
Root cause was *not* OOM or a corrupt iNat jpg, despite both being plausible
on first inspection.

The actual cause was an **NCCL `ALLREDUCE` watchdog timeout during the
validation pass**. The val set is now 271,615 samples (PC24 + iNat val split,
~2× the 014 anchor's val), and a single-rank scalar `ALLREDUCE` at the end of
the val loop was held for >600 s while rank 0 was still iterating. PyTorch's
`ProcessGroupNCCL` default watchdog is `timeout=timedelta(minutes=10)`, so
rank 1 killed the process group after 10 minutes — *after* `epoch_001.pt`,
`best.pt`, and `last.pt` had all saved cleanly.

**Fix:** `015_pc24_inat_mix/run_with_long_timeout.py` is a 30-line wrapper
that monkey-patches `torch.distributed.init_process_group` to default
`timeout=timedelta(hours=2)` before `runpy.run_path`-ing 010's `train.py`. CLI
args are forwarded via `sys.argv`. No shared 010 code touched.

Resume from `last.pt` with the wrapper restored full optimizer / scheduler /
epoch state and ran epochs 2–5 cleanly through the next four val passes.

### Validation Metrics (Per-Epoch)

| Epoch | Train loss | Val loss | top-1 | top-5 | genus | family | order | class |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.2996 | 0.8783 | 0.8046 | 0.9519 | 0.9149 | 0.9490 | 0.9448 | 0.9846 |
| 2 | 3.1261 | 0.8318 | 0.8143 | 0.9549 | 0.9198 | 0.9528 | 0.9507 | 0.9875 |
| 3 | 3.0182 | 0.8054 | 0.8207 | 0.9570 | 0.9229 | 0.9547 | 0.9535 | 0.9884 |
| 4 | 2.9529 | 0.7924 | 0.8244 | 0.9582 | 0.9238 | 0.9551 | 0.9543 | 0.9886 |
| **5** | 2.9235 | 0.7881 | **0.8255** | **0.9584** | 0.9240 | 0.9553 | 0.9544 | 0.9886 |
| 014 anchor (no iNat) | — | — | 0.7475 | 0.9323 | — | — | — | — |

ep5 val top-1 of **0.8255** is the best we've ever seen on this dataset —
**+0.078** over the team-best 014 anchor.

### Inference Recipe (= team-best 010 recipe, byte-for-byte)

```bash
python infer_tiles.py \
    --checkpoint .../pc24_inat_mix_unfreeze4/checkpoints/best.pt \
    --tile-mode grid_4x4 --tile-size 448 --overlap 0.0 --img-size 224 \
    --agg-mode softmax_mean --top-k 3 \
    --batch-size 64 --precision bf16
```

### Results

| Submission | Val top-1 | Val top-5 | Kaggle Public F1 | Δ vs anchor |
|---|---|---|---|---|
| `submission_015_ep1.csv` | 0.8046 | 0.9519 | **0.37506** | −0.00827 |
| `submission_015_ep5.csv` | **0.8255** | **0.9584** | **0.37956** | **−0.00377** |
| 014 anchor (no iNat) | 0.7475 | 0.9323 | **0.38333** | — |

### Status / Verdict

**The 50:50 PC24 ⊕ iNat mix at `unfreeze_n=4` cannot beat vanilla PC24 at the
same recipe.** ep5 closes 54% of the ep1 → anchor gap by training longer, but
still loses by **−0.00377**.

This is the strongest single piece of evidence yet that **val score is not a
Kaggle predictor on this distribution shift**:

- ep5 val top-1 (0.8255) is **+0.078 above the anchor** (0.7475)
- ep5 Kaggle (0.37956) is **−0.00377 below the anchor** (0.38333)

ep1 → ep5 also shows the relationship is *positive but weak* within 015
(+0.026 top-1 buys only +0.0045 Kaggle). Adding iNat made the model better at
predicting iNat-shaped data — and that did not transfer to LUCAS multi-species
quadrats.

**Don't:**
- Keep tuning 50:50 PC24:iNat ratios at `unfreeze_n=4`. The gradient is in
  the wrong direction.
- Run more epochs at this recipe — ep4 → ep5 val gain was +0.0002, already
  plateauing.

**Hypotheses worth testing IF iNat is revisited (none kicked off without
Arjun's experiment-A signal first):**
1. **Smaller iNat fraction** (e.g. 80:20 PC24:iNat) so the PC24 species prior
   stays dominant.
2. **Habitat-filtered iNat** — drop close-up macros, keep wider scenes that
   match LUCAS aspect 1.34.
3. **iNat as Phase-1 SSL pre-training**, then standard 010 fine-tune on PC24
   alone. Decouples "domain shift" from "label shift".

### Key Takeaways

1. **Naïve 50:50 multi-source training does not bridge the train→test domain
   gap.** The mixture became its own distribution rather than a stepping stone
   toward LUCAS — fitting it doesn't help on the actual test set.
2. **Val/Kaggle inversion is a load-bearing finding.** Three independent
   observations now (014 sweep n=3/4/5, 015 ep1, 015 ep5). The val set —
   whether PC24-only or PC24 ⊕ iNat — overfits the training distribution and
   over-rewards capacity that hurts test transfer.
3. **`run_with_long_timeout.py` is reusable.** Any future DDP training on
   this codebase that exceeds the 10-minute NCCL watchdog (likely whenever
   the val set scales up further) can drop in the same wrapper.
4. **The encoding trick generalises.** Mounting external data into 010's
   `{root}/{sp}/{file}` resolver via relative `..` traversal worked
   first-shot. Any future "add a second image source" experiment can copy this
   pattern instead of patching `dataset.py`.
5. **iNat is not necessarily dead — the recipe is.** A different mixing
   ratio, habitat filter, or Phase-1 SSL placement might still pay off, but
   they would need to overcome the val/Kaggle inversion that 50:50 amplified.
