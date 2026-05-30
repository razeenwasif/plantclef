# PlantCLEF 2026 — ANU R&D Fork

This is the active R&D fork of the PlantCLEF 2026 ANU submission. The
frozen paper artifact (the version that scored 7th private on the
leaderboard) lives in `~/Research/PlantCLEF2026`; that repo is what was
submitted for assessment, this repo is where the post-submission work
continues.

Public Macro F1 of the headline submission was **0.41826**; corresponding
private score **0.40283**, with the project's best private being
**0.40600** (logit-adjusted single-resolution variant).

The report lives in [`report/main.tex`](report/main.tex) (mirrored from
the frozen repo so this codebase remains self-contained).

---

## Repository layout

The codebase is split into two research streams plus a shared core:

```
.
├── shared/                            Code reused by both streams
│   ├── bioclip25_multitask/           BioCLIP 2.5 multitask training
│   │                                  (model + dataset + train loop)
│   └── tools/
│       ├── cache_features.py          Pre-compute backbone features
│       └── score_submission.py        Metrics + diagnostic plots
│
├── quadrat/                           PlantCLEF-style quadrat work
│   ├── inf_script.py                  Headline anchor recipe (0.41826)
│   ├── inf_script_phen.py             Pivot 3: seasonal phenology prior
│   ├── checkpoints/                   Paper anchor + cRT heads
│   ├── data/plantclef_test/           PlantCLEF 2026 test quadrats
│   ├── experiments/                   i001/i002/i003 + earlier work
│   ├── outputs/                       Quadrat inference outputs
│   └── docs/exp_reports/              Per-experiment writeups
│
├── single_plant/                      PlantNet-style single-plant work
│   ├── inf_script_whole.py            Whole-image (no tile) inference
│   ├── data/plantnet300k_manifest.csv
│   ├── outputs/                       PlantNet inference + caches
│   ├── tools/build_plantnet300k_manifest.py
│   └── docs/exp_reports/              Per-experiment writeups
│
├── src/                               Top-level entry points (shims)
│   ├── train.py                       -> shared/bioclip25_multitask/train.py
│   ├── inf_quadrat.py                 -> quadrat/inf_script.py
│   ├── inf_quadrat_phen.py            -> quadrat/inf_script_phen.py
│   └── inf_single_plant.py            -> single_plant/inf_script_whole.py
│
├── engines/rust/                      SIMD-parallel resize + WebDataset
│                                      pack + DALI index crates
├── coordinator/                       Multi-node training control plane
├── docs/                              Project-wide docs (algorithms,
│                                      backlog cross-refs, architecture)
├── report/                            LaTeX source of the working note
├── legacy/                            Pre-split / archived material
├── BACKLOG.md                         Future-novelty work
├── CHANGELOG.md                       Post-submission history
└── website/                           Project landing page
```

**Why the split?** PlantCLEF and PlantNet are different problems —
quadrat aggregation, vegetation filtering, phenology priors, and
multi-species CSV emission all matter only for the quadrat track,
while PlantNet-300K is a clean single-plant classification benchmark
where those layers actively hurt. Keeping the streams next to each
other lets them share `shared/bioclip25_multitask/` (model, loss,
training loop) without inheriting each other's quirks.

---

## Final paper system

The paper's headline configuration is a single BioCLIP 2.5 ViT-H/14
backbone with per-head MLPs for species, genus and family.

* **Backbone**: BioCLIP 2.5 ViT-H/14 (`hf-hub:imageomics/bioclip-2.5-vith14`).
  The lower 28 transformer blocks stay frozen to preserve the
  Tree-of-Life prior; only the last 4 blocks plus the final layer norm
  and projection are unfrozen.
* **Heads**: three independent MLPs of the form
  `LayerNorm -> Linear(1024 -> 1024) -> GELU -> Dropout(0.2)`,
  feeding linear classifiers of sizes 7,806 / 1,446 / 181 for species /
  genus / family.
* **Loss**: weighted joint cross-entropy with label smoothing 0.1,
  weights `1.0 * L_species + 0.30 * L_genus + 0.15 * L_family`;
  missing taxonomy labels encoded as `-1` and masked.
* **Training data**: the i001 manifest, 2,653,781 single-plant images
  across 7,806 species (PlantCLEF 2024 + a research-grade iNaturalist
  pull, deduplicated, genus / family pre-filled). No per-species cap.
* **Schedule**: two stages of ten epochs each. Stage 1 trains the head
  MLPs and classifiers with the backbone fully frozen; Stage 2 resumes
  the weights and additionally unfreezes the last 4 transformer blocks
  + `ln_post` + `proj`.
* **Optimiser**: AdamW (head LR `1e-4`, backbone LR `1e-6`, weight decay
  `1e-4`), one epoch of linear warmup then cosine decay to 1% of peak,
  global-norm gradient clip 1.0.
* **Precision**: bfloat16 AMP (no GradScaler), DDP across 2x RTX 5090
  via `torchrun --nproc_per_node=2`.
* **Inference**: each quadrat is partitioned into a 4x4 grid of 16
  tiles; every tile is forwarded through the encoder at both 224 and
  336 pixels (the ViT-H/14 pos-embed is bicubically resampled for the
  336 px pass). Per-tile softmax probabilities are averaged across
  tiles and across the two resolutions, class-prior logit adjustment
  with `tau = 0.25` is applied against the Laplace-smoothed training
  prior, and every species with post-adjustment probability above
  `T = 0.03` is emitted, clamped to `[k_min = 2, k_max = 10]`.

Full hyperparameter, augmentation, and split specification:
[report Appendix B (Table 7)](report/sections/appendix_development_trace.tex).

---

## Quick start

### Environment

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `open_clip_torch` (BioCLIP 2.5 weights),
`pandas`, `Pillow`, `torchvision`.

### Train the paper model

`src/train.py` is a thin shim that forwards to
`shared/bioclip25_multitask/train.py`. The two stages from the paper:

```bash
# Stage 1: head only, 10 epochs, backbone frozen
torchrun --nproc_per_node=2 src/train.py \
    --metadata-csv  path/to/metadata_filled_genus_family.csv \
    --train-image-root path/to/images \
    --epochs 10 --batch-size 512 --grad-accum-steps 2 \
    --precision bf16 --freeze-backbone \
    --head-lr 1e-4 --weight-decay 1e-4 \
    --output-dir outputs/stage1_head_only

# Stage 2: resume + unfreeze last 4 blocks, 10 epochs
torchrun --nproc_per_node=2 src/train.py \
    --metadata-csv  path/to/metadata_filled_genus_family.csv \
    --train-image-root path/to/images \
    --epochs 10 --batch-size 128 --grad-accum-steps 4 \
    --precision bf16 --unfreeze-last-n-blocks 4 \
    --backbone-lr 1e-6 --head-lr 1e-4 --weight-decay 1e-4 \
    --resume outputs/stage1_head_only/checkpoints/best.pt \
    --resume-weights-only \
    --output-dir outputs/stage2_last4_blocks
```

Head-only training can be massively accelerated by caching backbone
features once:

```bash
python shared/tools/cache_features.py \
    --metadata-csv path/to/manifest.csv \
    --output       outputs/feature_cache.pt

python src/train.py --feature-cache outputs/feature_cache.pt \
    --freeze-backbone --no-epoch-snapshots ...
```

### Reproduce the headline quadrat submission

```bash
python src/inf_quadrat.py \
    --checkpoint    quadrat/checkpoints/paper_anchor/best.pt \
    --image-dir     quadrat/data/plantclef_test \
    --metadata-csv  path/to/metadata_filled_genus_family.csv \
    --output        submission.csv
```

This is the fixed 4x4 grid + 224+336 + LA (tau=0.25) +
adaptive-threshold recipe that scored 0.41826 public.

### Phenology pivot (paper Pivot 3)

```bash
python src/inf_quadrat_phen.py \
    --checkpoint    quadrat/checkpoints/paper_anchor/best.pt \
    --image-dir     quadrat/data/plantclef_test \
    --metadata-csv  path/to/metadata_filled_genus_family.csv \
    --phenology-csv path/to/gbif_month_histograms.csv \
    --output        submission_phen.csv
```

Adds the four phenology-specific stages from Appendix C: multi-scale
tiling at {1.0, 0.8}, ExG vegetation filter (drop tiles below 15%
green), entropy-weighted Bayesian aggregation
`w_t ∝ exp(-H_t) · ExG_t`, and a circular-Gaussian DOY prior
(sigma=18 d, epsilon=0.05, beta=1.0) built from the GBIF month
histograms.

### Single-plant whole-image inference (PlantNet)

```bash
python src/inf_single_plant.py \
    --checkpoint  outputs/stage2_last4_blocks/checkpoints/best.pt \
    --image-root  /mnt/d/PlantNet-300k/images/images/test \
    --output      outputs/single_plant_whole/submission.csv \
    --resolutions 224 336 --top-k 5
```

No tiling, no LA, no adaptive thresholding — just softmax-mean over
the requested resolutions on the centre crop. Writes a CSV with
`image_id, top1_species_id, top1_prob, topk_species_ids`. Score it
with `shared/tools/score_submission.py` for Macro F1 + plot suite.

---

## Where to read next

* `report/main.tex` — the working note (paper).
* `BACKLOG.md` — planned novel inference + architecture experiments.
* `CHANGELOG.md` — post-submission history.
* `quadrat/docs/exp_reports/` — per-experiment writeups for the
  quadrat side (paper anchor, phenology pivot, cRT pivot, etc.).
* `single_plant/docs/exp_reports/` — per-experiment writeups for the
  PlantNet side (head-only cached run, scout runs, etc.).

---

## Citation

```bibtex
@inproceedings{anu-plantclef2026,
  title  = {Fine-Tuning of BioCLIP 2.5 with Taxonomic Heads for Multi-Species Plant Identification},
  author = {Raj, Arjun and de Mel, Manindra and Wasif, Razeen and Brake, William},
  booktitle = {CLEF 2026 Working Notes},
  year   = {2026},
}
```
