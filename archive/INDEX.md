# PlantCLEF 2026 — Archived Project Tree

Snapshot of the team's working repository (`PlantCLEF2026-main/`) preserved
here on **2026-05-16** before the working copy was deleted. Cruft removed:
the 416 Windows `*:Zone.Identifier` metadata files and a stray empty `0.5`
placeholder. Everything else is byte-for-byte intact, including the 16 MB
EDA notebook and the i001 explore notebook.

The headline source — `PlantCLEF2026-main/report/main.tex/.pdf` with the
Path A corrections applied on 2026-05-16 — also lives at the **primary**
location:

  `../report/main.tex` (canonical official paper, kept in sync with
  `report/main.tex` in this archive at the moment of archiving).

The personal / portfolio paper is unchanged and lives at:

  `../paper/plantclef2026_research_paper.tex/.pdf`

---

## Where to look

### Headline paper artefacts

| File | What it is |
|---|---|
| `report/main.tex` | Official CLEF 2026 working note — Path-A-corrected |
| `report/main.pdf` | Compiled PDF, 388 KB, 9 pages |
| `report/plantclef-refs.bib` | Bibliography |
| `report/ceurart.cls`, `*.bst` | CEUR-ART class file and style |
| `report/Deprecated/` | Older drafts (sample-1col / sample-2col templates) |

### Master documentation

| File | What it is |
|---|---|
| `docs/experiments_summary.md` | Master roll-up of every numbered experiment — recipes, scores, deltas, takeaways. **The single most important doc in the repo for understanding the project.** |
| `docs/inference_experiments_report.md` | The three orthogonal-pivot inference experiments (cRT, genus rerank, phenology) — full recipes, diagnostic gates, deltas. Source of the pivot stack details. |
| `docs/challenge_overview.md` | PlantCLEF 2026 task spec |
| `docs/dataset_description.md` | Train / test data structure |
| `docs/evaluation.md` | Macro-F1 metric definition |
| `docs/exp_reports/` | Per-experiment writeups in detail |

### Project-level docs

| File | What it is |
|---|---|
| `README.md` | High-level project README |
| `research_proposal.md` | Original COMP3242 research proposal |
| `background.txt` | Background notes on the challenge |
| `todo.md` | Working TODO list |
| `requirements.txt` | Python dependencies |

### Source code (training pipeline)

| Path | What it is |
|---|---|
| `src/` | Shared training infrastructure (`config.py`, `data/`, `models/`, `training/`, `train.py`) |
| `src_experiments/` | Per-experiment subdirectories — each has its own `model.py`, `dataset.py`, `train.py`, `infer_tiles.py`, plus `report.md` and score CSVs where applicable |
| `tools/` | Standalone utilities — `check_ckpt.py`, `generate_metadata.py`, `rescue_checkpoint.py`, `stratify_dataset.py`, `verify_images.py` |
| `scripts/` | Top-level helper scripts |
| `Infastructure/` | Cluster / deployment config |

### Key per-experiment reports (the audit trail)

The numbered experiments under `src_experiments/`. Best-scoring ones first:

| Folder | Best public F1 | Notes |
|---|---|---|
| `i002_bioclip25_cap_image/` | **0.418265** (224+336 ensemble + LA τ=0.25, T=0.03) | Headline anchor. `report.md` is authoritative; the in-folder `README.md`/`EXPERIMENT_REPORT.md` are stale copies of 010's docs. Five score CSVs in this folder. |
| `i003_bioclip25_cap_image_extra500/` | 0.40041 | 500-cap with 161 K extra under-100-img observations. Capping hurt; uncapped i002 wins. |
| `010_bioclip25_end_to_end_finetune_multitask/` | 0.38333 | Original 4-block anchor on the 1.4 M manifest. The 010 recipe is the bedrock of everything that followed. |
| `014_unfreeze_sweep/` | n=3: 0.37455 · n=5: 0.36919 | Brackets the n=4 anchor; n=5 has the highest val top-5 but lowest Kaggle. First observation of the val/Kaggle inversion. `summary.md` is the writeup. |
| `015_pc24_inat_mix/` | ep5 0.37956 | 50:50 PC24 + iNat mix; underperforms the 010 anchor. "Adding capacity to the wrong distribution." |
| `009_bioclip25_plantnet_finetune/` | 0.20777 | Full fine-tune. Collapses the Tree-of-Life prior. |
| `006_bioclip25_finetune/` | 0.330 | Frozen prototype (head only). Anchor for the head-only ablation row. |
| `011_bioclip25_aggregation_sweep/` | 0.38278 | Inference-only α-sweep of mean × noisy-or on 010 logits. Inside noise. |
| `008_dinov3_plantnet_finetune/` | 0.34671 (fused with 006) | DINOv3 PlantNet PhaseA. Retired. |
| `005_dinov3_multilabel/` | 0.13 | Early DINOv3 multilabel baseline. Retired. |

### Leaderboard data (the canonical scoreboard)

| File | What it is |
|---|---|
| `website/components/Leaderboard.jsx` | Public team-website leaderboard component. The `SUBMISSIONS` array near the top is the canonical public + private F1 history for the team's top 25 submissions. Recipe strings here are the final authoritative recipe labels. |

### EDA / analysis notebooks (heavy)

| File | Size | What it is |
|---|---|---|
| `eda/plantclef2026_eda.ipynb` | 16 MB | Master EDA notebook — tile-aggregation analysis, per-quadrat species spread, etc. |
| `src_experiments/i001_data_download/explore.ipynb` | 7.9 MB | i001 manifest exploration |

### Submissions corpus

| File | What it is |
|---|---|
| `outputs/submission_009_*.csv`, `submission_010_*.csv` | Sample submitted CSVs (per-row `quadrat_id, species_ids`). The full corpus of inference output dirs lives under `src_experiments/i002_bioclip25_cap_image/outputs/`. |

---

## What was NOT preserved

- The original Git history of `PlantCLEF2026-main` — this is a flat snapshot, not a clone of the git tree.
- The Windows `:Zone.Identifier` metadata (deliberately deleted — 416 files of cruft).
- An empty `0.5` placeholder file at the root.
- Anything that was already gitignored or untracked outside the directory tree.

## Authoritative numbers to remember

| Quantity | Value | Source |
|---|---|---|
| Public Macro-F1 (best) | **0.418265** | i002 224+336 ensemble, LA τ=0.25, T=0.03 |
| Private Macro-F1 (best) | **0.40283** | Same submission |
| 010 anchor (1.4 M, T=0.05 single-res) | 0.38333 | Team's first stable score |
| n=3 / n=5 deltas vs n=4 | −0.00878 / −0.01414 | 014b sweep |
| Total training images (i002 manifest) | 2,653,781 | i001 redownload |
| Species in taxonomy | 7,806 | Stable across all experiments |
| Total submitted configurations on hidden test | 4 | anchor + cRT + genus + phenology |
| Pivot deltas (cRT / genus / phenology) | −0.036 / −0.006 / −0.005 | Path-A corrected |
| Validation vs Kaggle Pearson r | ≈ 0.4 (across 12 strongest runs) | Discussion §6 |
