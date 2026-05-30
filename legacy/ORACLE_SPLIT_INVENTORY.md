# Oracle split inventory

Working classification of `~/Oracle` for the three-repo split:

* **`~/Research/plantclef`** = active ML R&D (training, inference, models, data, configs, research tooling)
* **`~/Oracle`** = product / business logic (Identify upload flow, Atlas globe, Journal, dashboard, Firebase)
* **`~/Research/PlantCLEF2026`** = frozen paper artifact (already done)

## Legend

| Tag             | Meaning                                                                                          |
|-----------------|--------------------------------------------------------------------------------------------------|
| **ML**          | Belongs in `~/Research/plantclef`                                                                |
| **Product**     | Stays in `~/Oracle`                                                                              |
| **Shared**      | Needed by both - one repo owns it, the other consumes via a clean API. Needs a call.             |
| **Already**     | Already ported into `~/Research/PlantCLEF2026` (and therefore mirrored in `~/Research/plantclef`); decide whether the active copy lives here or there. |
| **Decide**      | Genuinely ambiguous; needs your call.                                                            |
| **Drop**        | Cache / build artefacts / scratch; do not move.                                                  |

## Top-level files

| Path                            | Tag      | Notes                                                                                          |
|---------------------------------|----------|------------------------------------------------------------------------------------------------|
| `README.md`                     | Decide   | Currently the polyglot ML/Blackwell pitch. Needs splitting: ML half -> R&D README, product half -> a new Oracle README. |
| `ROADMAP.md`                    | Product  | Identify / Atlas / Journal / dashboard product pillars. Stays in Oracle.                       |
| `CHANGELOG.md`                  | Decide   | Mixed (ML phase work + dashboard releases). Recommend splitting per repo's actual history.     |
| `PROGRESS.md`                   | Drop     | Working-note scratch; probably stale.                                                          |
| `Dockerfile`                    | Decide   | Whether it builds the training image or the dashboard image. Inspect before classifying.       |
| `firebase.json`, `.firebaserc`  | Product  | Firebase project config.                                                                       |
| `firestore.rules`               | Product  | Firestore security rules.                                                                      |
| `plantclef.py`                     | ML       | The training / inference CLI. Rename when ported.                                              |
| `inf_script.py`                 | Already  | Headline-recipe inference (paper pivot 0). Already in `src/inf_script.py` here.                |
| `inf_script_phen.py`            | Already  | Phenology pivot. Already in `src/inf_script_phen.py` here.                                     |
| `requirements.txt`              | Decide   | Currently mixes ML + product deps. Split: ML deps go to R&D, JS deps already isolated under apps/. |
| `PlantCLEF2026.pdf:Zone.Identifier` | Drop | Windows alternate-data-stream junk.                                                            |
| `nexus-current.png`, `palette-*.png` | Decide | Looks like dashboard mockups. Probably Product.                                              |

## Top-level directories

### Clearly ML

| Path                      | Tag    | Notes                                                                                          |
|---------------------------|--------|------------------------------------------------------------------------------------------------|
| `phases/`                 | ML     | `foundation_caching/`, `head_warmup/`, `expert_specialization/`, `student_distillation/`, `asymmetric_distillation/`, `crt_train/`, `inference/`, `experimental/`. Core training lifecycle. |
| `src/training/`           | ML     | `trainer.py`, `loops.py`, `losses.py`, `checkpoints.py`, `accelerator.py`, `cache.py`, `liveness.py`, `pca_utils.py`, `query_expansion.py`, `telemetry.py`, `utils.py`. |
| `src/inference/`          | ML     | `pipeline.py`, `aggregation.py`, `ensemble.py`, `frank_wolfe.py`, `phenology.py`, `ising_model.py`, `llm_arbiter.py`, `geochem.py`, `retrieval.py`, `ttt.py`, `sam_extractor.py`, `pseudo_labeler.py`, `pav_solver.py`, `postprocess.py`, `submission.py`, `tile_filter.py`, `tiling.py`, `model_runner.py`, `image_loader.py`, `self_healing.py`, `types.py`, `config.py`. |
| `src/models/`             | ML     | `bioclip.py`, `bioclip_multitask.py`, `dinov2.py`, `dinov3.py`, `convnext.py`, `ensemble.py`, `sahi.py`, `vit_backbone.py`, `hotspot.py`, `biomass/`, `layers/`, `uncertainty/`. |
| `src/data/`               | ML     | `dataloader.py`, `wds_loader.py`, `shard_manager.py`, `hash_indexer.py`, `auto_discover.py`, `collage_generator.py`, `preprocess.py`, `sam_extractor.py`. |
| `src/evaluation/`         | ML     | `evaluator.py`.                                                                                |
| `src/config/`             | ML     | Pydantic config schemas (`loader.py`, `registry.py`, `schema.py`).                             |
| `configs/`                | ML     | Training/inference YAMLs: `p1_extract.yaml`, `p2a_warmup.yaml`, `p2b_*.yaml`, `inference*.yaml`, `training.yaml`, `high_res_*.yaml`, `experimental/`, `datasets.yaml`, `experiment.yaml`, `quick_test.yaml`. `cluster.example.yaml` is **Already** (lives in `coordinator/configs/`). |
| `tools/audit/`            | ML     | 25 model / data / weight audit scripts.                                                        |
| `tools/baseline_infer/`   | ML     | Pivot inference scripts: `infer_crt.py`, `infer_tiles_adaptive.py`, `model_i002.py`, `model_010.py`, `phenology_rerank.py`, `fetch_phenology.py`, `ensemble_logits.py`, `postprocess_la_sweep.py`, `transforms.py`, `utils.py`. |
| `tools/data/`             | ML     | 21 dataset-construction scripts (taxonomy graph, ecological DB, EIVE matrix, environmental priors, PAV tree, phylo adjacency, seasonality prior, species prototypes, traits, WorldClim, DALI index, val index, anchors, collage harmonisation, etc.). |
| `tools/inference/`        | ML     | Calibration, SAHI predict, conformal, F1 simulation, 3D quadrat viz, RRF.                      |
| `tools/modeling/`         | ML     | DeepSpeed-to-pth, Optuna optimisation, threshold optimisation, checkpoint rescue, species health audit. |
| `tools/submission/`       | ML     | CSV fixers + submission validators.                                                            |
| `tools/infrastructure/`   | ML     | `vram_watchdog.py`, `pulsar.py` (heartbeat emitter), `multi_seed_sprint.sh`, `balance_shards.sh`. |
| `tools/legacy/`           | ML     | Old SAM extractors. Drop into the R&D repo's own `legacy/`.                                    |
| `submissions/`            | ML     | Submission CSV outputs.                                                                        |
| `models/`                 | ML     | `registry.json` + checkpoint store. (See coupling note 3 below.)                               |
| `paper/`                  | Decide | A separate CVPR-style draft (`plantclef2026_research_paper.tex`). Probably belongs alongside the active R&D repo, not the frozen one. |
| `tests/`                  | ML     | Architectural / dynamic-loader / extension / core / optim tests.                               |
| `data/`                   | ML     | 9 GB of raw datasets + Kaggle splits. **Probably do not move - point both repos at a shared location.** |

### Clearly Product

| Path                      | Tag      | Notes                                                                                          |
|---------------------------|----------|------------------------------------------------------------------------------------------------|
| `apps/`                   | Product  | `oracle-plant/` (full Vite + React + Three.js + Firebase frontend, ~866 MB with `node_modules`), and stubs `oracle-bird/`, `oracle-fungi/`, `oracle-land/`, `oracle-marine/`. |
| `src/apps/`               | Product  | Currently empty - placeholder for Python service backends to the apps.                         |
| `assets/`                 | Product  | 5.7 MB of dashboard assets.                                                                    |
| `scripts/`                | Product  | `redeploy.sh`, `redeploy_frontend.sh`.                                                         |
| `.firebase/`              | Product  | Firebase cache.                                                                                |

### Shared / already-ported

| Path                      | Tag      | Notes                                                                                          |
|---------------------------|----------|------------------------------------------------------------------------------------------------|
| `src/cluster/`            | Already  | Cluster manifest module. Already at `coordinator/python/cluster/` in this repo. **Pick an owner** (suggested: R&D, with Product consuming via env vars). |
| `src/setup/launch_oracle.sh` | ML    | Multi-phase launcher. The coordinator README here documents a `torchrun` drop-in; the active R&D repo can keep the full launcher since the phase modules go with it. |
| `src/setup/` (rest)       | Mixed    | `download_datasets.sh`, `feature_extract.sh`, `cleanup_models.sh`, `seafile_downloader.py`, `upload_to_kaggle.py`, `submit_predictions_kaggle.sh` -> **ML**. `runpod_full_setup.sh`, `setup_environment.sh`, `_remote_setup.sh`, `QUICKSTART.md`, `init_db.sql` -> **Shared** (infra setup; pick the canonical home). |
| `src/setup/oracle_*`      | Drop     | Compiled Go binaries; rebuild from source.                                                     |
| `orchestrator/`           | Already  | Go control plane (`main.go`, `telemetry.go`, `expert/`, `inf/`). Already at `coordinator/go/` here. Same ownership call as `src/cluster/`. |
| `engines/cpp_cuda/`       | ML       | Custom CUDA kernels (Fused GFAM, SAHI tiling, Retinex). Belongs in R&D.                        |
| `engines/rust/`           | Already  | The seven Rust crates. Already at `engines/rust/` here (renamed). **Pick the canonical copy.** |
| `engines/native/`         | Decide   | `classical_ai/` (A* / Minimax / K-Means C++) and `haskell/` are for symbolic-reasoning research; `data_auditor/` is Rust. Probably ML.                |
| `docs/`                   | Mixed    | See per-file split below.                                                                      |
| `legacy/`                 | Mixed    | Old launchers, configs, indexer, patches, inference v1, control script, watchdog. The patches and old inference configs are ML-history; nothing here is Product. **ML, but stash into the R&D repo's own `legacy/`.** |
| `archive/`                | Decide   | 41 MB. Contains an older snapshot (`INDEX.md`, `Infastructure/`, `src/`, `experiments/`, `eda/`, `report/`, `tools/`, `website/`) - looks like a previous PlantCLEF2026 layout. Probably collapses entirely once we accept the frozen `~/Research/PlantCLEF2026`. |
| `.git`, `.github`         | Per-repo | Each repo needs its own.                                                                       |
| `.compile_cache`, `__pycache__`, `.pytest_cache`, `.playwright-mcp`, `.antigravitycli`, `.claude` | Drop | Caches and tool configs. |

### `docs/` per-file split

| File                                    | Tag      | Notes                                                                              |
|-----------------------------------------|----------|------------------------------------------------------------------------------------|
| `ARCHITECTURE_UML.md`                   | ML       | Class diagrams of the training/inference stack.                                    |
| `ASYMMETRIC_DISTILLATION.md`            | ML       | Dual-teacher distillation design.                                                  |
| `CODEBASE_GUIDE.md`                     | ML       | Source-tree guide; mostly the Python ML packages.                                  |
| `DB_ENGINEERING_PROGRESS.md`            | ML       | Ecological / botanical DB build status.                                            |
| `HIGH_RES_STRATEGY.md`                  | ML       | Training resolution strategy.                                                      |
| `LLM_CONTEXT.md`                        | ML       | LLM-arbiter design.                                                                |
| `NOVELTY_BACKLOG.md`                    | ML       | Already mirrored into `BACKLOG.md` here.                                           |
| `OPTIMIZATIONS_LOG.md`                  | ML       | Kernel / throughput notes.                                                         |
| `PIPELINE.md`                           | ML       | End-to-end training pipeline doc.                                                  |
| `SESSION_HANDOFF_2026-05-22.md`         | ML       | Working note; probably ML.                                                         |
| `SQ_ROOT_SPACE_PROGRESS.md`             | ML       | Williams 2502.17779 angle; ML.                                                     |
| `TPU.md`                                | ML       | TPU launch guide.                                                                  |
| `UNIVERSAL_NITRO_ROADMAP.md`            | Decide   | Title sounds product-ish; inspect before classifying.                              |
| `VARIATIONAL_RESEARCH_STRATEGY.md`      | ML       | Research planning.                                                                 |
| `CLOUD_RUN_DEPLOY.md`                   | Product  | Deployment of the frontend / API.                                                  |
| `MULTI_ORACLE_ARCHITECTURE.md`          | Decide   | Sounds like the multi-domain product expansion (plant / bird / fungi / marine / land). Probably Product. |
| `ROADMAP.md`                            | Product  | Already in repo root.                                                              |
| `PROGRESS.md`                           | Drop     | Stale.                                                                             |
| `README.md`                             | Decide   | Split per repo.                                                                    |
| `challenge_overview.md`, `dataset_description.md`, `evaluation.md`, `submission_format.md` | ML | PlantCLEF-task references. |
| `inference_experiments_report.md`       | ML       | Experiment write-up.                                                               |
| `research_notes.txt`                    | ML       | Working notes.                                                                     |
| `todo.md`                               | Drop     | Working notes.                                                                     |

## Cross-cutting coupling to decide

These are the boundary calls that determine how clean the split is.

1. **The cluster manifest + Go hub** (`coordinator/`). Both ML training and the Product dashboard read its telemetry today. Recommend: **R&D owns it.** Product consumes the HTTP telemetry API via the URL only (no source dependency). The `CLUSTER_CLI_PATH` env-var hook already keeps the Go hub project-agnostic.

2. **The Rust engines** (`engines/rust/`). They preprocess data for training but the Product backend never decodes 1.4 M images. Recommend: **R&D owns it.** Already mirrored here.

3. **The model registry** (`models/registry.json` + the `ModelArtifact` / `ModelRegistry` classes in `src/config/registry.py`). Training writes; the dashboard reads to show progress. Two viable models:
   - **R&D owns the registry; Product reads via a thin HTTP endpoint** the orchestrator already exposes (`/api/registry`). Cleanest.
   - **Shared schema package** that both repos depend on. Heavier.

4. **The `plantclef.py` CLI.** Today it's the single launch surface. Splitting means the Product dashboard's `/api/launch` endpoint can no longer call `./plantclef.py` directly. The `CLUSTER_CLI_PATH` env var already provides the indirection; just point the Product copy at the R&D CLI's installed location.

5. **The training/inference data** (`data/`, ~9 GB). **Do not move with either repo.** Treat it as a host-local cache; both repos point at the same path.

6. **The CVPR paper draft** (`paper/`). If you intend to keep writing on PlantCLEF, this belongs alongside the R&D repo. If it's done, archive it.

7. **The `archive/` directory.** Looks like a duplicate of an older PlantCLEF2026 tree. If `~/Research/PlantCLEF2026` is the canonical paper artifact, this can probably be deleted from Oracle entirely.

## Suggested next moves

Once you've reviewed the table above and resolved the **Decide** rows, the move can run in three passes:

1. **Pure-ML pass** (no Product touch points): `phases/`, `src/training/`, `src/inference/`, `src/models/`, `src/data/`, `src/evaluation/`, `src/config/`, `configs/` (except `cluster.example.yaml`), `tools/audit/`, `tools/baseline_infer/`, `tools/data/`, `tools/inference/`, `tools/modeling/`, `tools/submission/`, `tools/infrastructure/`, `tests/`, `engines/cpp_cuda/`, `engines/native/`, `submissions/`, `models/`, ML `docs/*`, `plantclef.py`. -> `~/Research/plantclef`.

2. **Shared-with-owner-pick pass**: `src/cluster/`, `orchestrator/`, `engines/rust/`, the registry, `launch_oracle.sh`, `src/setup/`. -> R&D unless you want Product to own a piece.

3. **Product-only pass** (no ML): `apps/`, `firebase.*`, `firestore.rules`, `assets/`, `scripts/`, `ROADMAP.md`, product `docs/*`, dashboard images. -> Stay in Oracle.

Each pass should land in its own commit so any later cherry-pick or revert is clean.
