# Changelog

All notable changes to the PlantCLEF 2026 codebase after the paper was
submitted. Dates are in `YYYY-MM-DD`.

## [Unreleased] - 2026-06-11

Adds a torch-free smoke path so the full Nexus loop (request → spawner
→ shim → telemetry → monitor → dashboard) can be exercised end-to-end
in ~20 s on a machine with no torch and no GPU, and hardens how the
shim locates `torchrun`. No model, data, or schedule changes;
standalone `torchrun` invocations are untouched. (Nexus-side
counterparts — the Start-Run modal's "Smoke test" toggle, the
`start_smoke_pod.sh` launcher, the full-tail OOM-classification fix,
and the sweep-child retry gating — are in the Nexus CHANGELOG.)

### Nexus integration

- **Add `src/nexus_smoke.py`** — stdlib-only fake trainer driven by
  the *real* `src/training/telemetry.py`. Binds rank 0 / world 1
  (phase from `PLANTCLEF_PHASE`), emits `run.start` with a small
  snapshot, a 5 s heartbeat, then one `step` event per step
  (deterministic decaying loss, ramping `local_acc`) every
  `--step-interval` seconds (default 0.6) for `--steps` steps
  (default 25), writing `{"step": i}` to
  `<output-dir>/checkpoints/last.pt` every 5 steps, then
  `validation.end` (acc 0.87) and `run.end` status `ok`. `--resume`
  continues from the checkpoint's step + 1. `--oom-at <step>` prints
  a realistic multi-line `torch.cuda.OutOfMemoryError` traceback
  (containing the literal `CUDA out of memory`) to stderr and exits 1
  *without* a `run.end` — exactly the shape Nexus's failure
  classification and auto-retry key on. Unknown flags are ignored
  (`parse_known_args`), so spawner-injected args like `--batch-size`
  are accepted as logged no-ops.
- **`nexus_entry.py` grows a consumed `--smoke` flag.** When set, the
  shim launches `sys.executable src/nexus_smoke.py` with the same env
  mapping, cwd, and exit-code propagation as the real path, instead
  of `torchrun` — the smoke trainer is stdlib-only, so whatever
  python runs the shim works.
- **Robust `torchrun` resolution (review fix).** The non-smoke path
  no longer assumes a bare `torchrun` on `PATH`: resolution order is
  `$PLANTCLEF_TORCHRUN` if set, else `<repo>/.venv/bin/torchrun` if
  it exists, else `shutil.which("torchrun")`; if none resolve, the
  shim prints one clear line to stderr ("torchrun not found; set
  PLANTCLEF_TORCHRUN or install torch in `<repo>/.venv`") and exits 3
  before spawning anything. `--print-cmd` dry-runs keep working on
  torch-less boxes — an unresolved launcher prints the placeholder
  `torchrun` instead of exiting.

## [Unreleased] - 2026-06-10

Wires the repository into Nexus (the Firestore-mediated training
control plane) so single-node runs of the shared multitask trainer can
be launched and monitored from the Nexus dashboard. No model, data, or
schedule changes; standalone `torchrun` invocations keep working
exactly as before — telemetry is soft-imported and defaults its sink to
`none` (no files written, no heartbeat thread, no atexit handler) unless
the run is launched under Nexus (`PLANTCLEF_RUN_ID_TEMPLATE` /
`PLANTCLEF_TELEMETRY` set), and binding is guarded so a sink failure can
never abort training.

### Nexus integration

- **Add `src/nexus_entry.py`** — adapter shim used as the Nexus pod's
  `--trainer-cmd`. Consumes the spawner-injected flags (`--name`,
  `--phase`, `--seed`, `--telemetry-dir`, optional `--dataset` →
  `--train-image-root`), passes everything else through verbatim to
  `shared/bioclip25_multitask/train.py`, maps the Nexus env protocol
  onto ours (`PLANTCLEF_TELEMETRY_DIR`, `PLANTCLEF_SEED`,
  `PLANTCLEF_NAME`, `PLANTCLEF_PHASE`,
  `CLUSTER_HOST_ID=$ORACLE_HOST_ID`, `PLANTCLEF_TELEMETRY=file`,
  `PYTHONPATH` including the repo root so `src.training.telemetry`
  imports under torchrun), and launches single-node
  `torchrun --standalone` with inherited stdio so worker OOM text
  reaches Nexus's stderr capture. `--print-cmd` /
  `PLANTCLEF_DRYRUN=1` print the resolved argv + env without running.
- **Instrument `shared/bioclip25_multitask/train.py`** with the JSONL
  telemetry the Nexus run monitor tails (mirroring the pattern already
  in `src/training/trainer.py`): `run.start` with a config/git
  snapshot after distributed setup, `step` (step + smoothed loss) on
  the step-log cadence, `validation.end` (top-1 acc) after each
  validation pass, informational `checkpoint.save` after each
  `last.pt` write, `run.end` with status `ok`/`failed` (the training
  body is wrapped so exceptions emit `failed` and re-raise), plus a
  15 s heartbeat thread. Imports are soft — the trainer runs unchanged
  when the telemetry module or env vars are absent.
- **Nexus-compatible run-id template bridge.** The shim forwards
  Nexus's `ORACLE_RUN_ID_TEMPLATE` verbatim as
  `PLANTCLEF_RUN_ID_TEMPLATE` (already honoured by
  `src/training/telemetry.py`'s `{rank}` substitution), so per-rank
  JSONL filenames keep the `oracle_…_r{rank}_…` stem that Nexus's
  `run_monitor.py` globs (`oracle_*.jsonl`) and rank-strips — every
  rank aggregates into the single Firestore run doc the spawner
  pre-created, with zero changes on the Nexus monitor side.

## [Unreleased] - 2026-05-30

Structural split into two research streams (`quadrat/` and
`single_plant/`) plus a shared model+training core (`shared/`). No
paper numbers change; this is a reorganisation that keeps PlantCLEF and
PlantNet work cleanly separable while continuing to share one BioCLIP
2.5 multitask implementation.

### Structural reorganisation

- **Add `shared/bioclip25_multitask/`** — the BioCLIP 2.5 multitask
  model + dataset + transforms + training loop (formerly
  `experiments/i002_bioclip25_cap_image/`). Both research streams now
  import from here, so the model definition has a single source of
  truth and either stream can pick up multitask improvements without
  copy-pasting.
- **Add `shared/tools/cache_features.py`** (formerly
  `tools/cache_features_i002.py`) and **`shared/tools/score_submission.py`** —
  utilities used by both streams. The feature-cache builder is the
  load-bearing 50x head-only speedup for 24 GB GPUs; the scorer
  computes Macro F1 plus the full diagnostic plot suite (top-K curve,
  per-species F1 histogram, prediction distribution, family confusion,
  train-count vs accuracy, top confused pairs).
- **Add `quadrat/`** — everything PlantCLEF-specific: the headline
  anchor inference script, the phenology pivot, paper checkpoints,
  the 2026 test set, prior PlantCLEF experiments, and the per-experiment
  writeups under `quadrat/docs/exp_reports/`.
- **Add `single_plant/`** — everything PlantNet-specific: the new
  whole-image inference script, the PlantNet-300K manifest, the
  PlantNet output directories, the manifest builder, and PlantNet-side
  experiment writeups.
- **Refactor `src/` to be a shim layer.** `src/train.py`,
  `src/inf_quadrat.py`, `src/inf_quadrat_phen.py`, and
  `src/inf_single_plant.py` are thin `runpy` shims forwarding to the
  real implementations. Repository-root entry points remain stable for
  anyone working from the README.
- **Add `single_plant/inf_script_whole.py`** — new whole-image
  single-plant inference. Loads a `shared/bioclip25_multitask`
  checkpoint, runs softmax-mean across one or more centre-crop
  resolutions (default 224 + 336), writes a
  `image_id, top1_species_id, top1_prob, topk_species_ids` CSV that
  `shared/tools/score_submission.py` can score directly. Deliberately
  strips the tile / anchor / LA / adaptive-K logic that only matters
  for quadrats.

### Path / reference fixes from the move

- Update `EXP_DIR` in `quadrat/inf_script.py`,
  `quadrat/inf_script_phen.py`, and `shared/tools/cache_features.py`
  to resolve to the new shared location.
- Update docstrings + the training-startup banner to say
  `shared/bioclip25_multitask` instead of the old
  `experiments/i002_bioclip25_cap_image` path.
- Update `README.md` to describe the new layout and document all four
  entry points.

## [Unreleased] - 2026-05-29

Audit pass to bring the repository state into agreement with
`report/main.tex` and make the codebase navigable for the course
assessment. No paper numbers change; this is an alignment + cleanup
release.

### Paper-code alignment

- **Rewrite `src/inf_script_phen.py`** to implement the full
  seasonal-phenology pipeline described in Appendix C: 4x4 grid tiling
  at scales 1.0 *and* 0.8, ExG vegetation filter (drop tiles below
  15% green pixels), entropy-weighted Bayesian aggregation
  `w_t proportional to exp(-H_t) * ExG_t`, and a circular Gaussian
  day-of-year prior built from the GBIF month histograms
  (sigma = 18 d, epsilon = 0.05) combined multiplicatively in log
  space at beta = 1.0. The previous version had only a simplified
  month-bin multiplicative prior and would crash before producing
  output (`amp_autocast` arg-count bug, missing 336-px positional
  embedding resampling).
- **Replace `src/train.py`** with a thin `runpy` shim that forwards
  the command line to the actual paper training script at
  `experiments/i002_bioclip25_cap_image/train.py`. The previous
  `src/train.py` was an unrelated DALI + DeepSpeed + LoRA ensemble
  pipeline that the paper does *not* describe.
- **Resolve hardcoded `/workspace/scratch_space_arjun/...` paths**
  in `src/inf_script.py` and `src/inf_script_phen.py` to a path
  computed relative to `__file__`, so the scripts find the i002
  sources on any host that preserves the repository layout.
- **Reconstruct the missing `data/metadata_utils.py`** for the i002
  and i003 experiments. The i002 / i003 `train.py`, `dataset.py`,
  `verify_dataset_cap.py`, and `prepare_combined_manifest.py` all
  import from `data.metadata_utils`, but the file had never been
  committed; the paper training script crashed at import time. The
  reconstruction (`apply_max_images_per_species_cap`,
  `apply_max_train_rows_cap`, `load_metadata_csv`,
  `print_species_distribution`, `build_weighted_sampler`,
  `_pick_col`, `_sniff_delimiter`, and the `_*_CANDIDATES`
  constants) is driven by the import surface of the call sites and
  matches the bin-histogram format used in the existing experiment
  reports.

### Repository layout

- **New `legacy/`** directory at the repo root, with a `README.md`
  explaining what's in it and that none of it is used by the paper
  code. Contents:
  - `legacy/{config.py,data,models,training,extensions}` - the
    abandoned triple-backbone (BioCLIP + DINOv2-Large +
    ConvNeXt-V2-Large) LoRA-ensemble pipeline. Moved with `git mv`
    so commit history follows the files.
  - `legacy/{research_proposal.md,todo.md,background.txt}` - the
    initial-plan documents that describe the ensemble approach the
    paper does not use.
- **New top-level `engines/rust/`** with seven Rust crates
  (`train_resizer`, `test_resizer`, `train_packer`, `val_packer`,
  `fused_resizer_packer`, `train_indexer`, `val_indexer`) that
  implement the SIMD-parallel preprocessing layer the project
  actually used to resize and shard the 2.65 M-image manifest. Each
  crate is built on Rayon + `fast_image_resize` + the `tar` crate;
  the train resizer also exposes a C-FFI library variant. `target/`
  build artefacts are excluded; `Cargo.lock` files are kept where
  they existed.
- **New top-level `coordinator/`** with the multi-node training
  control plane that the project used for the parallel-pod
  experiments behind the paper's appendix (within-backbone
  saturation diagnostics, the i003 GBIF tail-augmentation arm,
  the cross-checkpoint ensembling sweep). Contents:
  - `coordinator/python/cluster/` - typed cluster-manifest module
    (`Manifest`, `HostEntry`, `TrainingTask`, `load_manifest`); one
    third-party dep (`pyyaml`).
  - `coordinator/go/` - Go control plane (no external deps): TCP
    master/worker NK/GO/HB handshake (`main.go`), UDP heartbeat
    receiver + telemetry HTTP API (`telemetry.go`), and standalone
    `expert/` and `inf/` launchers. All three Go packages build
    clean under `go 1.22`.
  - `coordinator/configs/cluster.example.yaml` - multi-host
    topology example.
  - Environment variables use a `CLUSTER_*` prefix and the
    project-specific CLI path is lifted to a `CLUSTER_CLI_PATH`
    env var. The original launcher script and Python CLI are not
    ported because they route through phase modules that do not
    exist in this repo; the coordinator's README documents a
    `torchrun` drop-in.
- **New `CHANGELOG.md`** (this file).
- **New `BACKLOG.md`** documenting the future-work directions
  (saturated / high-priority / frontier) that would have shipped
  with more time.
- **Trim `requirements.txt`** to the dependencies actually needed by
  the i002 paper pipeline (`torch`, `torchvision`, `open_clip_torch`,
  `pandas`, `numpy`, `Pillow`, `tqdm`, `python-dotenv`, `wandb`).
  Dropped the legacy stack (`deepspeed`, `peft`, `nvidia-dali-cuda120`,
  `cudf-cu12`, `cupy-cuda12x`, `albumentations`, `opencv-python`,
  `torchmetrics`, `accelerate`, `timm`, `scipy`, `scikit-learn`,
  `segment-anything`, `sahi`); they live in `legacy/` now.
- **Rewrite top-level `README.md`** to describe the i002 paper
  system: layout map, final-system spec (backbone, heads, schedule,
  optimiser, augmentations, inference recipe), two-stage `torchrun`
  invocation, anchor and phenology inference commands, and notes for
  assessors on which directories carry paper code vs. development
  trace.
- **Delete junk files**: empty `0.5` placeholder, scratch `PROGRESS.md`.

### Verified, no changes

The following pieces of the paper code were audited and confirmed to
already match the report. No edits required.

- `experiments/i002_bioclip25_cap_image/{model,dataset,train,
  transforms,utils}.py`: per-head MLPs (LayerNorm -> Linear(1024 ->
  1024) -> GELU -> Dropout 0.2), last-N transformer blocks plus
  `ln_post` + `proj` unfreeze, auxiliary loss weights 0.30 (genus) +
  0.15 (family), bf16 AdamW with head LR 1e-4 and backbone LR 1e-6,
  one-epoch linear warmup then cosine decay to 1% of peak, stratified
  10% validation split (seed 42; species with fewer than five images
  stay in train), augmentation pipeline.
- `experiments/i001_data_download/final_data.ipynb`: confirmed
  the 1,408,033 + 1,245,748 = 2,653,781 row, 7,806-species manifest
  counts reported in Table 1.
- `experiments/i002_*/scores_*.csv`: confirmed every public-Macro-F1
  number quoted in Tables 4-6 (tau x T sweep, fixed top-k, gap /
  relative-threshold selection rules) traces directly to a Kaggle
  submission archived in this repo.
- `src/inf_script.py` recipe constants (4x4 grid, 448-px tiles,
  224 + 336 dual resolution, softmax-mean, logit adjustment
  tau = 0.25, adaptive threshold T = 0.03, k in [2, 10]).
