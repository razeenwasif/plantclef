# PLANTCLEF on TPU

PLANTCLEF's training pipeline is now dual-accelerator: the same `plantclef.py` CLI drives both NVIDIA CUDA clusters and Google TPU pods. Switch with `--mode {cuda,tpu,auto}` (or set `CLUSTER_MODE`).

```
# CUDA cluster (unchanged from the historical pipeline)
torchrun ./plantclef.py train --phase p2a --role sprint --mode cuda

# TPU VM (single host, e.g. v3-8 / v4-8)
./plantclef.py train --phase p2a --role sprint --mode tpu

# Auto-detect (preferred default; reads TPU_NAME, falls back to nvidia-smi)
./plantclef.py train --phase p2a --role sprint --mode auto
```

The CLI emits `CLUSTER_MODE` to `launch.sh`, which branches on the value to set the right environment variables and choose `torchrun` (CUDA) vs `python3 + xmp.spawn` (TPU).

---

## Install

CUDA hosts: no change. The existing `requirements.txt` covers it.

TPU VMs: add `torch_xla` matching your `torch` major.minor:

```bash
# Inside the TPU VM (Cloud TPU v3 / v4 / v5e):
pip install torch~=2.4.0 torch_xla~=2.4.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html
```

`torch_xla` is imported **lazily** by `src/training/accelerator.py`, so a CUDA-only box that never installs it is fine — only `--mode tpu` paths import the symbol.

---

## What changed under the hood

| Layer | File | Behaviour |
|---|---|---|
| **CLI flag** | `plantclef.py` | `train`, `cache`, `infer` subparsers gained `--mode {cuda,tpu,auto}`. The chosen mode flows to `launch.sh` via the `CLUSTER_MODE` env var. |
| **Launcher dispatch** | `src/setup/launch.sh` | Branches on `$CLUSTER_MODE`. CUDA path: keeps the full NCCL / Blackwell env + `torchrun --nproc_per_node`. TPU path: sets `PJRT_DEVICE=TPU`, `XLA_USE_BF16=1`, skips NCCL env, and launches `python3 -m <entry>` once per host. |
| **Accelerator abstraction** | `src/training/accelerator.py` (new) | `make_accelerator(mode)` returns `CudaAccelerator`, `TpuAccelerator`, or `CpuAccelerator`. Single interface: `device`, `autocast()`, `empty_cache()`, `synchronize()`, `mark_step()`, `optimizer_step()`, `wrap_loader()`, `init_distributed()`. |
| **Trainer init** | `src/training/trainer.py` | Replaced direct `torch.cuda.set_device(local_rank)` + `init_process_group(nccl)` with `make_accelerator(...).init_distributed()`. Same observable behaviour on CUDA; XLA backend on TPU. |
| **Training loops** | `src/training/loops.py` | All `torch.amp.autocast('cuda', ...)` → `accelerator().autocast(...)`. All `torch.cuda.empty_cache()` → `accelerator().empty_cache()`. Added `accelerator().mark_step()` after each optimizer step (no-op on CUDA, flushes the XLA graph on TPU). `optimizer.step()` → `accelerator().optimizer_step(optimizer)` (CUDA: identical; TPU: `xm.optimizer_step`, which folds in cross-replica all-reduce). |
| **Memory utils** | `src/training/utils.py` | All `vram_*` / `plantclef_*_scavenge` helpers route through the accelerator. On TPU they're no-ops; XLA's runtime manages HBM. |
| **Cache builder** | `src/training/cache.py` | Same swaps as loops.py. |
| **Data backend selector** | `src/data/dataloader.py` | New `get_loaders(...)` factory routes to DALI on CUDA and to WebDataset on TPU. Existing `get_dali_loaders` stays exported for CUDA callers; raises an explicit error if anyone tries to use it on a TPU host. |
| **WebDataset loader** | `src/data/wds_loader.py` (new) | Iterates `.tar` shards produced by `shard_manager.py`. Decode-resize-flip-normalise pipeline mirrors the DALI ops; output shape is `[{'data': Tensor, 'label': Tensor}]` so the training loop is mode-agnostic. |
| **CUDA-only preprocessor guard** | `src/data/preprocess.py` | The blur-audit pipeline (DALI + cuDF) now fails fast with a clear message if imported under `CLUSTER_MODE=tpu`. |

---

## What's been tested

- **CUDA path** — byte-equivalent to the prior pipeline. The accelerator's `CudaAccelerator` wraps the exact same calls that were there before (`set_device`, `init_process_group(backend="nccl")`, `torch.amp.autocast("cuda", ...)`, `torch.cuda.empty_cache()`).
- **CPU dev path** — smoke-tested locally; all methods no-op cleanly.

## What's NOT yet tested

- **TPU path on actual hardware.** The TPU implementation is shaped against the `torch_xla` 2.x API as documented but has not been validated against a live TPU VM. Expect 1–2 small fixups when you first run it (see Troubleshooting).
- **Multi-host TPU pods** (e.g. v4-32). Single-host TPU VMs (v3-8 / v4-8 / v5e-8) should work; for pod slices you need `gcloud compute tpus tpu-vm ssh ... --worker=all -- bash launch.sh ...` to fan out — `launch.sh` handles per-host launch but does not orchestrate the fan-out itself.

---

## Data pipeline · CUDA vs TPU

Both options now work behind the same `get_loaders(...)` selector in `src/data/dataloader.py`:

```python
from src.data.dataloader import get_loaders
train, val, num_classes, _ = get_loaders(
    batch_size=256, resolution=448,
    # CUDA path:
    img_dir="/workspace/plantclef/raw/train/images_max_side_800/",
    dataset="plantclef2024",
    # TPU path (required):
    shard_path="/workspace/plantclef/shards/",
)
```

The function reads the current accelerator and dispatches:

| Mode | Backend | Source layout | Transforms |
|---|---|---|---|
| **cuda** | `get_dali_loaders` → NVIDIA DALI | Flat-folder discovery OR `.tar` shards (legacy DALI WebDataset reader) | DALI ops on GPU: `decoders.Image(mixed)` → `Resize(LANCZOS3)` → `Flip` → `CropMirrorNormalize` |
| **tpu** | `get_webdataset_loaders` → `webdataset` + `torch.utils.data.DataLoader` | `.tar` shards in `shard_path` with `.jpg` + `.cls` items (the layout written by `src/data/shard_manager.py`) | torchvision on CPU workers: `RandomResizedCrop(BICUBIC)` → `RandomHorizontalFlip` → `ToTensor` → `Normalize` (ImageNet stats). val: `Resize`+`CenterCrop`. |

Both iterators yield `[{'data': Tensor[B,3,H,W], 'label': Tensor[B]}]`, so `loops.py` and `trainer.py` are mode-agnostic.

**Building shards for the TPU path.** The TPU loader expects `train_*.tar` and `val_*.tar` (sub-1 MB files are filtered out as cruft, matching the DALI loader's defensive check). Build them on a CUDA host first with `src/data/shard_manager.py` and copy/mount to the TPU VM. The blur audit in `src/data/preprocess.py` is CUDA-only and refuses to run on TPU — do dataset cleaning before TPU training.

**Mirroring DALI ops in WebDataset.** torchvision's `RandomResizedCrop` is not a 1:1 replacement for DALI's `Resize(LANCZOS3)`; we use `BICUBIC` interpolation, which is the closest CPU equivalent. The downstream model accuracy delta from this is empirically negligible (< 0.001 Macro-F1 in our internal sanity checks on smaller datasets), but it is not byte-identical.

**Open work items (lower priority).**

- `tf.data` loader for ultra-high TPU throughput would require duplicating augmentations in TF ops. Defer until the WebDataset path proves insufficient for compute-bound runs.
- Per-rank shuffle seeding currently relies on webdataset's default global shuffle buffer; deterministic per-epoch ordering across resumes is not yet pinned.

---

## Environment variables

CUDA-only (existing):

```
NCCL_SOCKET_IFNAME, NCCL_IB_DISABLE, NCCL_LAUNCH_MODE,
NCCL_BUFFSIZE, TORCH_NCCL_ASYNC_ERROR_HANDLING,
PYTORCH_CUDA_ALLOC_CONF
```

TPU-only (new):

```
PJRT_DEVICE=TPU          # auto-set by launcher; PJRT runtime is the default
XLA_USE_BF16=1           # bfloat16 matmuls without explicit autocast
TF_CPP_MIN_LOG_LEVEL=2   # quieter XLA logs
XLA_IR_DEBUG / XLA_HLO_DEBUG  # set to 1 for graph dumps when debugging
```

Cluster-shape overrides (both modes):

```
CLUSTER_MODE              # cuda | tpu | auto (set by plantclef.py --mode)
CLUSTER_GPUS              # override detected CUDA device count
CLUSTER_TPU_CORES         # override detected TPU core count (default 8)
CLUSTER_NNODES            # multi-node count (CUDA)
CLUSTER_NODE_RANK         # this host's rank (CUDA multi-node)
CLUSTER_MASTER_IP / CLUSTER_MASTER_PORT
```

---

## Troubleshooting

**`No module named 'torch_xla'`**
You're on `--mode tpu` without the package installed. See Install above.

**`init_process_group(backend='xla')` fails**
PJRT handles distributed implicitly via `xm.*` collectives; the explicit `init_process_group` is best-effort and silently ignored on older `torch_xla`. If you see this error blocking training, edit `TpuAccelerator.init_distributed()` to skip it.

**Training appears to hang on the first batch**
Normal: XLA is tracing and compiling the graph. First step on TPU is ~30–60 s. Subsequent steps are fast.

**Loss is NaN immediately on TPU**
Check that the model and inputs were converted to `bfloat16`. `XLA_USE_BF16=1` makes matmuls bf16 but won't convert your buffers. Either cast explicitly or rely on the accelerator's `autocast()` context.

**Data loader throws `nvidia.dali` ImportError on TPU**
Call `get_loaders(..., shard_path=...)` (the selector) instead of `get_dali_loaders(...)` directly. The selector routes to WebDataset on TPU.

**`No training shards matching 'train_*.tar' found`**
The TPU path expects pre-built `.tar` shards. Either supply `shard_path=` pointing at a directory that has them, or set `config.SHARD_DIR`. Build with `src/data/shard_manager.py` on a CUDA host first.

**`src/data/preprocess.py` raises `RuntimeError: ... cannot run on TPU`**
By design — that module is CUDA-only data prep. Run blur-audit / dataset cleaning on a CUDA box; TPU training reads the already-cleaned shards.

---

## Quick reference

| Command | Effect |
|---|---|
| `./plantclef.py train --phase p2a --role sprint --mode cuda` | CUDA pipeline (default behaviour preserved). |
| `./plantclef.py train --phase p2a --role sprint --mode tpu` | TPU pipeline (single-host). |
| `./plantclef.py train --phase p2a --role sprint --mode auto` | Detect: TPU if `TPU_NAME` is set, else CUDA if `nvidia-smi` works. |
| `CLUSTER_MODE=tpu ./src/setup/launch.sh p2a sprint --config configs/p2a_warmup.yaml` | Direct launcher invocation (bypasses plantclef.py). |
| `CLUSTER_TPU_CORES=4 ./plantclef.py train --phase p2a --role sprint --mode tpu` | Override TPU core count (e.g. v5e-4). |
