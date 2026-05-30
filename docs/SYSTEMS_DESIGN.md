# Systems-design notes

Catalogue of the systems-, hardware-, and distributed-training
techniques considered or implemented across the project. Some are
live in the active codebase (`engines/rust/`, `coordinator/`); others
remain reference notes for future work, in which case
[`BACKLOG.md`](../BACKLOG.md) is the forward-looking pointer and this
file is the *technique* description.

For ML / mathematical techniques (loss functions, calibration,
selection rules, post-processing priors), see
[`ALGORITHMS.md`](ALGORITHMS.md) instead.

## Table of contents

1. [Polyglot system architecture](#1-polyglot-system-architecture)
2. [Data plane](#2-data-plane)
3. [Distributed-training optimisation](#3-distributed-training-optimisation)
4. [Numerical resilience](#4-numerical-resilience)
5. [Hardware-specific compute optimisation](#5-hardware-specific-compute-optimisation)
6. [Inference acceleration](#6-inference-acceleration)
7. [Fused LoRA engine](#7-fused-lora-engine)
8. [Cluster network satiation](#8-cluster-network-satiation)

---

## 1. Polyglot system architecture

Three concurrent execution planes, each in the language best suited
to its workload.

| Plane              | Language     | Role                                                                                  | Code                |
|--------------------|--------------|---------------------------------------------------------------------------------------|---------------------|
| **Compute plane**  | Python + CUDA / C++ | Tensor ops, model fine-tuning, inference. PyTorch + DALI for Blackwell-specific kernel execution. | `phases/`, `src/training/`, `src/inference/`, `engines/cpp_cuda/` (future) |
| **Data plane**     | Rust         | Image resize, WebDataset packing, DALI-index generation, metadata audit. SIMD intrinsics + zero-copy byte streaming, no GIL. | `engines/rust/`     |
| **Control plane**  | Go           | Cluster sync + heartbeat, node-health monitoring, telemetry API. Goroutines + channels avoid the barrier deadlocks common in monolithic DDP. | `coordinator/go/`   |

The split exists because each language eliminates a distinct
overhead: the GIL for I/O parallelism, Python's tracing-allocator
jitter for high-resolution kernels, and Python's blocking sockets for
cluster coordination. Crossing language boundaries is restricted to a
small number of well-typed surfaces (a C-FFI on the train resizer,
`CLUSTER_*` env vars between Go and Python).

---

## 2. Data plane

### 2.1 WebDataset sharding

The 1.4 M-image PlantCLEF 2024 corpus (plus the 2.65 M-row i001
manifest) is converted into 5,000-sample `.tar` shards keyed by
species ID. The pipeline becomes a sequential read rather than a
random access over millions of small files. Same strategy OpenAI used
for CLIP training. See `engines/rust/train_packer` and
`engines/rust/val_packer`.

### 2.2 nvJPEG hardware decoding via DALI

100 % of JPEG decompression runs on the GPU's dedicated nvJPEG silicon
through NVIDIA DALI, eliminating CPU-to-GPU copy latency and allowing
~90 % HBM occupancy on training steps. The Python `DataLoader` in the
i002 paper code is the fallback when DALI is unavailable.

### 2.3 SSD + RAM-disk satiation

At pod launch, the resized 800-px training tree is rsync'd to the
local 3.6 TB NVMe; for hot runs it is additionally synced to
`/dev/shm` so the DALI prefetch queue (8 batches) never blocks. The
1.4 M-file dataset-initialisation scan that historically took 20 min
drops to under 30 s through this *SPEED-INIT* bypass that prefers
`/dev/shm` over network volume.

### 2.4 Rust Polars + zero-copy CSV / metadata

The 2.65 M-row manifest CSV is read through Rust Polars rather than
pandas. Polars' Arrow-backed columnar IO + SIMD shuffling reduces
metadata randomisation for 1.4 M images from ~150 ms to ~3 ms. The
Python training loop reads via the standard pandas surface but the
heavy parse path bypasses the GIL.

### 2.5 mmap teacher logit cache

The 22 GB teacher logit cache (BioCLIP + DINOv3 + ConvNeXt-V2-L
ensemble logits cached for the distillation phase) is opened with
`mmap=True`. Multi-pod read sync becomes near-instantaneous because
the OS page cache, not a userspace copy, is the broadcast surface.

### 2.6 NVIDIA DALI v1.2 indexing

The Rust `train_indexer` / `val_indexer` produce DALI v1.2 `.idx`
files from the shards so a DALI-based loader can start in under one
second on a millions-of-files dataset (the file-listing scan that
would otherwise dominate startup is replaced by a direct seek-table).

---

## 3. Distributed-training optimisation

### 3.1 Dual-mapped feature engine

Combines the Rust Polars zero-copy CSV path (3.4), the mmap teacher
cache (3.5), and the SPEED-INIT `/dev/shm` bypass (3.3) into one
pipeline. The net effect is that a fresh pod can resume a multi-node
training run in under a minute regardless of how many files the
underlying dataset has.

### 3.2 ZeRO++ INT8 quantised gradient communication

DeepSpeed ZeRO Stage 2's all-reduce broadcasts the full bf16 gradient
payload every step. With LoRA-only training (rank R = 512 plus the
GCN head), only a small slice of parameters has live gradients - we
quantise the all-reduce payload to INT8 before transmission
(`zero_quantized_gradients: true`). 4x less TCP traffic per step;
INT8 quantisation noise is well below the optimiser's inherent
stochasticity at the relevant batch scale.

### 3.3 NCCL TCP tuning for Ethernet pods

For pods connected over standard Ethernet (no InfiniBand), the
following NCCL env vars get set whenever `WORLD_SIZE > 1`:

```
NCCL_IB_DISABLE=1                # prevent wasted InfiniBand polling
NCCL_SOCKET_NTHREADS=2           # multiple parallel TCP sockets ...
NCCL_NSOCKS_PERTHREAD=4          # ... per NCCL thread, saturating bandwidth
NCCL_SOCKET_FAMILY=AF_INET
```

Set automatically inside the config loader so individual training
scripts do not have to think about it.

### 3.4 Accumulation-aware barrier reduction

With `gradient_accumulation_steps = 2` under DeepSpeed ZeRO Stage 2
the all-reduce is deferred to every second micro-step, halving
barrier frequency versus a single-GPU baseline and maximising
compute / communication overlap on Ethernet-connected nodes.

---

## 4. Numerical resilience

### 4.1 Coordinated Resilient Bypass (distributed NaN handling)

If any GPU detects an unstable batch (e.g. a corrupt LUCAS frame, a
zero-norm CutMix sample), it broadcasts a 1-byte "skip" signal via
`torch.distributed.all_reduce`. Every rank then drops the same batch
together, preventing the DDP weight divergence that you get when one
rank does a step and the others do not. Cheap (single `all_reduce`
per micro-step) and required for training on 160 GB of
field-collected data with non-trivial bad-row rates.

### 4.2 FP32-isolated loss paths

Backbones run in bf16, but the loss aggregation, logit adjustments,
and label-smoothing arithmetic are all cast to fp32. Bf16 in the
loss path silently miscomputes long-tail gradients (the relative
error on a `log(1e-6)` is much larger in bf16 than in fp32). Almost
free in flops; the loss accounts for far less than 0.1 % of compute.

### 4.3 Taxonomic CutMix regularisation

Standard CutMix blends arbitrary image pairs. For botanical data this
creates "franken-plants" (desert cactus into swamp lily) that punish
the model for not predicting visually impossible classes. Restricting
CutMix to species within the same *genus* forces the model to ignore
shared generic backgrounds and focus on fine-grained morphology -
leaf serration, vein pattern - exactly the trait set needed to
discriminate closely related taxa.

### 4.4 Stochastic Weight Averaging (SWA)

The final 5 epochs of Phase 2B aggregate model weights along the Lion
optimiser trajectory. Smooths the loss landscape, escapes sharp
minima, and lifts long-tail recall in particular. Implemented as a
moving average over snapshots taken every N optimiser steps.

### 4.5 Autonomous VRAM Watchdog

Background process monitors HBM in real time, dynamically scales
batch size / chunking to maintain ~98 % physical saturation
(~94.5 GB / 96 GB on a Blackwell PRO 6000), and crucially
*discriminates between hardware limits and software logic errors* so
it does not endlessly halve the batch size while a real OOM bug
sits in the code path. Lives in `tools/infrastructure/vram_watchdog.py`
(in the active R&D repo).

---

## 5. Hardware-specific compute optimisation

### 5.1 Blackwell FP8 surge

`transformer-engine` running in native `e4m3fn` precision on the
projection and classification heads gives a 2x throughput lift on
tensor cores vs bf16, with dynamic scaling preserving identification
accuracy. Only the heads run in fp8; backbones stay bf16 because the
attention path is more sensitive.

### 5.2 FlashAttention-4 (Blackwell-native)

Uses the `TCGEN05` tensor-core instruction and the Tensor Memory
(`TMEM`) feature for warp-group matrix-multiply overlap with softmax.
1.5x-2x attention throughput on high-res 672-px ViT-L backbones
versus FA3. Loadable as a drop-in replacement for the attention
module on Blackwell.

### 5.3 Parallel CUDA stream orchestration

Custom C++ orchestrator launches the BioCLIP, DINOv3, and ConvNeXt-V2
backbones on three separate CUDA streams in parallel rather than the
default sequential execution. The three forward passes overlap on
the SM scheduler, taking the wall-clock from `sum(t_i)` to
`max(t_i)` for the slowest backbone. Required for the cRT and AD-TD
ensemble paths.

### 5.4 Warp-specialised GFAM fusion

Partition the thread block into specialised warps: *producer* warps
use `cp.async` to stage backbone features into shared memory while
*consumer* warps simultaneously compute the softmax gating weights
and LayerNorm statistics. Hides HBM latency entirely on multi-issue
Blackwell SMs.

### 5.5 L2 cache persistence for the adjacency matrix

The 7,806 x 7,806 ecological adjacency matrix used by the
neuro-symbolic GCN head is ~244 MB. With `cudaAccessPropertyPersisting`
and a 40 % hit-ratio hint, Blackwell's 96 MB L2 keeps the most-active
taxonomic rows resident across graph-traversal passes. ~35 % less
HBM bandwidth for the GCN forward.

### 5.6 Memory allocator hardening

* `jemalloc` as the primary system allocator eliminates the
  fragmentation that glibc's allocator induces under high-concurrency
  DALI pipelines.
* `MKL_DEBUG_CPU_TYPE=5` forces MKL onto the "Zen path" on EPYC.
* `cuda_malloc_async` (the stream-ordered allocator) removes the
  synchronisation point at every intermediate buffer allocation,
  letting Blackwell stay in pure async execution.

### 5.7 Zen 4 AVX-512 branchless filtering

The Rust data-auditor and Polars metadata engine are compiled with
the `znver4` target. Branchless taxonomic filtering uses AVX-512
intrinsics (`_mm512_mask_storeu_ps`) so misprediction penalties
during large-scale species masking disappear. Inline `prefetcht0`
assembly hides the memory latency of the 6,656-d feature lookup.

### 5.8 Atomic RAM indexing

160 GB dataset, 4 ranks each scanning the same tree, race conditions
guaranteed. Atomic write-and-rename indexing strategy: one rank
builds the index file under a `.tmp` extension, all ranks `rename`
to the canonical path - the FS guarantees atomicity at single-byte
granularity. Reduces the per-rank initialisation walk to <1 s
regardless of file count.

---

## 6. Inference acceleration

### 6.1 Vectorised multi-scale TTA

Standard TTA executes sequentially: identification -> hflip ->
vflip -> rotate. Each pass blocks. We construct a *composite batch*
containing four views (original, hflip, vflip, 180-deg rotate) of
every tile at multiple zoom levels (1.0 / 0.75 / 0.5) and process
the whole thing in one hardware pass. The species probabilities
average across orientation- and scale-invariant features without the
sequential latency tax.

### 6.2 torch.compile + CUDA Graphs (`reduce-overhead`)

Kernel fusion (BioCLIP / DINOv3 attention blocks fused into monolithic
kernels), TensorRT backend compilation, and static graph capture into
CUDA Graphs. The static graph eliminates the per-step launch overhead
that dominates at small inference batches. Combined with weight-only
INT8 quantisation via `torchao`, the per-tile latency drops to a
small constant.

### 6.3 Entropy-gated TTA with CUDA Graphs

The full 4-view TTA only triggers on high-entropy tiles. Confident
tiles bypass TTA and ride a static CUDA Graph for the single forward
pass; uncertain tiles trigger a separate static graph capturing the
4-view batch. Compute is allocated dynamically without paying the
launch overhead of a dynamic graph.

### 6.4 Async GPU tiling + heuristic filter

Image decode goes straight to GPU tensors via `torchvision`. Before
the encoder runs, a Laplacian-variance + Excess-Green (ExG) heuristic
drops empty tiles (sky, dirt, sampling-frame edges). 20-40 % fewer
tiles enter the encoder; throughput goes up by the same factor. The
ExG vegetation aggregation that uses these per-tile vegetation scores
in the prediction-pooling step is documented in
[`ALGORITHMS.md`](ALGORITHMS.md).

---

## 7. Fused LoRA engine

Standard LoRA reads three weight matrices per layer per forward pass:
the frozen backbone `W`, the down-projection `A`, and the up-projection
`B`. The HBM traffic for `W` dominates because it is by far the
largest.

In Phase 2B (LoRA-only fine-tune, backbone frozen) we pre-compute the
fused matrix once per optimiser step:

```
W_fused = W + (B @ A) * (alpha / r)
```

All forward passes within that step reduce to a single GEMM against
`W_fused`. HBM bandwidth drops ~60 %.

The backward path stays correct by way of a custom
`torch.autograd.Function`: the gradient of `W_fused` decomposes as a
sparse update on `A` and `B` only - the frozen `W` never needs to be
re-read or its gradient stored. Net effect: the EPYC CPUs no longer
stall the GPU memory bus.

---

## 8. Cluster network satiation

For multi-pod training on Blackwell NVLink-Switch fabric:

### 8.1 NVLink SHARP (in-switch reduction)

`NCCL_NVLS_ENABLE=1` offloads the all-reduce summation logic from the
GPU SMs to the NVLink Switch's physical fabric. LoRA gradient
synchronisation for payloads under 256 MB becomes near-free.

### 8.2 Tree reduction topology

`NCCL_ALGO=Tree` combined with `NCCL_LAUNCH_MODE=PARALLEL` minimises
point-to-point hops across a 4-GPU cluster. ~22 % lower synchronisation
latency than ring reduction at this scale.

### 8.3 GPUDirect RDMA satiation

```
NCCL_NET_GDR_LEVEL=5     # maximally aggressive memory-mapping
NCCL_MAX_NCHANNELS=4     # one channel per physical GPU
NCCL_BUFFSIZE=16777216
```

Aligns NCCL's software-level collective streams with the physical
Blackwell hardware lanes - the channels exactly match the GPU count
so neighbouring ranks never contend for the same lane. Eliminates HBM
bandwidth thrashing during gradient sync.

---

## Cross-references

* [`BACKLOG.md`](../BACKLOG.md) - the future-work directions that
  call into these techniques: AD-TD distillation (5.3 + 5.4), GFNet
  (5.2), Frank-Wolfe selection (3.2), uncertainty-gated reasoning
  (6.3).
* [`ALGORITHMS.md`](ALGORITHMS.md) - GFAM, neuro-symbolic GCN, HTL,
  PAV-tree calibration, Multi-Scale Retinex, ExG vegetation
  aggregation, and the other principled ML / math techniques whose
  systems implementations are catalogued here.
* `coordinator/README.md` - the cluster-coordinator subsystem the
  distributed-training optimisations in §3 and §8 ride on top of.
* `engines/rust/README.md` - the SIMD-parallel Rust implementations
  of the data-plane techniques in §2.
