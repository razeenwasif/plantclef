# PlantCLEF 2026 Codebase Guide

This document provides a technical overview of the high-performance plant identification pipeline.

## 1. Architectural Synthesis (Deep Saturation Phase)

The system utilizes an **Adaptive Triple Ensemble** fused via a learned gating network. To maximize species-level recall, we employ a multi-rank LoRA strategy:
- **Primary Foundation:** Rank-512 LoRA for extreme detail and species specificity.
- **Diversity Member (Proposed):** Rank-128 LoRA for high-level generalization and noise smoothing.
- **Ensemble Result:** Probability-space averaging of both ranks during inference.

### Model Components
- **BioCLIP 2 (ViT-L/14):** Taxonomic foundation expert.
- **DINOv3 (ViT-L/14):** Geometric and structural visual grounding expert.
- **ConvNeXt-V2-L:** Local translation invariance expert.

### Advanced Training Strategies
- **Gated Feature Aggregation (GFAM):** A learned gating network predicts per-image weights for the three backbones. Fusion is executed via a custom **Constant-Memory CUDA Kernel**.
- **Multi-Crop Aggregation:** Features are extracted from the **Center and all 4 Corners** of the quadrat images (6x richer representation).
- **Contrastive Warmup:** Zero-shot reliance starts at **70%** and decays to 10% over 5 epochs to seed the model with foundation knowledge.
- **Expert Dropout:** 10% chance to drop an entire backbone during training to force expert independence.
- **Genus-Level Smoothing:** Distributes 10% of target probability mass to all species in the same genus to stabilize gradients.

## 2. High-Performance Systems Stack

To process 1.4M images (160GB) at high resolution (672px), we bypass standard Python bottlenecks.

### Rust Engine (`engines/rust/data_auditor`)
- **Polars Metadata Engine:** Multi-threaded CSV/Feather parsing that memory-maps files for near-instant loading.
- **JWalk Integrity Scanner:** OS-level directory crawling using work-stealing parallelism to audit the dataset in <5 seconds.
- **Taxonomic Filter:** SIMD-accelerated bitset operations for microsecond species compatibility checks.

### I/O & Memory
- **Hybrid RAM Disk:** A background process populates **87GB of `/dev/shm`** with the dataset. The loader prioritizes these RAM-based images for zero-latency I/O.
- **DALI Network Shock-Absorber:** Increased prefetch queues and disabled memory-mapping for network-based files to prevent SIGBUS crashes.

## 3. Numerical Fortress (Stability Suite)

Designed for stable BFloat16 training on **NVIDIA RTX 5090 / PRO 6000** clusters.

- **LogitNorm:** Maps predictions onto a stable unit hypersphere instead of hard-clamping.
- **Lion Optimizer:** Uses evolved sign momentum to provide robust convergence in Phase 1 and 2A, mitigating the noise from extreme class sparsity (7,806 labels).
- **Gradient Centralization:** Centers gradients to zero-mean before the optimizer step to flatten the loss landscape.
- **Coordinated Resilient Bypass:** Distributed `all_reduce` protocol to synchronize batch skips across the entire GPU cluster if NaNs are detected.
- **Atomic Progress Checkpointing:** Saves both weights and the **Full DeepSpeed Engine** (Optimizer/LR) every 5% of the epoch with resume-aware markers.

## 4. Execution (Ultra-PLANTCLEF Protocol)

The system is designed for **Diverse Ensembling** as its standard operating mode. The primary ground-truth entry point is the `plantclef.py` CLI, which orchestrates all backend logic via `src/setup/launch.sh`.

### The Ground-Truth Workflow
Instead of training a single monolithic model, PLANTCLEF trains three independent experts with diverse seeds (`42`, `1337`, `2026`) and merges them at inference time.

#### Unified Phase Launching
```bash
# 1. Foundation Caching (Pre-compute features)
./plantclef.py train --phase p1 --role sprint

# 2. Head Warmup (Train MLP/GCN heads)
./plantclef.py train --phase p2a --role sprint

# 3. Student Distillation (Deep Fine-Tuning)
./plantclef.py train --phase p2b-student --role sprint

# 4. Asymmetric Dual-Teacher Distillation (The Trinity)
./plantclef.py train --phase ad-td --role sprint -- --mode extract
./plantclef.py train --phase ad-td --role sprint -- --mode train
```

#### Standard Multi-Seed Inference
```bash
./plantclef.py infer --ensemble
```
## 5. Centralized Orchestration (Mission Control)

The system has been overhauled to provide a single, type-safe command center.

### Unified Configuration (`src/config/`)
All hyperparameters and hardware settings are defined in **Python Dataclasses**. 
- **`schema.py`:** Defines the strict structure for Hardware, Model, and Training settings.
- **`loader.py`:** A deterministic loader that auto-probes GPU hardware (Blackwell/5090) and applies performance overrides at runtime.
- **`__init__.py`:** Exports legacy variables to ensure backward compatibility for all scripts.

### Model Registry (`src/config/registry.py`)
Instead of raw filesystem scanning, PLANTCLEF uses a formal **JSON Registry** to track model artifacts. This ensures the dashboard and ensemble inference pass always use verified, healthy checkpoints.

### plantclef CLI (`plantclef.py`)
The unified entry point for all system operations.
- **Training:** `./plantclef.py train --phase p2b --seed 42`
- **Inference:** `./plantclef.py infer --ensemble`
- **Registry:** `./plantclef.py registry --list`

## 6. Real-Time Observability

### Pulsar Telemetry (`tools/infrastructure/pulsar.py`)
A zero-latency heartbeat system that broadcasts training vitals (Loss, FPS, Temperature, VRAM) via UDP to the master orchestrator.

### Neon Dashboard (`dashboard/`)
A hyper-futuristic React command center served by the Go orchestrator. It provides real-time cluster-wide monitoring, hardware HUDs, and remote mission dispatching.

---

### Hardware-Aware Optimization (PLANTCLEF Core)
...

The pipeline is designed for absolute performance on Blackwell/Zen4 clusters:
- **Rust Data Auditor:** High-speed directory indexing and metadata parsing.
- **AVX-512 Hot Paths:** Branchless species masking using `_mm512_mask_storeu_ps`.
- **Assembly Prefetching:** Manual L1-cache warming using inline `prefetcht0`.
- **Atomic RAM Shield:** Zero-latency initialization for 1.4M image collections.
- **Blackwell FP8 Surge:** Direct `e4m3fn` precision for 2x tensor core throughput.
- **Parallel Stream Orchestration:** Simultaneous multi-backbone execution.
