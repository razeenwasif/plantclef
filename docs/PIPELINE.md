# PlantCLEF 2026 - PLANTCLEF Training & Inference Pipeline

This document describes the state-of-the-art "PLANTCLEF" pipeline designed for the PlantCLEF 2026 competition, optimized for high-resolution quadrat identification on Blackwell (RTX 5090) hardware.

---

## 🚀 Hardware Stack
*   **Cluster:** 4x Pods (8x NVIDIA GeForce RTX 5090 32GB).
*   **CPU:** 96-Core AMD EPYC (Preprocessing & Orchestration).
*   **Storage:** 3.6TB Local NVMe Cache + Shared Network Volume.
*   **Precision:** Native FP8 Forward Surge + BFloat16 Gradients.

---

## 🛠️ Phase 1: Foundation Caching (Sub-1-Hour)
*   **Goal:** Convert 1.4 million images into a 20-30GB `.pt` feature cache.
*   **Speed:** ~400 images/second across the 8-GPU cluster.
*   **Logic:** Backbones are frozen; only final pooling layers are active.
*   **Output:** `/workspace/models/phase1_feature_cache.pt`.
*   **Launch:** `./plantclef.py train --phase p1 --role sprint`

---

## 🧬 Phase 2a: Head Warmup (The "Speed of Light")
*   **Goal:** Train the MLP and GCN heads on extracted features.
*   **Duration:** ~2 seconds per epoch.
*   **Optimization:** Fused Lion optimizer + RAM-disk caching.
*   **Accuracy:** Targets >95% training accuracy before unfreezing backbones.
*   **Launch:** `./plantclef.py train --phase p2a --role sprint`

---

## 🔥 Phase 2b-expert: Expert Specialization (Deep Saturation)
*   **Goal:** Heavy LoRA fine-tuning of specialized backbones (BioCLIP or DINOv3).
*   **Launch:** 
```bash
./plantclef.py train --phase p2b-expert --role sprint --config configs/p2b_teacher_bioclip.yaml
./plantclef.py train --phase p2b-expert --role sprint --config configs/p2b_teacher_dinov3.yaml
```

---

## 🎓 Phase 2b-student: Student Distillation
*   **Goal:** LoRA (R=512) fine-tuning of the Triple-Backbone Ensemble with Knowledge Distillation from the specialists.
*   **Strategy:** **High-Res Teacher / Low-Res Student Distillation.**
    *   **Teachers:** Two frozen SWA-averaged experts (`expert_bioclip_512`, `expert_dinov3_512`) at 512px.
    *   **Student:** Curriculum from 224px → 448px → 512px, inherits fine-grained features via KL divergence.
*   **Launch:** `./plantclef.py train --phase p2b-student --role sprint`

---

## ⚡ Data Infrastructure (The Perfect Stack)
To eliminate the JPEG decoding and network I/O bottlenecks, the following stack is implemented:

1.  **Rust-Resizing:** Images are pre-resized to 700px using a SIMD-optimized Rust swarm (550+ imgs/sec).
2.  **WebDataset Sharding:** Data is packed into high-speed `.tar` shards to convert random I/O into sequential streams.
3.  **SSD Satiation:** Shards are prefetched to the local 3.6TB NVMe drive (`/tmp/shards`) at pod launch.
4.  **nvJPEG ASIC Decoding:** DALI uses the Blackwell 5090's hardware decoders (`device="mixed"`) for 100% GPU-based decompression.
5.  **Elastic Hash Registry:** We utilize a custom `XXHash` Funnel Hashing index to maintain an $O(1)$ amortized, non-reordering lookup table for 1.4M+ paths. This ensures zero VRAM/RAM metadata overflow and no Garbage Collection stutters in low-RAM WSL2 environments.

---

## 🚀 plantclef Launch Sequence (Industrial)

### Complete Pipeline

```
[Phase 1] foundation_caching (p1)
    ↓
[Phase 2a] head_warmup (p2a)  ←── parallel ──→  [Expert Specialization] (p2b-expert)
    ↓                                                   ↓
[Phase 2b-student] student_distillation  ←─ uses ─  [Teacher cache] (one-time build)
    ↓
[Phase 2.5] ad-td (Asymmetric Dual-Teacher Distillation)
    ↓
[Infer] SWA → threshold opt → calibration → ensemble inference
```

---

### 1. Foundation Caching (Phase 1)
Build the foundation feature cache (`phase1_feature_cache.pt`):
```bash
./plantclef.py train --phase p1 --role sprint
```

### 1b. Expert Specialization (Phase 2b-expert)
Trains the frozen teacher models (`expert_bioclip_512`, `expert_dinov3_512`):
```bash
./plantclef.py train --phase p2b-expert --role sprint --config configs/p2b_teacher_bioclip.yaml
```

### 2. Head Warmup (Phase 2a)
Trains MLP + GCN head on the feature cache:
```bash
./plantclef.py train --phase p2a --role sprint
```

### 3. Student Distillation (Phase 2b-student)
Multi-seed student training with Knowledge Distillation:
```bash
./plantclef.py train --phase p2b-student --role sprint
```

### 3.5 Asymmetric Dual-Teacher Distillation (Phase 2.5 - "The Trinity")
Trains a high-efficiency DeiT student using taxonomic (BioCLIP) and structural (DINOv3) teachers.

```bash
# 1. Pre-compute teacher logits (BioCLIP + DINO)
./plantclef.py train --phase ad-td --role sprint -- --mode extract --variant elite

# 2. Train the dual-token student
./plantclef.py train --phase ad-td --role sprint -- --mode train --variant elite
```

### 4. High-Resolution Inference (SAHI)
Inference on multi-species quadrats using a sliding-window pattern. Optimized for sub-100ms latency per tile.
- `NCCL_IB_DISABLE=1` + `NCCL_SOCKET_NTHREADS=2` + `NCCL_NSOCKS_PERTHREAD=4` (TCP tuning)
- `zero_quantized_gradients: true` in DeepSpeed (int8 LoRA grad allreduce, 4× less TCP payload)

### 4.5 Agentic CV: LLM Arbitration (VCoT)
Before mathematical constraints are applied, the pipeline supports an optional **Agentic LLM Arbiter**.
- **Gemma 4 (Local):** Acts as a visual tie-breaker when the ViT ensemble is uncertain (Top 1 and Top 2 are close).
- **Nemotron-3-Super (Local):** Acts as a Logic Integrator to resolve conflicts between visual predictions and ecological math (pH, GDD).
- *Implementation:* `src/inference/llm_arbiter.py`

### 3. High-Resolution Specialists
Train the 512px expert models:
```bash
./launch_high_res_expert.sh
```

## 🏆 Final Submission Recipe
The winning recipe used for the final PlantCLEF 2026 submission:

1.  **Primary Model:** **i002 DINOv3-L** (fine-tuned ViT-L/14) - Our most robust geometric feature extractor.
2.  **Secondary Model:** **cRT Gated Student** (BioCLIP + DINOv3 + ConvNeXt) - Trained using Classifier Re-Training to balance rare species.
3.  **Ensemble Method:** Multi-resolution probability averaging (scales: 448px, 672px).
4.  **Inference Strategy:** Sliding-window tiling (SAHI) with 518px crops and 172px stride.
5.  **Post-Processing:**
    *   **Thermodynamic Phenology:** Seasonal Boltzmann prior (β=1.0).
    *   **Agentic LLM Arbiter:** Gemma 4 tie-breaking for Top-1/Top-2 gaps < 0.15.
    *   **Taxonomic AC-3:** Hard-constraint pruning.

### Launching the Final Submission
```bash
./plantclef.py infer --ensemble --role sprint --config configs/inference.yaml
```

### 5. Automated Hyperparameter Tuning (Optuna)
Once you have generated raw logits during inference, you can optimize the ecological post-processing parameters without re-running the heavy visual backbones.

```bash
# 1. Run inference to generate and cache raw logits
./plantclef.py infer --role sprint --config configs/inference.yaml

# 2. Run Optuna to find the optimal F1 score parameters
python tools/modeling/optimize_optuna.py
```
*Note: This script explores `global_threshold`, `top_k`, `phenology_beta`, and `fw_lambda` using bayesian optimization over the cached logit arrays.*

### 6. Enhanced Ablation Grid
To systematically evaluate the impact of different resolutions, post-processing techniques, and parameter synergies, run the automated ablation suite:

```bash
python tools/audit/run_ablation_full.py
```
This generates a combinatoric grid of configurations, runs inference for each, and aggregates the resulting Macro-F1 scores into `ablation_results.csv`.

---

## 🔮 Future Work: Post-Competition Research

While the core plantclef system is saturated, the following experimental architectures are in active development for the next iteration:

### 1. The "Holy Trinity": Asymmetric Dual-Teacher Distillation (AD-TD)
To achieve elite performance without the massive latency of running three models at once, we use an Asymmetric Distillation (AD-TD) framework.
*   **The Setup:** Use the fine-tuned BioCLIP and DINOv3 "Experts" as teachers.
*   **The Student:** Train a single, hyper-efficient **DeiT (Data-efficient Image Transformer)** student.
*   **Visual Chain-of-Thought:** The student is modified with two unique "distillation tokens." One token learns the biological relationships from BioCLIP, while the second token learns the spatial/structural precision of DINOv3. The result is a single model that "sees" like a cartographer but "reasons" like a botanist.

### 2. Agentic LLM Arbiter (VCoT)
For high-entropy or ambiguous samples, a local LLM (Gemma 4 / Nemotron) performs a **Visual Chain-of-Thought** analysis.
*   **Role:** Acts as a human field-expert to break ties between visually similar species.
*   **Implementation:** Triggered dynamically when Top-1 and Top-2 species confidence gaps are < 0.15.

### 3. Spectral Botanical Vision (GFNet)
Replacing Self-Attention with **Complex-Valued Fourier Filters**.
*   **Physics:** Learns to filter image components in the frequency domain. High frequencies capture sharp botanical details; low frequencies represent background soil noise.
*   **Performance:** $O(N \log N)$ complexity allows for extremely high-resolution training (1024px+) without quadratic VRAM explosions.

---
*   **Heuristic Tile Filtering & GPU Decoding:** OpenCV/NumPy based filtering drops empty patches; `torchvision` provides direct-to-tensor GPU decoding.
*   **Entropy-Gated Multi-Scale TTA:** High-confidence tiles bypass rotations. TTA only applied to uncertain tiles ($p < 0.9$) across `[1.0, 0.75, 0.5]` scales.
*   **Blackwell INT8 & TensorRT:** Model weights quantized to INT8 (`torchao`) and compiled with TensorRT backend for maximal throughput.
*   **Few-Shot Retrieval Augmentation:** Low-confidence predictions ($p < 0.4$) trigger a FAISS nearest-neighbor search using BioCLIP embeddings to vote and bump logits.
*   **Bayesian-Veg Aggregator:** Fusion of prediction entropy and Excess Green (ExG) vegetation index to filter noise.
*   **Neuro-Symbolic AC-3:** Post-processing consistency filter using the taxonomic graph.
*   **RRF Blending:** Reciprocal Rank Fusion ($k=60$) for final model ensembling.

---

## 🔬 Physics-Informed Preprocessing (Retinex)

Field quadrat images suffer from spatially non-uniform illumination (canopy shadows, specular highlights, sky gradients). **Multi-Scale Retinex (MSR)** is applied between float32 conversion and backbone normalization inside `_TileDataset._preprocess()`:

- **Formula:** `r(x,y) = log(I) − mean(log(Gσ * I))` for σ ∈ {15, 80, 250} pixels
- **Effect:** Separates surface reflectance from illumination at three frequency scales
- **Output:** Per-channel min-max normalized to [0,1], backbone-compatible
- **Config:** `use_retinex: true`, `retinex_sigmas: [15.0, 80.0, 250.0]` in YAML

Enable via config YAML:
```yaml
architecture:
  use_retinex: true
  retinex_sigmas: [15.0, 80.0, 250.0]
```

---

## 🧬 Phylogenetic Graph Construction (Offline, Once)

Build the 7806×7806 phylogenetic adjacency matrix before training:
```bash
python tools/data/build_phylo_adj.py
# Output: data/phylo_adj.npy  (~230MB, float32)
```

**Weighting scheme (RBF on taxonomic distance):**
| Relationship | Weight |
|:---|:---|
| Same genus | 1.000 (exp(−0)) |
| Same family | 0.368 (exp(−1)) |
| Different family | 0.135 (exp(−2)) |

The `EcologicalGCNHead` auto-detects `data/phylo_adj.npy` at init and blends: `A_final = 0.7 × A_eco + 0.3 × A_phylo`.

---

## ⚡ Frank-Wolfe with Island Biogeography Prior

The postprocessing `FrankWolfeSolver` now incorporates a **MacArthur-Wilson species count prior** (K̄ ≈ 8 species per 0.25m² quadrat):

```
effective_gradient = logits − 2λ(||y||₁ − K̄)
```

When the current solution selects more species than expected, the effective gradient is reduced uniformly, biasing the LMO toward sparsity. Parameters:
- `species_count_k: 8.0` — expected species richness per quadrat
- `count_lambda: 0.05` — penalty strength

---

## 🏆 Elite Post-Processing & Refinement (The "Winning Tier" Strategies)
These advanced strategies are designed to push the F1 score beyond the 33.45 private F1 benchmark by leveraging biological priors and high-precision segmentation.

### 1. Dual-Scale Tiling (Global vs. Local Context)
Instead of a fixed grid, inference uses a two-pass approach to capture multi-scale botanical traits:
*   **Fine-Scale Pass:** 518×518px windows with 172px stride (~144 tiles/image). Captures textures, hairs (trichomes), and leaf serration.
*   **Coarse-Scale Pass:** 732×732px windows on the same grid, down-scaled to 518px. Captures larger inflorescences and growth habits that span multiple fine crops.
*   **Aggregation Rule:** **Max-Pooling Across Scales.** We take the single highest confidence observed for every species in either stream. Probabilities are not summed or averaged, ensuring that a sharp "botanical hit" is never diluted by a blurred view.

### 2. GroundingDINO + SAM Noise Masking (+2.64pp Gain)
Non-botanical noise (rulers, fingers, labels, roulette wheels) is the largest source of systematic error.
*   **Detection:** GroundingDINO is prompted with: *"stone, shell, roulette, plastic, metal, hand, ice, snow, measure, ruler, wood, board, paper"*.
*   **Segmentation:** SAM (Segment Anything) produces pixel-accurate masks for detected objects.
*   **Soft Penalty Formula:** Confidence scores are rescaled by: $p' = p(1 - 0.35m)$, where $m$ is the fraction of noise pixels in the window. 
*   **Logic:** Windows are only discarded if $m=1$ (fully covered); otherwise, they are softly penalized.

### 3. Niche Overlap Filtering via EIVE Ellenberg Indicators (+1.28pp combined)
An upgrade to the binary AC-3 solver that utilizes continuous ecological indicators (Light, Temp, Moisture, Soil pH, Oraclegen).
*   **Metric:** Niche Similarity Index $S$ (Mahalanobis distance kernel + Bhattacharyya coefficient).
*   **Refinement:**
    *   **Pruning:** Species with mean similarity $<0.015$ to other predictions are removed.
    *   **Recovery:** Reserve list species with mean similarity $>0.750$ are promoted.
*   **Effect:** This acts as "Soft Belief Propagation" on ecological indicator space, ensuring quadrat coherence.

### 4. Cross-Year Plot Propagation (+0.12pp)
Leveraging the temporal redundancy in the dataset:
*   If multiple images of the same plot exist across different years (determined via filename metadata), the highest-confidence species from one year are propagated to the other years.

---

## 📈 Summary of Efficiency Gains
| Optimization | Speedup | Notes |
| :--- | :--- | :--- |
| **WebDataset + SSD RAM-disk** | 5.0× | Zero I/O wait, sequential shard reads |
| **Rust Preprocessing** | 8.0× | SIMD-parallel resize at 550+ img/s |
| **Teacher Logit Cache** | 3.0× | Eliminates 2× full teacher forward per step |
| **torch.compile (student + teachers)** | 1.6× | Triton kernel fusion on Blackwell |
| **Curriculum-Adaptive Teacher Res** | 1.2× | 448px vs 512px in first 5 epochs |
| **ZeRO++ int8 Grad Allreduce** | 1.4× net | 4× TCP payload reduction across 4 pods |
| **NCCL TCP Tuning** | 1.1× | Multi-socket NCCL on Ethernet |
| **4-Pod DDP** | ~3× | Near-linear scaling with ZeRO Stage 2 |

**Phase 2b Epoch Time Targets:**

| Configuration | Steps/epoch | Time/epoch |
|:---|:---|:---|
| 1 GPU, no cache (baseline) | 1,039 | ~75 min |
| 1 GPU + teacher cache | 1,039 | ~26 min ✓ |
| 4 GPU + teacher cache | 260/GPU | ~9 min |
