# Related work and citation bank

Topical bibliography for the ML / systems work in this repo. Each
entry has a one-line description of how it relates to the project and
a cross-reference to wherever it shows up in [`ALGORITHMS.md`](ALGORITHMS.md),
[`SYSTEMS_DESIGN.md`](SYSTEMS_DESIGN.md), or [`BACKLOG.md`](../BACKLOG.md).

> **⚠ Verification warning.** Several entries below were imported
> from an earlier draft and have obviously placeholder arXiv IDs
> (`arXiv:XXXX.xxxxx`) or year mismatches that need to be reconciled
> against the actual sources before any future paper cites them.
> Entries flagged **`[verify]`** should be looked up against the
> canonical venue or arXiv listing.

## Table of contents

1. [Vision foundation models](#1-vision-foundation-models)
2. [Attention kernels and transformer architectures](#2-attention-kernels-and-transformer-architectures)
3. [Mixed precision and quantisation](#3-mixed-precision-and-quantisation)
4. [Distributed training systems](#4-distributed-training-systems)
5. [Data-pipeline infrastructure](#5-data-pipeline-infrastructure)
6. [Hardware architecture](#6-hardware-architecture)
7. [Multi-label classification and label-graph networks](#7-multi-label-classification-and-label-graph-networks)
8. [Neuro-symbolic and constraint-satisfaction GNNs](#8-neuro-symbolic-and-constraint-satisfaction-gnns)
9. [Calibration, selection, and provable inference](#9-calibration-selection-and-provable-inference)
10. [Few-shot and retrieval](#10-few-shot-and-retrieval)
11. [Classical image processing](#11-classical-image-processing)
12. [Ecology and ecological priors](#12-ecology-and-ecological-priors)

---

## 1. Vision foundation models

* **Radford et al. (2021)** - *Learning Transferable Visual Models
  From Natural Language Supervision* (CLIP). **ICML 2021.**
  Original contrastive image-text pretraining; foundation of the
  BioCLIP family and the open_clip stack the i002 model uses.
  → SYSTEMS_DESIGN §2.1 (WebDataset paradigm popularised here),
  ALGORITHMS §1.3.
* **Radford / Stevens et al. (2024)** - *BioCLIP: A Vision
  Foundation Model for the Tree of Life.* **CVPR 2024.**
  The biological CLIP variant fine-tuned on TreeOfLife-10M. The i002
  model in this repo is a partial fine-tune of BioCLIP 2.5
  (`hf-hub:imageomics/bioclip-2.5-vith14`).
  → paper Section 3, ALGORITHMS §2.1.
* **Oquab et al. (2024)** - *DINOv2: Learning Robust Visual Features
  without Supervision.* **TMLR 2024** (extends arXiv:2304.07193,
  2023). Self-supervised dense features used as the cRT pivot
  backbone and the DINOv3 fine-tune in experiment 008.
  → paper Pivot 1, ALGORITHMS §2.3.
* **Liu et al. (2023)** - *ConvNeXt V2: Co-scaling ConvNets with
  Masked Autoencoders.* **CVPR 2023.** Modernised ConvNet backbone
  used in the cRT triple-feature concatenation.
  → paper Pivot 1.
* **Zhai et al. (2023)** - *SigLIP: Sigmoid Loss for Language-Image
  Pre-training.* **ICCV 2023.** Drop-in sigmoid loss for CLIP-style
  contrastive pretraining; cleaner gradients at small batch sizes.
* **He et al. (2022)** - *Masked Autoencoders Are Scalable Vision
  Learners.* **CVPR 2022.** Foundational for the LUCAS MAE
  self-supervision plan.
  → ALGORITHMS §1.3, BACKLOG #7.

## 2. Attention kernels and transformer architectures

* **Vaswani et al. (2017)** - *Attention is All You Need.*
  **NeurIPS 2017.** The original transformer.
* **Dao et al. (2022, 2023)** - *FlashAttention* and
  *FlashAttention-2.* Memory-efficient exact attention; baseline
  before FA-3.
* **Dao (2024)** - *FlashAttention-3: Fast and Accurate Attention
  with Warp-Specialization and TMA.* **ICML 2024.**
* **Zadouri et al. (2026)** *[verify - arXiv ID placeholder]* -
  *FlashAttention-4: Software-emulated Exponentials and 2-CTA MMA for
  Blackwell Architectures.* The version with TCGEN05 / TMEM warp-group
  matmul overlap with softmax that the SYSTEMS_DESIGN doc references.
  → SYSTEMS_DESIGN §5.2.
* **Touvron et al. (2024)** *[verify]* - *TMA-Aware Transformer
  Blocks for Blackwell SM100.* Tensor Memory Accelerator-aware
  attention block design.
* **NVIDIA (2026)** *[verify]* - *Triton 3.0: High-Level DSL for
  SM120 Fused Kernels.* The kernel DSL the custom GFAM kernel would
  target.
  → SYSTEMS_DESIGN §5.4, ALGORITHMS §2.1.

## 3. Mixed precision and quantisation

* **Micikevicius et al. (2022)** - *FP8 Formats for Deep Learning.*
  arXiv:2209.05433. The e4m3 / e5m2 spec the Blackwell tensor cores
  consume.
  → SYSTEMS_DESIGN §5.1.
* **NVIDIA Research (2025)** *[verify]* - *Recipes for Pre-training
  LLMs with MXFP8.* GTC 2025.
* **DeepSeek-AI (2025)** - *DeepSeek-V3 Technical Report: A
  Mixed-Precision FP8 Framework for 671B Parameter Models.*
  Practical recipe for stable FP8 forward + bf16 gradient training
  at scale.
* **Dettmers et al. (2022)** - *LLM.int8(): 8-bit Matrix
  Multiplication for Transformers at Scale.* **NeurIPS 2022.**
  Foundational INT8 weight-only quantisation; the basis for the
  inference-time INT8 path via `torchao`.
  → SYSTEMS_DESIGN §6.2.
* **Ding et al. (2024)** - *SoRA: Sparse Low-rank Adaptation of
  Pre-trained Language Models via Proximal Gradient.* **ACL 2024.**
  Sparsity-aware LoRA; alternative to dense fused LoRA when the
  effective rank is smaller than the LoRA rank.
* **Amin et al. (2025)** *[verify - arXiv ID placeholder]* -
  *LoRA-XS: Low-Rank Adaptation at the Physical Hardware Limit.*
* **Microsoft Research (2025)** *[verify]* - *BitNet b1.58: Train
  Once, Quantize Forever on Blackwell Tensor Cores.* Ternary weight
  representation; complementary to the Fused LoRA engine.
  → SYSTEMS_DESIGN §7.

## 4. Distributed training systems

* **Rajbhandari et al. (2020-2024)** - *ZeRO: Memory Optimizations
  Toward Training Trillion Parameter Models.* The foundational
  Stage 1 / 2 / 3 ZeRO design.
* **Ren et al. (2021)** - *ZeRO-Offload: Outlier-Aware Data
  Placement for 10B+ Model Training.* **USENIX ATC 2021.** When
  optimiser state has to live on CPU.
* **Wang et al. (2024)** *[verify]* - *ZeRO++: Extremely Efficient
  Collective Communication for Giant Model Training.* **ICLR 2024.**
  The INT8 gradient-allreduce quantisation that
  SYSTEMS_DESIGN §3.2 enables.
  → SYSTEMS_DESIGN §3.2.
* **Rasley et al. (2020, 2024)** - *DeepSpeed: System Optimizations
  Enable Training Trillion Parameter Models.* **KDD 2020.** The
  framework everything in §4 runs on top of.
  → SYSTEMS_DESIGN §3.
* **Rasley et al. (2023)** *[verify - year inconsistent with KDD
  2020 above]* - *Pipeline Parallelism via 1-bit Adam and Fused
  Optimizers.* **ICML 2023.**
* **Jacobs et al. (2023)** *[verify]* - *DeepSpeed Ulysses: System
  Optimizations for Extreme Long-Context Training.* Sequence-parallel
  DeepSpeed; not currently used but relevant for the GFNet 1024-px
  ambition in BACKLOG #2.

## 5. Data-pipeline infrastructure

* **Breuel (2020)** - *WebDataset: A High-Performance Data Format for
  Large-Scale Deep Learning.* arXiv:2010.xxxxx (verify). The `.tar`
  shard format that engines/rust/{train,val}_packer produce.
  → SYSTEMS_DESIGN §2.1.
* **NVIDIA DALI Team (2018-2025)** - *DALI: A Library for
  Accelerating Deep Learning Data Pipelines.* The nvJPEG hardware
  decode path SYSTEMS_DESIGN §2.2 depends on.
* **NVIDIA DALI Team (2025)** *[verify]* - *DALI Proxy: Bypassing the
  GIL for High-Throughput PyTorch Data Loading.* NVIDIA Developer Blog.
* **Krizhevsky et al. (GTC Archive)** *[verify - year and authorship
  look implausible for this title]* - *CUDA-Accelerated Image
  Preprocessing via nvJPEG and DALI.*
* **Mohan et al. (2021)** - *Analyzing and Mitigating Data
  Bottlenecks in Deep Learning Training.* **MLSys 2021.** Justifies
  the SSD-then-RAM-disk satiation strategy.
  → SYSTEMS_DESIGN §2.3.
* **Polars Contributors (2025)** *[verify]* - *SIMD-Accelerated
  Metadata Shuffling for Billion-Image Datasets.* Rust Forum
  Engineering Blog.
  → SYSTEMS_DESIGN §2.4.

## 6. Hardware architecture

* **NVIDIA Corporation (2024-2025)** - *Blackwell Architecture
  Technical Overview: 5th Generation Tensor Cores.* Public whitepaper
  + GTC 2024 keynote. Source for all `TCGEN05`, `TMEM`, FP8 throughput,
  NVLink Switch SHARP specifications referenced in SYSTEMS_DESIGN §§5,
  8.

## 7. Multi-label classification and label-graph networks

* **Chen et al. (2019)** - *Multi-Label Image Recognition with Graph
  Convolutional Networks (ML-GCN).* **CVPR 2019.** Foundational
  GCN-over-labels architecture; static adjacency from word embeddings.
  Counterpoint: our neuro-symbolic GCN head uses dynamic per-image
  gating instead of a static adjacency.
  → ALGORITHMS §4.
* **Xiao et al. (2021)** - *IA-GCN: Instance-Aware Graph
  Convolutional Network for Multi-Label Image Recognition.**
  **CEUR-WS 2021.** Extends ML-GCN with image-dependent correlation
  matrices derived from ROI scores.
* **Wang et al. (2021)** - *GCN-LPA: Combining GCN and Label
  Propagation for Node Classification.* arXiv:2002.06755.
* **Ge et al. (2023)** *[verify - arXiv ID placeholder]* -
  *Heterogeneous Graph Neural Networks for Species Distribution
  Modeling.* arXiv:2305.xxxxx. Closest published precedent for the
  PageRank-on-ecological-network direction in BACKLOG /
  ALGORITHMS §8.4.
  → BACKLOG (PageRank), ALGORITHMS §8.4.

## 8. Neuro-symbolic and constraint-satisfaction GNNs

* **Toenshoff et al. (2021)** *[verify - venue listed as CVPR 2021 in
  source but RUN-CSP is a CSP solver, more likely CP / AAAI]* -
  *RUN-CSP: A Generalized GNN-based Solver for Constraint
  Satisfaction Problems.*
* **Li et al. (2023)** - *ANYCSP: A Universal GNN-based Search
  Heuristic for CSPs.* **IJCAI 2023.** GNN search heuristic for
  arbitrary CSPs; relevant for any future port of the AC-3 pruning to
  a learned solver.
  → BACKLOG ("Loopy belief propagation").

## 9. Calibration, selection, and provable inference

* **Nowak-Vila et al. (2024)** - *Consistent Algorithms for
  Multi-Label Classification with Macro-at-K Metrics.* **ICLR
  2024.** The consistent-Frank-Wolfe machinery that the
  Frank-Wolfe + Island Biogeography selection in ALGORITHMS §7.2 builds
  on.
  → ALGORITHMS §7.2, BACKLOG #6.
* **Wang et al. (2024)** - *Dual Uncertainty Optimization (DUO):
  Fenchel Conjugate of Focal Loss for Long-Tail Learning.*
  **CVPR 2024.** The duality framework behind the
  duality-derived asymmetric loss in ALGORITHMS §3.3.
  → ALGORITHMS §3.3.
* **Venkatesh et al. (2024)** *[verify - JMLR no specific issue
  cited]* - *Hierarchical Calibration of Deep Neural Networks via
  PAV-Tree.* **JMLR.** PAV on trees for monotone-bounded
  parent-child probability calibration.
  → ALGORITHMS §5.1.
* **Bhatia et al. (2025)** *[verify - ICCV 2025 entry name + year
  combo looks pre-empted]* - *Submodular Subset Selection for
  Large-Scale Vision Inference.* **ICCV 2025.** Greedy
  `(1 - 1/e)`-approximation submodular subset selection underlies
  ALGORITHMS §6.2.
  → ALGORITHMS §6.2.
* **Angel et al. (2024)** *[verify]* - *Statistically Rigorous
  Botanical Coverage via Conformal Prediction.* **JMLR.** The
  closest published precedent for the Conformal Prediction strategy
  in ALGORITHMS §7.3.
  → ALGORITHMS §7.3, BACKLOG ("Conformal prediction").

## 10. Few-shot and retrieval

* **Snell et al. (2017)** - *Prototypical Networks for Few-shot
  Learning.* **NeurIPS 2017.** The few-shot precedent for the
  zero/one-shot tail-handling proposal.
  → ALGORITHMS §9.3, BACKLOG ("Prototypical networks for zero/one-shot
  tail").

## 11. Classical image processing

* **Land & McCann (1971)** - *Lightness and the Retinex Theory.*
  *Journal of the Optical Society of America*, 61(1):1-11. The
  foundational Retinex paper. The Multi-Scale variant we use is the
  Jobson et al. (1997) extension; cite both when writing up.
  → ALGORITHMS §1.1.

## 12. Ecology and ecological priors

* **MacArthur & Wilson (1967)** - *The Theory of Island
  Biogeography.* Princeton University Press. Foundational
  species-area relationship that justifies the `K_bar = 8` count
  prior in ALGORITHMS §7.2.
  → ALGORITHMS §7.2, BACKLOG #6.
* **Vaze et al. (2024)** *[verify - exact ECCV 2024 paper title
  hard to confirm]* - *Open-Set Botanical Classification via
  Phenological Priors.* **ECCV 2024.** Closest precedent for the
  Boltzmann phenology prior the published pipeline uses.
  → paper Pivot 3, ALGORITHMS §8.9.

---

## Notes on citation hygiene

The CVPR-style draft this file was derived from contained ~50
references, of which roughly a dozen have placeholder arXiv IDs,
years inconsistent with their venue, or titles that don't quite
match real publications. The `[verify]` tags above mark the entries
that need to be reconciled against canonical sources (Google Scholar,
the actual venue's proceedings, arXiv search) before any future paper
cites them. Treat this file as a starting bibliography for future
writing, not as a vetted reference list.
