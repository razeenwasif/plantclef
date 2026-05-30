# Research Proposal: BioCLIP, DINOv2 & ConvNeXt-V2 Ensemble for High-Resolution Multi-Label Plant Identification in Vegetation Plots

**Date:** March 24, 2026 (Updated for Blackwell V2)

## 1. Title
**BioCLIP, ConvNeXt-V2 & DINOv2 Ensemble with Blackwell-Native LoRA Adaptation for High-Resolution Multi-Label Plant Identification**

## 2. Research Questions
1. How can domain-specific foundation models (BioCLIP) be adapted to identify rare species in complex, overlapping vegetation plots?
2. Does Blackwell-native FP8 compute provide a significant throughput advantage without degrading botanical identification accuracy?
3. Can high-rank LoRA ($R=64$) achieve 80%+ accuracy on 7,800 classes where lower-rank methods plateau?
4. Does GPU-resident feature caching eliminate I/O bottlenecks in foundation model adaptation?

## 3. National Interest Statement
Automating biodiversity monitoring is critical for climate resilience. This research provides environmental organizations with high-throughput tools to monitor ecosystems, using the latest Blackwell hardware to process millions of images with unprecedented efficiency.

## 4. Introduction
Ecological reality consists of complex "vegetation plots" where species overlap. This project builds a high-performance Blackwell-native AI system using biological foundation models and high-capacity parameter-efficient fine-tuning.

## 5. Research Problem
1. **The Domain Gap:** Models trained on single plants fail in dense plots.
2. **The Long-Tail Problem:** Rare species are often ignored by generic AI.
3. **Blackwell Utilization:** Standard PyTorch pipelines fail to fully saturate the massive compute potential of the RTX 5090.

## 6. Research Design & Methodology

### A. Feature Fusion Ensemble
Three complementary backbones:
1. **BioCLIP (ViT-L/14):** Taxonomic expert, fully frozen.
2. **DINOv2 (ViT-L/14):** Geometric/structural expert, LoRA-adapted.
3. **ConvNeXt-V2 (Large):** Local context expert, LoRA-adapted.

**Blackwell Refactoring:** Projection layers are refactored into a single **Grouped GEMM** to maximize Tensor Core utilization and reduce kernel launch overhead.

### B. Blackwell-Native Precision
- **FP8 Training:** Utilizing `float8_e4m3fn` for forward and backward passes, effectively doubling throughput relative to BF16.
- **Max-Autotune Compilation:** Application of `torch.compile` with Triton backend to generate optimized Blackwell kernels.

### C. GPU-Resident Data Engineering
- **VRAM Caching:** The entire 22.5GB Phase 1 feature cache is loaded directly into the 5090's 32GB VRAM. This eliminates all PCIe and Disk I/O bottlenecks.
- **NVIDIA DALI:** Full GPU image decoding and augmentation for Phase 2.

### D. Phase 1: Feature Caching & Deep Head Warmup
- **High-Fidelity PCA:** Compression retention increased to 1024 components (from 512) to preserve fine-grained signals.
- **Deep Linear Probe:** 3-layer MLP (2048-1024-7800) for complex non-linear feature mapping.
- **Strategy:** 20 epochs of rapid head training on frozen, GPU-resident features.

### E. Phase 2: Blackwell-Native LoRA Fine-Tuning
- **High-Capacity Adaptation:** LoRA Rank increased to $R=64$ ($\alpha=128$) for 7,800-class discrimination.
- **Natural Sampling:** Optimizing for competition distribution with 1,000,000 samples per epoch.
- **Pure GPU Training:** Zero DeepSpeed CPU offloading to keep optimizer states in 32GB VRAM.
- **Target:** 80% validation accuracy within 5 epochs.

### F. Inference & Optimization
- **SAHI Tiling:** Overlapping 512× patches with Max Pooling.
- **Logit Adjustment:** Calibration for 7,800-class frequency bias.
- **Cross-Entropy:** Logit-adjusted Softmax for robust single-label "winner-take-all" classification.

## 7. Technical Stack Summary
- **Backbones:** BioCLIP, DINOv2-L, ConvNeXt-V2-L.
- **Fine-Tuning:** LoRA via PEFT (R=64, α=128).
- **Compilation:** `torch.compile(mode="max-autotune")`.
- **Precision:** FP8 (`e4m3fn`), BF16, TF32.
- **Hardware:** NVIDIA RTX 5090 (32GB), Blackwell Architecture.
