# PlantCLEF 2026 - Master TODO & Roadmap

## Phase 1: Foundations & Training
- [x] **Triple Ensemble:** BioCLIP, DINOv2-L, ConvNeXt-V2-L backbones.
- [x] **Configurable Backbones:** "5090" / "4090" hardware modes.
- [x] **BF16 & TF32 Precision:** High-throughput Blackwell tensor core utilization.
- [x] **DALI Pipeline:** Full GPU image decode/resize/augment; no CPU bottleneck.
- [x] **Residual MLP Head:** Upgraded with skip connections and Multi-Sample Dropout.
- [x] **High-Fidelity PCA:** 1024 components to preserve fine-grained detail.
- [x] **Auto-Skip Phase 1:** Training skips Phase 1 if a compatible 1024-d checkpoint exists.
- [x] **Fast Cache Warmup:** 10-minute head training bypass using pre-extracted features.
- [x] **Validation Feature Caching:** Instant validation passes (~1s) during warmup epochs.

### RTX 5090 Blackwell Optimization (Completed)
- [x] **Parallel Backbone Streams:** simultaneous execution of all 3 backbones via custom CUDA Stream Orchestrator.
- [x] **Fused Slotted Projection:** Zero-copy feature fusion kernel; eliminates `torch.cat()` bottlenecks.
- [x] **VRAM Balancing:** Optimized Batch 128 / Accumulation 4 for 32GB 5090.
- [x] **Inductor Max-Autotune:** `torch.compile` enabled for Blackwell native kernels.

## Phase 2: Synthetic Complexity & The Long-Tail
- [x] **High-Rank LoRA ($R=64$):** Increased adaptation capacity for 7,800-class discrimination.
- [x] **Stratified Sampling:** Balanced training set (max 500/species) to boost Macro-F1.
- [x] **Asymmetric Loss (ASL):** Fused CUDA kernel for multi-label long-tail classification.
- [x] **Lookahead Optimizer:** Finding flatter minima for better test generalization.
- [x] **Taxonomic Distillation:** Knowledge transfer from BioCLIP (KD_ALPHA=0.3).
- [ ] **Taxonomic Hierarchy Loss (CUDA):** Custom kernel to penalize cross-family misclassifications more heavily than cross-genus.
- [ ] **Uncertainty-Aware Rare Species Detection:** C++/CUDA implementation of predictive variance to improve recall on the tail.

## Phase 3: Domain Adaptation & High-Res Tiling
- [ ] **Pseudo-Labeling:** Teacher-Student loop on unlabeled LUCAS quadrats.
- [x] **CUDA SAHI Engine:** VRAM-native image slicing and fused max-pooling (10x speedup).
- [ ] **Test-Time Augmentation (CUDA):** Fused kernel for multi-view (flip/rotate) inference with zero-latency aggregation.
- [ ] **Vegetation Plot Structure Modelling:** Spatial attention kernel to model the 3D layout of plants within the quadrat.

## Phase 4: Metric Optimization & Ecological Guardrails
- [x] **C++ Taxonomic Filter:** Bitset-based co-occurrence validation (runs in <1ms).
- [ ] **Ecological Co-occurrence Prior (C++):** Bayesian prior engine using community-wide co-occurrence matrices.
- [ ] **Cross-Modal Phenological Reasoning:** CUDA-fused attention between visual features and seasonal/climate metadata.
- [ ] **Metadata Integration (GPS/Soil/Altitude):** High-speed C++ lookup for geographic constraints.
- [ ] **Logit Ensemble Weighting (CUDA):** Learnable, per-class backbone weighting for optimal ensemble blending.
- [ ] **Threshold Optimization:** Brent's Method for per-class F1 thresholds.
- [ ] **Final Ensembling:** Model Soup or multi-model averaging.

---
*Last updated: March 30, 2026*
