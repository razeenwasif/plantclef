# PlantCLEF 2026 - Task Registry

## Completed Systems & Architecture
- [x] **Rust Data Engine:** Polars metadata loader and JWalk scanner implemented.
- [x] **Hybrid RAM Cache:** `/dev/shm` populate script and hybrid loader active.
- [x] **Numerical Fortress v2:** LogitNorm and Gradient Centralization implemented.
- [x] **GFAM Fusion:** Learned gating with constant-memory CUDA kernel.
- [x] **Resilient Checkpointing:** Resume-aware DeepSpeed engine saves (every 5%).
- [x] **Quadrat Optimizations:** Multi-crop feature aggregation (5-crop ensemble).
- [x] **Stabilization:** Expert Dropout and Genus-Level Smoothing active.

## Active Phase: Deep Saturation (Road to 80%)
- [x] Launch high-resolution (672px) training run.
- [x] Phase 2B Multi-Node Scale-Out (8x RTX 5090 Cluster).
- [ ] Enable "Extreme Mode" (SAM + Rank-512 LoRA) for final accuracy push.
- [ ] **Rank-Diversity Ensemble:** Train a second high-throughput model with Rank-128 LoRA to ensemble with the current Rank-512 foundation.
- [ ] Evaluate Diversity Ensembling (Multi-Seed / Cross-Backbone).
- [ ] Monitor validation for 60% and 70% breakthroughs.
- [ ] Execute SWA (Stochastic Weight Averaging) on best 5 checkpoints.
- [ ] Run Brent's Method for per-class threshold optimization.
- [ ] **Mega-Ensemble:** Aggregate saturation model with teammate retrains during inference.
- [ ] Final Submission Generation using Rust Taxonomic Filter.

## Next Phase: Elite Refinement (The 33.45 Private F1 Wall)
- [ ] **Dependency GNN / Label Co-occurrence:** Learn species co-occurrence relationships from training data and refine logits using a learned adjacency graph.
- [ ] **Pairwise Ranking Refinement:** Implement a cross-encoder style reranker for pairwise comparisons between top candidate species.
- [ ] **Cascaded Expert Heads:** Route uncertain predictions to specialized genus/family expert models (e.g., Carex specialist).
- [ ] **Multi-Instance Attention Decoding:** Replace max-pooling with learned attention over tile predictions.
- [ ] **Dual-Scale Tiling Engine:** Implement 518px fine and 732px coarse passes with max-pooling aggregation.
- [ ] **Noise Masking Pipeline:** Integration of GroundingDINO detection and SAM segmentation in the inference loop.
- [ ] **Ellenberg Indicator Integration:** Download EIVE indicator clouds and implement Niche Similarity Index $S$.
- [ ] **Temporal Propagation:** Implement cross-year plot score boosting based on metadata matching.
- [ ] **Loopy BP Upgrade:** Transition from binary AC-3 to the full Loopy Belief Propagation solver using Niche Similarity weights.

