# Oracle: Novelty Backlog & Research Frontiers

This document tracks the advanced botanical and mathematical concepts integrated into the Oracle system, distinguishing between **Saturated** (fully implemented) and **Unsaturated** (exploratory/theoretical) ideas.

## 🟢 Saturated (Final Submission Core)
- **Thermodynamic Phenology:** Seasonal prior based on circular Gaussian smoothing of GBIF data and Boltzmann distribution.
- **Neuro-Symbolic AC-3:** Rust-backed hard-constraint satisfaction over the taxonomic graph to prune impossible predictions.
- **Frank-Wolfe Sparsity Gating:** Post-processing optimization finding the sparsest set of species that explains visual features.
- **Retinex Illumination Normalization:** Multi-scale Retinex (MSR) preprocessing for handling complex field lighting.
- **FAISS Retrieval-Augmented Classification:** Nearest-neighbor boosting for long-tail species using BioCLIP prototype embeddings.
- **SAM Noise Masking:** Segment Anything (SAM) integration to isolate plant instances.
- **Hardware-Aware Orchestration:** Auto-detection of Blackwell (RTX 6000) vs Lovelace (4090).
- **Dynamic Shard Management:** Low-RAM sliding-window data loading for consumer GPUs.
- **Optuna Hyperparameter Optimization:** Bayesian optimization of inference post-processing parameters over cached un-adjusted logits.
- **Combinatorial Ablation Grids:** Automated generation and execution of multi-resolution, multi-parameter post-processing ablation studies.

## 🟡 Future Work (High Priority Research)

### 1. The "Holy Trinity": Asymmetric Dual-Teacher Distillation (AD-TD)
To achieve elite performance without the massive latency of running three models at once, we are developing an Asymmetric Distillation framework.
*   **The Setup:** Use fine-tuned BioCLIP and DINOv3 "Experts" as teachers.
*   **The Student:** Train a single, hyper-efficient **DeiT (Data-efficient Image Transformer)** student.
*   **Dual-Token VCoT:** The student is modified with two unique "distillation tokens." One token learns biological relationships from BioCLIP, while the second learns spatial precision from DINOv3. The result is a single model that "sees" like a cartographer but "reasons" like a botanist.

### 2. Agentic LLM Arbiter (VCoT)
For high-entropy or ambiguous samples, a local LLM (Gemma 4 / Nemotron) will perform a **Visual Chain-of-Thought** analysis.
*   **Role:** Acts as a human field-expert to break ties between visually similar species.
*   **Logic:** Integrates visual features with ecological metadata (pH, GDD) to veto impossible predictions.

### 3. Spectral Vision: Global Filter Networks (GFNet)
Transitioning from spatial $O(N^2)$ Self-Attention to $O(N \log N)$ frequency-domain filtering.
*   **Fourier Token Mixing:** Utilizing 2D FFT to map image features into the **Complex Plane**.
*   **Complex-Valued Weighting:** Learning filters that operate directly on the frequency spectrum. This allows the model to mathematically "erase" low-frequency background noise (soil, shadows) while perfectly preserving high-frequency diagnostic traits (leaf serrations, petal veins).
*   **Efficiency:** Enables massive batch sizes and high-resolution training on 96GB RTX 6000 hardware due to the superior scaling of the FFT over traditional transformers.

## 🔴 Unsaturated (The Frontier)

### 1. Ecological Physics & Logic
- **Ecological Pauli Exclusion Principle:** A repulsion layer in the GCN head that prevents the prediction of two species that occupy the same exact ecological niche in a single 1m² quadrat. 
- **Ising Model (Spin Glass Refiner):** Modeling species co-occurrence as a spin-glass system. Simulated Annealing is used to find the global "ground state" energy of a quadrat's species composition.
- **Allelopathic Phenology:** Modeling "chemical warfare" between plants where certain species suppress the detection probability of others.

### 2. Environmental & Edaphic Intelligence
- **Geochemical & Edaphic Masking:** Integrating SoilGrids data (pH, cation exchange capacity, clay content) as a hard filter on species proposals.
- **Ellenberg Indicator Refinement:** Using standardized botanical light, temperature, and moisture requirements to veto ViT hallucinations.
- **PageRank on Ecological Networks:** Identifying "Keystone Species" using PageRank to propagate confidence to associated minor species.

### 3. Mathematical Optimization & Acceleration
- **Square-Root Space Logic Evaluation:** Inspired by R. Ryan Williams' breakthrough (arXiv:2502.17779) in simulating time with square-root space, allowing us to perform Extreme Memory-Constrained Computing for Neuro-Symbolic reasoning. The methodology involves:
  1. **Partition the Logic:** Instead of evaluating the entire rule graph at once, we break the logical forward pass into discrete "blocks" (e.g., evaluating just the generic/family constraints first, then the spatial constraints).
  2. **The Tree Mapping:** We map the dependencies between these logical blocks into a Tree Evaluation problem.
  3. **Space-Efficient Verification:** Using the $O(\sqrt{t \log t})$ space-efficient algorithms (like those from Cook and Mertz), we can evaluate the gradients of these logic trees without storing the entire state history in RAM, preventing combinatorial RAM/VRAM explosions on hardware like the RTX 4090.
- **Entropy Gating (Advanced):** More aggressive "agentic" triggering of LLMs based on multi-sample predictive variance.
- **TensorRT & INT8 Quantization:** Moving beyond BFloat16 to full INT8 compilation for 1000+ FPS quadrat scanning on Blackwell hardware.
- **Loopy Belief Propagation (Full):** Transitioning from binary AC-3 to the full probabilistic BP solver for all quadrats.
