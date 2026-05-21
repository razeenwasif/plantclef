# Oracle (formerly Nitro): Project Context & Memory

This file serves as the definitive state-of-the-world and context preservation document for the Oracle project. When a new LLM session begins, reading this document provides immediate understanding of the system's architecture, hardware environment, and strategic roadmap.

## 1. Project Identity & Goal
*   **Name:** Oracle (Transitioned from "Nitro" on May 7, 2026).
*   **Original Scope:** A state-of-the-art, high-resolution botanical identification pipeline for the PlantCLEF 2026 challenge (7,806 species, extreme long-tail).
*   **Current Vision:** Transforming into a **Universal Neuro-Symbolic Knowledge Engine** capable of multimodal perception, formal logical reasoning, and dynamic ontology management across any domain (plants, automotive, pathology).

## 2. Hardware & Environment Profile
The system is engineered to maximize throughput while preventing bottlenecks on consumer-grade hardware.
*   **Environment:** Windows Subsystem for Linux (WSL2).
*   **GPU:** Single NVIDIA RTX 4090 (24GB VRAM). (Codebase also contains scaling logic for 4-pod RTX 6000 Ada / Blackwell clusters with FP8 support).
*   **Storage:** Datasets are located on an external Windows drive (`D:\`, mapped in WSL as `/mnt/d/`).
*   **LLM Backend:** Local inference via **Ollama** running in WSL (models: `gemma4`, `nemotron-3-super`, `minimax2.5`).

## 3. Core Architecture (The "Saturated" Stack)
The system employs a polyglot architecture (Python, Rust, CUDA, C++, Haskell).

### Data Plane (Zero-Latency I/O)
*   **Rust Engines (`engines/rust/`):** SIMD-optimized dataset auditing, resizing, and WebDataset (`.tar`) packing.
*   **NVIDIA DALI:** Direct-to-GPU nvJPEG decompression and augmentation, bypassing the CPU GIL.
*   **Dynamic Sharding & Elastic Hashing:** For the 4090/WSL environment, datasets are loaded using a sliding-window RAM-disk (`/dev/shm`) manager. File paths are converted to `uint64` via **XXHash** and stored in a novel **Elastic/Funnel Hash Registry** (O(1) open addressing without reordering) to prevent VRAM/RAM metadata overflow on massive datasets while maintaining zero-collision latency.
*   **Agentic Auto-Discovery:** A Python OS-scanner combined with **Gemma 4** dynamically locates datasets (like PlantNet-300K) across the filesystem and automatically registers them in `configs/datasets.yaml` with stratified splitting.

### Compute Plane (Perception)
Training is split into discrete phases managed by the `oracle.py` CLI and `src/setup/launch_oracle.sh`:
1.  **Foundation Caching (`phases/foundation_caching`):** Freeze backbones, cache 1.4M image features to NVMe.
2.  **Head Warmup (`phases/head_warmup`):** Rapidly train MLP/GCN heads on cached features.
3.  **Expert Specialization (`phases/expert_specialization`):** Heavy LoRA fine-tuning of individual backbones.
4.  **Student Distillation (`phases/student_distillation`):** Distill knowledge into a combined **Gated Feature Aggregation (GFAM)** ensemble (BioCLIP, DINOv3, ConvNeXt).
*   *Note: Inference uses a Multi-Resolution SAHI (Sliding Window) strategy (224px to 672px).*

### Reasoning Plane (Post-Processing)
*   **Bayesian-ExG Aggregator:** Fuses visual certainty (negative entropy) with physical Excess Green (ExG) index to ignore background noise.
*   **Thermodynamic Phenology:** A Boltzmann-distributed seasonal prior driven by GBIF growing degree days (GDD).
*   **AC-3 Taxonomic Pruning:** A hard mathematical filter checking predictions against the phylogenetic tree.
*   **Frank-Wolfe Sparsity:** Post-processing optimization anchored by MacArthur-Wilson island biogeography priors.

### Modeling Tools
*   **Optuna Hyperparameter Optimization:** Bayesian optimization of inference post-processing parameters over cached un-adjusted logits (`tools/modeling/optimize_optuna.py`).
*   **Combinatorial Ablation Grids:** Automated generation and execution of multi-resolution, multi-parameter post-processing ablation studies (`tools/audit/run_ablation_full.py`).

## 4. The Immediate Frontier (Future Work / Active R&D)
These components are implemented but isolated from the main "competition" baseline, acting as the foundation for the next generation of the engine:

*   **Asymmetric Dual-Teacher Distillation (AD-TD / "The Holy Trinity"):** A hyper-efficient pipeline distilling taxonomic knowledge (BioCLIP) and geometric precision (DINOv3) into a single dual-token **DeiT** student.
*   **Agentic LLM Arbiter (VCoT):** Using local Ollama models (Gemma 4 / Nemotron) to perform **Visual Chain-of-Thought** analysis. Triggered dynamically to act as a human field-expert and break ties on high-entropy (uncertain) predictions.
*   **Spectral Vision (GFNet):** Exploring $O(N \log N)$ Complex-Valued Fourier Filtering (2D FFT) to replace $O(N^2)$ spatial self-attention, allowing the model to mathematically isolate high-frequency plant features from low-frequency soil noise.

## 5. The "Grand Vision" Roadmap
We are currently executing a major transition from static JSON/NPY files to a dynamic relational database backend. See `docs/UNIVERSAL_NITRO_ROADMAP.md` (renamed to `UNIVERSAL_ORACLE_ROADMAP.md` contextually).

*   **Phase 1 (Current): The Relational Backbone.** Replacing static taxonomies with a PostgreSQL database using the `ltree` extension for dynamic ontology modeling. *(Database `oracle-db` is currently running via Docker).*
*   **Phase 2: The Feature Store.** Using `pgvector` to store and query multi-domain embeddings.
*   **Phase 3: Universal Ingestion (ETL).** Building a Pydantic/Parquet-based data contract system.
*   **Phase 4: Neuro-Symbolic (NeSy) Integration.** Replacing hard AC-3 logic with fully differentiable Probabilistic Logic Programming (using Dolphin or LTNtorch) with a strict CPU/GPU compute split to avoid OOM on the 4090.
*   **Phase 5: Automated Model Lineage.** MLOps tracking in Postgres.

## 6. Recent Structural Changes
*   Renamed entire codebase from "Nitro" to "Oracle".
*   Consolidated C++/CUDA extensions, Rust data engines, and Haskell logic into the `engines/` root directory.
*   Implemented Dynamic Dataset Discovery and WSL optimizations (Ollama integration, Hash Indexing).
*   Created classical AI baseline examples (`engines/native/classical_ai/`).
*   Added Optuna and enhanced ablation workflows.
*   Updated `.github/workflows/oracle_ci.yml` and Python pathing to reflect the new structure.

---
**INSTRUCTIONS FOR LLM ON RESTART:**
When starting a new session, acknowledge you have read `docs/LLM_CONTEXT.md` and check the relevant progress files:
- `docs/DB_ENGINEERING_PROGRESS.md`: For the transition to a dynamic relational database (Postgres).
- `docs/SQ_ROOT_SPACE_PROGRESS.md`: For the Extreme Memory-Constrained Computing (Square-Root Space Logic Evaluation) systems engineering project.
