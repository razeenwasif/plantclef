# plantclef: High-Resolution Expert Strategy

This document outlines the "Diversity + Detail" ensembling strategy used to maximize performance for the PlantCLEF 2026 challenge.

## 1. Architectural Rationale
The competition identifies plants in high-resolution quadrat images. We utilize two distinct tiers of models to capture different features:

*   **Diversity Tier (3x Seeds):** Trained at 192px/224px. These models are fast and provide a robust "majority vote" on the global structure of the quadrat. Multiple seeds reduce variance caused by stochastic optimization.
*   **Detail Tier (2x Experts):** Trained at 512px on RTX PRO 6000 (48GB). These specialists focus on fine-grained botanical traits (leaf serration, trichomes) that are lost at lower resolutions.

## 2. Training Workflow
The experts are trained sequentially using specialized configurations:
1.  **BioCLIP Expert:** Captures taxonomic signals using the biological foundation model.
2.  **DINOv3 Expert:** Captures structural and boundary signals via the EVA-02/DINOv3 backbone.

Both use a higher LoRA rank (`R=512`) to accommodate the increased complexity of the 512px input space.

## 3. The Ultimate Ensemble
The final inference pass combines all 5 checkpoints:
*   **Resolution:** All tiles are resized to 512px. The low-res models benefit from high-quality interpolation, while the experts run at their native resolution.
*   **Consistency:** We use **Loopy Belief Propagation (BP)** with 10 iterations to ensure that predictions across the 25 tiles of the quadrat are ecologically coherent based on the species co-occurrence matrix.
*   **Aggregation:** Results are fused using `bayesian_veg` aggregation, which weights tiles based on their botanical relevance and information entropy.

## 4. Troubleshooting
If the PRO 6000 Pod OOMs, reduce `p2_batch_size` in `configs/high_res_*.yaml` to `32`. The current value of `64` is optimized for 48GB VRAM with gradient accumulation.
