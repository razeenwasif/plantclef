# Source (src)

This directory contains the core Python library and building blocks for the Oracle engine.

It provides the shared infrastructure used across all execution `phases/`:
- **`models/`**: PyTorch definitions for the GFAM ensemble, BioCLIP/DINOv3 wrappers, GCN heads, and the custom Fused LoRA engine.
- **`data/`**: The DALI-based WebDataset loaders, dynamic dataset discovery (with Gemma 4 VCoT), and hash-based indexers.
- **`inference/`**: Core logic for post-processing, including Thermodynamic Phenology, Agentic LLM Arbitration, Frank-Wolfe sparsity, and AC-3 constraint satisfaction.
- **`training/`**: Reusable loss functions (Asymmetric Loss, HTL), optimizers, and checkpointing logic.
- **`setup/`**: The `launch_oracle.sh` orchestrator script that manages DDP execution and hardware environment variables.
- **`config/`**: Pydantic schemas enforcing strict configuration contracts across the system.