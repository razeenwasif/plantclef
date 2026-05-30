# Phases

This directory contains the primary execution scripts for the plantclef training and inference lifecycle.

Each subdirectory represents an isolated phase of the pipeline, ensuring clean modularity:
- **`foundation_caching/`**: Freezes backbones and extracts 1.4M image features to an NVMe cache.
- **`head_warmup/`**: Rapidly trains the MLP/GCN classification heads on the cached features.
- **`expert_specialization/`**: Conducts heavy LoRA fine-tuning on individual backbones (e.g., DINOv3, BioCLIP) to create high-resolution teachers.
- **`student_distillation/`**: Distills the experts' knowledge into the unified GFAM ensemble using a multi-resolution curriculum.
- **`asymmetric_distillation/`**: (R&D) The "Holy Trinity" setup distilling taxonomic and geometric teachers into a dual-token DeiT.
- **`inference/`**: The multi-resolution, sliding-window (SAHI) evaluation engine equipped with ecological reasoning filters.