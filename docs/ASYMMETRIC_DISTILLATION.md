# Asymmetric Dual-Teacher Distillation (AD-TD)

## Overview
AD-TD is a high-performance distillation framework designed to combine the strengths of three specialized architectures into a single, efficient inference model.

### The "Holy Trinity" of Botanical AI
1.  **BioCLIP (The Botanist):** Foundation model pre-trained on biological taxonomy. Provides deep semantic and phylogenetic understanding.
2.  **DINOv3 (The Cartographer):** Self-supervised vision transformer optimized for geometric grounding and dense feature separation.
3.  **DeiT (The Ultimate Student):** Data-efficient Image Transformer with native support for distillation tokens, engineered to absorb knowledge from multiple teachers.

## Architecture: Dual-Token Student
The student model is a modified DeiT that utilizes two distillation tokens: `[DIST_tax]` and `[DIST_geo]`.

-   **[CLS] Token:** Anchored to Ground Truth labels via Asymmetric Cross-Entropy (ASL).
-   **[DIST_tax] Token:** Supervised by BioCLIP's soft probabilities via KL-Divergence. Captures taxonomic relationships.
-   **[DIST_geo] Token:** Supervised by DINOv3's dense structural features via KL-Divergence. Captures spatial boundaries and occlusions.

## Pipeline Strategy
To maximize throughput on a 4-pod cluster (RTX PRO 6000), we employ an **Offline Feature Extraction** strategy.

### Step 1: Teacher Logit Extraction
We pre-compute and cache the teacher logits for the entire training set. This eliminates the need to run the massive BioCLIP and DINOv3 models during the student training phase, reducing VRAM usage by ~70%.

```bash
python src/models/asymmetric_distillation/extract_teachers.py
```

### Step 2: Asymmetric Distillation
The student model is trained on all 4 pods. The dataloader reads the image and the pre-computed teacher vectors.

```bash
python src/models/asymmetric_distillation/train_trinity.py
```

## Inference Mode
During inference, the massive teacher models are discarded. Only the lightweight DeiT student is deployed. 
-   The model processes quadrats using a sliding-window SAHI pattern.
-   Outputs from the three tokens (`[CLS]`, `[DIST_tax]`, `[DIST_geo]`) are averaged to produce the final logit vector.
-   **Visual Chain-of-Thought (VCoT):** For low-confidence or ambiguous detections, the **Agentic LLM Arbiter** is triggered. It uses a local LMM (Gemma 4 / Nemotron) to perform a multi-step visual reasoning process to resolve ties between similar species or veto ecologically impossible predictions.
-   The resulting logits are passed to the **Thermodynamic Phenology** and **AC-3 Taxonomic** filters.
