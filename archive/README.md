# PlantCLEF 2026: BioCLIP Multi-Species Detection Pipeline

This project implements a state-of-the-art botanical identification system designed to identify multiple plant species within complex vegetation plots (quadrats).

## Core Design Choices

### 1. Foundation Model: BioCLIP (ViT-L/14)
We utilize **BioCLIP**, a domain-specific foundation model pre-aligned with the Tree of Life. This provides superior feature discrimination for morphologically similar species compared to general-purpose models like DINOv2.

*   **High-Resolution Tuning (448px):** To capture fine-grained botanical features (hairs, leaf margins) in 50x50cm quadrats, we increased the input resolution from 224px to 448px.
*   **Positional Embedding Interpolation:** We implemented bicubic interpolation for the Vision Transformer's positional embeddings to ensure the backbone remains spatially coherent at higher resolutions.

### 2. Loss Function: Asymmetric Loss (ASL)
To address the **Long-Tail distribution** and the **Multi-Species** nature of vegetation plots, we use Asymmetric Loss:
*   **Imbalance Handling:** ASL applies different focusing parameters ($\gamma_+$ and $\gamma_-$) to positive and negative samples, preventing the "sea of negatives" (7,800+ species) from overwhelming the gradients.
*   **Multi-Label Readiness:** By using sigmoid activation and ASL, the model is prepared to identify "collages" where multiple species overlap, rather than being forced to choose just one (Softmax).

### 3. GPU-Accelerated Pipeline: NVIDIA DALI & cuDF
Training on 1.4 million images requires high-throughput data loading.
*   **NVDEC Decoding:** JPEGs are decoded directly on the GPU.
*   **Mixed Device Transforms:** All spatial augmentations (RandomResizedCrop, Flip, Normalization) occur on the GPU, ensuring 100% GPU utilization and eliminating CPU bottlenecks.
*   **cuDF:** Fast metadata manipulation using GPU-accelerated DataFrames.

### 4. Inference Strategy: Tiled Slicing (SAHI-style)
Identifying tiny plants in large quadrat images requires maintaining high resolution.
*   **Slicing:** Images are divided into 512x512 tiles with overlap.
*   **Aggregation:** We perform classification on each tile and aggregate the results (Max Pooling of probabilities) to detect all species present in the entire plot.

## Project Structure

```text
├── src/
│   ├── model.py        # BioCLIP + Positional Interpolation
│   ├── dataloader.py   # NVIDIA DALI Pipeline
│   ├── train.py        # ASL Training Loop + WandB
│   └── predict_sahi.py # Tiled Inference Script
├── data/               # Symlinked to your datasets
└── Infastructure/      # Setup and download scripts
```

## Getting Started

1.  **Environment:** Ensure you have `nvidia-dali-cuda120` and `open_clip_torch` installed.
2.  **Training:** Run `python src/train.py` to start the BioCLIP baseline with ASL.
3.  **Inference:** Run `python src/predict_sahi.py` for high-resolution quadrat analysis.

## Key Performance Indicators (KPIs)
*   **Micro-F1 Score:** Evaluated across 7,800 species.
*   **Rare Species Recall:** Monitored to ensure the model doesn't just guess common species.
# PlantCLEF 2026 Project

## Report
- [Course Report](https://www.overleaf.com/project/69bdfbb96387d38e48604120)
- [Competition Working Notes](https://www.overleaf.com/project/69be00b57b35c09ee8357334)
