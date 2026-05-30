# Variational Research Strategy: ELBO in Plant Identification

This document outlines a research proposal and technical design for integrating **Evidence Lower Bound (ELBO)** and variational inference into our plant-identification and distributed-training workflows. 

This strategy directly addresses the primary bottlenecks highlighted in our PlantCLEF 2026 paper: **single-plant to multi-species domain shift, fragile heuristic vegetation filtering (ExG), long-tail calibration, and the noise floor of post-hoc priors**.

---

## 1. Core Mathematical Framework: The ELBO

In deterministic deep learning, we map an image $x$ to a class probability distribution using cross-entropy. In a variational framework, we introduce a latent representation $z$ (e.g., biological structure, Linnaean ancestry) and maximize the likelihood of the data by optimizing the **Evidence Lower Bound (ELBO)**.

The log-likelihood of our observed data $x$ (image tiles and species sets) is bounded by:

$$\log p(x) \ge \text{ELBO}(\theta, \phi; x) = \mathbb{E}_{q_{\phi}(z \mid x)}[\log p_{\theta}(x \mid z)] - D_{\text{KL}}(q_{\phi}(z \mid x) \parallel p(z))$$

Where:
*   **$q_{\phi}(z \mid x)$ (The Variational Posterior)**: A neural network encoder (our partially unfrozen BioCLIP 2.5) that predicts the distribution of latent biological/taxonomical features given a tile.
*   **$p_{\theta}(x \mid z)$ (The Generative Decoder / Classifier)**: A generative or classification network that maps the latent space to observed plant properties or species IDs.
*   **$p(z)$ (The Prior)**: The structural Linnaean Tree of Life or geographical co-occurrence constraints.
*   **$D_{\text{KL}}$ (Kullback-Leibler Divergence)**: A regularization term that measures how much our predicted visual latent space deviates from our structured biological prior.

---

## 2. Research Initiatives

### Initiative A: Variational Taxonomic Priors (Structured KL Regularization)

#### Problem
In our paper, we use auxiliary deterministic cross-entropy losses at coarser taxonomic levels (genus, family) to regularize the species encoder. This is a heuristic approximation of hierarchy; the gradients can still conflict, and the model does not learn a mathematically unified latent hierarchy.

#### Variational Solution
Instead of auxiliary heads, we treat the taxonomic hierarchy as a tree-structured latent space $z$:
1.  **Prior Model $p(z)$**: We construct a hierarchical prior where family generates genus, which generates species. This prior is fixed and derived from the GBIF Linnaean taxonomy.
2.  **Posterior Model $q_{\phi}(z \mid x)$**: BioCLIP embeds the image into a continuous latent space. We model this space as a hierarchical variational distribution.
3.  **The Loss**:
    $$\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_{\phi}(z \mid x)}[\log p(y_{\text{species}} \mid z)] + D_{\text{KL}}(q_{\phi}(z \mid x) \parallel p(z))$$
    *   Optimizing this objective mathematically forces the continuous visual representations learned by the encoder to occupy a latent space structured exactly like the Linnaean Tree of Life.

---

### Initiative B: Structure-Aware OOD Vegetation Filter (Replacing ExG)

#### Problem
Our current pipeline uses an **Excess Green index** ($ExG = 2G - R - B$) to drop non-vegetation tiles. This color heuristic is highly fragile, failing completely on dry, brown, or autumnal vegetation (e.g., litter-covered quadrats) or under strong lighting domain shifts.

#### Variational Solution
We train a standard Variational Autoencoder (VAE) exclusively on clean, high-quality, single-specimen plant images (the 2.65M `i002` manifest):
1.  **The VAE**: Learns a tight generative model of what a "plant structure" looks like.
2.  **Inference-Time OOD Scoring**:
    For each tile $x_t$ in a test quadrat, we compute its ELBO:
    $$\text{ELBO}(x_t) = \mathbb{E}_{q_{\phi}(z \mid x_t)}[\log p_{\theta}(x_t \mid z)] - D_{\text{KL}}(q_{\phi}(z \mid x_t) \parallel p(z))$$
3.  **Filter Logic**:
    *   If $\text{ELBO}(x_t) > \text{threshold}$, the tile contains structured botanical elements (green leaves, brown stalks, flowers) and is kept.
    *   If $\text{ELBO}(x_t) \le \text{threshold}$, the tile is flagged as Out-of-Distribution (quadrat frames, bare soil, rocks, plastic tape, or shadows) and is immediately pruned.

```
       [ Input Quadrat Image ]
                 │
                 ▼
       [ Partition into Tiles ]
                 │
                 ▼
        [ Run VAE on Tiles ]
                 │
                 ├──► High ELBO  ──► [ Keep Tile: Forward to BioCLIP ]
                 └──► Low ELBO   ──► [ Drop Tile: Flagged as OOD Noise ]
```

---

### Initiative C: Variational Aggregation for Quadrat Tile Fusion

#### Problem
Our current tile aggregation (softmax-mean) averages the 16 tile-level species probabilities. If a rare species appears confidently in only one tile, its signature is heavily diluted by the other 15 background/litter tiles.

#### Variational Solution
We frame the quadrat's total species composition as a latent Dirichlet distribution $z$ that generates the observed tile-level predictions $x_1, \dots, x_{16}$:
1.  **Prior $p(z)$**: Encodes our "siblings stick together" species co-occurrence statistics.
2.  **Variational Aggregator**: We optimize the variational posterior $q(z \mid x_1, \dots, x_{16})$ using the ELBO.
3.  **The Advantage**: The optimization recognizes that a highly confident signature of a rare species in a single tile is an unlikely noise event under the prior, allowing the model to confidently emit the species without it getting drowned out by background dilution.

---

### Initiative D: Semi-Supervised Variational Pseudo-Labeling (LUCAS)

#### Problem
In our paper's future work, we proposed pseudo-labeling the unlabeled **LUCAS quadrat corpus** to bridge the single-plant-to-multi-species domain shift. However, standard pseudo-labeling suffers from confirmation bias, reinforcing early incorrect predictions.

#### Variational Solution
We frame semi-supervised training on combined labeled (PlantCLEF) and unlabeled (LUCAS) datasets as a variational expectation-maximization problem (Kingma's M2 VAE framework):
*   For labeled images, we optimize the standard supervised ELBO.
*   For unlabeled quadrats, the species label $y$ is treated as a **latent variable**. The loss function computes the ELBO by taking an expectation over the model's own predicted class distribution $q_{\phi}(y \mid x)$ while regularizing it with the taxonomic and co-occurrence priors:
    $$\mathcal{U}(x) = \sum_{y} q_{\phi}(y \mid x) \text{ELBO}(x, y) + \mathcal{H}(q_{\phi}(y \mid x))$$
*   This prevents the model from collapsing into high-confidence errors on unlabeled data by enforcing structural taxonomic bounds during pseudo-label generation.

---

## 3. RTX 4090 Feasibility & Implementation Plan

Because VAEs and Bayesian models can be computationally expensive to train from scratch, we leverage a highly efficient **two-stage training design** tailored for an RTX 4090 workstation:

1.  **Pre-Extract Frozen Features**:
    We freeze the lower 28 transformer blocks of BioCLIP 2.5, DINOv3, and ConvNeXt-V2-L. We run a single pass over the training corpus to save these static visual embeddings to a fast SSD.
2.  **Train Variational Heads Locally**:
    Since we are only training lightweight MLP heads (representing $q_{\phi}(z \mid \text{embedding})$ and $p_{\theta}(\text{species} \mid z)$) on static embeddings, the training takes **minutes** rather than days on an RTX 4090.
3.  **Software Stack**:
    *   Use `Pyro` or `PyTorch VAE` frameworks for variational layers.
    *   Reuse our existing `inf_script.py` tile-partitioning logic to feed features into the OOD ELBO filters.
