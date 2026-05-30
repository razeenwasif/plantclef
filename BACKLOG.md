# Backlog: future novelty directions

This file is a living record of research directions we scoped,
prototyped, or deferred for a future iteration of the PlantCLEF
system. Nothing here is implemented in the codebase that backs
`report/main.tex`; the working note's "Considered Approaches Not
Implemented" appendix and Section 6 (Discussion) cover the highest-
priority ones at a higher level. The detail in this file is drawn from
both the working note's appendices and from an internal long-form
research write-up that has since been decomposed into `BACKLOG.md`,
`docs/SYSTEMS_DESIGN.md`, `docs/ALGORITHMS.md`, and
`docs/RELATED_WORK.md`.

The three buckets below:

* **Saturated** - in the paper as a shipped feature.
* **Future work (high priority)** - directions with a concrete design
  or partial prototype that would have shipped with more time.
* **Unsaturated frontier** - directions we have a hypothesis for but
  no implementation or evidence yet.

---

## Saturated: in the paper's final system

The following appear in `src/inf_script.py`, `src/inf_script_phen.py`,
and the i002 training code, and are documented in the paper:

* **Partial-unfreeze BioCLIP 2.5 with taxonomic heads.** Last four
  transformer blocks plus `ln_post`/`proj` unfrozen on the i001
  manifest, with per-head MLPs for species / genus / family
  (Section 3 of the report).
* **Tiled inference with adaptive selection.** 4x4 grid, 224 + 336
  dual-resolution single-checkpoint ensemble, softmax-mean
  aggregation, class-prior logit adjustment (tau = 0.25), adaptive
  probability threshold (T = 0.03, k in [2, 10]) (Section 3.4-3.7).
* **Seasonal phenology prior (pivot 3).** Circular-Gaussian DOY pdf
  built from GBIF month histograms, multiplied into the visual
  posterior in log space at beta = 1.0; ExG vegetation filter and
  entropy-weighted Bayesian aggregation on a multi-scale (1.0 + 0.8)
  tiling (Appendix C).
* **Triple-backbone classifier retraining (cRT, pivot 1).** Frozen
  3328-d concatenation of BioCLIP 2.5 + DINOv3 + ConvNeXt-V2-L
  features with two MLP heads; reported as an unsuccessful pivot.
* **Genus/family co-occurrence reranking (pivot 2).** Sibling-prior
  built from the anchor posterior's own aggregated genus/family
  supports; reported as a near-neutral pivot.

---

## Future work (high priority)

Directions with a concrete design that we would carry into the next
iteration first.

### 1. Asymmetric Dual-Teacher Distillation (AD-TD)

Reach triple-backbone Macro-F1 without the inference-time cost of
running three encoders.

* **Teachers**: the i002 partially-unfrozen BioCLIP 2.5 and a similarly
  fine-tuned DINOv3 (the experiment 008 line in the appendix).
* **Student**: a single DeiT (Data-efficient Image Transformer)
  modified with two dedicated *distillation tokens*. One token is
  supervised by the BioCLIP posterior (biological / taxonomic
  reasoning); the second by DINOv3's dense features (spatial precision
  and segmentation cues). The resulting model "sees" like a
  cartographer but "reasons" like a botanist.
* **Why now**: the within-backbone saturation diagnostic
  (Section 5.3, Jaccard 0.923 between intra-BioCLIP variants) shows
  headroom does not live inside a single backbone family; AD-TD is a
  concrete way to combine two backbones at training time only.

### 2. Spectral Vision: Global Filter Networks (GFNet)

Replace `O(N^2)` self-attention in the encoder with `O(N log N)`
2-D FFT-based token mixing. The diagnostic case for this:

* Quadrats are mostly low-frequency context (soil, litter, shadow)
  with a few high-frequency diagnostic features (leaf serrations,
  petal venation). Frequency-domain filters can learn to suppress
  the former and amplify the latter directly.
* Complex-valued neural networks (CVNNs) let us learn spectral
  filters that operate directly on the Fourier-domain feature map -
  high-frequency botanical traits get amplified, low-frequency
  diffuse background gets suppressed, without ever leaving the
  complex plane.
* Memory linear in N enables training at 1024-px+ inputs with large
  batch sizes, capturing microscopic botanical detail that remains
  invisible to standard spatial-domain transformers.

### 3. Agentic LLM arbiter (Visual Chain-of-Thought)

For high-entropy quadrats where the visual posterior is uncertain
between morphologically similar species, route the candidate-species
shortlist + cropped tiles to a local LMM for a tie-break step.
Locally-deployed (no zero-shot generation, just arbitration) keeps
latency and hallucination risk bounded.

* **Trigger**: top-1 / top-2 probability gap below ~15 %, predictive
  entropy above a calibrated threshold, or Jaccard between the
  visual top-K and the phenology-reweighted top-K below a threshold.
* **Tier 1 - fine-grained visual arbiter (Gemma 3 / Gemma 4 class)**:
  receives the high-resolution tile crop and a structured prompt
  asking for a morphological comparison (e.g. leaf ligule structure,
  vein pattern). Output is a small logit boost on the target species.
* **Tier 2 - neuro-symbolic integrator (Nemotron-class)**: receives
  the shortlist plus continuous ecological variables (GDD, soil pH,
  Ellenberg L/T/F/N/R values) and acts as a logic gate, issuing
  a hard veto when the environmental math strongly contradicts the
  visual prediction.
* **Inputs**: image crops, per-tile attention maps, candidate-species
  shortlist with GBIF habitat / pH / GDD ranges, and the observation
  date.

### 4. External ecological co-occurrence prior

The genus / family rerank we shipped (pivot 2) used the model's *own*
posterior to build the co-occurrence prior, which makes the prior
statistically dependent on the visual posterior. A genuine
co-occurrence prior built from external community-survey data
(e.g. LUCAS, EVA, GBIF Plot summaries) would be information-theoretically
orthogonal in a stronger sense.

### 5. Per-class threshold optimisation

Instead of the single `T = 0.03` we sweep, fit a per-class threshold
`T_s` via Brent's method on a held-out validation slice. Expected to
help mid-frequency species where the long-tail logit adjustment
over-corrects.

### 6. Frank-Wolfe + Island Biogeography selection

Replace the adaptive threshold rule with a constrained convex
selection: pick the sparse label vector `y in [0,1]^N` that maximises
the inner product with the post-LA logits while satisfying an
ecological-co-occurrence polytope. Frank-Wolfe (Conditional Gradient)
sidesteps projection onto the polytope by solving linear
sub-problems. The gradient picks up a MacArthur-Wilson count
regulariser

```
nabla_eff = logits - 2 * lambda * (||y||_1 - K_bar) * 1
```

with `K_bar ~ 8` (empirical GBIF mean species count for a 0.25 m^2
quadrat) and `lambda = 0.05`. Equivalent to a soft L1 ball centred at
`K_bar`, but the LMO never needs an explicit projection. Selection
becomes provably consistent with known botanical patch-area species
richness rather than relying on a single global probability cutoff.

### 7. LUCAS MAE self-supervision pretraining

Close the train/test domain gap by pretraining the backbones with
Masked Autoencoding on the ~160 GB of unlabeled LUCAS quadrat
images *before* fine-tuning on PlantCLEF 2024. Adapts the ViT and
ConvNeXt textural representations to ground-level perspective, field
lighting, and seasonal vegetation density - a far stronger
initialisation than ImageNet-22k weights for the quadrat distribution
we actually score on. Slots in before the existing two-stage i002
schedule and changes nothing about the head architecture.

---

## Unsaturated frontier

Directions with a hypothesis but no implementation yet.

### Ecological physics and logic priors

* **Ecological Pauli exclusion / tile-level NMS.** Add a repulsion
  term to a GCN head on the predicted species set, penalising sets
  that contain two species occupying the same exact ecological niche
  in a 1 m^2 quadrat. Trained from Ellenberg indicator values + Grime
  CSR strategy labels. At the tile level: if a tile is confidently
  claimed (p > 0.9) by a single species, apply an exponential decay
  penalty to all other species' logits for that tile, forcing the
  model to explain the quadrat using physically distinct tiles.
* **Ising / spin-glass refinement.** Model the per-quadrat species
  set as a spin system whose interactions encode pairwise
  co-occurrence statistics; refine the visual posterior by simulated
  annealing toward the ground-state composition that jointly optimises
  neural confidence against the ecological interaction-energy matrix.
* **Allelopathic phenology.** Use known chemical-warfare relations
  between species (e.g. juglone from *Juglans* suppressing many
  understorey species) as a multiplicative suppression term on
  detection probability. Cleanly composes with the Frank-Wolfe
  selection above as negative edge weights: confident selection of an
  allelopathic species reduces the LMO score of its known victims.
* **Physical identity coherence via DSU clustering.** In dense
  quadrats, one plant body often spans several overlapping tiles.
  Cluster tiles by feature cosine similarity (`> 0.95`) using
  Disjoint Set Union and treat each cluster as an atomic unit before
  the AC-3 / BP solver runs. Prevents identity conflicts where the
  two halves of the same leaf get assigned to different species.

### Environmental and edaphic intelligence

* **SoilGrids-driven hard masking.** Veto species whose pH, cation
  exchange capacity, clay-content, or nitrogen-level tolerance ranges
  (per the global SoilGrids product) do not include the quadrat's
  coordinates. Acidic-obligate plants get a 0.0 multiplier on
  basic/limestone soils.
* **Ellenberg indicator refinement.** Apply the standardised
  Ellenberg L / T / F / N / R values as soft posteriors over the
  shortlist before threshold selection.
* **PageRank on ecological networks.** Treat the model's initial
  probabilities as a teleportation vector and run Random Walk with
  Restart on the taxonomic co-occurrence graph, so confidence from
  common "keystone" species flows naturally to associated rare
  undergrowth species.

### Mathematical optimisation and acceleration

* **Square-root-space logic evaluation.** Inspired by R. Ryan
  Williams' arXiv:2502.17779 (TIME with `O(sqrt(t log t))` SPACE).
  Three-step recipe: (1) partition the rule graph into blocks
  (generic/family constraints first, then spatial); (2) map
  inter-block dependencies into a tree-evaluation problem; (3)
  evaluate gradients with a Cook-Mertz-style space-efficient walker.
  The motivation is to let the neuro-symbolic constraint refinement
  run inside a 24 GB consumer GPU's working set even on large rule
  graphs.
* **Loopy belief propagation.** Reformulate the multi-tile
  identification task as a Markov Random Field. For each tile `i`,
  define unary potential `psi_i(s) = logit_i(s)` from the GFAM /
  i002 network; for adjacent tiles `(i, j)`, define pairwise
  potential `psi_ij(s, t) = w(s, t)` from ecological co-occurrence.
  Iterative message-passing reinforces weak but ecologically
  consistent signals in obscured tiles using confident neighbours.
  A Rust message-passing loop keeps the per-quadrat cost real-time.
* **Entropy-gated agentic triggering.** Replace the fixed-threshold
  trigger for the LLM arbiter (Future Work #3) with a multi-sample
  predictive variance trigger, so the LLM is invoked exactly when
  the visual ensemble is internally inconsistent.
* **TensorRT + INT8 quantisation.** Move beyond bf16 to INT8
  compilation of the BioCLIP backbone for high-throughput scanning
  on Blackwell hardware. Unblocks long-quadrat-streaming scenarios
  (1000+ FPS).
* **Elastic / Funnel hashing for zero-RAM metadata indexing.**
  Replace the standard `O(N log N)` unique-array rebuild used for
  metadata loading with an `O(1)`-amortised insertion hash structure
  based on elastic / funnel hashing - decoupled probing sequences
  across multiple logarithmically-decreasing sub-tables, with the
  guarantee that placed elements are never moved. Eliminates the
  GC stutters that currently appear when loading the 2.65 M-row
  manifest on consumer hardware.

### Test-time, self-supervised, and adaptive techniques

* **Test-time training (TTT) via SSL rotation / jigsaw.** At
  inference, run ~20 gradient steps per quadrat on a self-supervised
  task (rotation prediction + masked patch reconstruction) before the
  identification forward pass. Exploits the fact that one quadrat is
  many augmentable views of the same patch of ground, and forces the
  backbone weights to internalise the lighting and seasonal
  statistics of that quadrat.
* **Conformal prediction for set-valued output.** Calibrate
  per-quadrat prediction sets with a guaranteed coverage level using
  split-conformal nonconformity scores: produce `C(x)` such that
  `P(Y in C(x)) >= 1 - delta` for a chosen significance level
  (e.g. delta = 0.1). The set size automatically adapts to the visual
  ambiguity of the quadrat, which is a natural fit for the Macro-F1
  scoring rule.
* **Retrieval-augmented classification for the extreme tail.**
  Maintain a FAISS index over per-species prototype embeddings; for
  species with fewer than ten training samples, mix the GFAM / i002
  logits with a cosine-similarity score from the prototype bank,

  ```
  S_final = alpha * S_logit + (1 - alpha) * S_retrieval
  ```

  recovering one-shot visual similarities that gradient-based training
  on a 7,806-class softmax tends to wash out.
* **Prototypical networks for zero/one-shot tail.** For species with
  zero or one training image, perform episodic meta-learning in the
  BioCLIP latent space - identify by cosine distance to a single
  class prototype rather than relying on the softmax head. Composes
  with the FAISS retrieval idea above; the prototype is the natural
  source for the retrieval bank.
* **Uncertainty-gated selective reasoning.** Use Monte Carlo Dropout
  (N ~ 20 forward passes) as a per-tile uncertainty gate. High-variance
  tiles are routed through the expensive Loopy BP / AC-3 / LLM-arbiter
  path; low-variance tiles take a fast-path threshold filter. Targets
  compute at the genuinely ambiguous samples instead of spending it
  uniformly.
* **Asynchronous self-healing networks.** Inject a lightweight
  *Reflexive Layer* (an adapter) at the classification head. When the
  model predicts a species with high visual confidence but is vetoed
  by a neuro-symbolic filter (phenology / AC-3 / Pauli / Ising), the
  conflict is captured by a non-blocking background thread, which
  computes the KL divergence between the raw logits and the
  symbolically-corrected distribution and applies a small gradient
  step to the adapter only. Lets the model heal its domain blind
  spots during deployment without full retraining.

---

## Where to find more

* `report/sections/appendix_development_trace.tex`, "Considered
  Approaches Not Implemented" - the paper's own list of directions
  we scoped and did not run.
* `report/sections/discussion.tex` - shorter prose discussion of
  why the visual encoder is the right place to look for the next
  gains.
* `docs/ALGORITHMS.md` - principled techniques worth implementing
  next (GFAM, HR-KD, neuro-symbolic GCN, hierarchical taxonomic
  loss, PAV-tree calibration, submodular tile selection, Retinex
  preprocessing, etc.).
* `docs/SYSTEMS_DESIGN.md` - hardware, distributed-training, and
  data-engineering directions (Blackwell FP8 surge, ZeRO++ INT8
  allreduce, GPUDirect RDMA, warp-specialised GFAM fusion, L2 cache
  persistence, network satiation).
* `docs/RELATED_WORK.md` - citation bank for future paper writing.
