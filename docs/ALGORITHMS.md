# Algorithms reference

Catalogue of the ML and mathematical techniques considered or
implemented across the project, organised by lifecycle stage. Some
are in the published headline pipeline (BioCLIP 2.5 partial fine-tune,
4x4 tiled inference, ExG vegetation aggregation, circular-Gaussian
phenology prior); others are reference designs that
[`BACKLOG.md`](../BACKLOG.md) points at as future work.

For hardware / systems / distributed-training implementations, see
[`SYSTEMS_DESIGN.md`](SYSTEMS_DESIGN.md). For citations of the
external work cited below, see [`RELATED_WORK.md`](RELATED_WORK.md).

## Table of contents

1. [Preprocessing and domain alignment](#1-preprocessing-and-domain-alignment)
2. [Architecture and feature aggregation](#2-architecture-and-feature-aggregation)
3. [Training objectives](#3-training-objectives)
4. [Neuro-symbolic GCN head](#4-neuro-symbolic-gcn-head)
5. [Calibration](#5-calibration)
6. [Tile-level aggregation](#6-tile-level-aggregation)
7. [Selection rules](#7-selection-rules)
8. [Post-processing reasoning layers](#8-post-processing-reasoning-layers)
9. [Domain adaptation and tail handling](#9-domain-adaptation-and-tail-handling)
10. [Uncertainty gating](#10-uncertainty-gating)

---

## 1. Preprocessing and domain alignment

### 1.1 Multi-Scale Retinex (MSR)

Field quadrats suffer from spatially non-uniform illumination (canopy
shadow, specular reflection, overcast gradient). Per-channel
normalisation is a global op and cannot separate local illumination
from surface reflectance.

For each colour channel `c` and pixel `(x, y)`:

```
r_c(x, y) = log I_c(x, y) - (1/S) * sum_{s in S} log(G_s * I_c)(x, y)
```

with `S = {15, 80, 250}` pixels (Gaussian blur scales). The log
subtraction separates log-reflectance from a local log-illumination
estimate; per-channel min-max normalisation to `[0, 1]` keeps
backbone-compatible statistics. The three scales capture distinct
frequency bands: `sigma = 15` handles sharp shadow edges and specular
highlights, `sigma = 80` removes broad illumination gradients,
`sigma = 250` separates the global sky-to-ground luminance ramp
common in ground-level photography. Inserted between float32
conversion and backbone normalisation, no architecture change.

### 1.2 SAM noise masking

When the ExG vegetation filter (§6.1) is insufficient - e.g. tiles
that contain plants *and* a metric ruler, taxonomic label, finger, or
shadow - selectively deploy Segment Anything (SAM) with
GroundingDINO to identify and generate pixel-perfect negative masks
for the non-botanical objects. Prevents the encoder from overfitting
to artefacts that correlate with sampling protocol.

### 1.3 LUCAS MAE self-supervision

The LUCAS land-use survey provides ~160 GB of *unlabeled* European
field quadrats with the exact perspective and lighting our test set
inherits. Pre-train the ViT and ConvNeXt backbones with Masked
Autoencoding on this corpus before fine-tuning on the PlantCLEF
species labels. Strictly stronger initialisation than ImageNet-22k
for the quadrat distribution we actually score on, especially for the
seasonal-vegetation-density signal.

## 2. Architecture and feature aggregation

### 2.1 Gated Feature Aggregation (GFAM)

Rather than simple probability averaging across the BioCLIP / DINOv3 /
ConvNeXt-V2-L expert ensemble, learn a gating network `G(x)` that
emits a per-image 3-vector `alpha`. The fused representation is

```
f_fused = sum_i alpha_i * Proj_i(f_i)
```

with each `Proj_i` a small adapter that maps the expert's native
embedding dim into a shared latent. The gating weights cache in
constant memory in the custom CUDA kernel (§5.4 in
[`SYSTEMS_DESIGN.md`](SYSTEMS_DESIGN.md)) so the warp-level broadcast
is `O(1)`.

### 2.2 Mixed-resolution inference ensembling

Standard ensembling forces all members to a single inference
resolution. Mixed-resolution ensembling instead runs each member at
its *native* peak resolution (e.g. 224 + 336 for BioCLIP 2.5,
448 + 672 for DINOv3 / ConvNeXt experts) and fuses by weighted mean
across resolutions. Each model operates at its peak signal-to-noise
ratio; local texture and global botanical habit are both retained.
The headline submission uses the dual-resolution 224+336 variant
(see `src/inf_script.py`); the wider 448+672 cross-backbone version
is the future-work target (BACKLOG §1).

### 2.3 Asymmetric Dual-Teacher Distillation (AD-TD)

Single DeiT student modified with two distillation tokens:

* `[DIST_tax]` is supervised by a fine-tuned BioCLIP expert -
  captures the Tree-of-Life relationships.
* `[DIST_geo]` is supervised by a DINOv3 cartographer - captures
  dense spatial boundaries for tile-level segmentation.

At inference, `[CLS]` + `[DIST_tax]` + `[DIST_geo]` outputs are
averaged. Single-pass cost, dual-backbone information. See
[`BACKLOG.md`](../BACKLOG.md) §1.

## 3. Training objectives

### 3.1 High-Resolution Knowledge Distillation (HR-KD)

Composite loss for distilling a high-resolution teacher ensemble into
the student:

```
L_total = 0.3 * L_AsymCE + 0.7 * KL(softmax(z_s / T) || softmax(z_t / T))
```

with `T = 2.0`. The 0.7 weight on the KL term forces the student to
prioritise high-frequency taxonomic signals over its own labels.

A teacher-logit pre-cache is the load-bearing performance trick:
because the teacher weights are frozen, run a single offline pass
over the 1.33 M training images at 512 px on clean (non-augmented)
crops and store averaged teacher logits as an mmap'd float16 array
of shape `(N, 7808)` (~22 GB). The KD branch then reduces to an
indexed memory read rather than two full forward passes through the
triple-backbone ensemble, eliminating ~66 % of per-step FLOPS.

A multi-resolution curriculum progressively shifts the student from
224 px to 512 px over 20 epochs. During the low-resolution phase the
fallback live-teacher inference (used when the cache is unavailable)
is capped at 448 px rather than 512 px, cutting teacher FLOPS by
24 % for the first five epochs without degrading the
resolution-advantage signal.

### 3.2 Hierarchical Taxonomic Loss (HTL)

For a target species `s` at the species-level cross-entropy term,
smooth the label probability across `Genus(s)` so that misidentifying
a sibling species inside the correct genus is penalised less than
crossing a genus boundary. Forces the model to learn shared
morphological features within a genus and provides a "soft landing"
for errors. Composes with the auxiliary genus / family heads from
the i002 model (Section 3 of the paper) as the species-level
counterpart of those auxiliary signals.

### 3.3 Duality-derived asymmetric loss

Standard Asymmetric Loss uses global focusing parameters
`gamma_plus`, `gamma_minus`. Per-class optimal focuses derived from
the Fenchel-conjugate dual let each class get its own focus tuned to
its training frequency. Provides a theoretically grounded suppression
of negative noise on the extreme tail rather than a hand-picked
global hyperparameter. Drop-in replacement for the standard ASL
forward.

## 4. Neuro-symbolic GCN head

Bridge pixel-based deep learning with biological logic via a GCN head
on top of the encoder. Two static matrices are pre-computed once and
re-used at every step:

* A `7,806 x 19` matrix of standardised botanical traits (leaf
  shape, phyllotaxy, etc.) provides per-species feature anchors.
* A `7,806 x 7,806` ecological adjacency matrix `A_eco` derived from
  GBIF co-occurrence statistics provides the graph edges.

### 4.1 Phylogenetic adjacency blending

Add a second adjacency from phylogenetic distance:

```
A_phylo[i, j] = exp(-d_tax(i, j))
d_tax(i, j) = 0  if same genus
              1  if same family but different genus
              2  if different family
```

Row-normalise to `D^{-1} A_phylo` and blend 70 / 30 with the
ecological matrix:

```
A_final = 0.7 * A_eco + 0.3 * A_phylo
```

Two complementary signals: where species *appear* together
(ecological) and where species *evolved* together (phylogenetic).
Species in the same genus share full edge weight, so the GCN
propagates strong gradient signal from well-sampled congeners to
data-scarce tail species. The matrix is built by a chunked offline
script (`build_phylo_adj.py`) using 512-row memory blocks to stay
within a 2 GB RAM budget, then persisted to `data/phylo_adj.npy`.

### 4.2 Bf16 stabilisation for GCN

Three small but load-bearing tricks to keep the GCN forward
numerically stable in bf16:

* **Gradient anchoring** - a final LayerNorm after weight generation
  to preserve unit variance.
* **Symmetric normalisation** - `D^{-1/2} A D^{-1/2}` rather than
  `D^{-1} A` to prevent energy accumulation across graph passes.
* **Learnable temperature** - a clamped logit scale on the softmax
  to prevent exponential overflow.

## 5. Calibration

### 5.1 Hierarchical calibration via PAV-tree

Pool Adjacent Violators on a taxonomic tree is hierarchical isotonic
regression: the predicted probability of a species is *monotonically
bounded* by its parent genus and the genus by the family. The PAV
algorithm enforces this with a convex constraint, regularising
unreliable tail-species estimates with the more stable signal from
higher taxonomic ranks.

Stacks under the headline adaptive-threshold selection. Improves
calibration ECE; in stacked use with cRT / pivot 1 it can over-prune
(see paper Section 6 on the pruning trio).

## 6. Tile-level aggregation

### 6.1 ExG Bayesian vegetation aggregation (`bayesian_veg`)

The published phenology pipeline already uses this; documenting the
math here. For tile `t` compute

```
ExG_t = 2*G_t - R_t - B_t                              # per-pixel Excess Green
v_t   = fraction of pixels with ExG_t > 20 and G > R and G > B  # vegetation fraction
H_t   = - sum_s p_t(s) * log p_t(s)                    # predictive entropy
w_t   = exp(-H_t) * v_t                                # Bayesian weight
```

Tiles with `v_t < 0.15` are dropped before the encoder forward pass
(saves ~20-40 % of tile FLOPS). The surviving tiles aggregate by
weighted mean with weight `w_t`, so confident, vegetation-rich tiles
dominate. Plain softmax-mean (the i002 anchor) is the `w_t = 1` limit.

### 6.2 Submodular tile selection

For inference at 672 px the number of useful tiles can balloon past
200 per quadrat. Define a coverage function `F(S)` on the set of
tiles that exhibits diminishing returns - submodular. Greedy
maximisation selects the `k = 20` most informative tiles per quadrat,
achieving a provable `(1 - 1/e)` approximation of the full-coverage
ensemble while reducing inference compute 5-10x. Composes with the
ExG filter (§6.1) as a downstream selector.

## 7. Selection rules

### 7.1 Adaptive probability threshold (paper anchor)

The headline submission: emit every species with post-LA probability
above `T = 0.03`, clamped to `[2, 10]`. Detail in
`src/inf_script.py`; ablation in paper Section 5.

### 7.2 Frank-Wolfe with Island Biogeography prior

Replace the adaptive threshold with a constrained convex selection.
Pick `y in [0, 1]^N` to maximise `<logits, y>` subject to an
ecological-co-occurrence polytope. Frank-Wolfe (Conditional Gradient)
avoids projection onto the polytope by solving linear sub-problems.
Augment the linear objective with a MacArthur-Wilson count
regulariser

```
nabla_eff = logits - 2 * lambda * (||y||_1 - K_bar) * 1
```

with `K_bar = 8` (empirical GBIF mean species count per 0.25 m^2
plot) and `lambda = 0.05`. Equivalent to a soft `L1` ball centred at
`K_bar` without an explicit projection. Selection becomes provably
consistent with known patch-area species richness, not a global
probability cutoff. See [`BACKLOG.md`](../BACKLOG.md) §6.

### 7.3 Conformal prediction

Split-conformal nonconformity scores fit on a held-out validation
slice produce per-quadrat prediction sets `C(x)` satisfying

```
P(Y in C(x)) >= 1 - delta
```

for a chosen `delta` (e.g. 0.1). Set size automatically adapts to the
visual ambiguity of the quadrat. Natural fit for the Macro-F1
scoring rule because the F1-optimal set size is itself input-dependent.

## 8. Post-processing reasoning layers

### 8.1 Loopy belief propagation

Reformulate multi-tile identification as a Markov Random Field. For
each tile `i`, unary potential

```
psi_i(s) = logit_i(s)
```

For adjacent tiles `(i, j)`, pairwise potential

```
psi_ij(s, t) = w(s, t)        # ecological co-occurrence weight
```

Iterative message passing reinforces weak-but-consistent signals in
obscured tiles using confident neighbours. Soft probabilistic
inference replaces the binary AC-3 pruning the paper appendix
discusses; the message-passing loop lives in Rust to keep per-quadrat
cost real-time.

### 8.2 Disjoint Set Union (DSU) clustering for tile coherence

In dense quadrats one plant body often spans several overlapping
tiles. Cluster tiles by feature cosine similarity (`> 0.95`) using
DSU and treat each cluster as an atomic unit before the BP / AC-3
solver runs. Prevents identity conflicts where the two halves of the
same leaf get different species labels.

### 8.3 Ecological Pauli exclusion (tile-level NMS)

If a tile is confidently claimed (`p > 0.9`) by a single species,
apply an exponential decay penalty to all other species' logits for
that tile. Forces the model to explain the quadrat using *physically
distinct* tiles - one species can be supported by many tiles, but a
single tile cannot weakly endorse many species.

### 8.4 PageRank on the ecological network

Treat the model's initial per-species probabilities as a teleportation
vector and run Random Walk with Restart on the taxonomic
co-occurrence graph. Confidence from common "keystone" species flows
to associated rare undergrowth species through the graph structure
rather than through ad-hoc reweighting.

### 8.5 Allelopathic repulsion

Negative edges in the Frank-Wolfe selection objective (§7.2): if the
solver currently has a confident selection for an allelopathic
species (e.g. *Juglans* / juglone), the LMO score for its known
chemical victims gets penalised. Composes with the IB prior so the
final selected set is allelopathically consistent rather than
ecologically arbitrary.

### 8.6 Ising spin-glass refiner

Model the per-quadrat species set as a spin system: each species is
a spin `s_i in {+1, -1}`, pairwise interactions are the ecological
co-occurrence matrix entries, external field is the visual posterior.
Simulated annealing finds the ground state - jointly optimising
neural confidence against ecological interaction energy. Most
expensive of the reasoning layers but the only one that searches a
*combinatorial* refinement rather than a local one.

### 8.7 Geochemical and edaphic masking

SoilGrids (global soil-property database) gives soil pH, cation
exchange capacity (CEC), and total nitrogen at any coordinate. Hard
masking: an acidic-obligate species gets a `0.0` multiplier on
basic / limestone soils. Different multiplier matrices per soil
property combine multiplicatively.

### 8.8 Ellenberg indicator refinement

Standardised Ellenberg L / T / F / N / R values (light, temperature,
moisture, nitrogen, reaction-pH) for ~3,000 species. Compute a niche
similarity index per quadrat from the climate + soil context, prune
the shortlist by removing species whose indicators contradict the
quadrat's measured / inferred environment.

### 8.9 Thermodynamic phenological prior (Boltzmann form)

The headline submission's phenology pivot, written here in its
thermodynamic form. For species `s` and day-of-year `d in {1..365}`:

```
P(s | d) = (1/Z_s) * exp(-E(s, d) / kT)
E(s, d)  = min(|d - mu_s|, 365 - |d - mu_s|)^2
```

where `mu_s` is the species' phenological mean and `kT` the effective
"phenological temperature" (temporal variance). This is the
von-Mises / wrapped-Gaussian limit of the growing-degree-day model
plant ecologists use; the published pipeline uses the
circular-Gaussian smoothing with `sigma = 18` d that this form
implies, plus `epsilon = 0.05` uniform mass to prevent hard zeros in
the tails. Combined in log-space with the visual posterior at
`beta = 1.0`; see paper Appendix C and `src/inf_script_phen.py`.

## 9. Domain adaptation and tail handling

### 9.1 Test-time training (TTT)

Per-quadrat self-supervised adaptation. For each test quadrat, run
~20 gradient steps on a rotation-prediction + masked-patch-
reconstruction objective before the identification forward pass.
Forces the backbone to internalise the lighting and seasonal
statistics specific to that quadrat. Exploits the fact that one
quadrat is many augmentable views of the same patch of ground.

### 9.2 Retrieval-augmented classification (RAC)

For species in the extreme tail (`< 10` training samples), pre-compute
mean feature prototypes and store in a FAISS index. At inference
fuse the visual posterior with a cosine-similarity score from the
prototype bank:

```
S_final = alpha * S_logit + (1 - alpha) * S_retrieval
```

Recovers one-shot visual similarities that gradient-based training on
a 7,806-class softmax tends to wash out.

### 9.3 Prototypical networks

For zero- and one-shot tail species (no or single training image),
perform episodic meta-learning in the BioCLIP latent space. Identify
by cosine distance to a single class prototype rather than relying on
the softmax head. Composes with §9.2 - the prototype is the natural
source for the FAISS retrieval bank.

## 10. Uncertainty gating

### 10.1 MC Dropout selective reasoning

Run `N ~ 20` forward passes with dropout active. The per-tile predictive
variance is the gate: high-variance tiles get routed through the
expensive Loopy BP / AC-3 / LLM-arbiter path; low-variance tiles take
a fast-path threshold filter. Targets compute at the genuinely
ambiguous samples instead of spending it uniformly.

### 10.2 Entropy-gated agentic triggering

Generalises §10.1 from MC-Dropout variance to multi-sample predictive
variance across an ensemble. Invokes the LLM arbiter (BACKLOG §3)
exactly when the visual ensemble is internally inconsistent, rather
than at a fixed entropy threshold.

---

## Cross-references

* [`BACKLOG.md`](../BACKLOG.md) - the forward-looking entry point;
  techniques marked "future" or "frontier" there have their algorithmic
  detail catalogued here. Notable two-way pointers: GCN head (§4) is
  the locus for ecological Pauli exclusion (§8.3); RAC + Prototypical
  + TTT + MC-Dropout (§§9-10) form the tail-handling stack.
* [`SYSTEMS_DESIGN.md`](SYSTEMS_DESIGN.md) - hardware / kernel /
  distributed-training implementations of the above. GFAM (§2.1) has
  a warp-specialised CUDA kernel there; PAV-tree (§5.1) and
  Frank-Wolfe (§7.2) are CPU-bound and live in Polars / Rust paths.
* [`RELATED_WORK.md`](RELATED_WORK.md) - citation bank for the
  algorithmic foundations above (Land & McCann's Retinex,
  Nowak-Vila's consistent Frank-Wolfe, Snell et al's Prototypical
  Networks, etc.).
* `src/inf_script.py`, `src/inf_script_phen.py` - the two paper
  pipelines that ship some of these techniques (adaptive threshold,
  ExG aggregation, circular-Gaussian phenology).
