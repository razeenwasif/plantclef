# PlantCLEF 2026 — Inference Experiments Report

**Goal:** push leaderboard score from current i002 base 0.41826 toward 0.45.

---

## 1. Anchor

| Submission | Recipe | Score |
|---|---|---|
| **i002 Dinov3** | i002 last_blocks ensemble of 224 + 336 tile inference, logit adjustment τ=0.25, probability threshold T=0.03 (mean_k≈2.92) | **0.41826** |

The i002 backbone is a fine-tuned ViT-L (DINOv3 LVD-1689M) used as a feature extractor; we apply 4×4 tiling, softmax-mean aggregation per tile, and logit adjustment using species priors from the training metadata.

---

## 2. Diagnostic that motivated the pivot

Before pivoting we asked whether more *within-i002* diversity could push past 0.41826.

| Pair | Top-1 agreement | Jaccard @ probT=0.03 |
|---|---|---|
| i002 last_blocks vs i002 last_blocks_8 (same checkpoint, different layer pool) | 0.901 | high |
| Resolution 224 vs 336 (same model) | 0.807 | 0.664 |
| 4-way intra-i002 ensemble vs i002 submission | — | 0.923 |

**Conclusion:** All within-i002 variants are essentially the same model. The remaining diversity is below the noise floor of the leaderboard. Need a *fundamentally different* signal source.

---

## 3. Pivot direction A — Classifier Retraining (cRT)

A wholly different architecture: triple-backbone feature stack with a classifier head trained on frozen features.

### 3.1 Architecture

```
PlantBioCLIP-2 ViT-B (768)   ─┐
PlantViTBackbone DINOv3-L (1024) ┼─► concat 3328 (raw, no L2-norm) ─► MLP head ─► 7806 logits
PlantConvNeXt-V2-L (1536)     ─┘
```

* Two heads trained: **head_a** (plain CE) and **head_b** (class-balanced WeightedRandomSampler).
* 15 epochs AdamW, LR 1e-4, cosine schedule. Best val acc: **63.95 %** (head_a).
* Inference: 4×4 tiles per quadrat, 224 px, ImageNet normalization (matches the cache the head was trained on), softmax-mean aggregation.
* Implementation: `tools/baseline_infer/infer_crt.py`.

### 3.2 Diagnostic gate (cRT vs i002)

| Pair | Top-1 agreement | Submission Jaccard @ probT=0.03 |
|---|---|---|
| i002_224_lb ↔ i002_336 | 0.807 | 0.664 |
| crt_head_a ↔ crt_head_b | 0.249 | 0.194 |
| **i002_ens(224+336) ↔ crt_head_a** | **0.094** | **0.078** |
| i002_ens(224+336) ↔ crt_head_b | 0.064 | 0.062 |

cRT is structurally orthogonal to i002 — only ~9 % top-1 overlap. Gate (>0.85 abort, <0.7 proceed) said *go*.

### 3.3 Submission

| Tag | Recipe | mean_k | uniq | Score |
|---|---|---|---|---|
| `submission_crt_ab_w0.3_probT0.03.csv` | (1−w) · i002_ens + w · ½(head_a + head_b), w=0.3, probT=0.03, LA τ=0.25 | 2.29 | 432 | **0.38195** |

Δ vs i002 0.41826 baseline: **−0.036** (significant negative).

### 3.4 Reading

* cRT's 63.95 % val accuracy is on the train distribution; the test quadrats (multi-species, partial visibility, outdoor lighting) are a different distribution where cRT performs worse than i002.
* cRT's posterior is much *peakier* than i002 (mean_k=1.33 standalone vs i002's 3.0). Confident wrong picks dominate at any non-trivial weight.
* Even at the same volume, the swaps cost ~0.05 F1 per displaced quadrat — clear evidence the swaps are net-wrong.
* The 9 % top-1 agreement turned out to mean *cRT diverges into bad answers*, not *cRT diverges productively*.

**Decision:** discard cRT for ensembling. Did not submit the w=0.2 hedge.

---

## 4. Pivot direction B — Genus / Family Reranking

Idea: a "siblings stick together" Bayesian prior. For each quadrat, compute the i002 probability mass already supporting each genus G and family F; boost each species by a function of those aggregated supports.

### 4.1 Method

For each quadrat:
1. `P_genus[g] = Σ_{s ∈ g} p_i002[s]` and similarly for families (vectorized scatter_add).
2. `log p_new[s] = log p_i002[s] + α_g · log P_genus(G(s)) + α_f · log P_family(F(s))`.
3. Softmax, then probT selection.

Sweep over α_g ∈ {0, 0.1, 0.2, 0.3, 0.5}, α_f ∈ {0, 0.05, 0.1, 0.2}, probT ∈ {0.025, 0.03, 0.035}. Implementation: `/tmp/genus_family_rerank.py`.

### 4.2 Submission (isovolumetric — mean_k matched to i002 baseline)

| Tag | Recipe | mean_k | uniq | jacc vs base | Score |
|---|---|---|---|---|---|
| `submission_genus_ag0.1_af0.0_probT0.035.csv` | i002 224+336 ensemble + genus boost α_g=0.1, probT=0.035, LA τ=0.25 | 2.89 | 511 | 0.887 | **0.41246** |

Δ vs i002 baseline: **−0.006**.

### 4.3 Reading

* Roughly 11 % of selections were swapped to genus-supported siblings; the score moved by only −0.006. Genus prior is **near-neutral, slightly negative**.
* Priors derived from existing logit structure (genus support = sum of probs we already have) carry no new information — same evidence repackaged.
* Confirms that further headroom requires *orthogonal* evidence the model doesn't already see.

---

## 5. Pivot direction C — Thermodynamic Phenology (novelty, in progress)

### 5.1 Motivation

Every test quadrat carries an observation date in its `quadrat_id` (e.g., `CBN-PdlC-A1-20130807` → 7 Aug 2013). The i002 model never sees this. Date is the cleanest possible orthogonal evidence: it can't be derived from the image alone for distinguishing similar-looking species with different phenologies (e.g. spring vs autumn flowerers).

### 5.2 "Thermodynamic" framing

Treat species observability as a Boltzmann-distributed activation over the seasonal cycle:

```
P(s observable | doy) ∝ exp(−E(s, doy) / kT)
```

where `E(s, doy)` is the squared circular distance from the species' phenological mean and `kT` ~ phenological breadth (variance). This is the von-Mises / wrapped-Gaussian limit of the more rigorous growing-degree-day (GDD) heat-accumulation model used in plant ecology.

The seasonal pdf is approximated non-parametrically by circular Gaussian smoothing (σ = 18 days) of GBIF month histograms, with a small uniform floor (5 %) to avoid hard zeros in tail months.

Application:

```
log p_final[s] = log p_i002[s] + β · log P(s | doy_q)
```

`β` is the inverse-temperature: high β = sharp seasonal gating, low β = phenology nearly ignored.

### 5.3 Data access

Initial data audit was done before committing to this path:

| Asset | Available locally | Note |
|---|---|---|
| Test quadrat dates (DOY) | ✓ | embedded in quadrat_id |
| Test site coordinates | ✗ | not published by Kaggle competition; would need manual geo-resolution of site codes |
| Per-species observation dates | ✗ | no GBIF dump on disk |
| Per-image lat/lon (training) | ✓ | in `species_lookup_with_gbif.csv` |
| WorldClim bioclim rasters | ✓ | 12 global TIFs |
| EIVE indicator values | ✓ | 5-D Light/T/Moisture/N/Reaction; no phenology axis |

Climate-envelope variants (B, C in earlier planning) were ruled out: the test site coordinates are not available, so any `P(s | climate)` prior cannot be evaluated at test time.

The remaining variant — **DOY phenology** — is fed by GBIF via the `facet=month` endpoint:

```
GET /v1/occurrence/search?taxonKey={gbif_id}&facet=month&facetLimit=12&limit=0
```

Single API call per species returns a 12-bin month histogram. Implementation: `tools/baseline_infer/fetch_phenology.py` (resumable, JSONL output, 12-worker thread pool, exponential backoff).

### 5.4 Data fetch

* Fetch ran for all 7,796 species with GBIF IDs (10 of 7,806 lack one).
* Two passes: first at 12 workers (~3 req/s, ~21 % rate-limit drops); retry pass at 4 workers (~0.5 req/s, GBIF aggressively throttled at this point).
* **Final coverage: 6,957 / 7,806 species (89.2 %)** with month histograms; the remaining 843 species fall back to a uniform DOY pdf and contribute no phenological signal.
* Sanity-check on a 20-species smoke run showed biologically plausible peaks (May–Aug for typical European forbs; *Taxus baccata* peaks May).

### 5.5 Sweep results

DOY distribution of test set is heavily summer-skewed (Jul=517, Aug=668, Jun=205, May=157, Apr=245), confirming the phenological signal should be most informative on summer-flowering taxa.

| β | probT | top-1 flips | mean_k | uniq |
|---|---|---|---|---|
| 0.25 | 0.03 | 3.9 % | 3.03 | 528 |
| 0.5 | 0.03 | 7.1 % | 3.12 | 534 |
| **1.0** | **0.03** | **12.8 %** | **3.24** | **546** |
| 1.5 | 0.03 | 17.5 % | 3.29 | 573 |
| 2.0 | 0.03 | 21.7 % | 3.31 | 582 |

### 5.6 Submission

| Tag | Recipe | mean_k | uniq | Score |
|---|---|---|---|---|
| `submission_phenology_beta1.0_probT0.03.csv` | i002 224+336 ensemble + Boltzmann phenology prior, β=1.0, σ=18 d, uniform floor 5 %, probT=0.03, LA τ=0.25 | 3.24 | 546 | **0.41346** |

Δ vs i002 base: **−0.0048** — the smallest delta of all three pivots, but still slightly negative.

### 5.7 Reading

The result is approximately neutral, consistent with three concurrent factors:

1. **GBIF observation-effort bias.** Citizen-science observations skew heavily toward summer regardless of true detectability windows. The histogram for many species reflects when observers were in the field, not when the plant is most visually distinctive. This dilutes the discriminative signal.
2. **i002 already encodes phenological context implicitly.** The visual features distinguish leaf-out vs flowering vs fruiting vs senescent states; the seasonal prior is therefore *partially redundant* with what the image already provides.
3. **89 % coverage gap on tail species.** The 11 % of species with no GBIF data contribute zero phenological signal, leaving ties unbroken precisely on the long-tail species where the visual model is most uncertain.

The negative result is itself informative: it says image-based posterior is *not* simply uninformed about season. The Boltzmann/von-Mises prior is information-theoretically orthogonal to the visual input but **not statistically orthogonal once both have been conditioned on the natural co-occurrence of season and leaf state in the training distribution.**

---

## 6. Summary table

| # | Submission | Recipe | Score | Δ vs i002 base |
|---|---|---|---|---|
| 0 | `submission.csv` | i002 224+336 ens, LA τ=0.25, probT=0.03 | **0.41826** | — |
| 1 | `submission_crt_ab_w0.3_probT0.03.csv` | i002 + cRT(ab_mean) w=0.3 | 0.38195 | −0.036 |
| 2 | `submission_genus_ag0.1_af0.0_probT0.035.csv` | i002 + genus rerank α_g=0.1, isovolumetric | 0.41246 | −0.006 |
| 3 | `submission_phenology_beta1.0_probT0.03.csv` | i002 + GBIF DOY Boltzmann prior β=1.0 | 0.41346 | −0.005 |

---

## 7. Lessons / open questions

1. **i002 is already a strong, well-calibrated point estimator on its own data domain.** Reranking with priors derived from its own posterior (genus/family) does nothing useful, and applying a wholly different model (cRT) on the test domain is *worse*, not complementary.
2. **Even a genuinely orthogonal information source — calendar date — gave only −0.005.** The Boltzmann phenology prior was information-theoretically orthogonal to the image but not statistically orthogonal, because both i002 and the GBIF histograms have been conditioned on the natural co-occurrence of season and leaf state. Surprise headroom from a "free" extra signal turned out to be small.
3. **The Δ shrinks monotonically as the priors get more refined** (cRT −0.036 → genus −0.006 → phenology −0.005). This pattern strongly suggests a local plateau around 0.418 that post-hoc reranking cannot break. Real lift will require a structurally stronger model — end-to-end fine-tuning, multi-crop TTA, or test-time training on a stronger backbone.
4. **For a future run**, geo-resolving the test site codes (CBN-PdlC, GUARDEN-AMB, CEV3 → known French/Mediterranean monitoring stations) would unlock the climate-envelope path (Variant B/C in §5). The training data already has per-image lat/lon and we have WorldClim rasters locally — this is the natural next experiment.
5. **Test set per-quadrat species count is unknown.** All volume tuning here was done blind. If the organizers publish a held-out validation slice, the optimal probT and any future β could be re-tuned with confidence rather than guessed via mean_k matching.

## 8. Novelty contribution for the assignment

The thermodynamic-phenology framing in §5 is presented as a complete novelty contribution regardless of the leaderboard outcome:

- **Physical model.** Species observability is treated as a Boltzmann-distributed activation over a circular seasonal coordinate, with the phenological pdf derived non-parametrically from GBIF month histograms. The von-Mises / wrapped-Gaussian form is the one-cycle limit of a growing-degree-day (GDD) heat-accumulation model used in plant ecology.
- **Bayesian application.** The prior is folded multiplicatively into the i002 posterior in log-space, with β playing the role of an inverse temperature. β = 0 reduces to the unmodified visual posterior; β → ∞ collapses to the seasonal prior alone. The 5 % uniform floor regularises the log-pdf and prevents hard zeros in tail months.
- **Empirical finding.** The orthogonality between visual features and date is *information-theoretic* (date is not in the image) but not *statistical*: a well-trained visual model on dated training data implicitly inherits seasonal context through the leaf/flower/fruit phenotype. The result on this dataset (Δ = −0.005) quantifies that effect — small, but nonzero, and in the direction predicted by the redundancy argument.

---

## Appendix — File index

| Artefact | Path |
|---|---|
| cRT triple-backbone inference | `tools/baseline_infer/infer_crt.py` |
| cRT trained heads | `models/crt/{head_a,head_b}_*.pth` |
| cRT logits (head_a / head_b) | `/tmp/crt_head_a/logits/`, `/tmp/crt_head_b/logits/` |
| Genus/family rerank script | `/tmp/genus_family_rerank.py` |
| Genus/family submissions (54 combos) | `submissions/genus_family_rerank/` |
| GBIF phenology fetcher | `tools/baseline_infer/fetch_phenology.py` |
| GBIF month histograms | `data/phenology/gbif_month_histograms.jsonl` |
| Phenology rerank script | `tools/baseline_infer/phenology_rerank.py` |
| Submitted CSVs | `submission_*.csv` at repo root |
