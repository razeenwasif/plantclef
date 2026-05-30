# Experiment: Logit-Adjustment Collapse in Multi-Label Sigmoid Heads

**Status:** Resolved (2026-04-21)
**Scope:** Phase 2 LoRA fine-tune of DINOv3 ViT-L/16 on synthetic PlantCLEF mosaics (`src_experiments/005_dinov3_multilabel/train.py`).
**TL;DR:** Applying the softmax-CE long-tail logit-adjustment trick (Menon et al., 2021) to an `AsymmetricLoss` sigmoid head on a 7806-class problem drove the model to a degenerate "all-positive" optimum. Every class's sigmoid output saturated near 1.0 regardless of input. The fix was one line: do not add log-priors to the pre-sigmoid logits.

---

## 1. Setup

Phase 2 fine-tunes DINOv3 with LoRA on synthetic K-plant mosaics (K ∈ {1..5}) using `AsymmetricLoss` (`γ_pos = 1`, `γ_neg = 4`, `clip = 0.05`) — see `src/training/losses.py`. The head produces one logit per species; after sigmoid, each species is an independent Bernoulli prediction. The logit-adjustment term `τ · log π_c` (τ = 1, π_c = class frequency) is added to logits *before* the loss, intended to suppress frequent classes (Menon et al., "Long-tail learning via logit adjustment", ICLR 2021).

Training ran for 5 epochs × 12500 steps (1 M samples/epoch, eff. batch 512). Final reported `avg_loss = 3.58`. No obvious NaNs, no training errors.

## 2. Symptom

`run_validation.py` swept four aggregation modes × six thresholds over 1000 mosaics. Every cell in the resulting table returned the exact same macro-F1 per sample — not within noise, but bit-identical to 16 decimal places:

```
agg_mode     threshold   macro_F1_per_sample   n_mosaics
max          0.10        0.008085503482025221  1000
max          0.20        0.008085503482025221  1000
...
noisy_or     0.60        0.008085503482025221  1000
```

This is not a real metric. Four mathematically distinct aggregations (max, mean, mean-top-m, noisy-or) on six different thresholds cannot produce identical predictions unless the predictions do not depend on the score function at all.

## 3. Diagnostic pass

### 3.1 Is the predict path ignoring agg_mode or threshold?

Inspected `run_validation.py`. The scoring path correctly re-encodes probabilities per `agg_mode` (noisy-or re-logits; everything else uses raw probs) and applies threshold + top-N:

```python
sorted_vals, sorted_idx = image_scores.sort(descending=True)
sorted_vals = sorted_vals[: args.top_n]    # top_n = 20
sorted_idx  = sorted_idx[: args.top_n]
pred_set = {species_ids[idx] for val, idx in zip(...) if val >= thr}
```

So the code is correct. If the top-20 scores are always ≥ 0.6, threshold is effectively unused and `pred_set` size stays at 20 — that explains *threshold* constancy. But it does not explain *agg-mode* constancy.

### 3.2 Probe model output distribution

Wrote `diagnose_predictions.py`: forward 5 distinct mosaic canvases through the full model, print sigmoid statistics per class, per tile.

```
[mosaic 0] truth=['1392770','1458364','1509263'], n_tiles=1
  per-tile sigmoid prob: min=0.9458  median=0.9995  max=1.0000  mean=0.9995
  top-20 overlap across agg_modes (mosaic 0):
    max ∩ mean      = 20/20
    max ∩ noisy_or  = 20/20
    mean ∩ noisy_or = 20/20
```

Two findings:
1. **The sigmoid output is saturated near 1.0 for every class on every input.** Minimum probability across 7806 classes is 0.9458, median 0.9995. The model has collapsed to "predict everything as positive."
2. **With only 1 tile per 384-px mosaic at stride 256, max / mean / noisy-or are mathematically identical per-class.** That explains the bit-equal F1 across agg_modes: they *cannot* differ on this tiling config.

Across-mosaic top-20 sets *did* vary (1/20–5/20 overlap between mosaic 0 and others), confirming the model is not literally input-invariant — but the variation is driven by tiny floating-point deltas between values all in `[0.99994, 0.99997]`, which is noise, not discrimination.

### 3.3 Quantify the signal

`diagnose_recall.py` on 200 mosaics (avg truth size 2.25):

| K | recall@K | macro-F1 (top-K, no threshold) |
|---|---|---|
| 1 | 1.87% | 0.022 |
| 3 | 2.04% | 0.014 |
| 5 | 2.71% | 0.014 |
| 10 | 4.58% | 0.013 |
| 20 | 5.08% | 0.008 |
| 50 | 6.75% | 0.004 |

Chance recall@K on 7806 classes with truth size ~2 is `K × 2 / 7806`: 0.026% at K=1, 0.64% at K=50. So the model is ~70× above chance at K=1 but only ~10× above chance at K=50 — there is *some* ranking signal but it is too noisy to build a submission around.

## 4. Root cause

The pure-PyTorch ASL forward (`src/training/losses.py:88`):

```python
x_adj = x + self.logit_adjustments
xs_pos = torch.sigmoid(x_adj)
...
loss = weights * (y * log(xs_pos) + (1-y) * log(1 - xs_pos + clip))
```

And the adjustment tensor (`train.py:145-149`):

```python
def logit_adjustments_from_counts(counts, tau=1.0):
    priors = (counts + 1) / (counts.sum() + len(counts))
    return tau * torch.log(priors)
```

With 1.4 M samples across 7806 classes, `π_c ≈ 1/7806 = 1.28e-4`, so `log π_c ≈ −8.96`. The adjustment tensor is therefore a near-uniform vector of **−9** added to every logit prior to sigmoid.

### Why this is wrong for sigmoid loss

Menon et al.'s formulation is derived for **softmax** cross-entropy:

```
P(y=c | x) ∝ exp(z_c)    →   logit adjustment: z_c ← z_c + τ log π_c
```

The normalisation in the softmax (Σ_c exp(z_c) in the denominator) is what turns the log-prior shift into a per-class *suppression* of frequent classes. There is no coupling across classes in the sigmoid multi-label case: each class has its own independent Bernoulli loss.

Adding −9 to *every* sigmoid logit just says "start by assuming every class is massively negative." The loss-minimum response of the head is to push raw logits **up by +9 across the board** so that `x − 9` lands near the target distribution. There is no mechanism in the loss that distinguishes classes: the per-class ASL gradients for positives pull logits up, the per-class gradients for negatives barely move (ASL's `clip = 0.05` floors the negative loss exactly in this near-saturated regime).

Result: a head with raw output ~ +9 uniformly across all 7806 classes, sigmoid ≈ 1, no discrimination.

### Why Phase 1 did not collapse

Phase 1 used the same log-priors, but through `LogitAdjustmentLoss` wrapping **cross-entropy** (softmax). There, the log-prior shift is theoretically correct: it penalises frequent classes via the softmax normaliser. Phase 1 reached `val_top1 = 11.97%` / `val_top5 = 26.13%` on held-out single-plant images — a healthy result.

### Why the average loss (3.58) looked fine

With the head fully collapsed (sigmoid ≈ 0.999 for all 7806 classes), the per-class ASL loss is:

- **Positive targets** (~2 per sample): `−log(sigmoid(9)) ≈ 1.2e-4`, weighted by `(1 − 0.999)^1 = 1e-3` → negligible.
- **Negative targets** (~7804 per sample): `−log(1 − 0.999 + 0.05) ≈ −log(0.051) ≈ 2.98`, weighted by `(1 − 0.999)^4 ≈ 1e-12` → also ~0 in practice because of the `(1 − p_t)^γ` focal weight.

The `clip` term and focal weight together make the saturated regime approximately free in loss space. `avg_loss = 3.58` was therefore not informative about model quality.

## 5. Fix

One-line change in `src_experiments/005_dinov3_multilabel/train.py:349-355`:

```diff
- # Class priors -> logit adjustment for ASL
- counts = class_counts_from_metadata(df, species_ids)
- adj = logit_adjustments_from_counts(counts).to(args.device)
- criterion = AsymmetricLoss(
-     gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
-     logit_adjustments=adj, use_fused=False,
- ).to(args.device)
+ # Logit adjustment is the softmax-CE long-tail trick (Menon 2021); applying
+ # log(prior) ≈ -9 to a sigmoid head makes the loss-minimum collapse to
+ # all-positive (verified by diagnose_predictions.py). ASL's gamma_neg=4 +
+ # clip=0.05 already handle multi-label class imbalance — leave adj=None.
+ criterion = AsymmetricLoss(
+     gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
+     logit_adjustments=None, use_fused=False,
+ ).to(args.device)
```

ASL's `γ_neg = 4` focal weighting is itself the multi-label-safe imbalance mechanism: it down-weights easy negatives without creating a global logit bias. No per-class prior correction is needed.

## 6. Verification

After restart with `logit_adjustments=None`, Phase 2 logs:

```
11:55:42  P2 ep1 step0/62500     loss=107.17
11:57:22  P2 ep1 step550/62500   loss=126.48
12:01:35  P2 ep1 step3200/62500  loss=38.91
12:29:12  P2 ep1 step6950/62500  loss=14.95
13:01:44  P2 ep1 step16250/62500 loss=4.83
```

Compared with the broken run (loss stabilised near the collapsed floor almost immediately), the fixed run starts at ~100× higher loss (meaningful positive gradient) and decays smoothly — exactly what a healthy multi-label classifier looks like on 7806 classes.

Recall@K, final macro-F1 numbers, and Kaggle leaderboard comparison against the BioCLIP few-shot baseline are deferred until training completes.

## 7. Takeaways

1. **Logit-adjustment formulas are loss-specific.** The `z_c ← z_c + τ log π_c` trick applies to *softmax* CE, not independently-sigmoided multi-label losses. Always re-derive the correction for the actual loss you are using.
2. **Absolute loss values are not diagnostics in multi-label regimes.** With a focal weight and a clip, large regions of output space are approximately zero-cost. A "looks fine" final loss can hide total collapse. Validate by inspecting the actual sigmoid output distribution before trusting the model.
3. **Identical metrics across unrelated hyperparameters is the loudest possible signal that something is ignoring its inputs.** `run_validation.py` gave bit-equal F1 across 24 configurations — that is the first thing to look for when a metric table looks too clean.
4. **Keep the diagnostic script.** `diagnose_predictions.py` and `diagnose_recall.py` are committed alongside the fix; they are the fastest way to confirm "is my multi-label head actually discriminative?" on any future checkpoint.

## References

- Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2021). Long-tail learning via logit adjustment. *ICLR 2021*.
- Ridnik, T., Ben-Baruch, E., Zamir, N., Noy, A., Friedman, I., Protter, M., & Zelnik-Manor, L. (2021). Asymmetric Loss for Multi-Label Classification. *ICCV 2021*.
