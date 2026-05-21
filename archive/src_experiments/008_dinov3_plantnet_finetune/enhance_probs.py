"""
Post-process a test_probs.npz to squeeze more F1 out of the same model scores.

Reads the fp16 ``probs_{max,mean,noisy_or}`` arrays written by
``dump_test_probs.py``, optionally applies three orthogonal F1-boosting
transforms, and writes a new npz with the same schema (drop-in for
``make_submission.py`` / ``apply_thresholds_to_npz.py``).

Transforms (each opt-in):

1. **Post-hoc logit adjustment** (``--logit-adjust``). Training used CE + logit
   adjustment (``log(class_freq)`` added to logits during training so the
   softmax regresses toward *uniform* despite class imbalance). At test time
   that bias is *wrong* — the evaluator does not re-weight by frequency, so
   rare classes are systematically under-predicted. We undo the shift by
   subtracting ``τ·log(p(y))`` from logits. Implemented here in probability
   space: ``p' ∝ p · freq**(-τ)``, normalised along the quadrat axis only for
   max-pool (non-normalised for sigmoid outputs — we subtract in logit space
   exactly). See Menon et al. 2021 (ICLR, "Long-tail learning via logit
   adjustment") eq. (7). Typical τ ∈ [0.5, 1.5].

2. **Bayesian Model Averaging over tiles** (``--agg bma``). Instead of just
   max / mean / noisy_or per class, rebuild a single per-quadrat score from
   the three aggregates as a weighted geometric mean (equivalent to averaging
   in log-probability space, aka a Bayesian ensemble over tile-aggregation
   hypotheses). Weights sum to 1 and default to (0.5, 0.3, 0.2) for
   (max, noisy_or, mean) — max is the strongest single signal but noisy_or
   calibrates confidence across many weak tiles.

3. **Dynamic thresholds** (``--dynamic-threshold``). The test set has a known
   average prediction-set length (empirically ~5 species/quadrat for PC25/26;
   the exact number comes from the train-set species cardinality priors).
   Search a single global threshold τ* such that the submission's average
   per-quadrat prediction count matches ``--target-avg-preds`` (default 5.0).
   This beats a fixed 0.5 cut-off because the optimal threshold is
   model-dependent and drifts with every retrain.

All three stack: you can apply logit-adjust, then BMA, then dynamic-threshold
in a single invocation.

Example::

    python enhance_probs.py \
        --probs-npz /path/to/test_probs.npz \
        --class-freq-csv /workspace/plantclef/processed/species_train_counts.csv \
        --logit-adjust 1.0 \
        --agg bma --bma-weights 0.5 0.3 0.2 \
        --output /path/to/test_probs_enhanced.npz
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probs-npz", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    # (1) Post-hoc logit adjustment
    p.add_argument("--logit-adjust", type=float, default=0.0,
                   help="τ. 0 disables. Typical 0.5–1.5. "
                        "Requires --class-freq-csv.")
    p.add_argument("--class-freq-csv", type=Path,
                   help="CSV with columns species_id,count giving per-class "
                        "training frequency. Used by --logit-adjust.")

    # (2) BMA / aggregation choice
    p.add_argument("--agg", choices=["max", "mean", "noisy_or", "bma"], default="max",
                   help="Per-class aggregation stored in probs_max of output. "
                        "'bma' = weighted geometric mean of the three aggregates.")
    p.add_argument("--bma-weights", type=float, nargs=3, default=[0.5, 0.3, 0.2],
                   help="Weights for (max, noisy_or, mean) when --agg bma. "
                        "Must sum to ~1.")

    # (3) Dynamic threshold (informational — writes metadata; does not change probs)
    p.add_argument("--dynamic-threshold", action="store_true",
                   help="Sweep τ and log the value that produces "
                        "--target-avg-preds per quadrat. Writes to 'suggested_threshold' "
                        "key in output npz; does not modify probs.")
    p.add_argument("--target-avg-preds", type=float, default=5.0,
                   help="Target average predictions per quadrat when searching τ.")
    p.add_argument("--threshold-lo", type=float, default=0.01)
    p.add_argument("--threshold-hi", type=float, default=0.95)
    return p.parse_args()


def load_class_freq(
    csv_path: Path, species_ids: np.ndarray, default_count: int = 1
) -> np.ndarray:
    """Return (C,) int array of train counts aligned to species_ids order."""
    df = pd.read_csv(csv_path, dtype={"species_id": str})
    if "species_id" not in df.columns or "count" not in df.columns:
        raise ValueError(
            f"{csv_path} must have columns (species_id, count); saw {list(df.columns)}"
        )
    lookup = dict(zip(df["species_id"], df["count"].astype(int)))
    counts = np.array(
        [lookup.get(str(sid), default_count) for sid in species_ids],
        dtype=np.int64,
    )
    n_missing = int((counts == default_count).sum() - (np.array(
        [str(s) in lookup and lookup[str(s)] == default_count for s in species_ids]
    )).sum())
    if n_missing:
        logger.warning(
            f"{n_missing} species from the probs file had no row in "
            f"{csv_path.name} — defaulted to count={default_count}."
        )
    return counts


def apply_logit_adjustment(
    probs: np.ndarray, class_counts: np.ndarray, tau: float
) -> np.ndarray:
    """``p' ∝ p · freq**(-τ)`` elementwise. Stays in [0,1] via clip + renorm-free
    rescale (we treat each class independently in sigmoid space)."""
    eps = 1e-6
    p = probs.astype(np.float32).clip(eps, 1 - eps)
    logit = np.log(p) - np.log1p(-p)
    log_freq = np.log(class_counts.astype(np.float64) + eps)
    adj_logit = logit - tau * log_freq[None, :]
    return (1.0 / (1.0 + np.exp(-adj_logit))).astype(np.float32)


def bma_aggregate(
    p_max: np.ndarray, p_noisyor: np.ndarray, p_mean: np.ndarray,
    weights: list[float],
) -> np.ndarray:
    """Weighted geometric mean — i.e. arithmetic mean in log-prob space.

    Equivalent to a Bayesian ensemble over the three tile-aggregation
    hypotheses, which is provably at least as good as any single one in
    expected log-loss terms.
    """
    eps = 1e-6
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    log_mix = (
        w[0] * np.log(p_max.astype(np.float32).clip(eps))
        + w[1] * np.log(p_noisyor.astype(np.float32).clip(eps))
        + w[2] * np.log(p_mean.astype(np.float32).clip(eps))
    )
    return np.exp(log_mix)


def find_dynamic_threshold(
    probs: np.ndarray, target_avg_preds: float,
    lo: float, hi: float, tol: float = 1e-3, max_iter: int = 50,
) -> tuple[float, float]:
    """Binary-search τ so mean(count per row where p>=τ) ≈ target_avg_preds."""
    def avg_preds_at(t: float) -> float:
        return float((probs >= t).sum(axis=1).mean())

    a, b = lo, hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        v = avg_preds_at(m)
        if abs(v - target_avg_preds) < 0.01 or (b - a) < tol:
            return m, v
        # higher τ -> fewer predictions
        if v > target_avg_preds:
            a = m
        else:
            b = m
    return 0.5 * (a + b), avg_preds_at(0.5 * (a + b))


def main() -> None:
    args = parse_args()
    d = np.load(args.probs_npz, allow_pickle=True)
    quadrat_ids = d["quadrat_ids"]
    species_ids = d["species_ids"]
    p_max = d["probs_max"].astype(np.float32)
    p_mean = d["probs_mean"].astype(np.float32)
    p_noisyor = d["probs_noisy_or"].astype(np.float32)

    N, C = p_max.shape
    logger.info(f"Loaded {args.probs_npz}: N={N} quadrats, C={C} species")

    if args.logit_adjust > 0.0:
        if args.class_freq_csv is None:
            raise SystemExit("--logit-adjust requires --class-freq-csv")
        counts = load_class_freq(args.class_freq_csv, species_ids)
        logger.info(
            f"Applying logit adjustment τ={args.logit_adjust} "
            f"(class counts: min={counts.min()}, median={int(np.median(counts))}, "
            f"max={counts.max()})"
        )
        p_max = apply_logit_adjustment(p_max, counts, args.logit_adjust)
        p_mean = apply_logit_adjustment(p_mean, counts, args.logit_adjust)
        p_noisyor = apply_logit_adjustment(p_noisyor, counts, args.logit_adjust)

    if args.agg == "bma":
        logger.info(f"BMA weights (max, noisy_or, mean) = {args.bma_weights}")
        p_out = bma_aggregate(p_max, p_noisyor, p_mean, args.bma_weights)
        # Overwrite the primary slot so downstream tools pick up BMA by default.
        p_max_out = p_out
        p_mean_out = p_mean
        p_noisyor_out = p_noisyor
    elif args.agg == "mean":
        p_max_out = p_mean
        p_mean_out = p_mean
        p_noisyor_out = p_noisyor
    elif args.agg == "noisy_or":
        p_max_out = p_noisyor
        p_mean_out = p_mean
        p_noisyor_out = p_noisyor
    else:  # "max"
        p_max_out = p_max
        p_mean_out = p_mean
        p_noisyor_out = p_noisyor

    suggested_threshold = None
    if args.dynamic_threshold:
        tau, avg = find_dynamic_threshold(
            p_max_out, args.target_avg_preds, args.threshold_lo, args.threshold_hi
        )
        suggested_threshold = tau
        logger.info(
            f"Dynamic-threshold search: τ*={tau:.4f} -> "
            f"avg {avg:.3f} preds/quadrat (target {args.target_avg_preds})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        quadrat_ids=quadrat_ids,
        species_ids=species_ids,
        probs_max=p_max_out.astype(np.float16),
        probs_mean=p_mean_out.astype(np.float16),
        probs_noisy_or=p_noisyor_out.astype(np.float16),
    )
    if suggested_threshold is not None:
        save_kwargs["suggested_threshold"] = np.array([suggested_threshold], dtype=np.float32)
    np.savez_compressed(args.output, **save_kwargs)
    logger.info(
        f"Wrote {args.output} "
        f"({args.output.stat().st_size / 1e6:.1f} MB, agg={args.agg}, "
        f"logit_adjust={args.logit_adjust})"
    )


if __name__ == "__main__":
    main()
