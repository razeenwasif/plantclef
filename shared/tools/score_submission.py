#!/usr/bin/env python3
"""
Score a PlantCLEF-style submission CSV against ground-truth metadata and
produce a folder of diagnostic plots.

What it does
------------
Reads a `submission.csv` (`quadrat_id, species_ids` as written by
`shared/bioclip25_multitask/infer_tiles_adaptive.py`) and a
ground-truth CSV (PlantNet's `plantnet300K_metadata.csv` or a
PlantCLEF-equivalent). For every prediction it computes:

* overall top-1 / top-K accuracy at the species level,
* the same aggregated to genus and family levels,
* per-species top-1 accuracy distribution + long-tail diagnostics,
* the top-N most-confused species pairs,
* a family-level confusion matrix (~100x100, readable as a heatmap),
* train-image-count vs val-top-1-accuracy scatter, if a manifest is
  provided (the long-tail correlation diagnostic).

Outputs
-------
Under `--output-dir`:

* `summary.json` — all numerical metrics
* `per_species_accuracy.csv` — one row per species, sorted worst-first
* `top_confused_pairs.csv` — top-N most-confused species pairs
* `topk_curve.png` — top-K cumulative accuracy
* `per_species_accuracy_hist.png` — histogram of per-species top-1
* `prediction_distribution.png` — predicted-count vs GT-count per species
* `family_confusion_heatmap.png` — family-level confusion matrix
* `train_count_vs_accuracy.png` — long-tail scatter (if manifest given)
* `confused_pairs_bar.png` — top-N confused pairs as a horizontal bar chart

Usage
-----

    python tools/score_submission.py \\
        --submission        outputs/.../softmax_mean_top5/submission.csv \\
        --ground-truth      /mnt/d/PlantNet-300k/plantnet300K_metadata.csv \\
        --species-metadata  /mnt/d/PlantNet-300k/species_metadata.csv \\
        --output-dir        outputs/.../analysis \\
        [--manifest         data/plantnet300k_manifest.csv] \\
        [--split            test] \\
        [--top-n-confused   30]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

# Plot stack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path, split: str | None) -> dict[str, int]:
    """
    PlantNet metadata: PN_hash + split + species_id columns.
    Returns hash -> int species_id for the requested split.
    """
    df = pd.read_csv(path)
    if split is not None and "split" in df.columns:
        df = df[df["split"] == split]
    out = dict(zip(df["PN_hash"].astype(str), df["species_id"].astype(int)))
    return out


def load_species_meta(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    df["species_id"] = df["species_id"].astype(int)
    return df


def load_submission(path: Path) -> dict[str, list[int]]:
    """
    Submission format: quadrat_id, species_ids="[3, 12, 999]".
    Returns hash/stem -> ordered list of int species_ids.
    """
    df = pd.read_csv(path)
    out: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        ids_raw = str(row["species_ids"]).strip("[]")
        ids = [int(x.strip()) for x in ids_raw.split(",") if x.strip().lstrip("-").isdigit()]
        out[str(row["quadrat_id"])] = ids
    return out


def load_manifest(path: Path | None) -> dict[int, int] | None:
    """
    Optional training manifest -> species_id -> train_image_count.
    Used only for the long-tail diagnostic scatter.
    """
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, usecols=["species_id"])
    return dict(df.astype(str)["species_id"].astype(str).str.split(".").str[0].astype(int).value_counts())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    submission: dict[str, list[int]],
    ground_truth: dict[str, int],
    species_meta: pd.DataFrame | None,
    max_k: int = 20,
) -> dict:
    """
    Returns a dict of overall + per-species/genus/family metrics plus a
    DataFrame `per_species` for downstream plotting.
    """
    # Build species_id -> (genus, family) lookup
    sp_to_genus = {}
    sp_to_family = {}
    if species_meta is not None:
        for _, row in species_meta.iterrows():
            sp_to_genus[int(row["species_id"])] = row.get("genus", "?")
            sp_to_family[int(row["species_id"])] = row.get("family", "?")

    # Per-image evaluation
    rows = []
    n_total = 0
    correct_at_k = np.zeros(max_k, dtype=np.int64)
    per_sp_total = Counter()
    per_sp_top1_correct = Counter()
    per_genus_total = Counter()
    per_genus_top1_correct = Counter()
    per_family_total = Counter()
    per_family_top1_correct = Counter()
    pred_top1_count = Counter()
    # confusion pairs (truth, top1_pred) when wrong
    confusion = Counter()
    # y_true / y_pred (top-1) for sklearn metrics
    y_true_top1: list[int] = []
    y_pred_top1: list[int] = []

    for qid, truth in ground_truth.items():
        preds = submission.get(qid)
        if preds is None:
            continue
        n_total += 1
        per_sp_total[truth] += 1
        y_true_top1.append(truth)
        y_pred_top1.append(preds[0] if preds else -1)

        # top-K cumulative
        for k in range(max_k):
            if k < len(preds) and preds[k] == truth:
                correct_at_k[k:] += 1
                break
            # If truth ever appears at slot j, correct_at_k[j:] += 1 ↑ once
            # so we break after the first hit; if no hit, no slot increments.

        # top-1
        if preds and preds[0] == truth:
            per_sp_top1_correct[truth] += 1
        else:
            if preds:
                confusion[(truth, preds[0])] += 1

        if preds:
            pred_top1_count[preds[0]] += 1

        # Genus / family aggregation (top-1)
        truth_g = sp_to_genus.get(truth)
        truth_f = sp_to_family.get(truth)
        pred_g = sp_to_genus.get(preds[0]) if preds else None
        pred_f = sp_to_family.get(preds[0]) if preds else None
        if truth_g is not None:
            per_genus_total[truth_g] += 1
            if pred_g is not None and truth_g == pred_g:
                per_genus_top1_correct[truth_g] += 1
        if truth_f is not None:
            per_family_total[truth_f] += 1
            if pred_f is not None and truth_f == pred_f:
                per_family_top1_correct[truth_f] += 1

    topk_curve = (correct_at_k / max(n_total, 1)).tolist()

    # ------------------------------------------------------------------
    # F1 metrics (top-1 confusion, sklearn-driven)
    # ------------------------------------------------------------------
    y_true_arr = np.array(y_true_top1)
    y_pred_arr = np.array(y_pred_top1)
    # All classes that appear either as truth or as a top-1 prediction
    label_universe = sorted(set(y_true_arr) | set(int(p) for p in y_pred_arr if p >= 0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=label_universe,
                              average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=label_universe,
                                 average="weighted", zero_division=0))
    micro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=label_universe,
                              average="micro", zero_division=0))
    prec_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=label_universe, zero_division=0,
    )
    per_class_f1 = dict(zip(label_universe, f1_pc.tolist()))

    # Per-species table
    per_species = (
        pd.DataFrame({
            "species_id":     list(per_sp_total.keys()),
            "n_test":         [per_sp_total[s] for s in per_sp_total],
            "top1_correct":   [per_sp_top1_correct[s] for s in per_sp_total],
        })
        .assign(top1_acc=lambda d: d["top1_correct"] / d["n_test"])
        .sort_values("top1_acc")
        .reset_index(drop=True)
    )
    if species_meta is not None:
        per_species = per_species.merge(
            species_meta[["species_id", "species", "genus", "family"]],
            on="species_id",
            how="left",
        )

    # Per-genus / per-family tables (for printing)
    def _agg(total: Counter, correct: Counter) -> dict:
        if not total:
            return {"count": 0, "avg_top1": 0.0, "min_top1": 0.0, "max_top1": 0.0}
        rates = [correct.get(k, 0) / total[k] for k in total]
        return {
            "count":     len(total),
            "avg_top1":  float(np.mean(rates)),
            "min_top1":  float(np.min(rates)),
            "max_top1":  float(np.max(rates)),
        }

    summary = {
        "n_scored": n_total,
        "n_missing_predictions": len(ground_truth) - n_total,
        "n_classes_total":       len(per_sp_total),
        "top1_acc":              topk_curve[0] if topk_curve else 0.0,
        "top5_acc":              topk_curve[4] if len(topk_curve) >= 5 else None,
        "top10_acc":             topk_curve[9] if len(topk_curve) >= 10 else None,
        "topk_curve":            topk_curve,
        "macro_f1":              macro_f1,
        "weighted_f1":           weighted_f1,
        "micro_f1":              micro_f1,
        "per_class_f1_summary":  {
            "median": float(np.median(f1_pc)),
            "mean":   float(np.mean(f1_pc)),
            "min":    float(np.min(f1_pc)),
            "max":    float(np.max(f1_pc)),
            "n_zero": int((f1_pc == 0.0).sum()),
            "n_full": int((f1_pc == 1.0).sum()),
        },
        "per_species_summary":   {
            "min_top1":    float(per_species["top1_acc"].min()),
            "median_top1": float(per_species["top1_acc"].median()),
            "mean_top1":   float(per_species["top1_acc"].mean()),
            "max_top1":    float(per_species["top1_acc"].max()),
            "n_zero_top1": int((per_species["top1_acc"] == 0.0).sum()),
            "n_full_top1": int((per_species["top1_acc"] == 1.0).sum()),
        },
        "per_genus_summary":  _agg(per_genus_total, per_genus_top1_correct),
        "per_family_summary": _agg(per_family_total, per_family_top1_correct),
        "n_unique_top1_predictions": len(pred_top1_count),
        "most_frequent_top1_prediction": (
            pred_top1_count.most_common(1)[0] if pred_top1_count else None
        ),
    }

    # Add per-class F1 onto the per_species table for the CSV export
    per_species["f1"] = per_species["species_id"].map(per_class_f1).fillna(0.0)

    return {
        "summary":          summary,
        "per_species":      per_species,
        "confusion_counts": confusion,
        "pred_top1_count":  pred_top1_count,
        "per_family_total":   dict(per_family_total),
        "per_family_correct": dict(per_family_top1_correct),
        "sp_to_family":     sp_to_family,
        "sp_to_genus":      sp_to_genus,
        "per_class_f1":     per_class_f1,
        "f1_pc":            f1_pc,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.0)


def plot_topk_curve(topk_curve: list[float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = range(1, len(topk_curve) + 1)
    ax.plot(ks, [v * 100 for v in topk_curve], marker="o", color="#2563eb")
    ax.set_xlabel("k")
    ax.set_ylabel("Top-K accuracy (%)")
    ax.set_title("Top-K cumulative accuracy")
    ax.set_xticks(list(ks))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_per_species_hist(per_species: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(per_species["top1_acc"] * 100, bins=40, color="#16a34a", alpha=0.8)
    ax.set_xlabel("Per-species top-1 accuracy (%)")
    ax.set_ylabel("# species")
    ax.set_title(
        f"Per-species top-1 (n={len(per_species)} species)\n"
        f"median={per_species['top1_acc'].median()*100:.1f}%, "
        f"mean={per_species['top1_acc'].mean()*100:.1f}%, "
        f"{(per_species['top1_acc']==0).sum()} at 0%, "
        f"{(per_species['top1_acc']==1).sum()} at 100%"
    )
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_per_class_f1_hist(per_species: pd.DataFrame, summary: dict, out: Path) -> None:
    """Histogram of per-species F1 scores. Macro F1 = mean of this distribution."""
    f1 = per_species["f1"].to_numpy()
    macro = summary.get("macro_f1", float(f1.mean()))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(f1 * 100, bins=40, color="#0891b2", alpha=0.85)
    ax.axvline(macro * 100, color="black", linestyle="--", linewidth=1.5,
               label=f"Macro F1 = {macro*100:.2f}%")
    ax.set_xlabel("Per-species F1 (%)")
    ax.set_ylabel("# species")
    s = summary.get("per_class_f1_summary", {})
    ax.set_title(
        f"Per-species F1 distribution\n"
        f"macro={macro*100:.2f}%, median={s.get('median', 0)*100:.1f}%, "
        f"{s.get('n_zero', 0)} at 0, {s.get('n_full', 0)} at 100%"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_prediction_distribution(
    per_species: pd.DataFrame, pred_top1_count: Counter, out: Path
) -> None:
    """
    Per-species: # times predicted as top-1 (x) vs # times in ground truth (y).
    A perfectly calibrated model lies on y = x.
    """
    sp_to_n_test = dict(zip(per_species["species_id"], per_species["n_test"]))
    species_ids = sorted(set(sp_to_n_test) | set(pred_top1_count))
    gt   = np.array([sp_to_n_test.get(s, 0) for s in species_ids])
    pred = np.array([pred_top1_count.get(s, 0) for s in species_ids])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt, pred, alpha=0.45, s=18, color="#dc2626")
    lim = max(gt.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="black", alpha=0.5, label="y = x")
    ax.set_xlabel("Ground-truth count (per species, test split)")
    ax.set_ylabel("Predicted-as-top1 count")
    ax.set_title("Prediction distribution vs ground-truth distribution")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_family_confusion(
    confusion: Counter,
    per_family_total: dict,
    sp_to_family: dict,
    out: Path,
    annot_top_n: int = 0,
) -> None:
    """
    111x111 family-level confusion matrix.
    Row-normalised (each row sums to 1). Diagonal = within-family.
    """
    families = sorted(per_family_total)
    f_idx = {f: i for i, f in enumerate(families)}
    n = len(families)
    if n == 0:
        return

    M = np.zeros((n, n), dtype=np.float64)
    # The diagonal: all correctly-classified images at species level land
    # on the right family by construction; but the matrix from `confusion`
    # only has misclassifications. We need to count correct-by-family
    # separately.
    # Step 1: total per family from per_family_total. Step 2: subtract
    # off-diagonal mass and put the rest on the diagonal.
    for (truth, pred), cnt in confusion.items():
        tf = sp_to_family.get(truth)
        pf = sp_to_family.get(pred)
        if tf in f_idx and pf in f_idx:
            M[f_idx[tf], f_idx[pf]] += cnt

    # Add the diagonal: per-family correct = total - off-diagonal mass
    for f, total in per_family_total.items():
        i = f_idx[f]
        off_diag = M[i, :].sum() - M[i, i]   # misclass count for this family
        M[i, i] = max(0, total - off_diag)

    # Row-normalise
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    Mn = M / row_sums

    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(
        Mn, cmap="rocket_r", vmin=0, vmax=1,
        xticklabels=False, yticklabels=False,
        cbar_kws={"label": "P(predicted family | true family)"},
        ax=ax,
    )
    ax.set_xlabel(f"Predicted family ({n} families)")
    ax.set_ylabel("True family")
    ax.set_title("Family-level confusion (row-normalised)\nDarker on the diagonal = better")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_train_count_vs_accuracy(
    per_species: pd.DataFrame, train_counts: dict, out: Path
) -> None:
    if train_counts is None:
        return
    per_species = per_species.copy()
    per_species["train_count"] = per_species["species_id"].map(train_counts)
    per_species = per_species.dropna(subset=["train_count"])
    if per_species.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        per_species["train_count"], per_species["top1_acc"] * 100,
        alpha=0.45, s=16, color="#7c3aed",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Training images per species (log)")
    ax.set_ylabel("Test top-1 accuracy (%)")
    ax.set_title("Long-tail diagnostic: training count vs test accuracy")
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_top_confused_pairs(
    confusion: Counter, species_meta: pd.DataFrame | None,
    out: Path, top_n: int = 30,
) -> pd.DataFrame:
    if not confusion:
        return pd.DataFrame()

    rows = []
    for (truth, pred), cnt in confusion.most_common(top_n):
        truth_name = pred_name = None
        if species_meta is not None:
            tt = species_meta.loc[species_meta["species_id"] == truth, "species"]
            pp = species_meta.loc[species_meta["species_id"] == pred, "species"]
            truth_name = tt.iloc[0] if not tt.empty else str(truth)
            pred_name  = pp.iloc[0] if not pp.empty else str(pred)
        else:
            truth_name = str(truth)
            pred_name  = str(pred)
        rows.append({
            "true_species":  truth_name, "true_id":  truth,
            "pred_species":  pred_name,  "pred_id":  pred,
            "n_confused":    cnt,
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(df) + 1)))
    labels = [f"{r['true_species']}  →  {r['pred_species']}" for _, r in df.iterrows()]
    ax.barh(range(len(df)), df["n_confused"], color="#ea580c", alpha=0.85)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("# images misclassified")
    ax.set_title(f"Top-{len(df)} most-confused species pairs")
    ax.grid(True, axis="x", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Score a submission CSV and produce diagnostic plots.",
    )
    p.add_argument("--submission",       type=Path, required=True)
    p.add_argument("--ground-truth",     type=Path, required=True,
                   help="PlantNet plantnet300K_metadata.csv (or equivalent: "
                        "must have PN_hash, species_id, split).")
    p.add_argument("--species-metadata", type=Path, default=None,
                   help="Optional per-species taxonomy CSV (species_id, "
                        "species, genus, family).")
    p.add_argument("--manifest",         type=Path, default=None,
                   help="Optional training manifest used to derive "
                        "train_count per species for the long-tail plot.")
    p.add_argument("--split",            default="test",
                   help="Which split column value to score against. "
                        "Pass '' to skip the split filter.")
    p.add_argument("--top-n-confused",   type=int, default=30,
                   help="How many most-confused species pairs to plot.")
    p.add_argument("--max-k",            type=int, default=20,
                   help="Compute top-K curve up to this K.")
    p.add_argument("--output-dir",       type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = args.split if args.split else None

    print(f"[score] loading submission     : {args.submission}")
    submission = load_submission(args.submission)
    print(f"[score]  -> {len(submission):,} predictions")

    print(f"[score] loading ground truth   : {args.ground_truth} (split={split!r})")
    gt = load_ground_truth(args.ground_truth, split)
    print(f"[score]  -> {len(gt):,} ground-truth labels")

    print(f"[score] loading species meta   : {args.species_metadata}")
    sp_meta = load_species_meta(args.species_metadata)

    train_counts = load_manifest(args.manifest)
    if train_counts:
        print(f"[score] manifest: {len(train_counts):,} species with train counts")

    print(f"[score] computing metrics ...")
    out = compute_metrics(submission, gt, sp_meta, max_k=args.max_k)
    summary = out["summary"]

    # ------------------------------------------------------------------
    # Persist tables
    # ------------------------------------------------------------------
    out["per_species"].to_csv(args.output_dir / "per_species_accuracy.csv", index=False)
    confused_df = plot_top_confused_pairs(
        out["confusion_counts"], sp_meta,
        args.output_dir / "confused_pairs_bar.png",
        top_n=args.top_n_confused,
    )
    if not confused_df.empty:
        confused_df.to_csv(args.output_dir / "top_confused_pairs.csv", index=False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_topk_curve(summary["topk_curve"],
                    args.output_dir / "topk_curve.png")
    plot_per_species_hist(out["per_species"],
                          args.output_dir / "per_species_accuracy_hist.png")
    plot_per_class_f1_hist(out["per_species"], summary,
                           args.output_dir / "per_species_f1_hist.png")
    plot_prediction_distribution(out["per_species"], out["pred_top1_count"],
                                 args.output_dir / "prediction_distribution.png")
    plot_family_confusion(out["confusion_counts"], out["per_family_total"],
                          out["sp_to_family"],
                          args.output_dir / "family_confusion_heatmap.png")
    if train_counts:
        plot_train_count_vs_accuracy(out["per_species"], train_counts,
                                     args.output_dir / "train_count_vs_accuracy.png")

    # ------------------------------------------------------------------
    # summary.json + stdout
    # ------------------------------------------------------------------
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("  Scoring summary")
    print("=" * 60)
    print(f"  Scored             : {summary['n_scored']:,} / "
          f"{summary['n_scored'] + summary['n_missing_predictions']:,} "
          f"({100*summary['n_scored']/(summary['n_scored']+summary['n_missing_predictions']):.2f}% coverage)")
    print(f"  Classes in test    : {summary['n_classes_total']:,}")
    print(f"  Top-1 accuracy     : {summary['top1_acc']*100:.2f}%")
    if summary["top5_acc"] is not None:
        print(f"  Top-5 accuracy     : {summary['top5_acc']*100:.2f}%")
    if summary["top10_acc"] is not None:
        print(f"  Top-10 accuracy    : {summary['top10_acc']*100:.2f}%")
    print(f"  Macro F1           : {summary['macro_f1']*100:.2f}%   "
          f"(mean of per-species F1 — PlantCLEF-style metric)")
    print(f"  Weighted F1        : {summary['weighted_f1']*100:.2f}%   "
          f"(class frequency weighted)")
    print(f"  Micro F1           : {summary['micro_f1']*100:.2f}%   "
          f"(== top-1 accuracy for single-label)")
    pf1 = summary["per_class_f1_summary"]
    print(f"  Per-species F1     : median={pf1['median']*100:.2f}%  "
          f"mean={pf1['mean']*100:.2f}%  "
          f"min={pf1['min']*100:.2f}%  max={pf1['max']*100:.2f}%")
    print(f"                       {pf1['n_zero']} species at 0%, "
          f"{pf1['n_full']} at 100%")
    ps = summary["per_species_summary"]
    print(f"  Per-species top-1  : median={ps['median_top1']*100:.2f}%  "
          f"mean={ps['mean_top1']*100:.2f}%  "
          f"min={ps['min_top1']*100:.2f}%  max={ps['max_top1']*100:.2f}%")
    print(f"                       {ps['n_zero_top1']} species at 0%, "
          f"{ps['n_full_top1']} at 100%")
    pg = summary["per_genus_summary"]
    pf = summary["per_family_summary"]
    print(f"  Per-genus  top-1   : avg={pg['avg_top1']*100:.2f}%  "
          f"(over {pg['count']} genera)")
    print(f"  Per-family top-1   : avg={pf['avg_top1']*100:.2f}%  "
          f"(over {pf['count']} families)")
    print(f"  Top-1 prediction diversity : "
          f"{summary['n_unique_top1_predictions']:,} unique species predicted")
    if summary.get("most_frequent_top1_prediction"):
        sp, n = summary["most_frequent_top1_prediction"]
        print(f"  Most-frequent top-1: species {sp}  "
              f"({n:,} / {summary['n_scored']:,} = {100*n/summary['n_scored']:.2f}%)")
    print(f"  Outputs           -> {args.output_dir}/")


if __name__ == "__main__":
    main()
