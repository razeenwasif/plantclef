"""Thermodynamic phenology rerank.

Treat species observability as a Boltzmann-distributed activation over the
day-of-year (DOY) cycle. We approximate the activation pdf non-parametrically
from GBIF month histograms with circular Gaussian smoothing.

For each test quadrat at DOY d:
    log p_final[s] = log p_i002[s] + beta * log P(s | d)

This is a multiplicative prior in probability space (log-additive). beta
controls the temperature of the phenological "energy" relative to the visual
log-likelihood; high beta = cold (sharp seasonal gating), low beta = hot
(phenology nearly ignored).

Sweep over beta and a small probT grid, write submissions per combo.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/PlantCLEF2026/tools/baseline_infer")
from postprocess_la_sweep import build_logit_adjustment, select_probT

METADATA = "/workspace/scratch_space_arjun/PlantCLEF2026/src_experiments/i001_data_download/data/training_usage/metadata_filled_genus_family.csv"
TAU = 0.25
EPS = 1e-12
DAYS = 365

LOGITS = {
    "i002_224_lb": "/tmp/la_sweep_logits/logits/softmax_mean_logits.pt",
    "i002_336":    "/tmp/hires_336/logits/softmax_mean_logits.pt",
}

# Approximate DOY at the centre of each calendar month (non-leap year)
MONTH_CENTRE_DOY = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=np.float32)


def circular_smooth(month_counts: np.ndarray, sigma_days: float) -> np.ndarray:
    """Build a length-365 pdf from a 12-bin month histogram via circular Gaussian smoothing."""
    pdf = np.zeros(DAYS, dtype=np.float32)
    if month_counts.sum() <= 0:
        return np.full(DAYS, 1.0 / DAYS, dtype=np.float32)
    x = np.arange(DAYS, dtype=np.float32)
    for m in range(12):
        c = month_counts[m]
        if c <= 0:
            continue
        # circular distance to month centre (wrap at DAYS)
        d = np.abs(x - MONTH_CENTRE_DOY[m])
        d = np.minimum(d, DAYS - d)
        pdf += c * np.exp(-0.5 * (d / sigma_days) ** 2)
    pdf /= pdf.sum() + 1e-12
    return pdf


def build_species_pdf(jsonl_path: str, idx_to_species: list[str], sigma_days: float, uniform_floor: float) -> np.ndarray:
    """Returns [n_species, DAYS] pdf, one row per ref-species index."""
    sid_to_hist: dict[str, dict[int, int]] = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec:
                continue
            sid_to_hist[str(rec["species_id"])] = rec.get("months", {})

    n = len(idx_to_species)
    out = np.zeros((n, DAYS), dtype=np.float32)
    n_missing = 0
    for i, sid in enumerate(idx_to_species):
        h = sid_to_hist.get(sid)
        if not h:
            n_missing += 1
            out[i] = 1.0 / DAYS  # uniform fallback
            continue
        counts = np.zeros(12, dtype=np.float32)
        for k, v in h.items():
            counts[int(k) - 1] = v
        pdf = circular_smooth(counts, sigma_days=sigma_days)
        # blend with uniform floor so tail months never have log(0) energy
        pdf = (1 - uniform_floor) * pdf + uniform_floor / DAYS
        pdf /= pdf.sum()
        out[i] = pdf
    print(f"  species pdfs built. missing/uniform: {n_missing}/{n}")
    return out


def parse_doy(qid: str) -> int:
    """Extract YYYYMMDD from end of quadrat_id and return day-of-year (1-365)."""
    suffix = qid[-8:]
    d = dt.datetime.strptime(suffix, "%Y%m%d").date()
    doy = d.timetuple().tm_yday
    # Map leap-year DOY 366 onto 365 for index safety
    return min(doy, DAYS)


def write_submission(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for qid, sp in rows:
            w.writerow([qid, "[" + ", ".join(sp) + "]"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phenology", default="/workspace/PlantCLEF2026/data/phenology/gbif_month_histograms.jsonl")
    p.add_argument("--out-dir",   default="/workspace/PlantCLEF2026/submissions/phenology")
    p.add_argument("--betas",     nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--prob-thresholds", nargs="+", type=float, default=[0.025, 0.03, 0.035])
    p.add_argument("--sigma-days", type=float, default=18.0,
                   help="Phenology smoothing kernel; smaller = sharper seasonal gating")
    p.add_argument("--uniform-floor", type=float, default=0.05,
                   help="Mix this much uniform mass into each species pdf to avoid hard zeros")
    p.add_argument("--min-k", type=int, default=1)
    p.add_argument("--max-k", type=int, default=10)
    args = p.parse_args()

    # ---- Base i002 ensemble ----
    print("Loading i002 logits...")
    blobs = {k: torch.load(v, weights_only=False) for k, v in LOGITS.items()}
    ref_qids = blobs["i002_224_lb"]["quadrat_ids"]
    ref_species = blobs["i002_224_lb"]["idx_to_species"]
    n = len(ref_qids)
    print(f"  {n} quadrats x {len(ref_species)} species")

    la = build_logit_adjustment(METADATA, ref_species, tau=TAU)

    def reorder(blob):
        qid_to_i = {q: i for i, q in enumerate(blob["quadrat_ids"])}
        perm = torch.tensor([qid_to_i[q] for q in ref_qids], dtype=torch.long)
        return blob["logits"][perm]

    p_224 = F.softmax(reorder(blobs["i002_224_lb"]) - la, dim=1)
    p_336 = F.softmax(reorder(blobs["i002_336"]) - la, dim=1)
    p_base = (p_224 + p_336) / 2
    log_p_base = torch.log(p_base.clamp_min(EPS))

    # ---- Species phenology pdf ----
    print(f"Building species pdfs (sigma={args.sigma_days}d, floor={args.uniform_floor})...")
    species_pdf = build_species_pdf(args.phenology, ref_species, args.sigma_days, args.uniform_floor)
    species_log_pdf = torch.from_numpy(np.log(species_pdf + EPS))   # [S, 365]

    # ---- DOY per quadrat ----
    test_doys = np.array([parse_doy(q) - 1 for q in ref_qids], dtype=np.int64)  # 0..364
    doy_idx = torch.from_numpy(test_doys)
    print(f"  DOY range over test set: {test_doys.min()+1} ... {test_doys.max()+1}")
    print(f"  DOY histogram (months):  " + ", ".join(
        f"{m+1}:{(test_doys // 31 == m).sum()}" for m in range(12)))

    # log_prior_per_quadrat: [N, S]  =  species_log_pdf.T[doy_of_quadrat, :]
    # Equivalent: pick species_log_pdf[:, doy].T row-by-row
    log_prior = species_log_pdf[:, doy_idx].T  # [N, S]

    # ---- Baseline ----
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print("\nBaseline (beta=0):")
    for thr in args.prob_thresholds:
        rows = []
        for i in range(n):
            idxs = select_probT(p_base[i], thr, args.min_k, args.max_k)
            rows.append((ref_qids[i], [ref_species[j] for j in idxs]))
        mean_k = sum(len(r[1]) for r in rows) / n
        uniq = len({s for r in rows for s in r[1]})
        sub = out_root / f"baseline_probT{thr}" / "submission.csv"
        write_submission(rows, sub)
        print(f"  baseline probT={thr}  mean_k={mean_k:.2f}  uniq={uniq}")

    # ---- Sweep ----
    print("\nPhenology sweep:")
    for beta, thr in product(args.betas, args.prob_thresholds):
        if beta == 0.0:
            continue
        log_p_new = log_p_base + beta * log_prior
        p_new = F.softmax(log_p_new, dim=1)

        rows = []
        for i in range(n):
            idxs = select_probT(p_new[i], thr, args.min_k, args.max_k)
            rows.append((ref_qids[i], [ref_species[j] for j in idxs]))
        mean_k = sum(len(r[1]) for r in rows) / n
        uniq = len({s for r in rows for s in r[1]})
        # quick measure of how much beta actually changed top-1
        base_top1 = p_base.argmax(dim=1)
        new_top1 = p_new.argmax(dim=1)
        flips = (new_top1 != base_top1).float().mean().item()
        tag = f"beta{beta}_probT{thr}"
        sub = out_root / tag / "submission.csv"
        write_submission(rows, sub)
        print(f"  {tag:24s}  mean_k={mean_k:.2f}  uniq={uniq}  top1_flips={flips:.3f}")


if __name__ == "__main__":
    main()
