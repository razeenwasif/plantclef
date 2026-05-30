"""
Offline 2D sweep: la_tau × probT, given pre-saved softmax_mean logits.

Loads saved aggregated logits (no LA), applies logit adjustment at each tau,
runs prob_threshold selection at each threshold, writes a submission per combo.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


def build_logit_adjustment(metadata_csv: str, idx_to_species: list[str], tau: float, eps: float = 1e-6) -> torch.Tensor:
    df = pd.read_csv(metadata_csv, usecols=["species_id"])
    df["species_id"] = df["species_id"].astype(str).str.split(".").str[0].str.strip()
    counts = df["species_id"].value_counts()
    n = len(idx_to_species)
    raw_counts = torch.tensor([counts.get(sid, 0) for sid in idx_to_species], dtype=torch.float32)
    priors = (raw_counts + 1) / (raw_counts.sum() + n)
    return tau * torch.log(priors + eps)


def select_probT(probs: torch.Tensor, threshold: float, min_k: int, max_k: int) -> list[int]:
    sorted_probs, sorted_idx = probs.sort(descending=True)
    k = int((sorted_probs >= threshold).sum().item())
    k = max(min_k, min(max_k, k, probs.shape[0]))
    return sorted_idx[:k].tolist()


def write_submission(rows: list[tuple[str, list[str]]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for qid, species in rows:
            w.writerow([qid, "[" + ", ".join(species) + "]"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logits", required=True, help="Path to saved logits .pt")
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--la-taus", nargs="+", type=float, required=True)
    p.add_argument("--prob-thresholds", nargs="+", type=float, required=True)
    p.add_argument("--min-k", type=int, default=1)
    p.add_argument("--max-k", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    blob = torch.load(args.logits, weights_only=False)
    logits: torch.Tensor = blob["logits"]
    quadrat_ids: list[str] = blob["quadrat_ids"]
    idx_to_species: list[str] = blob["idx_to_species"]
    agg = blob["agg_mode"]

    print(f"Loaded {logits.shape[0]} quadrats × {logits.shape[1]} species (agg={agg})")
    out_root = Path(args.output_dir)

    for tau in args.la_taus:
        la = build_logit_adjustment(args.metadata_csv, idx_to_species, tau=tau)
        adj = logits - la.unsqueeze(0)
        probs = F.softmax(adj, dim=1)

        for thr in args.prob_thresholds:
            tag = f"la{tau}_probT{thr}"
            rows = []
            for i, qid in enumerate(quadrat_ids):
                idxs = select_probT(probs[i], thr, args.min_k, args.max_k)
                rows.append((qid, [idx_to_species[j] for j in idxs]))

            sub_path = out_root / tag / "submission.csv"
            write_submission(rows, sub_path)
            mean_k = sum(len(r[1]) for r in rows) / len(rows)
            uniq = len({s for r in rows for s in r[1]})
            print(f"  {tag}: mean_k={mean_k:.2f}  unique={uniq}  → {sub_path}")


if __name__ == "__main__":
    main()
