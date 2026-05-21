"""
Ensemble multiple saved softmax_mean logits files.

Loads N logits blobs, aligns by quadrat_id, computes a weighted mean of softmax
probabilities, applies optional logit adjustment, runs prob_threshold selection.

The saved logits were already aggregated as softmax_mean = log(mean(softmax(tile_logits))),
so re-applying softmax recovers the per-image probability vector. We then average
those probs across models (weighted), take log to feed back through LA, and run
the standard probT selection.
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


def load_logits(path: str):
    blob = torch.load(path, weights_only=False)
    return blob["quadrat_ids"], blob["logits"], blob["idx_to_species"]


def align_to_first(qids_list, logits_list, idx_list):
    """Reorder all subsequent (quadrat_ids, logits) to match the first list's order
    and verify idx_to_species is identical across all blobs."""
    base_qids = qids_list[0]
    base_idx = idx_list[0]
    for k, idx in enumerate(idx_list[1:], start=1):
        if idx != base_idx:
            raise ValueError(f"idx_to_species mismatch between logits[0] and logits[{k}]")
    aligned = [logits_list[0]]
    for k in range(1, len(logits_list)):
        qid_to_row = {q: i for i, q in enumerate(qids_list[k])}
        if set(qid_to_row.keys()) != set(base_qids):
            raise ValueError(f"quadrat_ids mismatch between logits[0] and logits[{k}]")
        order = [qid_to_row[q] for q in base_qids]
        aligned.append(logits_list[k][order])
    return base_qids, aligned, base_idx


def select_probT(probs_row: torch.Tensor, threshold: float, min_k: int, max_k: int) -> list[int]:
    sorted_probs, sorted_idx = probs_row.sort(descending=True)
    k = int((sorted_probs >= threshold).sum().item())
    k = max(min_k, min(max_k, k, probs_row.shape[0]))
    return sorted_idx[:k].tolist()


def write_submission(rows, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for qid, species in rows:
            w.writerow([qid, "[" + ", ".join(species) + "]"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logits", nargs="+", required=True, help="Saved logits .pt files (one per model).")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Per-model ensemble weights (defaults to uniform).")
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--la-taus", nargs="+", type=float, required=True)
    p.add_argument("--prob-thresholds", nargs="+", type=float, required=True)
    p.add_argument("--min-k", type=int, default=1)
    p.add_argument("--max-k", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    if args.weights is None:
        args.weights = [1.0 / len(args.logits)] * len(args.logits)
    elif len(args.weights) != len(args.logits):
        raise ValueError("--weights length must match --logits")
    else:
        s = sum(args.weights)
        args.weights = [w / s for w in args.weights]

    qids_list, logits_list, idx_list = [], [], []
    for path in args.logits:
        q, l, i = load_logits(path)
        print(f"  loaded {path}  shape={tuple(l.shape)}")
        qids_list.append(q)
        logits_list.append(l)
        idx_list.append(i)

    quadrat_ids, aligned_logits, idx_to_species = align_to_first(qids_list, logits_list, idx_list)
    print(f"Aligned: {len(quadrat_ids)} quadrats × {aligned_logits[0].shape[1]} species")

    # softmax_mean format: stored logit = log(mean_tile(softmax(tile_logits)))
    # Recover per-model probs by softmax over species axis.
    probs_per_model = [F.softmax(l, dim=1) for l in aligned_logits]
    avg_probs = sum(w * p for w, p in zip(args.weights, probs_per_model))
    avg_log = torch.log(avg_probs.clamp_min(1e-12))

    out_root = Path(args.output_dir)
    for tau in args.la_taus:
        la = build_logit_adjustment(args.metadata_csv, idx_to_species, tau=tau)
        adj_log = avg_log - la.unsqueeze(0)
        adj_probs = F.softmax(adj_log, dim=1)

        for thr in args.prob_thresholds:
            tag = f"la{tau}_probT{thr}"
            rows = []
            for i, qid in enumerate(quadrat_ids):
                idxs = select_probT(adj_probs[i], thr, args.min_k, args.max_k)
                rows.append((qid, [idx_to_species[j] for j in idxs]))
            sub = out_root / tag / "submission.csv"
            write_submission(rows, sub)
            mean_k = sum(len(r[1]) for r in rows) / len(rows)
            uniq = len({s for r in rows for s in r[1]})
            print(f"  ensemble[{tag}]: mean_k={mean_k:.2f}  unique={uniq}  → {sub}")


if __name__ == "__main__":
    main()
