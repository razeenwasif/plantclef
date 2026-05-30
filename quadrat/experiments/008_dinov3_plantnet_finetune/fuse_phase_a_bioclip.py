"""
Late-fuse Phase A (DINOv3) per-species probs with BioCLIP-2.5 per-species scores
and emit top-K submission CSVs.

Strategy: rank fusion. Both inputs have different score scales (softmax probs
vs cosine similarities), so we convert each to per-quadrat ranks and combine
via a weighted reciprocal-rank-fusion variant. This is scale-free.

Inputs
------
--phase-a   008 npz with quadrat_ids, species_ids, probs_max (2105, 7806)
--bioclip   004 npz with quadrat_ids, species_ids, scores_max (2105, 7806)

Output
------
One submission_*.csv per (alpha, top_k) combination in args.output_dir.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_matrix(npz_path: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    qids = np.array([str(x) for x in d["quadrat_ids"]])
    sids = np.array([str(x) for x in d["species_ids"]])
    mat = d[key].astype(np.float32)
    return qids, sids, mat


def to_ranks(mat: np.ndarray) -> np.ndarray:
    """Per-row ranks, descending (rank 0 = best)."""
    # argsort of -mat gives indices in descending order; we want rank-of-each-element
    n_rows, n_cols = mat.shape
    order = np.argsort(-mat, axis=1, kind="stable")  # (n_rows, n_cols)
    ranks = np.empty_like(order)
    arange_cols = np.arange(n_cols)
    for i in range(n_rows):
        ranks[i, order[i]] = arange_cols
    return ranks


def rrf_score(ranks: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal-rank-fusion score per cell: 1 / (k + rank)."""
    return 1.0 / (k + ranks.astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-a", required=True)
    p.add_argument("--phase-a-key", default="probs_max")
    p.add_argument("--bioclip", required=True)
    p.add_argument("--bioclip-key", default="scores_max")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6, 0.7],
                   help="Weight on Phase A (1-alpha goes to BioCLIP)")
    p.add_argument("--top-k", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--tag", default="fuse")
    args = p.parse_args()

    print(f"Loading Phase A: {args.phase_a}  key={args.phase_a_key}")
    qa, sa, pa = load_matrix(args.phase_a, args.phase_a_key)
    print(f"  shape={pa.shape}  range=[{pa.min():.4f}, {pa.max():.4f}]")

    print(f"Loading BioCLIP: {args.bioclip}  key={args.bioclip_key}")
    qb, sb, pb = load_matrix(args.bioclip, args.bioclip_key)
    print(f"  shape={pb.shape}  range=[{pb.min():.4f}, {pb.max():.4f}]")

    # Align species axis
    if not np.array_equal(sa, sb):
        # Build a permutation that maps Phase A species order -> BioCLIP species order
        sa_to_idx = {s: i for i, s in enumerate(sa)}
        missing = [s for s in sb if s not in sa_to_idx]
        if missing:
            raise SystemExit(
                f"BioCLIP has {len(missing)} species not in Phase A "
                f"(e.g. {missing[:3]}); cannot align."
            )
        perm = np.array([sa_to_idx[s] for s in sb])
        pa = pa[:, perm]
        sa = sa[perm]
        print(f"Aligned Phase A species axis to BioCLIP order.")
    else:
        print("Species axes already aligned.")

    # Align quadrat axis
    if not np.array_equal(qa, qb):
        qa_to_idx = {q: i for i, q in enumerate(qa)}
        missing = [q for q in qb if q not in qa_to_idx]
        if missing:
            raise SystemExit(
                f"BioCLIP has {len(missing)} quadrats not in Phase A "
                f"(e.g. {missing[:3]})."
            )
        qperm = np.array([qa_to_idx[q] for q in qb])
        pa = pa[qperm]
        qa = qa[qperm]
        print(f"Aligned Phase A quadrat axis to BioCLIP order.")
    quadrat_ids = qb

    print("Computing per-row ranks (this is the slow step)...")
    ranks_a = to_ranks(pa)
    ranks_b = to_ranks(pb)

    rrf_a = rrf_score(ranks_a, k=args.rrf_k)
    rrf_b = rrf_score(ranks_b, k=args.rrf_k)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    species_arr = sb  # final species order

    for alpha in args.alphas:
        fused = alpha * rrf_a + (1.0 - alpha) * rrf_b
        # Top-K extraction
        top_indices = np.argsort(-fused, axis=1, kind="stable")
        for k in args.top_k:
            sub_path = out_dir / f"submission_{args.tag}_a{alpha:.2f}_top{k}.csv"
            with open(sub_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["quadrat_id", "species_ids"],
                                   quoting=csv.QUOTE_ALL)
                w.writeheader()
                for i, qid in enumerate(quadrat_ids):
                    top = top_indices[i, :k]
                    sids = [species_arr[j] for j in top]
                    ids_str = "[" + ", ".join(sids) + "]"
                    w.writerow({"quadrat_id": qid, "species_ids": ids_str})
            print(f"  wrote {sub_path.name}  (alpha={alpha:.2f}, k={k})")

    # Also dump pure-BioCLIP rebuilt top-K as a sanity baseline (alpha=0)
    print("\nSanity: pure-BioCLIP top-K (should reproduce ~0.33):")
    top_b = np.argsort(-pb, axis=1, kind="stable")
    for k in args.top_k:
        sub_path = out_dir / f"submission_{args.tag}_pure_bioclip_top{k}.csv"
        with open(sub_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["quadrat_id", "species_ids"],
                               quoting=csv.QUOTE_ALL)
            w.writeheader()
            for i, qid in enumerate(quadrat_ids):
                top = top_b[i, :k]
                sids = [species_arr[j] for j in top]
                ids_str = "[" + ", ".join(sids) + "]"
                w.writerow({"quadrat_id": qid, "species_ids": ids_str})
        print(f"  wrote {sub_path.name}")


if __name__ == "__main__":
    main()
