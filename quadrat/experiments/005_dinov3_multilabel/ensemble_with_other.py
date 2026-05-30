"""
Ensemble blender: combine the DINOv3 Phase-2 probs (test_probs_phase2.npz)
with another model's per-quadrat probability vectors and emit a submission.

Expected other-model format (.npz):
    quadrat_ids     (N,)       object array of quadrat stems
    species_ids     (C,)       object array of species_id strings
    probs           (N, C)     fp16/fp32 — pre-aggregated per-quadrat probs
  OR one of probs_max / probs_mean / probs_noisy_or for per-agg-mode inputs.

Alignment: both files must share the same set of quadrat_ids and species_ids
(order-independent — we reindex). Any missing ID in either side aborts.

Blend: per-class mean of the two probability vectors (simple arithmetic mean
is surprisingly robust; swap to geometric mean via --geometric if needed).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dinov3-npz", required=True,
                   help="Output of dump_test_probs.py.")
    p.add_argument("--other-npz", required=True,
                   help="Other model's per-quadrat probs (same format).")
    p.add_argument("--dinov3-key", default="probs_max",
                   choices=("probs_max", "probs_mean", "probs_noisy_or"))
    p.add_argument("--other-key", default="probs",
                   help="Key inside --other-npz to use.")
    p.add_argument("--weight-dinov3", type=float, default=0.5,
                   help="Mix weight on DINOv3 (other gets 1 - this).")
    p.add_argument("--geometric", action="store_true",
                   help="Use geometric mean instead of arithmetic.")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Drop species with blended prob below this.")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--output", required=True,
                   help="Path for the submission CSV.")
    return p.parse_args()


def load_probs(path: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    if key not in d.files:
        raise KeyError(f"{path} has keys {list(d.files)} — '{key}' missing.")
    return d["quadrat_ids"], d["species_ids"], d[key].astype(np.float32)


def align(ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    """Return index array so that ids_a[idx] == ids_b (element-wise)."""
    pos = {str(s): i for i, s in enumerate(ids_a)}
    missing = [s for s in ids_b if str(s) not in pos]
    if missing:
        raise ValueError(f"{len(missing)} IDs missing from first file: {missing[:5]}...")
    return np.array([pos[str(s)] for s in ids_b], dtype=np.int64)


def main() -> None:
    args = parse_args()
    q_a, s_a, p_a = load_probs(args.dinov3_npz, args.dinov3_key)
    q_b, s_b, p_b = load_probs(args.other_npz, args.other_key)

    # Reorder A to match B
    q_idx = align(q_a, q_b)
    s_idx = align(s_a, s_b)
    p_a = p_a[q_idx][:, s_idx]

    w = float(args.weight_dinov3)
    if args.geometric:
        eps = 1e-8
        p_blend = np.exp(w * np.log(p_a + eps) + (1 - w) * np.log(p_b + eps))
    else:
        p_blend = w * p_a + (1 - w) * p_b

    # Rank per quadrat, apply threshold + top-N
    N, C = p_blend.shape
    rows: list[tuple[str, str]] = []
    for i in range(N):
        order = np.argsort(-p_blend[i])
        keep = []
        for j in order[: args.top_n * 2]:  # scan a bit past top_n to allow threshold filtering
            if p_blend[i, j] < args.threshold:
                break
            keep.append(int(s_b[j]))
            if len(keep) >= args.top_n:
                break
        qid = str(q_b[i])
        rows.append((qid, "[" + ", ".join(str(x) for x in keep) + "]"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w_csv = csv.writer(f, quoting=csv.QUOTE_ALL)
        w_csv.writerow(["quadrat_id", "species_ids"])
        for qid, sids in rows:
            w_csv.writerow([qid, sids])
    print(f"Wrote {out}  ({N} rows, top-{args.top_n}, threshold={args.threshold})")


if __name__ == "__main__":
    main()
