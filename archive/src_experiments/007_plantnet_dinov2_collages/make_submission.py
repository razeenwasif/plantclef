"""
Build a PlantCLEF submission CSV from test_probs.npz (dump_test_probs.py output).

Simple global-threshold + top-K submission. For calibrated per-class thresholds,
run 005's calibrate_thresholds.py on a held-out mosaic val set first and feed
the JSON through 005's apply_thresholds_to_npz.py — same npz schema.

Usage
-----
  python make_submission.py \
    --probs-npz /path/to/test_probs.npz \
    --probs-key probs_max \
    --threshold 0.5 \
    --top-k 10 \
    --min-preds 1 \
    --output /path/to/submission.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probs-npz", required=True)
    p.add_argument("--probs-key", default="probs_max",
                   choices=("probs_max", "probs_mean", "probs_noisy_or"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=10,
                   help="Upper bound on predictions per quadrat.")
    p.add_argument("--min-preds", type=int, default=1,
                   help="If no species passes threshold, force this many (top-scored).")
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = np.load(args.probs_npz, allow_pickle=True)
    quadrat_ids = d["quadrat_ids"]
    species_ids = d["species_ids"]
    probs = d[args.probs_key].astype(np.float32)

    N, C = probs.shape
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pred = 0
    hist = np.zeros(args.top_k + 1, dtype=np.int64)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for i in range(N):
            score = probs[i]
            order = np.argsort(-score)
            passed = [j for j in order[: args.top_k] if score[j] >= args.threshold]
            if len(passed) < args.min_preds:
                passed = list(order[: args.min_preds])
            sids = [int(species_ids[j]) for j in passed]
            w.writerow([str(quadrat_ids[i]), str(sids)])
            total_pred += len(sids)
            hist[min(len(sids), args.top_k)] += 1

    print(f"Wrote {out_path}  ({N} quadrats, {total_pred} total predictions, "
          f"avg {total_pred / max(N, 1):.2f}/quadrat)")
    print("Histogram of predictions/quadrat:")
    for k in range(len(hist)):
        if hist[k] > 0:
            print(f"  {k:>3}: {hist[k]}")


if __name__ == "__main__":
    main()
