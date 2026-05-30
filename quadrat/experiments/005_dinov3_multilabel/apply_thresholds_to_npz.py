"""
Apply a per-species threshold JSON (from calibrate_thresholds.py) to a
precomputed probability npz (from dump_test_probs.py) and emit a PlantCLEF
submission CSV. Zero GPU required.

Input:
  --probs-npz       dump_test_probs.py output (test_probs_phase2.npz)
  --probs-key       one of probs_max | probs_mean | probs_noisy_or
  --thresholds-json calibrate_thresholds.py output

Output:
  submission CSV with quadrat_id, species_ids (Python list literal, top-N cap).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probs-npz", required=True)
    p.add_argument("--probs-key", default="probs_max",
                   choices=("probs_max", "probs_mean", "probs_noisy_or"))
    p.add_argument("--thresholds-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--min-preds", type=int, default=1,
                   help="If no species passes threshold, still emit this many (top-scored).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = np.load(args.probs_npz, allow_pickle=True)
    quadrat_ids = d["quadrat_ids"]
    species_ids = d["species_ids"]
    probs = d[args.probs_key].astype(np.float32)

    with open(args.thresholds_json) as f:
        payload = json.load(f)
    thr_map = payload["thresholds"] if "thresholds" in payload else payload
    C = probs.shape[1]
    thresholds = np.zeros(C, dtype=np.float32)
    missing = 0
    for j, sid in enumerate(species_ids):
        key = str(sid)
        if key in thr_map:
            thresholds[j] = float(thr_map[key])
        else:
            thresholds[j] = 0.5
            missing += 1
    if missing:
        print(f"[warn] {missing} species not in thresholds JSON — defaulted to 0.5")

    N = probs.shape[0]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_pred = 0
    hist = np.zeros(args.top_n + 1, dtype=np.int64)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for i in range(N):
            score = probs[i]
            order = np.argsort(-score)
            keep: list[int] = []
            for j in order:
                if score[j] < thresholds[j]:
                    # Keep scanning; species later in ranking could still pass
                    # their (lower) threshold. But because order is by score,
                    # any later j has lower score — no later j will pass any
                    # threshold greater than its own score. Early-out when
                    # score[j] is below the minimum threshold across all species.
                    pass
                else:
                    keep.append(int(species_ids[j]))
                    if len(keep) >= args.top_n:
                        break
            if len(keep) < args.min_preds:
                # Fallback: pick top-min_preds by raw score.
                fallback = [int(species_ids[j]) for j in order[: args.min_preds]]
                keep = fallback
            total_pred += len(keep)
            hist[min(len(keep), args.top_n)] += 1
            w.writerow([
                str(quadrat_ids[i]),
                "[" + ", ".join(str(x) for x in keep) + "]",
            ])
    print(f"Wrote {out_path}  ({N} rows, total predictions={total_pred}, avg/quadrat={total_pred/N:.2f})")
    print(f"Prediction-count histogram (0..{args.top_n}): {hist.tolist()}")


if __name__ == "__main__":
    main()
