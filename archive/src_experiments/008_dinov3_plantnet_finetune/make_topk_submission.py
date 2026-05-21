"""
Convert a dump_test_probs.py .npz to a PlantCLEF submission CSV using a fixed
top-K selection per quadrat. No threshold calibration required.

Output rows:
  quadrat_id, "[sid_1, sid_2, ...]"   (top-K species by probs, csv.QUOTE_ALL)
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
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = np.load(args.probs_npz, allow_pickle=True)
    quadrat_ids = d["quadrat_ids"]
    species_ids = d["species_ids"]
    probs = d[args.probs_key].astype(np.float32)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N, C = probs.shape
    K = min(args.top_k, C)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for i in range(N):
            order = np.argsort(-probs[i])[:K]
            keep = [int(species_ids[j]) for j in order]
            w.writerow([
                str(quadrat_ids[i]),
                "[" + ", ".join(str(x) for x in keep) + "]",
            ])
    print(f"Wrote {out_path}  ({N} rows, top-{K}, key={args.probs_key})")


if __name__ == "__main__":
    main()
