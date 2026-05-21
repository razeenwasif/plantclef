"""
Convert a probabilities npz dump (probs_mean / probs_max / probs_noisy_or)
into a PlantCLEF submission CSV by taking top-K species per quadrat.

Format matches the team baseline:
  "quadrat_id","species_ids"
  "abc123","[1395806, 1351284, 1234567]"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probs", required=True, help="Path to npz with probs_max/probs_mean/probs_noisy_or, quadrat_ids, species_ids")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--key", default="probs_mean",
                   choices=["probs_max", "probs_mean", "probs_noisy_or"])
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz = np.load(args.probs, allow_pickle=True)
    probs = npz[args.key]
    quadrat_ids = npz["quadrat_ids"]
    species_ids = npz["species_ids"]

    # full deterministic sort — argpartition can tie-break differently than
    # argsort when multiple species share an fp16-quantized probability,
    # producing tiny but non-zero score differences vs other tools' CSVs.
    k = args.top_k
    topk_idx = np.argsort(-probs, axis=1, kind="stable")[:, :k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["quadrat_id", "species_ids"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for i, qid in enumerate(quadrat_ids):
            preds = [str(species_ids[j]) for j in topk_idx[i]]
            ids_str = "[" + ", ".join(
                str(int(s)) for s in preds if str(s).isdigit()
            ) + "]"
            writer.writerow({"quadrat_id": str(qid), "species_ids": ids_str})

    print(f"Wrote {out_path} ({probs.shape[0]} rows, key={args.key}, top-{k})")


if __name__ == "__main__":
    main()
