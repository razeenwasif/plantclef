"""
011 BioCLIP-2.5 aggregation α-sweep.

Take the 010 last_blocks `test_probs_010_last_blocks_grid4x4.npz` (which already
contains `probs_max`, `probs_mean`, `probs_noisy_or` over 16 grid_4x4 tiles per
quadrat) and emit one PlantCLEF submission CSV per α in the sweep.

Hybrid:
    hybrid_α = α · probs_mean + (1 − α) · probs_other

The team's current best (0.38333 / 0.37431 reproduced) uses α=1.0, i.e. pure
`probs_mean` (= softmax_mean). EDA showed that "localized but strong" species
have max/mean ≈ 5–15× — softmax_mean averages them down. `probs_noisy_or`
fires when *any* tile is confident, so a small (1 − α) noisy_or term is the
hypothesised lever.

Output filenames follow the convention recognised by `scripts/submit_predictions_kaggle.sh`:

    submission_011_aggMxNO_a{ALPHA}_top{K}.csv

with QUOTE_ALL-quoted bracketed species_id lists.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probs", required=True,
                   help="Path to 010 npz with probs_mean, probs_noisy_or, probs_max, quadrat_ids, species_ids.")
    p.add_argument("--out-dir", required=True, help="Directory for submission CSVs.")
    p.add_argument("--mix", default="mean_x_noisy_or",
                   choices=["mean_x_noisy_or", "mean_x_max"],
                   help="Which two aggregations to mix.")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="α values; α=1.0 ⇒ pure probs_mean (= baseline).")
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def write_submission(path: Path, quadrat_ids, species_ids, topk_idx) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["quadrat_id", "species_ids"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for i, qid in enumerate(quadrat_ids):
            ids_str = "[" + ", ".join(
                str(int(species_ids[j])) for j in topk_idx[i]
                if str(species_ids[j]).isdigit()
            ) + "]"
            writer.writerow({"quadrat_id": str(qid), "species_ids": ids_str})


def main() -> None:
    args = parse_args()
    npz = np.load(args.probs, allow_pickle=True)
    probs_mean = npz["probs_mean"].astype(np.float32)
    if args.mix == "mean_x_noisy_or":
        probs_other = npz["probs_noisy_or"].astype(np.float32)
        mix_tag = "MxNO"
    else:
        probs_other = npz["probs_max"].astype(np.float32)
        mix_tag = "MxMAX"
    quadrat_ids = npz["quadrat_ids"]
    species_ids = npz["species_ids"]
    print(f"Loaded {args.probs}  shape={probs_mean.shape}  mix={args.mix}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k = args.top_k
    for alpha in args.alphas:
        hybrid = alpha * probs_mean + (1.0 - alpha) * probs_other
        topk_idx = np.argsort(-hybrid, axis=1, kind="stable")[:, :k]
        a_str = f"{alpha:.2f}".replace(".", "p")
        out = out_dir / f"submission_011_agg{mix_tag}_a{a_str}_top{k}.csv"
        write_submission(out, quadrat_ids, species_ids, topk_idx)
        # quick top-1 distribution diagnostic
        top1 = topk_idx[:, 0]
        unique = len(np.unique(top1))
        print(f"  α={alpha:.2f}  → {out.name}  (#unique top-1 species: {unique} / {probs_mean.shape[1]})")


if __name__ == "__main__":
    main()
