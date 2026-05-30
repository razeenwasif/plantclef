"""Build a phylogenetic adjacency matrix from existing taxonomy data.

Uses the already-available species→genus and genus→family index mappings
to compute a continuous taxonomic-distance weight matrix that can replace or
blend with the binary GBIF co-occurrence adjacency used by the GCN.

Weight function (RBF on taxonomic distance):
    same genus                          → 1.0
    same family, different genus        → exp(-1) ≈ 0.368
    different family                    → exp(-2) ≈ 0.135

The matrix is row-normalised (D^{-1}A style) so GCN messages sum to 1.

Output: data/phylo_adj.npy  — float32, shape (N, N), row-normalised
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
S2G_PATH = DATA_DIR / "species_to_genus.json"
G2F_PATH = DATA_DIR / "genus_to_family.json"
ECO_ADJ_PATH = DATA_DIR / "ecological_adj.npy"
OUT_PATH = DATA_DIR / "phylo_adj.npy"

# RBF weights for each taxonomic level of separation
W_SAME_GENUS  = 1.0
W_SAME_FAMILY = math.exp(-1.0)   # ≈ 0.368
W_DIFF_FAMILY = math.exp(-2.0)   # ≈ 0.135

CHUNK = 512  # rows per chunk — keeps peak RAM < 2 GB


def build_phylo_adj(n_species: int) -> np.ndarray:
    """Return a (n_species, n_species) float32 row-normalised adjacency."""
    print(f"[PhyloAdj] Loading taxonomy data...")
    with open(S2G_PATH) as f:
        s2g: list[int] = json.load(f)
    with open(G2F_PATH) as f:
        g2f: list[int] = json.load(f)

    # Pad or truncate to n_species
    if len(s2g) < n_species:
        s2g = s2g + [s2g[-1]] * (n_species - len(s2g))
    s2g_arr = np.array(s2g[:n_species], dtype=np.int32)

    g2f_arr = np.array(g2f, dtype=np.int32)
    # Genus indices may exceed g2f length due to padding — clamp safely
    g_safe = np.clip(s2g_arr, 0, len(g2f_arr) - 1)
    f_arr = g2f_arr[g_safe]  # family index per species

    print(f"[PhyloAdj] Building {n_species}×{n_species} adjacency in chunks of {CHUNK}...")
    adj = np.empty((n_species, n_species), dtype=np.float32)

    for i_start in range(0, n_species, CHUNK):
        i_end = min(i_start + CHUNK, n_species)
        g_row = s2g_arr[i_start:i_end]  # (chunk,)
        f_row = f_arr[i_start:i_end]    # (chunk,)

        # Broadcast comparisons: shape (chunk, n_species)
        same_genus  = (g_row[:, None] == s2g_arr[None, :])
        same_family = (f_row[:, None] == f_arr[None, :])

        block = np.where(
            same_genus,
            W_SAME_GENUS,
            np.where(same_family, W_SAME_FAMILY, W_DIFF_FAMILY)
        ).astype(np.float32)

        adj[i_start:i_end] = block

        if (i_start // CHUNK) % 4 == 0:
            pct = 100 * i_end / n_species
            print(f"  {pct:.0f}%  ({i_end}/{n_species} rows)", end="\r")

    print()

    # Self-loops: ensure diagonal is 1.0 (species is its own genus peer)
    np.fill_diagonal(adj, W_SAME_GENUS)

    # Row-normalise: D^{-1} A
    row_sums = adj.sum(axis=1, keepdims=True)
    adj /= np.maximum(row_sums, 1e-8)

    return adj


def main():
    if not S2G_PATH.exists() or not G2F_PATH.exists():
        sys.exit(f"[PhyloAdj] ERROR: taxonomy files not found in {DATA_DIR}")

    # Match the shape of the existing ecological adjacency if present
    if ECO_ADJ_PATH.exists():
        eco = np.load(ECO_ADJ_PATH, mmap_mode='r')
        n_species = eco.shape[0]
        del eco
        print(f"[PhyloAdj] Matched n_species={n_species} from ecological_adj.npy")
    else:
        n_species = 7806
        print(f"[PhyloAdj] ecological_adj.npy not found; defaulting to n_species={n_species}")

    adj = build_phylo_adj(n_species)

    np.save(OUT_PATH, adj)
    print(f"[PhyloAdj] Saved {adj.shape} matrix to {OUT_PATH}")
    print(f"[PhyloAdj] Stats: min={adj.min():.4f}, max={adj.max():.4f}, "
          f"mean={adj.mean():.4f}")


if __name__ == "__main__":
    main()
