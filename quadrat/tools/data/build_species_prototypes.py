#!/usr/bin/env python3
"""Build a [num_classes, feat_dim] prototype tensor for FAISS retrieval.

For each species class, average the (concat-of-3-backbone) features of all
training samples. Stored as `models/species_prototypes.pt`. RetrievalEngine
loads this at inference startup.

Source features come from `models/cuda_deep_sat/phase1_feature_cache.pt`,
which is the output of phase 1 extraction (frozen pretrained backbones).

Usage:
    python scripts/build_species_prototypes.py \
        --cache models/cuda_deep_sat/phase1_feature_cache.pt \
        --out   models/species_prototypes.pt \
        --num-classes 7808
"""
from __future__ import annotations
import argparse
import os
import sys

import torch
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",       default="models/cuda_deep_sat/phase1_feature_cache.pt")
    ap.add_argument("--out",         default="models/species_prototypes.pt")
    ap.add_argument("--num-classes", type=int, default=7808)
    ap.add_argument("--mapping",     default="/workspace/plantclef/processed/species_ids_mapping.csv",
                    help="Used to derive class indices from species IDs in labels (if needed).")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        print(f"FATAL: phase1 feature cache not found: {args.cache}", file=sys.stderr)
        return 1

    print(f"[Prototypes] Loading {args.cache} ...")
    sd = torch.load(args.cache, map_location="cpu", weights_only=False)
    bio  = sd["bio"]    # [N, 768]
    dino = sd["dino"]   # [N, 1024]
    conv = sd["conv"]   # [N, 1536]
    labels = sd["labels"]  # [N]  — these are species IDs OR class indices, will detect

    N = bio.shape[0]
    feat_dim = bio.shape[1] + dino.shape[1] + conv.shape[1]   # 3328
    print(f"[Prototypes] N={N:,} samples, feat_dim={feat_dim} (bio+dino+conv)")

    # Detect label format: if max(labels) > num_classes, they're species IDs (need remap)
    lbl_max = int(labels.max().item())
    if lbl_max >= args.num_classes:
        print(f"[Prototypes] Labels look like species IDs (max={lbl_max}); remapping via {args.mapping}")
        if not os.path.exists(args.mapping):
            print(f"FATAL: mapping CSV not found: {args.mapping}", file=sys.stderr)
            return 1
        # Build species_id → class_index lookup
        species_ids = []
        with open(args.mapping) as f:
            for line in f:
                line = line.strip()
                if line:
                    species_ids.append(int(line))
        id_to_idx = {sid: i for i, sid in enumerate(species_ids)}
        # Vectorize remap on CPU
        labels_np = labels.numpy()
        remapped = np.zeros_like(labels_np)
        for i, sid in enumerate(labels_np):
            remapped[i] = id_to_idx.get(int(sid), -1)
        # Drop unmapped
        valid = remapped >= 0
        if not valid.all():
            n_drop = int((~valid).sum())
            print(f"[Prototypes] Dropping {n_drop:,} samples with unmapped species IDs")
        labels_idx = torch.from_numpy(remapped[valid])
        bio  = bio[valid]
        dino = dino[valid]
        conv = conv[valid]
        N = bio.shape[0]
    else:
        labels_idx = labels.long()
        print(f"[Prototypes] Labels look like class indices (max={lbl_max}); using directly")

    # Concat features per sample: [N, 3328]
    print(f"[Prototypes] Concatenating bio+dino+conv features ...")
    feats = torch.cat([bio, dino, conv], dim=1).float()

    # L2-normalize per sample (cosine similarity prep)
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)

    # Per-class mean
    print(f"[Prototypes] Computing class means over {args.num_classes} classes ...")
    prototypes = torch.zeros(args.num_classes, feat_dim, dtype=torch.float32)
    counts     = torch.zeros(args.num_classes, dtype=torch.long)
    # Use index_add for speed
    prototypes.index_add_(0, labels_idx.long(), feats)
    counts.index_add_(0, labels_idx.long(), torch.ones_like(labels_idx, dtype=torch.long))

    nonzero = counts > 0
    prototypes[nonzero] = prototypes[nonzero] / counts[nonzero].float().unsqueeze(1)

    # Re-normalize means (mean of unit vectors is not unit-length)
    prototypes[nonzero] = torch.nn.functional.normalize(prototypes[nonzero], p=2, dim=1)

    n_empty = int((~nonzero).sum())
    if n_empty:
        print(f"[Prototypes] WARN: {n_empty} classes had zero samples (left as zero vectors)")

    print(f"[Prototypes] Saving → {args.out}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.save(prototypes, args.out)
    sz_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[Prototypes] Done. Shape={tuple(prototypes.shape)}  size={sz_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
