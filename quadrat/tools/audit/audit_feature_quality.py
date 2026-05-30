import torch
import numpy as np
import os
import glob
from tqdm import tqdm

def audit_shard(shard_path, anchor_path="models/zero_shot_anchors.pt"):
    print(f"=== plantclef Feature Quality Audit: {os.path.basename(shard_path)} ===")
    
    # 1. Load Data
    data = torch.load(shard_path, weights_only=False)
    bio_feats = data['bio'].float().cuda()
    labels = data['labels'].cuda()
    
    # 2. Basic Sanity Check
    print(f"[Sanity] Samples: {len(labels)} | Shape: {bio_feats.shape}")
    nan_count = torch.isnan(bio_feats).sum().item()
    zero_count = (bio_feats.abs().sum(dim=1) == 0).sum().item()
    print(f"[Sanity] NaNs: {nan_count} | Dead Vectors: {zero_count}")
    
    if nan_count > 0 or zero_count > 10:
        print("!! CRITICAL WARNING: Feature collapse detected !!")
    
    # 3. Intra-Class Similarity (The "Tightness" Test)
    # Pick a species with multiple samples in this shard
    unique_labels, counts = torch.unique(labels, return_counts=True)
    test_labels = unique_labels[counts > 1][:5] # Take first 5 species with >1 sample
    
    print("\n[Quality] Intra-Class Cosine Similarity (Higher is better):")
    bio_feats_norm = torch.nn.functional.normalize(bio_feats, dim=1)
    
    for label in test_labels:
        mask = (labels == label)
        class_feats = bio_feats_norm[mask]
        # Compute all-to-all similarity within the class
        sim_matrix = torch.mm(class_feats, class_feats.t())
        avg_sim = (sim_matrix.sum() - len(class_feats)) / (len(class_feats) * (len(class_feats) - 1))
        print(f"  - Species {label.item()}: {avg_sim.item():.4f}")

    # 4. Zero-Shot Probe
    if os.path.exists(anchor_path):
        print("\n[Accuracy] Running Zero-Shot Baseline Accuracy...")
        anchors = torch.load(anchor_path, weights_only=True).float().cuda()
        anchors_norm = torch.nn.functional.normalize(anchors, dim=1)
        
        # Logits = Features @ Anchors.T
        logits = torch.mm(bio_feats_norm, anchors_norm.t()) / 0.07
        preds = logits.argmax(dim=1)
        
        acc = (preds == labels).float().mean() * 100
        print(f"  - Shard Zero-Shot Accuracy: {acc.item():.2f}%")
        
        if acc > 30:
            print("\n>>> VERDICT: FEATURES ARE EXCELLENT (High Signal)")
        elif acc > 10:
            print("\n>>> VERDICT: FEATURES ARE GOOD (Usable)")
        else:
            print("\n>>> VERDICT: WARNING - Low zero-shot signal. Check preprocessing.")

if __name__ == "__main__":
    # Audit the very first shard from Rank 1 (which we know is finished)
    shard_files = sorted(glob.glob("models/cuda_deep_sat/phase1_feature_cache.pt_shards/shard_rank1_*.pt"))
    if shard_files:
        audit_shard(shard_files[0])
    else:
        print("No shards found to audit.")
