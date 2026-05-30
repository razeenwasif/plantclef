"""Evaluates the cRT dual-head ensemble on the Phase 1 feature cache.

Usage:
    python phases/crt_train/eval.py --cache_path models/cuda_deep_sat/phase1_feature_cache.pt
"""
import argparse
import sys
import time
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader

from phases.crt_train.dataset import CachedFeatureDataset
from phases.crt_train.model import ClassificationHead


def evaluate_model(model_a, model_b, loader, device, num_classes, global_class_counts):
    model_a.eval()
    model_b.eval()

    correct_a = correct_b = correct_ensemble = total = 0

    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)

    print("Evaluating over validation set...")
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)

            logits_a = model_a(feats)
            logits_b = model_b(feats)

            # Accuracy A
            preds_a = logits_a.argmax(1)
            correct_a += (preds_a == labels).sum().item()

            # Accuracy B
            preds_b = logits_b.argmax(1)
            correct_b += (preds_b == labels).sum().item()

            # Ensemble (Average Probabilities)
            probs_a = torch.softmax(logits_a, dim=-1)
            probs_b = torch.softmax(logits_b, dim=-1)
            ensemble_probs = (probs_a + probs_b) / 2.0
            preds_ens = ensemble_probs.argmax(1)

            correct_ensemble += (preds_ens == labels).sum().item()
            total += labels.size(0)

            # Class-wise accuracy for tail analysis
            for c in range(num_classes):
                mask = (labels == c)
                class_total[c] += mask.sum()
                class_correct[c] += (preds_ens[mask] == labels[mask]).sum()

    acc_a = correct_a / total * 100
    acc_b = correct_b / total * 100
    acc_ens = correct_ensemble / total * 100

    print(f"\n--- Results ---")
    print(f"Head A (Standard) Accuracy: {acc_a:.2f}%")
    print(f"Head B (Balanced) Accuracy: {acc_b:.2f}%")
    print(f"Ensemble Accuracy:          {acc_ens:.2f}%")

    # Tail analysis
    # Sort classes by their true global frequency to define head vs tail accurately
    sorted_totals, indices = torch.sort(global_class_counts.to(device), descending=True)
    
    # Let's say top 20% are 'head' classes, bottom 80% are 'tail' classes
    head_cutoff = int(num_classes * 0.2)
    
    head_indices = indices[:head_cutoff]
    tail_indices = indices[head_cutoff:]

    head_correct = class_correct[head_indices].sum().item()
    head_total_count = class_total[head_indices].sum().item()

    tail_correct = class_correct[tail_indices].sum().item()
    tail_total_count = class_total[tail_indices].sum().item()

    if head_total_count > 0:
        head_acc = head_correct / head_total_count * 100
        print(f"\nEnsemble Accuracy on Head Classes (top 20%): {head_acc:.2f}%")
    
    if tail_total_count > 0:
        tail_acc = tail_correct / tail_total_count * 100
        print(f"Ensemble Accuracy on Tail Classes (bottom 80%): {tail_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default="models/cuda_deep_sat/phase1_feature_cache.pt")
    parser.add_argument("--head_a", type=str, default="models/crt/head_a_standard_best.pth")
    parser.add_argument("--head_b", type=str, default="models/crt/head_b_balanced_best.pth")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=7808)
    parser.add_argument("--in_features", type=int, default=3328)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading cached features from {args.cache_path}...")
    t0 = time.time()
    full_ds = CachedFeatureDataset(args.cache_path)
    print(f"Loaded {len(full_ds)} samples in {time.time() - t0:.2f}s.")

    # Split train/val deterministically to match training
    indices = torch.randperm(len(full_ds), generator=torch.Generator().manual_seed(42)).tolist()
    val_len = max(1, int(len(full_ds) * 0.05))
    val_indices = indices[:val_len]
    
    val_ds = torch.utils.data.Subset(full_ds, val_indices)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Compute true global class counts for accurate head/tail definition
    global_class_counts = torch.bincount(full_ds.labels, minlength=args.num_classes)

    print("Loading models...")
    head_a = ClassificationHead(in_features=args.in_features, out_features=args.num_classes).to(device)
    ckpt_a = torch.load(args.head_a, map_location=device)
    head_a.load_state_dict(ckpt_a.get("model_state", ckpt_a))

    head_b = ClassificationHead(in_features=args.in_features, out_features=args.num_classes).to(device)
    ckpt_b = torch.load(args.head_b, map_location=device)
    head_b.load_state_dict(ckpt_b.get("model_state", ckpt_b))

    evaluate_model(head_a, head_b, val_loader, device, args.num_classes, global_class_counts)

if __name__ == "__main__":
    main()
