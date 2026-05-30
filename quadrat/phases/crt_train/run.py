"""Classifier Retraining (cRT) entry point.

Trains two separate heads on cached Phase 1 features:
  - Head A (Common): Standard Cross-Entropy training on natural distribution.
  - Head B (Rare):   Weighted Random Sampling to oversample tail classes.

Usage:
    python phases/crt_train/run.py --cache_path models/cuda_deep_sat/phase1_feature_cache.pt
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from phases.crt_train.dataset import CachedFeatureDataset
from phases.crt_train.model import ClassificationHead


def _train_head(
    head_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    out_dir: str = "models/crt",
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    start_epoch = 0
    
    file_prefix = head_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    best_path = os.path.join(out_dir, f"{file_prefix}_best.pth")
    latest_path = os.path.join(out_dir, f"{file_prefix}_latest.pth")

    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(latest_path):
        print(f"Resuming {head_name} from {latest_path}...")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        print(f"Resumed at epoch {start_epoch} with best_val_acc {best_val_acc:.2f}%")

    print(f"\n--- Training {head_name} ---")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_correct = train_total = 0
        train_loss = 0.0

        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)
                logits = model(feats)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total * 100 if train_total else 0.0
        val_acc = val_correct / val_total * 100 if val_total else 0.0

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save latest checkpoint
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": max(best_val_acc, val_acc),
        }
        
        tmp_latest = latest_path + ".tmp"
        torch.save(state, tmp_latest)
        os.replace(tmp_latest, latest_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            tmp_best = best_path + ".tmp"
            torch.save(state, tmp_best)
            os.replace(tmp_best, best_path)

    print(f"[{head_name}] Done. Best Val Acc: {best_val_acc:.2f}% -> Saved best to {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default="models/cuda_deep_sat/phase1_feature_cache.pt")
    parser.add_argument("--out_dir", type=str, default="models/crt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=7808)
    parser.add_argument("--in_features", type=int, default=3328)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"Loading cached features from {args.cache_path}...")
    t0 = time.time()
    full_ds = CachedFeatureDataset(args.cache_path)
    print(f"Loaded {len(full_ds)} samples in {time.time() - t0:.2f}s.")

    # Split train/val deterministically
    indices = torch.randperm(len(full_ds), generator=torch.Generator().manual_seed(42)).tolist()
    val_len = max(1, int(len(full_ds) * 0.05))
    
    val_indices = indices[:val_len]
    train_indices = indices[val_len:]

    train_ds = torch.utils.data.Subset(full_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_ds, val_indices)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 1. Train Head A (Standard Distribution)
    train_loader_a = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    head_a = ClassificationHead(in_features=args.in_features, out_features=args.num_classes)
    _train_head(
        "Head A Standard", head_a, train_loader_a, val_loader, device,
        epochs=args.epochs, out_dir=args.out_dir
    )

    # 2. Train Head B (Class-Balanced / Rare Expert)
    print("\nComputing class weights for Head B...")
    # Extract labels for the train subset
    train_labels = full_ds.labels[train_indices]
    class_counts = torch.bincount(train_labels, minlength=args.num_classes)
    
    # Avoid division by zero for missing classes
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights[class_counts == 0] = 0.0

    sample_weights = class_weights[train_labels]
    sampler_b = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader_b = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler_b, num_workers=4, pin_memory=True)
    
    # We apply more weight decay for Head B because oversampling can cause overfitting
    head_b = ClassificationHead(in_features=args.in_features, out_features=args.num_classes)
    _train_head(
        "Head B (Balanced)", head_b, train_loader_b, val_loader, device,
        epochs=args.epochs, weight_decay=0.05, out_dir=args.out_dir
    )

    print("\nAll cRT training complete!")


if __name__ == "__main__":
    main()
