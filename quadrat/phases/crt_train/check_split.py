import torch
import sys
from phases.crt_train.dataset import CachedFeatureDataset

full_ds = CachedFeatureDataset("models/cuda_deep_sat/phase1_feature_cache.pt")
val_len = max(1, int(len(full_ds) * 0.05))
val_labels = full_ds.labels[:val_len]
train_labels = full_ds.labels[val_len:]

print(f"Total samples: {len(full_ds)}")
print(f"Val samples: {len(val_labels)}")
print(f"Train samples: {len(train_labels)}")

val_unique = torch.unique(val_labels)
train_unique = torch.unique(train_labels)

print(f"Unique classes in Val: {len(val_unique)}")
print(f"Unique classes in Train: {len(train_unique)}")

overlap = len(set(val_unique.tolist()).intersection(set(train_unique.tolist())))
print(f"Classes in BOTH Val and Train: {overlap}")
