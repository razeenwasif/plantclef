"""
PCA compression for cached backbone features.

Fits IncrementalPCA on the concatenated raw backbone features:
  [bio_full + bio_crop + dino_full + dino_crop + conv_full + conv_crop]
  = (512+512+1024+1024+1536+1536) = 6144-d  (with region features)
  or [bio + dino + conv] = 3072-d            (without region features)

Compresses to PCA_COMPONENTS-d (default 512), which:
  - Removes inter-backbone redundancy
  - Makes Phase 1 classifier tiny (512 → num_classes)
  - Speeds up Phase 1 training significantly
  - Retains >95% of explained variance at 512 components
"""
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

import config as _cfg
PCA_COMPONENTS      = getattr(_cfg, "PCA_COMPONENTS",     512)
PCA_TRANSFORM_PATH  = getattr(_cfg, "PCA_TRANSFORM_PATH",  "models/pca_transform.pkl")
USE_REGION_FEATURES = getattr(_cfg, "USE_REGION_FEATURES", False)


def _build_feature_matrix(cache):
    """
    Concatenate all cached feature vectors into a single (N, D) matrix.
    D = 6144 with region features, 3072 without.
    """
    if USE_REGION_FEATURES:
        parts = [
            cache['bio'],      cache['bio_crop'],
            cache['dino'],     cache['dino_crop'],
            cache['conv'],     cache['conv_crop'],
        ]
    else:
        parts = [cache['bio'], cache['dino'], cache['conv']]
    return torch.cat(parts, dim=1)   # (N, D)


def fit_pca(cache, batch_size=50000):
    """
    Fit IncrementalPCA on cached features in CPU-friendly batches.
    IncrementalPCA avoids loading all N×D floats into RAM simultaneously.

    Returns the fitted sklearn IncrementalPCA object.
    """
    from sklearn.decomposition import IncrementalPCA

    X   = _build_feature_matrix(cache)  # (N, D) float32 tensor on CPU
    N, D = X.shape
    n_components = min(PCA_COMPONENTS, D, N)

    print(f"[PCA] Fitting IncrementalPCA: {N:,} samples × {D}-d → {n_components}-d ...")
    pca = IncrementalPCA(n_components=n_components)

    for start in tqdm(range(0, N, batch_size), desc="PCA fit"):
        batch = X[start:start + batch_size].numpy().astype(np.float32)
        pca.partial_fit(batch)

    var_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[PCA] Explained variance: {var_explained:.1f}%  "
          f"(components={n_components})")
    return pca


def apply_pca(cache, pca, batch_size=50000):
    """
    Transform cached features using a fitted PCA object.
    Adds cache['features_pca'] (N, PCA_COMPONENTS) float32 tensor.
    """
    X = _build_feature_matrix(cache)
    N = len(X)
    out = np.zeros((N, pca.n_components_), dtype=np.float32)

    for start in tqdm(range(0, N, batch_size), desc="PCA transform"):
        batch       = X[start:start + batch_size].numpy().astype(np.float32)
        out[start:start + batch_size] = pca.transform(batch)

    cache['features_pca'] = torch.from_numpy(out)
    print(f"[PCA] Transformed {N:,} samples → shape {cache['features_pca'].shape}")
    return cache


def save_pca(pca, path=PCA_TRANSFORM_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"[PCA] Saved transform to {path}")


def load_pca(path=PCA_TRANSFORM_PATH):
    with open(path, 'rb') as f:
        pca = pickle.load(f)
    print(f"[PCA] Loaded transform from {path} "
          f"({pca.n_components_} components)")
    return pca


def fit_and_save(cache):
    """Convenience: fit PCA, apply to cache, save transform. Returns updated cache."""
    pca   = fit_pca(cache)
    cache = apply_pca(cache, pca)
    save_pca(pca)
    return cache, pca
