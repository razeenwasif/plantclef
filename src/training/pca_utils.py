"""PCA compression for cached backbone features.

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
from typing import Dict, Any, Tuple, Optional

from src import config as _cfg

PCA_COMPONENTS = getattr(_cfg, "PCA_COMPONENTS", 512)
PCA_TRANSFORM_PATH = getattr(_cfg, "PCA_TRANSFORM_PATH", "models/pca_transform.pkl")
USE_REGION_FEATURES = getattr(_cfg, "USE_REGION_FEATURES", False)


def _get_batch(cache: Dict[str, torch.Tensor], start: int, end: int) -> np.ndarray:
    """
    Extract a slice of concatenated features without materializing the full matrix.
    """
    if USE_REGION_FEATURES:
        parts = [
            cache['bio'][start:end], cache['bio_crop'][start:end],
            cache['dino'][start:end], cache['dino_crop'][start:end],
            cache['conv'][start:end], cache['conv_crop'][start:end],
        ]
    else:
        parts = [cache['bio'][start:end], cache['dino'][start:end], cache['conv'][start:end]]
    
    # Concatenate only this batch and convert to float32 for sklearn
    return torch.cat(parts, dim=1).float().numpy()


def fit_pca(cache: Dict[str, torch.Tensor], batch_size: int = 50000) -> Any:
    """
    Fit IncrementalPCA on cached features in memory-efficient batches.

    Parameters
    ----------
    cache : Dict[str, torch.Tensor]
        Feature cache dictionary.
    batch_size : int, default 50000
        Size of batches for fitting.

    Returns
    -------
    Any
        The fitted sklearn IncrementalPCA object.
    """
    from sklearn.decomposition import IncrementalPCA

    num_samples = len(cache['labels'])
    # Determine input dimension by looking at first sample
    first_batch = _get_batch(cache, 0, 1)
    input_dim = first_batch.shape[1]
    
    n_components = min(PCA_COMPONENTS, input_dim, num_samples)

    print(f"[PCA] Fitting IncrementalPCA: {num_samples:,} samples × {input_dim}-d "
          f"→ {n_components}-d ...")
    pca = IncrementalPCA(n_components=n_components)

    for start in tqdm(range(0, num_samples, batch_size), desc="PCA fit"):
        end = min(start + batch_size, num_samples)
        batch = _get_batch(cache, start, end)
        pca.partial_fit(batch)

    return pca


def apply_pca(cache: Dict[str, torch.Tensor], pca: Any, batch_size: int = 100000) -> Dict[str, torch.Tensor]:
    """
    PLANTCLEF: Accelerated PCA transformation using optimized tensor views.
    """
    num_samples = len(cache['labels'])
    
    # plantclef: Attempt GPU-accelerated transformation if CuPy is installed
    try:
        import cupy as cp
        print("[PCA] Using CuPy for GPU-accelerated transformation...")
        
        # Move transformation matrix to GPU once
        pca_comp_gpu = cp.array(pca.components_.T, dtype=cp.float32)
        pca_mean_gpu = cp.array(pca.mean_, dtype=cp.float32)
        
        out_gpu = cp.zeros((num_samples, pca.n_components_), dtype=cp.float32)
        
        # Batch transform directly on GPU
        for start in tqdm(range(0, num_samples, batch_size), desc="GPU PCA transform"):
            end = min(start + batch_size, num_samples)
            batch = cp.array(_get_batch(cache, start, end), dtype=cp.float32)
            out_gpu[start:end] = (batch - pca_mean_gpu).dot(pca_comp_gpu)
            
        cache['features_pca'] = torch.as_tensor(out_gpu.get(), device='cpu')
        
    except ImportError:
        # Fallback to optimized NumPy views
        print("[PCA] CuPy not found. Using optimized NumPy views...")
        out = np.zeros((num_samples, pca.n_components_), dtype=np.float32)
        pca_comp = pca.components_.T
        pca_mean = pca.mean_

        for start in tqdm(range(0, num_samples, batch_size), desc="CPU PCA transform"):
            end = min(start + batch_size, num_samples)
            batch = _get_batch(cache, start, end)
            out[start:end] = (batch - pca_mean).dot(pca_comp)
        
        cache['features_pca'] = torch.from_numpy(out)

    print(f"[PCA] Transformed {num_samples:,} samples -> shape {cache['features_pca'].shape}")
    return cache


def save_pca(pca: Any, path: str = PCA_TRANSFORM_PATH) -> None:
    """
    Save fitted PCA object to disk.

    Parameters
    ----------
    pca : Any
        The fitted PCA object.
    path : str
        Path where to save the object.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"[PCA] Saved transform to {path}")


def load_pca(path: str = PCA_TRANSFORM_PATH) -> Any:
    """
    Load fitted PCA object from disk.

    Parameters
    ----------
    path : str
        Path to the saved PCA object.

    Returns
    -------
    Any
        The loaded PCA object.
    """
    with open(path, 'rb') as f:
        pca = pickle.load(f)
    print(f"[PCA] Loaded transform from {path} ({pca.n_components_} components)")
    return pca


def fit_and_save(cache: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Any]:
    """
    High-level helper to fit PCA, save it, and apply it to the cache.

    Parameters
    ----------
    cache : Dict[str, torch.Tensor]
        The raw feature cache.

    Returns
    -------
    Tuple[Dict[str, torch.Tensor], Any]
        A tuple of (updated_cache, fitted_pca).
    """
    pca = fit_pca(cache)
    save_pca(pca)
    cache = apply_pca(cache, pca)
    return cache, pca
