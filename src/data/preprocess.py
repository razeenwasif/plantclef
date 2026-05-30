"""Preprocesses images using GPU-accelerated blur detection and filtering.

CUDA-only data-preparation utility (NVIDIA DALI + cuDF). Intended to be
run once on a CUDA host to build the cleaned manifest that downstream
training consumes — whether that training runs on CUDA or TPU. TPU users
should NOT invoke this directly; build shards from the cleaned manifest
on a CUDA box first, then point the TPU training run at the shard
directory.

Fail-fast guard below blocks the imports if someone accidentally hits
this module under `CLUSTER_MODE=tpu`.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Union, Optional, Any

# Fail-fast: this whole module is CUDA-only. Detect the active accelerator
# mode (set by CLUSTER_MODE / plantclef.py --mode) and raise a clear error
# before the DALI / cuDF imports below produce a confusing stack trace.
_mode = os.environ.get("CLUSTER_MODE", "auto").lower()
if _mode == "tpu":
    raise RuntimeError(
        "src/data/preprocess.py is a CUDA-only data-prep pipeline (DALI + cuDF). "
        "It cannot run on TPU hosts. Run dataset cleaning on a CUDA box first "
        "(`CLUSTER_MODE=cuda python -m src.data.preprocess ...`), then point your "
        "TPU training run at the resulting shard directory."
    )

import cudf
import numpy as np
import nvidia.dali.types as types
import torch
import torch.nn.functional as F
from nvidia.dali import fn, pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
CLEANED_CSV = config.CLEANED_CSV
IMG_DIR = config.IMG_DIR
RAW_CSV = config.RAW_CSV


@pipeline_def
def blur_pipeline(paths: List[str]) -> Any:
    """
    DALI pipeline for decoding, resizing, and grayscaling images.

    Parameters
    ----------
    paths : List[str]
        List of image file paths to process.

    Returns
    -------
    nvidia.dali.pipeline.DataNode
        The processed image data node on GPU.
    """
    jpegs, _ = fn.readers.file(files=paths, random_shuffle=False, name="Reader")
    # 'mixed' = decode on GPU
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.GRAY)
    images = fn.resize(images, resize_x=512, resize_y=512, device='gpu')
    # Cast to float, keep values in [0, 255]
    images = fn.cast(images, dtype=types.FLOAT, device='gpu')
    # No normalisation -- keep values in [0, 255] so blur threshold is intuitive
    return images


def gpu_blur_audit(paths: List[str], batch_size: int = 512, gpu_id: int = 0) -> np.ndarray:
    """
    GPU-Accelerated Blur Detection using NVIDIA DALI.

    DALI keeps the entire pipeline (decode -> resize -> grayscale) on the GPU,
    eliminating the CPU bottleneck.

    Parameters
    ----------
    paths : List[str]
        List of image file paths.
    batch_size : int, default 512
        The number of images to process in each batch.
    gpu_id : int, default 0
        The GPU device ID to use.

    Returns
    -------
    np.ndarray
        An array of blur scores (Laplacian variance) for each image.
    """
    device = torch.device('cuda', gpu_id)

    # Build the DALI pipeline.
    pipe = blur_pipeline(
        paths=paths,
        device_id=gpu_id,
        batch_size=batch_size,
        num_threads=4,
        exec_async=True,
        exec_pipelined=True,
    )
    pipe.build()

    # Wrap in a PyTorch iterator -- outputs land directly on the GPU.
    loader = DALIGenericIterator(
        pipe,
        output_map=["images"],
        size=len(paths),
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )

    # 3x3 Laplacian + Sobel kernels -- stacked for a single batched conv.
    kernel_lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                              dtype=torch.float32)
    kernel_sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                             dtype=torch.float32)
    kernel_sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                             dtype=torch.float32)
    # Shape: (3, 1, 3, 3)
    kernels = torch.stack([kernel_lap, kernel_sx,
                           kernel_sy]).unsqueeze(1).to(device)

    scores = np.zeros(len(paths), dtype=np.float32)
    offset = 0

    print(f"[Blur Audit] Running on {device} ({len(paths):,} images)...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="GPU Processing", unit="batch"):
            # imgs shape from DALI: (B, H, W, 1) -- rearrange to (B, 1, H, W)
            imgs = batch[0]["images"].permute(0, 3, 1, 2)
            batch_size_actual = imgs.shape[0]

            # Single batched conv across all kernels
            responses = F.conv2d(imgs, kernels, padding=1)  # (B, 3, H, W)

            # Use Laplacian variance as the primary focus score.
            lap_var = torch.var(
                responses[:, 0:1, :, :], dim=(2, 3)).squeeze(1)  # (B,)

            scores[offset:offset + batch_size_actual] = lap_var.cpu().numpy()
            offset += batch_size_actual

    return scores


def audit_data(csv_path: str, img_dir: str, output_path: str, 
               max_images_per_species: int = 500, blur_threshold: float = 100) -> None:
    """
    Full GPU-Accelerated Preprocessing Pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the input metadata CSV file.
    img_dir : str
        Directory containing the plant images.
    output_path : str
        Path to save the cleaned and filtered CSV file.
    max_images_per_species : int, default 500
        Maximum number of images per species to retain (long-tail balancing).
    blur_threshold : float, default 100
        Minimum Laplacian variance threshold for blur detection.
    """
    # Normalise img_dir so path joins are always correct
    img_dir = img_dir.rstrip('/') + '/'

    print(f"[Audit] Loading metadata from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. GPU-Accelerated CSV Loading
    df = cudf.read_csv(csv_path, sep=';')
    print(f"Initial records: {len(df):,}")

    # 2. Fast File Discovery (Shell-assisted)
    print(f"Scanning {img_dir} for existing images (using shell find)...")
    find_cmd = (f"find '{img_dir}' -maxdepth 2 -type f -name '*.jpg' | "
                f"sed 's|{img_dir}||'")
    try:
        existing_files_str = subprocess.check_output(
            find_cmd, shell=True).decode('utf-8')
        existing_files = cudf.Series(
            existing_files_str.splitlines()).str.lstrip('/')
    except subprocess.CalledProcessError as e:
        print(f"Error scanning files: {e}")
        return

    # 3. GPU-Accelerated Join / Filter
    df['relative_path'] = df['species_id'].astype(str) + "/" + df['image_name']
    df = df[df['relative_path'].isin(existing_files)]
    print(f"After physical file verification: {len(df):,}")

    # 4. GPU Blur Audit via DALI
    if blur_threshold > 0:
        full_paths = (img_dir + df['relative_path']).to_arrow().to_pylist()
        blur_scores = gpu_blur_audit(full_paths)

        df['blur_score'] = cudf.Series(blur_scores)
        df = df[df['blur_score'] >= blur_threshold]
        print(f"After removing blurry images: {len(df):,}")

    # 5. Long-Tail Balancing
    if max_images_per_species > 0:
        print(f"Capping species at {max_images_per_species} images...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df['_rank'] = df.groupby('species_id').cumcount()
        df = df[df['_rank'] < max_images_per_species].drop(columns=['_rank'])
        print(f"Final records after balancing: {len(df):,}")

    # 6. Cleanup and Save
    cols_to_drop = [
        c for c in ['relative_path', 'blur_score'] if c in df.columns
    ]
    df = df.drop(columns=cols_to_drop)
    df.to_csv(output_path, sep=';', index=False)

    print(f"[Audit Complete] Final Unique Species: "
          f"{df['species_id'].nunique():,}")
    print(f"Cleaned metadata saved to: {output_path}")


def main() -> None:
    """
    Main entry point for preprocessing.
    """
    audit_data(
        RAW_CSV,
        IMG_DIR,
        CLEANED_CSV,
        max_images_per_species=500,
        blur_threshold=100)



if __name__ == "__main__":
    main()
