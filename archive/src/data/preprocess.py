import os
import subprocess
import cudf
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# NVIDIA DALI imports for GPU-accelerated image loading
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import nvidia.dali.types as types


# ---------------------------------------------------------------------------
# DALI Pipeline: decodes, resizes, and grayscales images entirely on the GPU
# Note: device_id is a reserved @pipeline_def constructor arg, so it must NOT
# appear as a function parameter -- pass it only via blur_pipeline(..., device_id=)
# ---------------------------------------------------------------------------

@pipeline_def
def blur_pipeline(paths):
    jpegs, _ = fn.readers.file(files=paths, random_shuffle=False, name="Reader")
    # 'mixed' = decode on GPU
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.GRAY)
    images = fn.resize(images, resize_x=512, resize_y=512, device='gpu')
    # Cast to float, keep values in [0, 255]
    images = fn.cast(images, dtype=types.FLOAT, device='gpu')
    # No normalisation -- keep values in [0, 255] so blur threshold is intuitive
    return images


def gpu_blur_audit(paths, batch_size=512, gpu_id=0):
    """
    GPU-Accelerated Blur Detection using NVIDIA DALI.

    DALI keeps the entire pipeline (decode -> resize -> grayscale) on the GPU,
    eliminating the CPU bottleneck that starved the GPU with the old
    PIL + torchvision approach.
    """
    device = torch.device('cuda', gpu_id)

    # Build the DALI pipeline.
    # device_id is passed as a @pipeline_def constructor kwarg, not a function param.
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
    # last_batch_policy must be the LastBatchPolicy enum, not a plain string.
    loader = DALIGenericIterator(
        pipe,
        output_map=["images"],
        size=len(paths),
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )

    # 3x3 Laplacian + Sobel kernels -- stacked for a single batched conv.
    # This gives the GPU more work per batch and keeps it utilised longer.
    kernel_lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    kernel_sx  = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    kernel_sy  = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    # Shape: (3, 1, 3, 3)
    kernels = torch.stack([kernel_lap, kernel_sx, kernel_sy]).unsqueeze(1).to(device)

    scores = np.zeros(len(paths), dtype=np.float32)
    offset = 0

    print(f"[Blur Audit] Running on {device} ({len(paths):,} images)...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="GPU Processing", unit="batch"):
            # imgs shape from DALI: (B, H, W, 1) -- rearrange to (B, 1, H, W)
            imgs = batch[0]["images"].permute(0, 3, 1, 2)
            B = imgs.shape[0]

            # Single batched conv across all kernels
            responses = F.conv2d(imgs, kernels, padding=1)  # (B, 3, H, W)

            # Use Laplacian variance as the primary focus score.
            # squeeze(1) only removes the channel dim, never collapses the batch dim.
            lap_var = torch.var(responses[:, 0:1, :, :], dim=(2, 3)).squeeze(1)  # (B,)

            scores[offset:offset + B] = lap_var.cpu().numpy()
            offset += B

    return scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def audit_data(csv_path, img_dir, output_path, max_images_per_species=500, blur_threshold=100):
    """
    Full GPU-Accelerated Preprocessing Pipeline.
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
    # Python's os.walk is too slow for 1.4M files.
    print(f"Scanning {img_dir} for existing images (using shell find)...")
    # Quote img_dir to handle spaces / special characters safely
    find_cmd = f"find '{img_dir}' -maxdepth 2 -type f -name '*.jpg' | sed 's|{img_dir}||'"
    try:
        existing_files_str = subprocess.check_output(find_cmd, shell=True).decode('utf-8')
        existing_files = cudf.Series(existing_files_str.splitlines()).str.lstrip('/')
    except Exception as e:
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

    # 5. Long-Tail Balancing -- shuffle first to avoid collection-order bias
    if max_images_per_species > 0:
        print(f"Capping species at {max_images_per_species} images...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df['_rank'] = df.groupby('species_id').cumcount()
        df = df[df['_rank'] < max_images_per_species].drop(columns=['_rank'])
        print(f"Final records after balancing: {len(df):,}")

    # 6. Cleanup and Save
    cols_to_drop = [c for c in ['relative_path', 'blur_score'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df.to_csv(output_path, sep=';', index=False)

    print(f"[Audit Complete] Final Unique Species: {df['species_id'].nunique():,}")
    print(f"Cleaned metadata saved to: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import RAW_CSV, IMG_DIR, CLEANED_CSV

    audit_data(RAW_CSV, IMG_DIR, CLEANED_CSV, max_images_per_species=500, blur_threshold=100)
