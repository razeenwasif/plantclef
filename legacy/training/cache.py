import os
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import config as _cfg
EXTRACT_CHUNK_SIZE  = getattr(_cfg, 'EXTRACT_CHUNK_SIZE',  64)
USE_REGION_FEATURES = getattr(_cfg, 'USE_REGION_FEATURES', False)


def chunked_backbone_forward(backbone, x, chunk_size):
    """
    Run a backbone on x in sub-batches of chunk_size to cap peak VRAM.
    Only chunk_size images worth of activations exist at any moment.
    """
    return torch.cat([backbone(chunk) for chunk in x.split(chunk_size)])


def _make_center_crop(images):
    """
    Produce a 75% center crop of each image and resize back to original resolution.
    All ops on GPU -- adds negligible overhead to extraction.
    images: (B, C, H, W) BF16 tensor on CUDA.
    """
    h, w      = images.shape[2], images.shape[3]
    crop_h    = int(h * 0.75)
    crop_w    = int(w * 0.75)
    top       = (h - crop_h) // 2
    left      = (w - crop_w) // 2
    cropped   = images[:, :, top:top + crop_h, left:left + crop_w]
    # Resize back to original resolution so backbone receives same-shape input
    resized   = F.interpolate(cropped.float(), size=(h, w),
                              mode='bilinear', align_corners=False)
    return resized.to(images.dtype)


def extract_and_cache_features(model, loader, device, cache_path, shard_size=50):
    """
    One-time frozen-backbone feature extraction.

    When USE_REGION_FEATURES=True, extracts BOTH full image and center crop
    features for each backbone.  The resulting raw feature concatenation is:
        [bio_full(512) + bio_crop(512) + dino_full(1024) + dino_crop(1024)
         + conv_full(1536) + conv_crop(1536)] = 6144-d per sample.
    PCA (fitted in pca_utils.py) then compresses this to PCA_COMPONENTS-d.

    Async shard saving keeps the GPU busy -- disk writes run in a background
    thread while the GPU processes the next batch.
    """
    mode = "full+crop" if USE_REGION_FEATURES else "full only"
    print(f"[Feature Cache] Extracting features ({mode}, async shards)...")
    model.eval()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    shard_dir  = cache_path + "_shards"
    ckpt_file  = cache_path + "_progress.json"
    os.makedirs(shard_dir, exist_ok=True)

    # Resume from checkpoint if one exists
    import json
    resume_from = 0
    shard_idx   = 0
    shard_paths = []

    if os.path.exists(ckpt_file):
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        resume_from = ckpt["batches_done"]
        shard_idx   = ckpt["shards_done"]
        # Reconstruct shard_paths for already-completed shards
        shard_paths = [os.path.join(shard_dir, f"shard_{i:04d}.pt")
                       for i in range(shard_idx)]
        print(f"[Feature Cache] Resuming from batch {resume_from} "
              f"({shard_idx} shards already saved)...")

    # Accumulation buffers -- flushed every shard_size batches
    buf = {k: [] for k in
           (['bio', 'bio_crop', 'dino', 'dino_crop', 'conv', 'conv_crop', 'labels']
            if USE_REGION_FEATURES else
            ['bio', 'dino', 'conv', 'labels'])}

    executor       = ThreadPoolExecutor(max_workers=1)
    pending_future = None

    def _write(data, path, progress):
        torch.save(data, path)
        # Write progress checkpoint atomically after shard is saved
        with open(ckpt_file + ".tmp", "w") as f:
            json.dump(progress, f)
        os.replace(ckpt_file + ".tmp", ckpt_file)

    def flush_async():
        nonlocal shard_idx, pending_future
        path       = os.path.join(shard_dir, f"shard_{shard_idx:04d}.pt")
        shard_data = {k: torch.cat(v) for k, v in buf.items()}
        # Correctly calculate batches_done: shard_idx starts from 0.
        # After finishing shard_idx, we have completed (shard_idx + 1) shards.
        progress   = {"batches_done": (shard_idx + 1) * shard_size,
                      "shards_done":  shard_idx + 1}
        shard_paths.append(path)
        for v in buf.values():
            v.clear()
        shard_idx += 1
        if pending_future is not None:
            pending_future.result()
        pending_future = executor.submit(_write, shard_data, path, progress)

    total_batches = len(loader)  # DALI iterator exposes __len__
    print(f"[Feature Cache] Total batches to process: {total_batches}")

    # Create dedicated streams for parallel backbone execution
    stream_bio  = torch.cuda.Stream()
    stream_dino = torch.cuda.Stream()
    stream_conv = torch.cuda.Stream()

    with torch.no_grad():
        pbar = tqdm(total=total_batches, desc="Extracting features",
                    initial=resume_from, unit="batch")
        for idx_in_this_run, data in enumerate(loader):
            # idx is the absolute batch index across all runs
            idx = idx_in_this_run + resume_from
            
            # Stop if we reached total_batches (safety for infinite iterators)
            if total_batches and idx >= total_batches:
                break

            # Skip batches already processed in a previous run
            # In DALI, loader starts from 0 each time it's created,
            # so the first batch from loader is absolute batch 0.
            if idx_in_this_run < resume_from:
                pbar.update(1)
                continue
            images = data[0]['data'].to(memory_format=torch.channels_last)
            labels = data[0]['label'].squeeze().long()

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Run backbones in parallel streams
                with torch.cuda.stream(stream_bio):
                    feat_bio  = chunked_backbone_forward(model.bioclip,  images, EXTRACT_CHUNK_SIZE)
                with torch.cuda.stream(stream_dino):
                    feat_dino = chunked_backbone_forward(model.dinov2,   images, EXTRACT_CHUNK_SIZE)
                with torch.cuda.stream(stream_conv):
                    feat_conv = chunked_backbone_forward(model.convnext, images, EXTRACT_CHUNK_SIZE)

                if USE_REGION_FEATURES:
                    crops = _make_center_crop(images)
                    # Reuse streams for crop features
                    with torch.cuda.stream(stream_bio):
                        feat_bio_c  = chunked_backbone_forward(model.bioclip,  crops, EXTRACT_CHUNK_SIZE)
                    with torch.cuda.stream(stream_dino):
                        feat_dino_c = chunked_backbone_forward(model.dinov2,   crops, EXTRACT_CHUNK_SIZE)
                    with torch.cuda.stream(stream_conv):
                        feat_conv_c = chunked_backbone_forward(model.convnext, crops, EXTRACT_CHUNK_SIZE)

            # Sync all streams before moving to CPU
            torch.cuda.synchronize()

            # Store as bfloat16 -- halves CPU RAM vs float32
            buf['bio'].append(feat_bio.cpu().to(torch.bfloat16))
            buf['dino'].append(feat_dino.cpu().to(torch.bfloat16))
            buf['conv'].append(feat_conv.cpu().to(torch.bfloat16))
            buf['labels'].append(labels.cpu())

            if USE_REGION_FEATURES:
                buf['bio_crop'].append(feat_bio_c.cpu())
                buf['dino_crop'].append(feat_dino_c.cpu())
                buf['conv_crop'].append(feat_conv_c.cpu())

            pbar.update(1)

            if (idx + 1) % shard_size == 0:
                flush_async()

        pbar.close()

    if any(buf.values()):
        flush_async()

    if pending_future is not None:
        pending_future.result()
    executor.shutdown(wait=False)
    loader.reset()

    # Merge shards
    print(f"[Feature Cache] Merging {shard_idx} shards...")
    merged = {k: [] for k in buf}
    for path in shard_paths:
        shard = torch.load(path, weights_only=False)
        for k in merged:
            merged[k].append(shard[k])
        del shard

    # Convert feature tensors back to float32 for PCA + training compatibility
    # Labels stay as-is (int32)
    cache = {}
    for k, v in merged.items():
        t = torch.cat(v)
        cache[k] = t.float() if k != 'labels' else t
    torch.save(cache, cache_path)
    n = len(cache['labels'])
    print(f"[Feature Cache] Saved {n:,} samples to {cache_path}")

    import shutil
    shutil.rmtree(shard_dir)
    # Remove progress checkpoint -- extraction is complete
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)
    return cache


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CachedFeatureDataset(torch.utils.data.Dataset):
    """
    Returns raw backbone features (bio, dino, conv) for Phase 1 proj-head training.
    Used when PCA is disabled or for Phase 2 head transfer.
    """
    def __init__(self, cache):
        self.bio    = cache['bio']
        self.dino   = cache['dino']
        self.conv   = cache['conv']
        self.labels = cache['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.bio[idx], self.dino[idx], self.conv[idx], self.labels[idx]


class CachedPCADataset(torch.utils.data.Dataset):
    """
    Returns PCA-compressed features for fast Phase 1 linear-probe training.
    Requires cache['features_pca'] to exist (populated by pca_utils.apply_pca).
    """
    def __init__(self, cache):
        self.features = cache['features_pca']
        self.labels   = cache['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
