import torch
import torch.nn.functional as F
from torch.amp import autocast
import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
from src import config as scfg

# Research flags
USE_REGION_FEATURES = getattr(scfg, 'USE_REGION_FEATURES', False)

def get_cache_resume_point(cache_path: str, rank: int = 0) -> Tuple[int, int]:
    """Detects how many batches have already been saved for a specific rank."""
    shard_dir = cache_path + "_shards"
    ckpt_file = os.path.join(shard_dir, f"progress_rank{rank}.json")
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            return ckpt["batches_done"], ckpt["shards_done"]
        except: pass
    return 0, 0

@torch.compiler.disable()
@torch.no_grad()
def chunked_backbone_forward(model, images, chunk_size=8):
    """Memory-efficient forward pass for large batches."""
    outputs = []
    # plantclef: Ensure features stay on GPU during training to prevent TransformerEngine crash
    keep_on_device = images.is_cuda
    
    for i in range(0, images.size(0), chunk_size):
        chunk = images[i:i+chunk_size]
        out = model(chunk)
        if isinstance(out, tuple): out = out[0]
        
        if keep_on_device:
            outputs.append(out)
        else:
            outputs.append(out.cpu())
            
    return torch.cat(outputs)

def extract_and_cache_features(model, loader, device, cache_path):
    """PLANTCLEF: High-speed multi-GPU feature extraction with absolute persistence."""
    import torch.distributed as dist
    from src import config as _cfg_local
    
    # 1. Identify Rank
    is_dist = dist.is_initialized()
    rank = _cfg_local.RANK
    is_master = (rank == 0)
    
    mode = "full+crop" if USE_REGION_FEATURES else "full only"
    print(f"[Feature Cache][Rank {rank}] Absolute Extraction Mode: {mode}")

    model.eval()
    if hasattr(model, 'ensure_backbones_loaded'):
        model.ensure_backbones_loaded()

    # 2. Shard Configuration
    shard_dir = cache_path + "_shards"
    os.makedirs(shard_dir, exist_ok=True)
    ckpt_file = os.path.join(shard_dir, f"progress_rank{rank}.json")

    # 3. Resume logic
    resume_from, shard_idx = get_cache_resume_point(cache_path, rank)
    total_batches = len(loader)
    
    # 4. Accumulation Buffers
    buf = {k: [] for k in (['bio', 'dino', 'conv', 'labels'])}
    executor = ThreadPoolExecutor(max_workers=2)
    pending_future = None

    def _write(data: Dict[str, torch.Tensor], path: str, progress: Dict[str, int]) -> None:
        """Atomic write without deletion."""
        start = time.time()
        torch.save(data, path)
        with open(ckpt_file + ".tmp", "w") as f:
            json.dump(progress, f)
        os.replace(ckpt_file + ".tmp", ckpt_file)
        print(f"[Feature Cache][Rank {rank}] Shard {progress['shards_done']} saved in {time.time()-start:.2f}s")

    # 5. Extraction Loop
    # plantclef: Use config-defined extraction chunk size
    extract_chunk_size = getattr(_cfg_local, 'EXTRACT_CHUNK_SIZE', 8)
    
    with torch.no_grad():
        pbar = tqdm(loader, initial=resume_from, desc=f"[Rank {rank}] Extracting", 
                    unit="batch", position=rank, leave=True)

        for rel_idx, data in enumerate(pbar):
            abs_idx = rel_idx + resume_from
            if abs_idx >= total_batches: break

            # Micro-scavenge to prevent fragmentation
            if abs_idx % 10 == 0:
                from src.training.accelerator import accelerator
                accelerator().empty_cache()

            images = data[0]['data'].to(device, memory_format=torch.channels_last)
            labels = data[0]['label'].squeeze().long()

            from src.training.accelerator import accelerator as _accel
            with _accel().autocast(dtype=torch.bfloat16):
                f_bio  = chunked_backbone_forward(model.bioclip,  images, extract_chunk_size)
                f_dino = chunked_backbone_forward(model.dinov3,   images, extract_chunk_size)
                f_conv = chunked_backbone_forward(model.convnext, images, extract_chunk_size)

            buf['bio'].append(f_bio)
            buf['dino'].append(f_dino)
            buf['conv'].append(f_conv)
            buf['labels'].append(labels.cpu())

            # Save Shard every 100 batches
            if (abs_idx + 1) % 100 == 0:
                if pending_future is not None: pending_future.result()
                
                shard_data = {k: torch.cat(v) for k, v in buf.items()}
                path = os.path.join(shard_dir, f"shard_rank{rank}_{shard_idx:04d}.pt")
                progress = {"batches_done": abs_idx + 1, "shards_done": shard_idx + 1}
                
                pending_future = executor.submit(_write, shard_data, path, progress)
                
                # Clear buffers
                for v in buf.values(): v.clear()
                shard_idx += 1

        # Final flush
        if len(buf['labels']) > 0:
            if pending_future is not None: pending_future.result()
            shard_data = {k: torch.cat(v) for k, v in buf.items()}
            path = os.path.join(shard_dir, f"shard_rank{rank}_{shard_idx:04d}.pt")
            _write(shard_data, path, {"batches_done": total_batches, "shards_done": shard_idx + 1})

    # plantclef: NO AUTOMATIC DELETION. ALL DATA IS PERMANENT.
    print(f"[Rank {rank}] Extraction Complete. ALL DATA PERSISTENT ON DISK.")
    return None # Merger will be done manually or via launcher

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class CachedFeatureDataset(torch.utils.data.Dataset):
    """
    High-performance dataset for training on cached backbone features.
    """
    def __init__(self, cache: Dict[str, torch.Tensor]):
        # plantclef: Logic to handle PCA vs Raw Concatenated features
        if 'features_pca' in cache:
            self.features = cache['features_pca']
        elif all(k in cache for k in ['bio', 'dino', 'conv']):
            # Concatenate experts: [Bio_Full, Bio_Crop, Dino_Full, Dino_Crop, Conv_Full, Conv_Crop]
            # This matches the 6,656-d dimension defined in PlantEnsemble.
            self.features = torch.cat([cache['bio'], cache['dino'], cache['conv']], dim=1)
        else:
            self.features = cache.get('bio') # Legacy fallback
            
        self.labels = cache['labels']
        
        if self.features is None:
            raise KeyError("Cache must contain 'features_pca' or full expert keys (bio, dino, conv).")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
