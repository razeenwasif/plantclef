"""Phase 1 — Feature extraction entry point.

Runs the three frozen backbones over the full training set and writes a
.pt cache consumed by Phase 2a.

Usage:
    torchrun --nproc_per_node=N -m phases.foundation_caching.run \
        --config configs/p1_extract.yaml
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from .config import load, P1Config
from .model  import FeatureExtractor


def _extract(cfg: P1Config, force: bool = False) -> None:
    # Guard: refuse to overwrite an existing feature cache unless --force is passed.
    # Extracting features takes hours; accidental overwrites are catastrophic.
    if os.path.exists(cfg.cache_path) and not force:
        print(f"[P1] ABORT: cache already exists at {cfg.cache_path}")
        print("[P1] Pass --force to re-extract and overwrite.")
        return

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = cfg.local_rank
    is_master  = (cfg.rank == 0)
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.backends.cudnn.benchmark     = True
    torch.backends.cuda.matmul.allow_tf32 = True

    extractor = FeatureExtractor(
        bioclip_name  = cfg.bioclip,
        dinov3_name   = cfg.dinov3,
        convnext_name = cfg.convnext,
        input_res     = cfg.resolution,
    ).to(device)
    extractor.load_backbones()

    if cfg.use_compile and hasattr(torch, "compile"):
        extractor = torch.compile(extractor, mode="reduce-overhead")

    from src.data.dataloader import get_dali_loaders
    train_loader, _, N, _ = get_dali_loaders(
        batch_size  = cfg.batch_size,
        resolution  = cfg.resolution,
        device_id   = local_rank,
        num_shards  = cfg.world_size,
        shard_id    = cfg.rank,
        csv_path    = cfg.csv_path,
        img_dir     = cfg.img_dir,
        num_threads = cfg.num_threads,
    )

    if is_master:
        print(f"[P1] Rank {cfg.rank}/{cfg.world_size}: "
              f"extracting features (bs={cfg.batch_size}, res={cfg.resolution})...")

    features_list: list[torch.Tensor] = []
    labels_list:   list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            imgs   = data[0]["data"].to(device, dtype=torch.bfloat16,
                                        memory_format=torch.channels_last)
            labels = data[0]["label"].squeeze().long().cpu()

            feats = extractor(imgs.float())   # [B, 3328], float32 L2-normalised
            features_list.append(feats.cpu().half())
            labels_list.append(labels)

            if (batch_idx + 1) % 50 == 0 and is_master:
                print(f"  [P1] Batch {batch_idx + 1}")

            if (batch_idx + 1) % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    shard_dir  = os.path.join(cfg.output_dir, "p1_shards")
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f"shard_{cfg.rank}.pt")
    from src.training.checkpoints import atomic_torch_save
    atomic_torch_save({
        "features": torch.cat(features_list),
        "labels":   torch.cat(labels_list),
    }, shard_path)
    if is_master:
        print(f"[P1] Rank {cfg.rank}: shard saved → {shard_path}")

    dist.barrier()

    # Master merges shards into a single cache file
    if is_master:
        print(f"[P1] Merging {cfg.world_size} shards...")
        all_feats:  list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for r in range(cfg.world_size):
            sd = torch.load(
                os.path.join(shard_dir, f"shard_{r}.pt"),
                weights_only=False,
            )
            all_feats.append(sd["features"])
            all_labels.append(sd["labels"])

        os.makedirs(cfg.output_dir, exist_ok=True)
        cache_path = cfg.cache_path
        atomic_torch_save({
            "features": torch.cat(all_feats),
            "labels":   torch.cat(all_labels),
        }, cache_path)
        total = torch.cat(all_feats).shape[0]
        print(f"[P1] Cache written → {cache_path}  ({total} samples)")
        # Per-rank shards no longer needed once merged.
        import shutil
        shutil.rmtree(shard_dir, ignore_errors=True)

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--force",      action="store_true",
                        help="Overwrite existing feature cache")
    args, _ = parser.parse_known_args()

    cfg = load(args.config)
    _extract(cfg, force=args.force)


if __name__ == "__main__":
    main()
