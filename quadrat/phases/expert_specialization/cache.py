"""Teacher logit cache builder — owned exclusively by expert_specialization.

Runs the SWA teacher in eval mode over the full training CSV and writes
teacher_logit_cache.npy (float16 memmap, shape [N, num_classes]).
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.distributed as dist

from .config import P2bTeacherConfig
from .model  import TeacherEnsemble


def build_cache(cfg: P2bTeacherConfig, output_npy: str, batch_size: int = 512) -> None:
    """Entry point: run after SWA. Writes output_npy on rank 0."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = cfg.local_rank
    is_master  = (cfg.rank == 0)
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    world_size = cfg.world_size

    # ── Load SWA teacher ─────────────────────────────────────────────────────
    swa_path = cfg.swa_ckpt
    if not os.path.exists(swa_path):
        raise FileNotFoundError(
            f"[Cache] SWA checkpoint not found: {swa_path}\n"
            "Run -m phases.expert_specialization.run swa first."
        )

    model = TeacherEnsemble(
        num_classes   = cfg.num_classes,
        input_res     = cfg.resolution,
        bioclip_name  = cfg.bioclip,
        dinov3_name   = cfg.dinov3,
        convnext_name = cfg.convnext,
    ).to(device)
    model.ensure_backbones_loaded()
    sd       = torch.load(swa_path, map_location=device, weights_only=False)
    model_sd = model.state_dict()
    compatible = {k: v for k, v in sd.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(compatible, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if is_master:
        print(f"[Cache] Teacher loaded from {swa_path}")

    # ── DALI loader ──────────────────────────────────────────────────────────
    from src.data.dataloader import get_dali_loaders
    csv = cfg.cleaned_csv if os.path.exists(cfg.cleaned_csv) else cfg.csv_path
    loader, _, _, _ = get_dali_loaders(
        batch_size   = batch_size,
        resolution   = cfg.resolution,
        device_id    = local_rank,
        num_shards   = world_size,
        shard_id     = cfg.rank,
        csv_path     = csv,
        img_dir      = cfg.img_dir,
        num_threads  = cfg.num_threads,
        with_sample_idx = False,
    )

    import pandas as pd
    df = pd.read_csv(csv, sep=";", low_memory=False)
    N  = len(df)

    # Allocate per-rank output buffer, then gather on master
    rank_logits = []
    rank_indices: list[int] = []

    if is_master:
        print(f"[Cache] Processing {N} rows (sharded across {world_size} GPUs)...")

    with torch.inference_mode():
        for batch_idx, data in enumerate(loader):
            imgs = data[0]["data"].to(device, dtype=torch.bfloat16,
                                      memory_format=torch.channels_last)
            row_labels = data[0]["label"].squeeze().long()  # row indices
            logits = model(imgs.float())                     # [B, C]
            rank_logits.append(logits.cpu().half())
            rank_indices.extend(row_labels.cpu().tolist())
            if is_master and (batch_idx + 1) % 100 == 0:
                print(f"  [Cache] Batch {batch_idx+1}")

    dist.barrier()

    # ── Gather all partial results onto master and write ──────────────────────
    # Serialise per-rank (index, logits) pairs via object_list gather
    local_data = list(zip(rank_indices, rank_logits))

    if world_size > 1:
        all_data_list = [None] * world_size
        dist.all_gather_object(all_data_list, local_data)
    else:
        all_data_list = [local_data]

    if is_master:
        cache = np.zeros((N, cfg.num_classes), dtype=np.float16)
        for rank_data in all_data_list:
            offset = 0
            for batch_logits in rank_data:
                if isinstance(batch_logits, tuple):
                    idxs, logits_t = batch_logits
                else:
                    # flat list of (idx, logit_tensor) pairs
                    for idx, lt in rank_data:
                        cache[idx] = lt.numpy().astype(np.float16)
                    break

        os.makedirs(os.path.dirname(output_npy), exist_ok=True)
        np.save(output_npy, cache)
        print(f"[Cache] Written → {output_npy}  shape={cache.shape}")
