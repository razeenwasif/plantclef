"""Phase 2b Teacher expert training entry point.

Trains the heavy-LoRA teacher at 512px with a 224→448→512 resolution
curriculum.  Save epoch checkpoints consumed by swa.py.

Usage:
    torchrun --nproc_per_node=3 -m phases.expert_specialization.run \
        --config configs/expert_specialization_bioclip.yaml
    torchrun --nproc_per_node=3 -m phases.expert_specialization.run \
        --config configs/expert_specialization_dinov3.yaml
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import io
import os
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import deepspeed

project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from .config import load, P2bTeacherConfig
from .model  import TeacherEnsemble
from .loss   import build_criterion

from src.training.checkpoints import (
    robust_load_state_dict, atomic_torch_save,
    prune_deepspeed_checkpoints, robust_deepspeed_resume,
    peek_checkpoint_metadata,
)
from src.data.dataloader import get_dali_loaders


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _ds_config(batch_size: int, accum_steps: int, world_size: int) -> dict:
    cfg: dict = {
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": True},
        "torch_autocast": {"enabled": True, "dtype": "bfloat16"},
        "gradient_clipping": 1.0,
        "gradient_accumulation_steps": accum_steps,
        "train_micro_batch_size_per_gpu": batch_size,
        "steps_per_print": 20,
    }
    if world_size > 1:
        cfg["zero_optimization"]["zero_quantized_gradients"] = True
    return cfg


def _curriculum_resolution(epoch: int, cfg: P2bTeacherConfig) -> int:
    if hasattr(cfg, "force_res") and getattr(cfg, "force_res") is not None:
        return cfg.force_res
    if epoch < cfg.warmup_epochs:
        return 224
    elif epoch < cfg.warmup_epochs + cfg.middle_epochs:
        return 448
    return cfg.resolution   # e.g. 512


def _validate(model_engine: "deepspeed.DeepSpeedEngine",
              val_loader, criterion, cfg: P2bTeacherConfig,
              device: torch.device, id_remap: torch.Tensor) -> tuple[float, float]:
    """Quick validation pass (capped at max_val_batches)."""
    model_engine.eval()
    correct = total = n_batches = 0
    loss_sum = 0.0
    is_master = (cfg.rank == 0)
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= cfg.max_val_batches:
                break
            # plantclef: Explicit device and dtype placement
            imgs    = data[0]["data"].to(device, dtype=torch.bfloat16, memory_format=torch.channels_last)
            raw_lbl = data[0]["label"].squeeze().long().to(device)
            # Remap shard species IDs → class indices
            labels = id_remap[raw_lbl.clamp(0, id_remap.shape[0] - 1)]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model_engine(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss_sum += criterion(outputs.float(), labels).item()
            
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            n_batches += 1
            
            # plantclef: One-time debug log to verify label alignment
            if i == 0 and is_master:
                lbl_min, lbl_max = labels.min().item(), labels.max().item()
                prd_min, prd_max = preds.min().item(), preds.max().item()
                print(f"\n[Debug:Val] Batch 0 Stats:")
                print(f"  -> Raw IDs:      {raw_lbl[:5].tolist()}")
                print(f"  -> Remapped:     {labels[:5].tolist()} (Range: {lbl_min}-{lbl_max})")
                print(f"  -> Predictions:  {preds[:5].tolist()} (Range: {prd_min}-{prd_max})")
                print(f"  -> Logit Stats:  mean={outputs.mean():.4f}, std={outputs.std():.4f}")
                sys.stdout.flush()

    model_engine.train()
    acc  = correct / total * 100 if total else 0.0
    loss = loss_sum / n_batches if n_batches else 0.0
    return acc, loss


def _train(cfg: P2bTeacherConfig, seed: int = 42) -> None:
    _set_seed(seed)

    local_rank = cfg.local_rank
    is_master  = (cfg.rank == 0)
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60), device_id=device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    swa_path = os.path.join(cfg.output_dir, "swa_model_final.pth")
    if is_master:
        if os.path.exists(swa_path):
            print(f"[Teacher:{cfg.expert_name}] WARNING: {swa_path} already exists. "
                  "Training will resume/continue into this directory. "
                  "Run swa.py with --force to overwrite the averaged model.")
        print(f"\n[Teacher:{cfg.expert_name}] Starting training  "
              f"LoRA r={cfg.lora_r}  res={cfg.resolution}px  "
              f"epochs={cfg.total_epochs}")

    # Peek the latest checkpoint (if any) so the freshly-built model architecture
    # matches what's on disk. Specifically, gating_network output dim depends on
    # whether FP8 was active at save time; mismatching it makes resume impossible.
    saved_meta = peek_checkpoint_metadata(cfg.output_dir, patch_size=14)
    if saved_meta and saved_meta.get("gating_dim") == 3:
        # Saved with non-FP8 path; force same architecture on rebuild.
        import src.config as _global_cfg
        if getattr(_global_cfg, "USE_FP8", False):
            _global_cfg.USE_FP8 = False
            if is_master:
                print(f"[Teacher] Saved checkpoint has gating_dim=3 → forcing non-FP8 path "
                      f"(USE_FP8 overridden to False) so architecture matches.")

    # Model — build at the resolution that the saved checkpoint was at, if known.
    # This avoids pos_embed shape mismatch during load. Falls back to 224 for fresh runs.
    initial_res = saved_meta.get("saved_res", 224)
    if is_master and saved_meta:
        print(f"[Teacher] Resume metadata: tag={saved_meta.get('epoch_tag')} "
              f"saved_res={saved_meta.get('saved_res')}px gating_dim={saved_meta.get('gating_dim')}.")

    model = TeacherEnsemble(
        num_classes   = cfg.num_classes,
        input_res     = initial_res,
        bioclip_name  = cfg.bioclip,
        dinov3_name   = cfg.dinov3,
        convnext_name = cfg.convnext,
    ).to(device)
    model.apply_lora(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
    model.freeze_backbones()
    model.freeze_stem(6)

    # plantclef: Compile FIRST to allow standard DeepSpeed/Dynamo harmony
    if cfg.use_compile and hasattr(torch, "compile"):
        import torch._inductor.config as inductor_cfg
        inductor_cfg.triton.cudagraphs = False
        inductor_cfg.fx_graph_cache    = True
        model = torch.compile(model, mode="default")
        if is_master:
            print("[Optim] Torch Dynamo compilation engaged.")

    from lion_pytorch import Lion
    optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer,
        config=_ds_config(cfg.batch_size, cfg.accumulation_steps, cfg.world_size),
        dist_init_required=False,
    )

    # Checkpoint directory
    out_dir = Path(cfg.output_dir)
    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Resume — robust: validates latest DS dir, falls back to older tags on corruption.
    start_epoch, start_step = robust_deepspeed_resume(model_engine, str(out_dir),
                                                      name=f"Teacher:{cfg.expert_name}")

    if start_epoch == 0 and start_step == 0:
        baseline = cfg.warmup_baseline
        if os.path.exists(baseline):
            ckpt = torch.load(baseline, map_location=device, weights_only=False)
            robust_load_state_dict(
                model_engine.module, ckpt.get("model_state") or ckpt, strict=False
            )
            if is_master:
                print(f"[Teacher] Loaded warmup baseline: {baseline}")

    model_engine.module.to(dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    gc.collect()

    # Criterion — plain hard-label ASL for teacher (no KD)
    criterion = build_criterion(cfg.num_classes, class_counts=None, device=device)

    # Build species-ID → class-index lookup tensor for shard label remapping.
    # Shard .cls files store raw species IDs (e.g. 1355868), not class indices.
    import pandas as pd
    _mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    _df_map  = pd.read_csv(_mapping_path, header=None, dtype={0: int})
    _id_remap = torch.zeros(int(_df_map[0].max()) + 1, dtype=torch.long, device=device)
    # Vectorized remap population
    _id_remap[_df_map[0].values] = torch.arange(len(_df_map), device=device)
    if is_master:
        print(f"[Teacher] Species-ID remap built: {len(_df_map)} classes.")

    current_res  = -1
    train_loader = val_loader = None
    
    # plantclef: Initial setup
    res = _curriculum_resolution(start_epoch, cfg)
    
    def get_dynamic_batch_config(res, base_batch, base_accum):
        """Scale batch size based on resolution while keeping global batch same."""
        # 224px uses ~1/10 the memory of 512px; we can safely triple the batch.
        multiplier = 1
        if res <= 224:
            multiplier = 3
        elif res <= 448:
            multiplier = 1
        
        new_batch = base_batch * multiplier
        new_accum = max(1, base_accum // multiplier)
        return new_batch, new_accum

    # plantclef: Calculate max epoch. If extra_epochs was provided via CLI, 
    # run for that many epochs BEYOND the start_epoch.
    max_epoch = cfg.total_epochs
    if hasattr(cfg, "extra_epochs") and cfg.extra_epochs is not None:
        max_epoch = start_epoch + cfg.extra_epochs
        if is_master:
            print(f"[Teacher] Running for {cfg.extra_epochs} additional epoch(s) (Target: ep{max_epoch-1})")

    for epoch in range(start_epoch, max_epoch):
        res = _curriculum_resolution(epoch, cfg)
        new_batch, new_accum = get_dynamic_batch_config(res, cfg.batch_size, cfg.accumulation_steps)

        if res != current_res:
            if is_master:
                print(f"\n[Teacher] Epoch {epoch}: switching to {res}px (Batch: {new_batch}, Accum: {new_accum})...")
            
            # 1. Cleanup old state
            if train_loader is not None:
                del train_loader, val_loader
                torch.cuda.empty_cache()
            
            # 2. Re-init DALI
            csv = cfg.cleaned_csv if os.path.exists(cfg.cleaned_csv) else cfg.csv_path
            train_loader, val_loader, _, _ = get_dali_loaders(
                batch_size   = cfg.batch_size,
                resolution   = res,
                device_id    = local_rank,
                num_shards   = cfg.world_size,
                shard_id     = cfg.rank,
                csv_path     = csv,
                img_dir      = cfg.img_dir,
                num_threads  = cfg.num_threads,
                seed         = seed,
                low_ram_mode = cfg.low_ram_mode,
                window_size  = cfg.shard_window_size,
                dataset      = getattr(args, "dataset", None),
            )

            
            # 3. Re-init DeepSpeed to apply new batch/accum settings
            # We must preserve the optimizer state if resuming mid-run, 
            # but DeepSpeed handles this via save/load.
            ds_config = _ds_config(new_batch, new_accum, cfg.world_size)
            
            # plantclef: Synchronize compute engine with dynamic loader
            # DeepSpeed engine uses a method for GA; update carefully to avoid TypeError
            model_engine.micro_batch_size = new_batch
            model_engine.gradient_accumulation_steps = lambda: new_accum
            
            # DeepSpeed's .config attribute is typically a dict
            if hasattr(model_engine, 'config') and isinstance(model_engine.config, dict):
                model_engine.config['gradient_accumulation_steps'] = new_accum
                model_engine.config['train_micro_batch_size_per_gpu'] = new_batch

            if is_master:
                print(f"[Teacher] Engine updated: MicroBatch={new_batch}, Accum={new_accum}")

            current_res = res

        model_engine.module.set_resolution(res)
        model_engine.train()
        total_steps = len(train_loader)

        # plantclef: Mid-epoch resume skip
        current_step = 0
        if epoch == start_epoch and start_step > 0:
            if is_master:
                print(f"[Teacher] Fast-forwarding {start_step} steps...")
            for _ in range(start_step):
                next(train_loader)
            current_step = start_step

        from tqdm import tqdm
        pbar = tqdm(train_loader,
                    desc=f"[Teacher:{cfg.expert_name}] E{epoch} R{cfg.rank}",
                    position=cfg.rank, leave=True, total=total_steps, initial=current_step)
        
        for i, data in enumerate(pbar):
            step_idx = i + current_step
            imgs    = data[0]["data"].to(device, dtype=torch.bfloat16,
                                         memory_format=torch.channels_last)
            raw_lbl = data[0]["label"].squeeze().long().to(device)
            # Remap shard species IDs → class indices
            labels  = _id_remap[raw_lbl.clamp(0, _id_remap.shape[0] - 1)]

            outputs = model_engine(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs.float(), labels)
            model_engine.backward(loss)
            model_engine.step()

            if model_engine.is_gradient_accumulation_boundary():
                model_engine.module.update_lora_fusion()
                
                # plantclef: Periodic progress checkpoint (every 1000 GA steps)
                # Since GA is handled inside model_engine.step, we count effective steps.
                if (model_engine.global_steps + 1) % 1000 == 0:
                    if is_master:
                        print(f"\n[Teacher] Step {step_idx}: periodic checkpointing...")
                    model_engine.save_checkpoint(
                        str(out_dir), tag=f"ep{epoch}",
                        client_state={"epoch": epoch, "step": step_idx + 1},
                    )
                    if is_master:
                        prune_deepspeed_checkpoints(str(out_dir), keep=2)

            pbar.set_postfix(loss=f"{loss.item():.4f}", res=f"{res}px")

        if dist.is_initialized():
            dist.barrier()

        try:
            train_loader.reset()
        except Exception:
            pass

        # Validation (skipped when no val shards exist)
        if val_loader is not None and is_master and (epoch + 1) % cfg.val_every_n_epochs == 0:
            val_acc, val_loss = _validate(model_engine, val_loader, criterion, cfg, device, _id_remap)
            print(f"[Teacher] Epoch {epoch}  val_acc={val_acc:.2f}%  val_loss={val_loss:.4f}")
            try:
                val_loader.reset()
            except Exception:
                pass

        # Save epoch checkpoint (consumed by swa.py) — atomic write.
        if is_master:
            ckpt_path = out_dir / f"epoch_ep{epoch}_final.pth"
            atomic_torch_save(model_engine.module.state_dict(), ckpt_path)
            print(f"[Teacher] Epoch {epoch} checkpoint → {ckpt_path}")

        # DeepSpeed checkpoint for resumability (collective: all ranks).
        model_engine.save_checkpoint(
            str(out_dir), tag=f"ep{epoch}",
            client_state={"epoch": epoch + 1},
        )

        # Prune old DS optimizer dirs (keep latest 2). Rank-0 only — all other
        # ranks have already returned from save_checkpoint.
        if is_master:
            n_pruned = prune_deepspeed_checkpoints(str(out_dir), keep=2)
            if n_pruned:
                print(f"[Teacher] Pruned {n_pruned} stale DeepSpeed checkpoint dir(s).")
        if dist.is_initialized():
            dist.barrier()

    if is_master:
        print(f"\n[Teacher] Training complete ({cfg.expert_name}).")
        print(f"[Teacher] Run  phases/expert_specialization/swa.py  to produce {cfg.swa_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--dataset",    type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--res",        type=int, default=None, help="Override training resolution")
    parser.add_argument("--epochs",     type=int, default=None, help="Override total epochs")
    args, _ = parser.parse_known_args()

    cfg = load(args.config)
    if args.res is not None:
        cfg.force_res = args.res
    if args.epochs is not None:
        # plantclef: Treat CLI --epochs as "additional epochs to run"
        cfg.extra_epochs = args.epochs

    _train(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
