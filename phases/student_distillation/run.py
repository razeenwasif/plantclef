"""Phase 2b Student fine-tuning entry point.

Trains the student ensemble with lighter LoRA, curriculum resolution, and
knowledge distillation from the teacher cache (or live teacher models).

Usage:
    torchrun --nproc_per_node=3 -m phases.student_distillation.run \
        --config configs/student_distillation.yaml
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
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import deepspeed

project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from .config import load, P2bStudentConfig
from .model  import StudentEnsemble
from .loss   import build_criterion, distillation_loss

from src.training.checkpoints import (
    robust_load_state_dict, atomic_torch_save,
    prune_deepspeed_checkpoints, robust_deepspeed_resume,
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


def _curriculum_resolution(epoch: int, cfg: P2bStudentConfig) -> int:
    """warmup_res for warmup epochs, then 448 for middle, 512 for high.
    With middle_epochs=high_epochs=0, the model stays at warmup_res throughout."""
    if epoch < cfg.warmup_epochs:
        return cfg.warmup_res
    elif epoch < cfg.warmup_epochs + cfg.middle_epochs:
        return 448
    return 512


def _load_teacher_cache(
    cfg: P2bStudentConfig,
) -> Optional[np.memmap]:
    if not os.path.exists(cfg.teacher_cache):
        return None
    C = cfg.num_classes
    N = os.path.getsize(cfg.teacher_cache) // (C * 2)   # float16 = 2 bytes
    return np.memmap(cfg.teacher_cache, dtype=np.float16, mode="r", shape=(N, C))


def _load_live_teachers(
    cfg: P2bStudentConfig, device: torch.device
) -> list:
    # Build teachers at the resolution they were SWA'd at. With option-A
    # training (warmup-only), that's cfg.warmup_res (224). Using 512 here
    # would force pos_embed surgery 16→36 and feed teachers an interpolated
    # input distribution they never saw → garbage logits.
    teacher_res = cfg.warmup_res
    teachers = []
    for path in cfg.teacher_paths:
        if not os.path.exists(path):
            continue
        teacher = StudentEnsemble(
            num_classes   = cfg.num_classes,
            input_res     = teacher_res,
            bioclip_name  = cfg.bioclip,
            dinov3_name   = cfg.dinov3,
            convnext_name = cfg.convnext,
        ).to(device)
        teacher.ensure_backbones_loaded()
        sd = torch.load(path, map_location=device, weights_only=False)

        # Strip compile/distributed prefixes if present.
        cleaned = {k.replace("_orig_mod.", "").replace("module.", ""): v
                   for k, v in sd.items()}
        # Drop TransformerEngine _extra_state (incompatible with plain nn.Linear).
        cleaned = {k: v for k, v in cleaned.items() if not k.endswith("._extra_state")}

        # Gating-dim surgery: FP8-trained teachers pad gating to 8 outputs for
        # TE matmul alignment, but only the first 3 are semantically meaningful
        # (one weight per backbone). Slice them down for non-FP8 student.
        for key in ("gating_network.3.weight", "gating_network.3.bias"):
            if key in cleaned and cleaned[key].shape[0] == 8:
                cleaned[key] = cleaned[key][:3].contiguous()

        # Final shape filter: drop any remaining mismatches (e.g., pos_embed
        # saved at wrong resolution). The freshly-built teacher's pretrained
        # backbone init is fine for those keys.
        model_sd = teacher.state_dict()
        compat = {k: v for k, v in cleaned.items()
                  if k in model_sd and v.shape == model_sd[k].shape}
        skipped = [k for k in cleaned if k in model_sd
                   and cleaned[k].shape != model_sd[k].shape]
        if skipped:
            print(f"[Student] Live teacher {os.path.basename(os.path.dirname(path))}: "
                  f"skipped {len(skipped)} shape-mismatched keys (e.g. {skipped[0]}).")
        teacher.load_state_dict(compat, strict=False)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teachers.append(teacher)
    return teachers


def _validate(model_engine: "deepspeed.DeepSpeedEngine",
              val_loader, criterion, cfg: P2bStudentConfig,
              device: torch.device, id_remap: torch.Tensor) -> tuple[float, float]:
    model_engine.eval()
    correct = total = n_batches = 0
    loss_sum = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= cfg.max_val_batches:
                break
            imgs    = data[0]["data"].to(device, dtype=torch.bfloat16,
                                          memory_format=torch.channels_last)
            raw_lbl = data[0]["label"].squeeze().long().to(device)
            labels  = id_remap[raw_lbl.clamp(0, id_remap.shape[0] - 1)]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model_engine(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss_sum += criterion(outputs.float(), labels).item()
            correct   += (outputs.argmax(1) == labels).sum().item()
            total     += labels.size(0)
            n_batches += 1
    model_engine.train()
    acc  = correct / total * 100 if total else 0.0
    loss = loss_sum / n_batches if n_batches else 0.0
    return acc, loss


def _train(cfg: P2bStudentConfig, seed: int = 42) -> None:
    _set_seed(seed)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

    local_rank = cfg.local_rank
    is_master  = (cfg.rank == 0)
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    if is_master:
        swa_path = os.path.join(cfg.output_dir, "swa_model_final.pth")
        if os.path.exists(swa_path):
            print(f"[Student] WARNING: {swa_path} already exists. "
                  "Training will resume/continue into this directory. "
                  "Run swa.py with --force to overwrite the averaged model.")
        print(f"\n[Student] Starting training  "
              f"LoRA r={cfg.lora_r}  epochs={cfg.total_epochs}")

    # Model
    model = StudentEnsemble(
        num_classes   = cfg.num_classes,
        input_res     = 224,
        bioclip_name  = cfg.bioclip,
        dinov3_name   = cfg.dinov3,
        convnext_name = cfg.convnext,
    ).to(device)
    model.apply_lora(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
    model.freeze_backbones()
    model.freeze_stem(6)

    if cfg.use_compile and hasattr(torch, "compile"):
        import torch._inductor.config as inductor_cfg
        inductor_cfg.triton.cudagraphs = False
        inductor_cfg.fx_graph_cache    = True
        model = torch.compile(model, mode="default")

    from lion_pytorch import Lion
    optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer,
        config=_ds_config(cfg.batch_size, cfg.accumulation_steps, cfg.world_size),
        dist_init_required=False,
    )

    out_dir = Path(cfg.output_dir)
    if is_master:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = robust_deepspeed_resume(model_engine, str(out_dir), name="Student")

    if start_epoch == 0:
        baseline = cfg.warmup_baseline
        if os.path.exists(baseline):
            ckpt = torch.load(baseline, map_location=device, weights_only=False)
            robust_load_state_dict(
                model_engine.module, ckpt.get("model_state") or ckpt, strict=False
            )
            if is_master:
                print(f"[Student] Loaded warmup baseline: {baseline}")

    model_engine.module.to(dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    gc.collect()

    # Build species-ID → class-index lookup tensor for shard label remapping.
    # Used when teacher cache is absent and shard (webdataset) labels are raw species IDs.
    import pandas as pd
    _mapping_path = "/workspace/plantclef/processed/species_ids_mapping.csv"
    _df_map  = pd.read_csv(_mapping_path, header=None, dtype={0: int})
    _id_remap = torch.zeros(int(_df_map[0].max()) + 1, dtype=torch.long, device=device)
    for _idx, _row in _df_map.iterrows():
        _id_remap[int(_row[0])] = int(_idx)
    if is_master:
        print(f"[Student] Species-ID remap built: {len(_df_map)} classes.")

    # Teacher setup
    teacher_cache = _load_teacher_cache(cfg)
    live_teachers  = [] if teacher_cache is not None else _load_live_teachers(cfg, device)
    use_sample_idx = teacher_cache is not None

    if is_master:
        if teacher_cache is not None:
            print(f"[Student] Teacher cache: {cfg.teacher_cache}  shape={teacher_cache.shape}")
        elif live_teachers:
            print(f"[Student] Live teachers: {len(live_teachers)} model(s)")
        else:
            print("[Student] No teacher — hard-label training only.")

    # Loss
    criterion = build_criterion(cfg.num_classes, class_counts=None, device=device)

    current_res  = -1
    train_loader = val_loader = None
    class_tensor: Optional[torch.Tensor] = None

    for epoch in range(start_epoch, cfg.total_epochs):
        res = _curriculum_resolution(epoch, cfg)

        if res != current_res:
            if is_master:
                print(f"\n[Student] Epoch {epoch}: switching to {res}px...")
            if train_loader is not None:
                del train_loader, val_loader
                torch.cuda.empty_cache()

            csv = cfg.cleaned_csv if os.path.exists(cfg.cleaned_csv) else cfg.csv_path
            train_loader, val_loader, _, ct = get_dali_loaders(
                batch_size      = cfg.batch_size,
                resolution      = res,
                device_id       = local_rank,
                num_shards      = cfg.world_size,
                shard_id        = cfg.rank,
                csv_path        = csv,
                img_dir         = cfg.img_dir,
                num_threads     = cfg.num_threads,
                seed            = seed,
                with_sample_idx = use_sample_idx,
                low_ram_mode    = cfg.low_ram_mode,
                window_size     = cfg.shard_window_size,
                dataset         = getattr(args, "dataset", None),
            )
            if class_tensor is None and ct is not None:
                class_tensor = ct.to(device)
            current_res = res

        model_engine.module.set_resolution(res)
        model_engine.train()
        total_steps = len(train_loader)

        from tqdm import tqdm
        pbar = tqdm(train_loader,
                    desc=f"[Student] E{epoch} R{cfg.rank}",
                    position=cfg.rank, leave=True, total=total_steps)
        for i, data in enumerate(pbar):
            imgs    = data[0]["data"].to(device, dtype=torch.bfloat16,
                                         memory_format=torch.channels_last)
            raw_lbl = data[0]["label"].squeeze().long().to(device)

            if use_sample_idx and class_tensor is not None:
                row_idx = raw_lbl.clamp(0, class_tensor.shape[0] - 1)
                labels  = class_tensor[row_idx]
            else:
                row_idx = None
                # Remap shard species IDs → class indices
                labels  = _id_remap[raw_lbl.clamp(0, _id_remap.shape[0] - 1)]

            # Teacher logits
            if teacher_cache is not None and row_idx is not None:
                cpu_idx  = row_idx.cpu().numpy()
                t_logits = torch.from_numpy(
                    teacher_cache[cpu_idx].astype(np.float32)
                ).to(device, dtype=torch.bfloat16)
            elif live_teachers:
                # Teachers were trained at the same resolution as the student
                # (option A). Feed them the same imgs — no upscale needed.
                with torch.inference_mode(), \
                        torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    t_logits = sum(t(imgs) for t in live_teachers) / len(live_teachers)
            else:
                t_logits = None

            # Forward
            outputs = model_engine(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if t_logits is not None:
                loss = distillation_loss(outputs, t_logits, labels, criterion,
                                         cfg.kd_alpha, cfg.kd_beta, cfg.kd_temp)
            else:
                loss = criterion(outputs.float(), labels)

            model_engine.backward(loss)
            model_engine.step()

            if model_engine.is_gradient_accumulation_boundary():
                model_engine.module.update_lora_fusion()

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
            print(f"[Student] Epoch {epoch}  val_acc={val_acc:.2f}%  val_loss={val_loss:.4f}")
            try:
                val_loader.reset()
            except Exception:
                pass

        # Save epoch checkpoint (consumed by swa.py) — atomic write.
        if is_master:
            ckpt_path = out_dir / f"epoch_ep{epoch}_final.pth"
            atomic_torch_save(model_engine.module.state_dict(), ckpt_path)
            print(f"[Student] Epoch {epoch} checkpoint → {ckpt_path}")

        # DeepSpeed checkpoint for resumability (collective: all ranks).
        model_engine.save_checkpoint(
            str(out_dir), tag=f"ep{epoch}",
            client_state={"epoch": epoch + 1},
        )

        if is_master:
            n_pruned = prune_deepspeed_checkpoints(str(out_dir), keep=2)
            if n_pruned:
                print(f"[Student] Pruned {n_pruned} stale DeepSpeed checkpoint dir(s).")
        if dist.is_initialized():
            dist.barrier()

    if is_master:
        print(f"\n[Student] Training complete.")
        print(f"[Student] Run  -m phases.student_distillation.swa  to produce the final model.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--dataset",    type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume",     type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = load(args.config)
    _train(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
