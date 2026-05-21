"""Student SWA — owned exclusively by student_distillation.

Averages the last N epoch checkpoints saved by run.py and writes
swa_model_final.pth to the student output directory.
"""
from __future__ import annotations
import glob
import os
from collections import OrderedDict

import torch
import torch.distributed as dist

from .config import P2bStudentConfig
from .model  import StudentEnsemble


def run_swa(cfg: P2bStudentConfig, force: bool = False) -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = cfg.local_rank
    is_master  = (cfg.rank == 0)
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Guard: skip if swa_model_final.pth already exists unless --force is passed.
    if is_master and os.path.exists(cfg.swa_ckpt) and not force:
        print(f"[SWA-Student] SKIP: {cfg.swa_ckpt} already exists.")
        print("[SWA-Student] Pass --force to re-average and overwrite.")
        dist.barrier()
        return

    # ── 1. Collect epoch checkpoints ────────────────────────────────────────
    pattern    = os.path.join(cfg.output_dir, "epoch_ep*_final.pth")
    all_paths  = sorted(glob.glob(pattern))
    if not all_paths:
        if is_master:
            print(f"[SWA-Student] No checkpoints found at {pattern}")
        return

    # Honor swa_start_epoch (parity with teacher SWA): skip pre-warmup epochs.
    filtered_paths = []
    for p in all_paths:
        try:
            fname  = os.path.basename(p)
            ep_num = int(fname.split("_ep")[1].split("_")[0])
            if ep_num >= cfg.swa_start_epoch:
                filtered_paths.append(p)
        except (IndexError, ValueError):
            filtered_paths.append(p)
    all_paths = filtered_paths
    if not all_paths:
        if is_master:
            print(f"[SWA-Student] No checkpoints at or after swa_start_epoch={cfg.swa_start_epoch}.")
        return
    # Cap at last 5 for diversity-aware averaging.
    if len(all_paths) > 5:
        all_paths = all_paths[-5:]

    # ── 2. Average weights (master only) ────────────────────────────────────
    avg_sd: OrderedDict = OrderedDict()
    if is_master:
        count = len(all_paths)
        print(f"[SWA-Student] Averaging {count} checkpoints...")
        for i, path in enumerate(all_paths):
            print(f"  [{i+1}/{count}] {path}")
            ckpt  = torch.load(path, map_location="cpu", weights_only=False)
            state = ckpt.get("module") or ckpt.get("model_state") or ckpt
            for k, v in state.items():
                # Remap FusedLoRALinear keys → plain nn.Linear keys
                nk = k.replace("_orig_mod.", "").replace("module.", "")
                if nk.endswith(".W_fused"):
                    nk = nk[: -len(".W_fused")] + ".weight"
                elif nk.endswith(".base_layer.bias"):
                    nk = nk[: -len(".base_layer.bias")] + ".bias"
                elif ".lora_A" in nk or ".lora_B" in nk or ".base_layer.weight" in nk:
                    continue
                if nk not in avg_sd:
                    avg_sd[nk] = v.float().clone()
                elif v.shape == avg_sd[nk].shape:
                    avg_sd[nk] += v.float()

        for k in avg_sd:
            avg_sd[k] /= count

    dist.barrier()

    # ── 3. Build model and load averaged weights ─────────────────────────────
    # Final resolution = the last curriculum stage that was actually trained.
    if cfg.high_epochs   > 0: final_res = 512
    elif cfg.middle_epochs > 0: final_res = 448
    else:                     final_res = cfg.warmup_res
    model = StudentEnsemble(
        num_classes   = cfg.num_classes,
        input_res     = final_res,
        bioclip_name  = cfg.bioclip,
        dinov3_name   = cfg.dinov3,
        convnext_name = cfg.convnext,
    ).to(device)

    model.ensure_backbones_loaded()

    if is_master:
        model_sd    = model.state_dict()
        compatible  = {k: v for k, v in avg_sd.items()
                       if k in model_sd and v.shape == model_sd[k].shape}
        skipped     = set(avg_sd) - set(compatible)
        if skipped:
            print(f"[SWA-Student] Skipped {len(skipped)} shape-mismatched keys "
                  f"(e.g. pos_embed at wrong resolution).")
        model.load_state_dict(compatible, strict=False)

    # Broadcast from master to all GPUs
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=0)

    # ── 4. Save ──────────────────────────────────────────────────────────────
    if is_master:
        from src.training.checkpoints import atomic_torch_save
        out_path = cfg.swa_ckpt
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        atomic_torch_save(model.state_dict(), out_path)
        print(f"[SWA-Student] Saved → {out_path}")


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)

    from .config import load

    parser = argparse.ArgumentParser(description="Student SWA weight averaging")
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--force",      action="store_true",
                        help="Overwrite existing swa_model_final.pth")
    args, _ = parser.parse_known_args()

    run_swa(load(args.config), force=args.force)


if __name__ == "__main__":
    main()
