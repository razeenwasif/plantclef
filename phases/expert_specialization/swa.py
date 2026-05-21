"""Teacher SWA — owned exclusively by expert_specialization.

Averages the last N epoch checkpoints saved by run.py and writes
swa_model_final.pth to the teacher output directory.

Key fixes applied in this file (do NOT copy-paste into student/swa.py
without verifying student-side checkpoint format):
  - ensure_backbones_loaded() called on ALL ranks before broadcast
  - shape-filtered load to handle curriculum pos_embed mismatches
  - BN calibration skipped (ViT/ConvNeXtV2 use LayerNorm only)
"""
from __future__ import annotations
import glob
import os
from collections import OrderedDict

import torch
import torch.distributed as dist

from .config import P2bTeacherConfig
from .model  import TeacherEnsemble


def run_swa(cfg: P2bTeacherConfig, force: bool = False) -> None:
    local_rank = cfg.local_rank
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)

    is_master  = (cfg.rank == 0)

    # Guard: skip if swa_model_final.pth already exists unless --force is passed.
    if is_master and os.path.exists(cfg.swa_ckpt) and not force:
        print(f"[SWA-Teacher] SKIP: {cfg.swa_ckpt} already exists.")
        print("[SWA-Teacher] Pass --force to re-average and overwrite.")
        if dist.is_initialized():
            dist.barrier()
        return

    # ── 1. Collect epoch checkpoints ────────────────────────────────────────
    pattern   = os.path.join(cfg.output_dir, "epoch_ep*_final.pth")
    all_paths = sorted(glob.glob(pattern))
    if not all_paths:
        if is_master:
            print(f"[SWA-Teacher] No checkpoints found at {pattern}")
        return
    
    # ORACLE: Honor swa_start_epoch from YAML
    filtered_paths = []
    for p in all_paths:
        try:
            fname = os.path.basename(p)
            # Pattern: epoch_ep{epoch}_final.pth
            ep_str = fname.split("_ep")[1].split("_")[0]
            ep_num = int(ep_str)
            if ep_num >= cfg.swa_start_epoch:
                filtered_paths.append(p)
        except (IndexError, ValueError):
            filtered_paths.append(p) # Fallback for unexpected naming
    
    all_paths = filtered_paths

    # ── 2. Average weights (master only) ────────────────────────────────────
    avg_sd: OrderedDict = OrderedDict()
    param_counts: Dict[str, int] = {}
    
    if is_master:
        count = len(all_paths)
        print(f"[SWA-Teacher] Averaging {count} checkpoints ({cfg.expert_name})...")
        for i, path in enumerate(all_paths):
            print(f"  [{i+1}/{count}] {path}")
            ckpt  = torch.load(path, map_location="cpu", weights_only=False)
            state = ckpt.get("module") or ckpt.get("model_state") or ckpt
            for k, v in state.items():
                nk = k.replace("_orig_mod.", "").replace("module.", "")
                if nk.endswith(".W_fused"):
                    nk = nk[: -len(".W_fused")] + ".weight"
                elif nk.endswith(".base_layer.bias"):
                    nk = nk[: -len(".base_layer.bias")] + ".bias"
                elif ".lora_A" in nk or ".lora_B" in nk or ".base_layer.weight" in nk:
                    continue
                
                if nk not in avg_sd:
                    avg_sd[nk] = v.float().clone()
                    param_counts[nk] = 1
                elif v.shape == avg_sd[nk].shape:
                    avg_sd[nk] += v.float()
                    param_counts[nk] += 1
                else:
                    # Shape mismatch (e.g. mixed resolutions). 
                    # We keep the one already in avg_sd and skip this one.
                    # robust_load_state_dict will handle interpolation later.
                    pass

        # Perform the actual averaging based on per-parameter occurrence counts
        for k in avg_sd:
            avg_sd[k] /= param_counts[k]

    dist.barrier()

    # ── Architecture-match: build the SWA target so it matches what the saved
    # epoch checkpoints were trained with. Uses the same peek-and-override
    # logic as expert_specialization/run.py — without it, SWA bakes in pos_embed at
    # cfg.resolution (e.g. 512) and gating_dim=8 (FP8 default on Blackwell),
    # which then can't be loaded by anything that expects 224 / non-FP8.
    from src.training.checkpoints import peek_checkpoint_metadata, robust_load_state_dict
    saved_meta = peek_checkpoint_metadata(cfg.output_dir, patch_size=14)
    if saved_meta and saved_meta.get("gating_dim") == 3:
        import src.config as _global_cfg
        if getattr(_global_cfg, "USE_FP8", False):
            _global_cfg.USE_FP8 = False
            if is_master:
                print(f"[SWA-Teacher] Saved checkpoints have gating_dim=3 → "
                      f"forcing USE_FP8=False so SWA architecture matches.")
    initial_res = saved_meta.get("saved_res", cfg.resolution) if saved_meta else cfg.resolution
    if is_master and saved_meta:
        print(f"[SWA-Teacher] Build target: input_res={initial_res}, "
              f"gating_dim={saved_meta.get('gating_dim')}.")

    # ── 3. Build model — ALL ranks must call ensure_backbones_loaded so the
    #       parameter set matches during the broadcast loop.
    model = TeacherEnsemble(
        num_classes   = cfg.num_classes,
        input_res     = initial_res,
        bioclip_name  = cfg.bioclip,
        dinov3_name   = cfg.dinov3,
        convnext_name = cfg.convnext,
    ).to(device)
    model.ensure_backbones_loaded()   # critical: must run on every rank

    if is_master:
        # ORACLE: Use robust_load_state_dict to handle pos_embed interpolation
        # if resolutions between averaged weights and build target differ.
        robust_load_state_dict(model, avg_sd, strict=False)

    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for buf in model.buffers():
        dist.broadcast(buf.data, src=0)

    # No BN calibration — all three backbones use LayerNorm exclusively.
    has_bn = any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                torch.nn.SyncBatchNorm))
                 for m in model.modules())
    if has_bn and is_master:
        print("[SWA-Teacher] WARNING: BN layers detected — skipping calibration. "
              "Add update_bn call here if needed.")

    # ── 4. Save ──────────────────────────────────────────────────────────────
    if is_master:
        from src.training.checkpoints import atomic_torch_save
        out_path = cfg.swa_ckpt
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        atomic_torch_save(model.state_dict(), out_path)
        print(f"[SWA-Teacher] Saved → {out_path}")
        print("=" * 60)
        print(f" TEACHER SWA COMPLETE: {out_path}")
        print("=" * 60)


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)

    from .config import load

    parser = argparse.ArgumentParser(description="Teacher SWA weight averaging")
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--force",      action="store_true",
                        help="Overwrite existing swa_model_final.pth")
    args, _ = parser.parse_known_args()

    run_swa(load(args.config), force=args.force)


if __name__ == "__main__":
    main()
