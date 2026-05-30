import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional
import time

# plantclef: PATH SYNC
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import load_config
from src.models.ensemble import PlantEnsemble

# DALI Imports
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False

# ── DALI Pipeline ─────────────────────────────────────────────────────────────

class TeacherCachePipeline(Pipeline):
    """Non-shuffled DALI pipeline. Labels = absolute mmap row indices."""

    def __init__(self, file_paths, row_indices, resolution,
                 batch_size, num_threads, device_id):
        # Honor the global PREFETCH_DEPTH (8 on B200, 6 on 6000 Blackwell).
        # Was hardcoded to 2 — starved B-class GPUs badly during cache build.
        try:
            from src import config as _cfg
            depth = getattr(_cfg, "PREFETCH_DEPTH", 5)
        except Exception:
            depth = 5
        super().__init__(batch_size, num_threads, device_id,
                         seed=0, prefetch_queue_depth=depth)
        self.input = ops.readers.File(
            files=file_paths,
            labels=row_indices,
            random_shuffle=False,
            num_shards=1,
            shard_id=0,
            pad_last_batch=True,
            prefetch_queue_depth=max(depth, 4),  # match outer
            name="Reader",
        )
        self.decode = ops.decoders.Image(
            device="mixed",
            output_type=types.RGB,
            hw_decoder_load=1.0,
            affine=True,
        )
        # CUBIC: visually indistinguishable from LANCZOS3 for inference,
        # ~1.5–2× faster on the GPU resize op. Cache is read-only output, so
        # tiny resize-quality differences don't propagate.
        self.resize = ops.Resize(
            device="gpu",
            size=[resolution, resolution],
            interp_type=types.INTERP_CUBIC,
        )
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    def define_graph(self):
        jpegs, row_idxs = self.input()
        images = self.resize(self.decode(jpegs))
        return self.normalize(images).gpu(), row_idxs.gpu()


# ── PIL fallback ──────────────────────────────────────────────────────────────

def _pil_loader(file_paths, row_indices, resolution, batch_size, num_workers):
    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader

    _norm = transforms.Compose([
        transforms.Resize((resolution, resolution), transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class _DS(Dataset):
        def __len__(self): return len(file_paths)
        def __getitem__(self, i):
            try:
                img = Image.open(file_paths[i]).convert("RGB")
                return _norm(img), row_indices[i]
            except Exception:
                return torch.zeros(3, resolution, resolution), row_indices[i]

    return DataLoader(_DS(), batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, pin_memory=True, drop_last=False,
                      persistent_workers=(num_workers > 0))


# ── Path helpers ──────────────────────────────────────────────────────────────

def fast_resolve(name, sp_id, img_dir, alt_dir):
    """Try three path candidates; return normalised path or None."""
    p1 = os.path.normpath(os.path.join(img_dir, str(name)))
    if os.path.isfile(p1) and os.path.getsize(p1) > 0: return p1
    p2 = os.path.normpath(os.path.join(img_dir, str(sp_id), str(name)))
    if os.path.isfile(p2) and os.path.getsize(p2) > 0: return p2
    p3 = os.path.normpath(os.path.join(alt_dir, str(name)))
    if os.path.isfile(p3) and os.path.getsize(p3) > 0: return p3
    return None


def _validate_path(p):
    """True if path is a non-empty, readable file."""
    try:
        return bool(p) and os.path.isfile(p) and os.path.getsize(p) > 0
    except OSError:
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build teacher logit cache (DALI multi-GPU)")
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--batch_size",  type=int, default=512,
                        help="Per-GPU batch size (reduce if OOM; larger = faster throughput)")
    parser.add_argument("--teacher_res", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DALI CPU threads (0 = auto: min(32, cpu_count))")
    parser.add_argument("--out_dir",     type=str, default=None)
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from done-mask checkpoint (instant — no re-iteration)")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing cache and done mask from scratch")
    parser.add_argument("--no_compile",  action="store_true",
                        help="Skip torch.compile (faster startup, ~same throughput due to "
                             "@torch.compiler.disable() on chunked_backbone_forward)")
    parser.add_argument("--checkpoint_every", type=int, default=10,
                        help="Flush mmap and done mask every N batches")

    rank       = int(os.environ.get("RANK", 0))
    is_main    = (rank == 0)

    args, unknown = parser.parse_known_args()
    if unknown and is_main:
        print(f"[build_teacher_cache] Ignoring unknown params: {unknown}")

    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))

    # NCCL interface detection
    if "NCCL_SOCKET_IFNAME" not in os.environ:
        import subprocess
        try:
            res = subprocess.check_output(
                "ip -4 addr show | grep 'inet 10.' | head -n1 | awk '{print $NF}'",
                shell=True, text=True).strip()
            if res:
                os.environ["NCCL_SOCKET_IFNAME"] = res
                os.environ["GLOO_SOCKET_IFNAME"]  = res
        except Exception:
            pass

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl", device_id=device)

    num_threads = args.num_workers or min(32, (os.cpu_count() or 8))
    config = load_config(args.config)

    # plantclef: CHUNK_SIZE = full batch so chunked_backbone_forward makes exactly 1
    # GPU call per forward pass.  The @torch.compiler.disable() on that function
    # runs once per batch (not batch_size/32 = 16 times), giving ~8-12× speedup.
    import src.config as _src_cfg_mod
    _src_cfg_mod.CHUNK_SIZE = args.batch_size

    # ── Output paths ──────────────────────────────────────────────────────────
    cache_base = args.out_dir if args.out_dir else os.path.dirname(config.FEATURE_CACHE_PATH)
    os.makedirs(cache_base, exist_ok=True)

    out_npy        = os.path.join(cache_base, "teacher_logit_cache.npy")
    out_order      = os.path.join(cache_base, "teacher_logit_cache_order.npy")
    resolved_csv   = os.path.join(cache_base, "resolved_build_df.csv")
    # Raw binary done-mask (no .npy header) for zero-copy memmap updates
    done_mask_path = os.path.join(cache_base, "cache_done_mask.bin")

    # ── Load teachers ─────────────────────────────────────────────────────────
    teacher_paths = [
        "models/cuda_deep_sat/expert_bioclip_512/swa_model_final.pth",
        "models/cuda_deep_sat/expert_dinov3_512/swa_model_final.pth",
    ]
    available = [p for p in teacher_paths if os.path.exists(p)]
    if not available:
        if is_main: print("[build_teacher_cache] No teacher checkpoints found — aborting.")
        return

    teachers = []
    for p in available:
        if is_main: print(f"[build_teacher_cache] Loading teacher: {p}")
        t = PlantEnsemble(num_classes=config.model.num_classes,
                          input_res=args.teacher_res).to(device)
        t.ensure_backbones_loaded()
        t.load_state_dict(torch.load(p, map_location=device, weights_only=False), strict=False)
        t.eval()
        for param in t.parameters():
            param.requires_grad = False
        if not args.no_compile:
            t = torch.compile(t, mode="reduce-overhead")
        teachers.append(t)

    if is_main:
        print(f"[build_teacher_cache] {len(teachers)} teachers loaded. "
              f"Running warmup (CUDA graph capture)...")

    # Warmup: capture CUDA graph for the steady-state batch shape.
    warmup_imgs = torch.empty(
        args.batch_size, 3, args.teacher_res, args.teacher_res,
        device=device, dtype=torch.bfloat16,
        memory_format=torch.channels_last,
    )
    with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for t in teachers:
            try:
                out = t(warmup_imgs)
            except Exception:
                pass
    del warmup_imgs
    torch.cuda.empty_cache()
    if is_main: print("[build_teacher_cache] Teachers ready.")

    # ── CSV → resolved file list ───────────────────────────────────────────────
    import pandas as pd
    csv_path = (config.dataset.cleaned_csv
                if os.path.exists(config.dataset.cleaned_csv)
                else config.dataset.raw_csv)
    df = pd.read_csv(csv_path, sep=';', dtype={'image_name': str}, low_memory=False)
    val_size = int(len(df) * 0.05)
    df_train = df.iloc[val_size:].reset_index(drop=True)
    img_dir  = config.dataset.img_dir.rstrip('/')
    alt_dir  = "/workspace/plantclef/processed/lucas_aspect_corrected"
    col      = 'species_ids' if 'species_ids' in df_train.columns else 'species_id'
    N_raw    = len(df_train)
    C        = config.model.num_classes

    if is_main:
        use_cached = (args.resume or not args.force) and os.path.exists(resolved_csv)
        if use_cached:
            print(f"[build_teacher_cache] Loading resolved paths from cache: {resolved_csv}")
            df_train = pd.read_csv(resolved_csv)
            # Normalise paths to eliminate double-slashes from stale cache entries
            df_train['resolved_path'] = df_train['resolved_path'].apply(
                lambda p: os.path.normpath(str(p)) if isinstance(p, str) else None
            )
            # Re-validate: drop any files that disappeared or are now empty/corrupted
            before = len(df_train)
            df_train = df_train[df_train['resolved_path'].apply(_validate_path)].reset_index(drop=True)
            if before != len(df_train):
                print(f"[build_teacher_cache] Dropped {before - len(df_train)} invalid cached paths.")
        else:
            print(f"[build_teacher_cache] Resolving {N_raw} image paths...")
            t0 = time.time()
            resolved = []
            for _, row in tqdm(df_train.iterrows(), total=N_raw, desc="  Resolving"):
                resolved.append(fast_resolve(row['image_name'], row[col], img_dir, alt_dir))
            df_train['resolved_path'] = resolved
            df_train = df_train[df_train['resolved_path'].notna()].reset_index(drop=True)
            df_train.to_csv(resolved_csv, index=False)
            print(f"[build_teacher_cache] {len(df_train)} valid images found "
                  f"({time.time() - t0:.1f}s).")

        if world_size > 1:
            dist.barrier()
    elif world_size > 1:
        dist.barrier()
        df_train = pd.read_csv(resolved_csv)
        df_train['resolved_path'] = df_train['resolved_path'].apply(
            lambda p: os.path.normpath(str(p)) if isinstance(p, str) else None
        )

    N          = len(df_train)
    file_paths = df_train['resolved_path'].tolist()

    if is_main:
        print(f"[build_teacher_cache] {N} training samples → ({N}, {C}) float16 cache")

    # ── Allocate mmap logit cache ─────────────────────────────────────────────
    if is_main:
        if not os.path.exists(out_npy) or args.force:
            np.memmap(out_npy, dtype=np.float16, mode='w+', shape=(N, C)).flush()
        if world_size > 1:
            dist.barrier()
    elif world_size > 1:
        dist.barrier()

    mmap = np.memmap(out_npy, dtype=np.float16, mode='r+', shape=(N, C))

    # ── Done mask (raw binary, per-row bool) ──────────────────────────────────
    # The mask is a flat bool array of length N, one entry per training row.
    # All ranks can memmap it safely because each rank writes to disjoint rows.
    if is_main:
        needs_reset = args.force or not os.path.exists(done_mask_path)
        if not needs_reset:
            # Sanity-check: mask must match current N
            existing_size = os.path.getsize(done_mask_path)
            if existing_size != N:
                print(f"[build_teacher_cache] Done-mask size mismatch "
                      f"({existing_size} vs {N}). Resetting.")
                needs_reset = True
        if needs_reset:
            np.memmap(done_mask_path, dtype=np.bool_, mode='w+', shape=(N,)).flush()
        if world_size > 1:
            dist.barrier()
    elif world_size > 1:
        dist.barrier()

    done_mask = np.memmap(done_mask_path, dtype=np.bool_, mode='r+', shape=(N,))

    # ── Signal handlers: flush mmaps before unclean exit (SIGTERM, SIGINT, SIGBUS) ──
    # SIGBUS is the common failure mode for mmap on flaky network volumes.
    # Without this, a kill mid-batch loses ALL un-flushed work since last
    # checkpoint_every flush. With it, we lose only the in-flight batch.
    import signal
    def _emergency_flush(signum, _frame):
        try:
            mmap.flush()
            done_mask.flush()
            print(f"\n[build_teacher_cache] Caught signal {signum}; flushed cache + done_mask.",
                  flush=True)
        except Exception as e:
            print(f"[build_teacher_cache] Emergency flush failed: {e}", flush=True)
        # Re-raise the default behavior for SIGTERM/SIGINT (exit). For SIGBUS
        # we can't continue safely — abort.
        os._exit(128 + signum)
    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGBUS):
        try:
            signal.signal(_sig, _emergency_flush)
        except (ValueError, OSError):
            pass  # SIGBUS not available on Windows / non-main thread

    # ── Per-rank pending set (instant resume — no iteration over done batches) ─
    # DALI sharding assigns rank R items at positions R, R+W, R+2W, ... in the file list.
    my_positions     = list(range(rank, N, world_size))
    pending_positions = [i for i in my_positions if not done_mask[i]]
    n_done_this_rank  = len(my_positions) - len(pending_positions)

    if is_main:
        total_done = int(done_mask.sum())
        pct = 100 * total_done / N if N > 0 else 0
        print(f"[build_teacher_cache] Checkpoint: {total_done}/{N} rows done ({pct:.1f}%).")

    if not pending_positions:
        if is_main:
            print(f"[build_teacher_cache] All rows cached. Nothing to do.")
            np.save(out_order, np.arange(N, dtype=np.int32))
        if world_size > 1:
            dist.barrier()
        return

    pending_files     = [file_paths[i] for i in pending_positions]
    # Labels = absolute mmap row indices (so DALI returns the right write target)
    pending_row_idxs  = [int(i) for i in pending_positions]
    n_pending_this    = len(pending_positions)

    if is_main or not (world_size > 1):
        print(f"[GPU {rank}] {n_done_this_rank} already done, "
              f"{n_pending_this} pending in this rank's shard.")

    # ── DALI or PIL pipeline (per-rank, no cross-rank sharding needed) ────────
    if HAS_DALI:
        pipe = TeacherCachePipeline(
            file_paths=pending_files,
            row_indices=pending_row_idxs,
            resolution=args.teacher_res,
            batch_size=args.batch_size,
            num_threads=num_threads,
            device_id=local_rank,
        )
        pipe.build()
        loader = DALIGenericIterator(
            [pipe], ['data', 'row_idx'],
            reader_name="Reader",
            auto_reset=False,
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
        total_steps = (n_pending_this + args.batch_size - 1) // args.batch_size
    else:
        loader = _pil_loader(pending_files, pending_row_idxs,
                             args.teacher_res, args.batch_size, num_threads)
        total_steps = len(loader)

    # Pre-allocate one stream per teacher (avoid per-batch allocation overhead)
    teacher_streams = [torch.cuda.Stream(device=device) for _ in teachers]

    written_this_run = 0

    with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        pbar = tqdm(
            total=total_steps,
            desc=f"[GPU {rank}] Building cache",
            position=rank,
            leave=True,
            disable=(not is_main and world_size > 1),
        )

        for i, batch in enumerate(loader):
            if HAS_DALI:
                imgs    = batch[0]['data'].to(dtype=torch.bfloat16,
                                              memory_format=torch.channels_last)
                row_idx = batch[0]['row_idx'].squeeze(-1).cpu().numpy().astype(np.int64)
            else:
                imgs, row_idx_t = batch
                imgs    = imgs.to(device, dtype=torch.bfloat16,
                                  memory_format=torch.channels_last)
                row_idx = row_idx_t.numpy().astype(np.int64)

            # Number of real (non-padded) samples in this batch
            real_in_batch = min(args.batch_size, n_pending_this - written_this_run)

            # ── Teacher ensemble forward ──────────────────────────────────────
            if len(teachers) > 1:
                t_outs = [None] * len(teachers)
                for idx, (t, s) in enumerate(zip(teachers, teacher_streams)):
                    with torch.cuda.stream(s):
                        out = t(imgs)
                        t_outs[idx] = out[0] if isinstance(out, tuple) else out
                torch.cuda.synchronize()
                logits = sum(t_outs) / len(teachers)
            else:
                logits = teachers[0](imgs)
                if isinstance(logits, tuple): logits = logits[0]

            # ── Write only real (non-padded) rows ─────────────────────────────
            valid_idx = row_idx[:real_in_batch]
            logits_np = logits[:real_in_batch].float().cpu().numpy().astype(np.float16)
            mmap[valid_idx] = logits_np
            done_mask[valid_idx] = True
            written_this_run += real_in_batch

            # Periodic checkpoint flush
            if (i + 1) % args.checkpoint_every == 0:
                mmap.flush()
                done_mask.flush()

            pbar.update(1)

            # Stop exactly when all real samples are processed
            if written_this_run >= n_pending_this:
                break

        pbar.close()

    mmap.flush()
    done_mask.flush()

    if is_main:
        np.save(out_order, np.arange(N, dtype=np.int32))
        total_done_final = int(np.memmap(done_mask_path, dtype=np.bool_,
                                         mode='r', shape=(N,)).sum())
        print(f"\n[build_teacher_cache] Run complete. "
              f"{total_done_final}/{N} rows cached ({100*total_done_final/N:.1f}%).")
        if total_done_final < N:
            print(f"[build_teacher_cache] {N - total_done_final} rows remain. "
                  f"Re-run with --resume to continue.")

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
