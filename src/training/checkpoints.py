import glob
import os
import shutil
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
from src import config as _cfg


def atomic_torch_save(obj, path: str) -> None:
    """torch.save with crash-safe tmp-then-rename. If the process dies mid-write,
    the original `path` (if any) is preserved and the caller sees no half-written file."""
    path = str(path)
    tmp  = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _ep_num(p: str) -> int:
    tail = os.path.basename(p).removeprefix("ep")
    return int(tail) if tail.isdigit() else -1


def _ds_checkpoint_looks_valid(ep_dir: str) -> bool:
    """Cheap integrity check: dir exists, has at least one non-empty .pt/.bin file,
    and contains a `mp_rank_*_model_states.pt` (DeepSpeed standard).
    Catches the common case of a SIGKILL during save leaving zero-byte shards."""
    if not os.path.isdir(ep_dir):
        return False
    files = os.listdir(ep_dir)
    if not files:
        return False
    payloads = [f for f in files if f.endswith((".pt", ".bin"))]
    if not payloads:
        return False
    for f in payloads:
        if os.path.getsize(os.path.join(ep_dir, f)) == 0:
            return False
    has_model = any("model_states" in f for f in payloads)
    return has_model


def prune_deepspeed_checkpoints(out_dir: str, keep: int = 2) -> int:
    """Delete `ep*/` DS checkpoint dirs except the most recent `keep` *valid* ones.
    Corrupted dirs are deleted on sight regardless of recency, so they cannot be
    auto-loaded as 'latest' on the next resume."""
    pattern  = os.path.join(str(out_dir), "ep*")
    all_dirs = sorted(
        [p for p in glob.glob(pattern) if os.path.isdir(p) and _ep_num(p) >= 0],
        key=_ep_num,
    )
    deleted = 0
    valid_dirs   = []
    for d in all_dirs:
        if _ds_checkpoint_looks_valid(d):
            valid_dirs.append(d)
        else:
            shutil.rmtree(d, ignore_errors=True)
            deleted += 1
    for old in valid_dirs[:-keep] if keep > 0 else valid_dirs:
        shutil.rmtree(old, ignore_errors=True)
        deleted += 1
    return deleted


def peek_checkpoint_metadata(out_dir: str, patch_size: int = 14) -> dict:
    """Read shape-revealing tensors from the latest DS checkpoint without
    actually loading model weights. Used at startup to align the freshly-built
    model's architecture/resolution with what was saved.

    Returns dict with optional keys:
      - saved_res:   inferred image resolution (from BioCLIP pos_embed shape)
      - gating_dim:  output dim of gating_network.3 (3 = non-FP8, 8 = FP8)
      - epoch_tag:   the latest tag string (e.g. "ep2")
    Empty dict on any failure (caller should treat as 'fresh run')."""
    out_dir = str(out_dir)
    latest = os.path.join(out_dir, "latest")
    if not os.path.exists(latest):
        return {}
    try:
        tag = open(latest).read().strip()
    except Exception:
        return {}
    state_pt = os.path.join(out_dir, tag, "mp_rank_00_model_states.pt")
    if not os.path.exists(state_pt) or os.path.getsize(state_pt) == 0:
        return {}
    try:
        sd = torch.load(state_pt, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    module = sd.get("module", sd) if isinstance(sd, dict) else {}
    if not isinstance(module, dict):
        return {}

    out: dict = {"epoch_tag": tag}
    for k, v in module.items():
        if not hasattr(v, "shape"):
            continue
        if k.endswith("gating_network.3.weight") and "gating_dim" not in out:
            out["gating_dim"] = int(v.shape[0])
        # Visual pos_embed (BioCLIP-style: [N+1, D])
        if (k.endswith("bioclip.model.visual.positional_embedding")
                or k.endswith("bioclip.backbone.positional_embedding")) and v.dim() == 2:
            n_total = int(v.shape[0])
            grid    = int(round((n_total - 1) ** 0.5))
            if grid > 0:
                out["saved_res"] = grid * patch_size
    return out


def robust_deepspeed_resume(model_engine, out_dir: str, name: str = "Resume") -> tuple[int, int]:
    """Try DeepSpeed resume from newest `ep*` tag, falling back to older tags
    if any are corrupt. Returns (start_epoch, start_step).

    Failure modes handled:
      - Latest DS dir killed mid-save (zero-byte shard files) → skip, try older
      - DS load_checkpoint raises an exception → skip, try older
      - All checkpoints fail → return (0, 0), log loudly
    """
    out_dir = str(out_dir)
    is_master = (int(os.environ.get("RANK", 0)) == 0)

    # Newest first
    pattern = os.path.join(out_dir, "ep*")
    candidates = sorted(
        [p for p in glob.glob(pattern) if os.path.isdir(p) and _ep_num(p) >= 0],
        key=_ep_num, reverse=True,
    )
    if not candidates:
        return 0, 0

    last_err = None
    for ep_path in candidates:
        tag = os.path.basename(ep_path)
        if not _ds_checkpoint_looks_valid(ep_path):
            if is_master:
                print(f"[{name}] Skip {tag}: integrity check failed (zero-byte or missing files).")
            continue
        try:
            lp, csd = model_engine.load_checkpoint(out_dir, tag=tag, load_module_strict=False)
            if lp:
                start_epoch = (csd or {}).get("epoch", _ep_num(ep_path))
                start_step  = (csd or {}).get("step", 0)
                if is_master:
                    print(f"[{name}] Resumed from {tag} → epoch={start_epoch}, step={start_step}")
                return start_epoch, start_step
            if is_master:
                print(f"[{name}] {tag}: load_checkpoint returned None; trying older.")
        except Exception as e:
            last_err = e
            if is_master:
                print(f"[{name}] {tag} load failed ({type(e).__name__}: {e}); trying older.")
            continue

    # ── Fallback: flat epoch_ep*_final.pth
    flat_pattern = os.path.join(out_dir, "epoch_ep*_final.pth")
    flat_files = []
    for p in glob.glob(flat_pattern):
        stem = os.path.basename(p).removeprefix("epoch_ep").removesuffix("_final.pth")
        if stem.isdigit() and os.path.getsize(p) > 0:
            flat_files.append((int(stem), p))
    flat_files.sort(reverse=True)

    for ep_num, flat_path in flat_files:
        try:
            sd = torch.load(flat_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model_state" in sd:
                sd = sd["model_state"]
            
            # ORACLE: Use robust_load_state_dict to handle pos_embed interpolation
            # and strip torch.compile prefixes automatically.
            module = model_engine.module
            res = robust_load_state_dict(module, sd, strict=False)
            
            if is_master:
                print(f"[{name}] Flat-file fallback: loaded weights from {os.path.basename(flat_path)}.")
                print(f"[{name}] Starting from epoch {ep_num + 1}")
            return ep_num + 1, 0
        except Exception as e:
            if is_master:
                print(f"[{name}] Flat fallback ep{ep_num} failed: {type(e).__name__}: {e}")
            continue

    if is_master:
        suffix = f" Last DS error: {last_err}" if last_err else ""
        print(f"[{name}] WARN: all DS and flat checkpoints unrecoverable; starting from epoch 0.{suffix}")
    return 0, 0

BASE_MODEL_DIR      = getattr(_cfg, "BASE_MODEL_DIR", "models")
P1_CKPT_PATH        = f"{BASE_MODEL_DIR}/phase1_checkpoint.pth"
P1_BEST_CKPT_PATH   = f"{BASE_MODEL_DIR}/phase1_checkpoint_best.pth"
P2_CKPT_DIR         = f"{BASE_MODEL_DIR}/phase2_checkpoint"
P2_EPOCH_CKPT       = getattr(_cfg, "P2_EPOCH_CKPT", "models/phase2_epoch_checkpoint.pth")


def phase1_is_complete() -> bool:
    """
    Checks if Phase 1 has already been finished.

    Returns
    -------
    bool
        True if the Phase 1 checkpoint file exists, False otherwise.
    """
    return os.path.exists(P1_CKPT_PATH)


def get_raw_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model to get the base nn.Module.
    Handles DeepSpeedEngine, torch.compile (OptimizedModule), and DDP.

    Parameters
    ----------
    model : torch.nn.Module
        The model to unwrap.

    Returns
    -------
    torch.nn.Module
        The base nn.Module.
    """
    # 1. Handle DeepSpeed or DDP
    if hasattr(model, 'module'):
        return get_raw_model(model.module)
    # 2. Handle torch.compile (OptimizedModule)
    if hasattr(model, '_orig_mod'):
        return get_raw_model(model._orig_mod)
    return model


def resample_pos_embed(checkpoint_state: Dict[str, torch.Tensor], 
                       model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Performs 'Surgery' on positional embeddings to handle resolution changes.
    Interpolates checkpoint embeddings to match the current model's grid size.

    Parameters
    ----------
    checkpoint_state : Dict[str, torch.Tensor]
        State dict from the saved checkpoint.
    model_state : Dict[str, torch.Tensor]
        State dict of the current model for shape matching.

    Returns
    -------
    Dict[str, torch.Tensor]
        Updated state dict with resampled positional embeddings.
    """
    updated_state = {}
    for k, v in checkpoint_state.items():
        if k in model_state and v.shape != model_state[k].shape:
            # Only resample known positional embedding keys
            if "positional_embedding" in k or "pos_embed" in k:
                print(f"[Surgery] Resampling {k}: {list(v.shape)} -> {list(model_state[k].shape)}")
                
                # Check for standard ViT format [1, N, D] or [N, D]
                if len(v.shape) == 3: # [1, N, D] (DINOv2)
                    # Exclude class token
                    pos_tokens = v[:, 1:, :]
                    d = v.shape[-1]
                    old_grid = int((v.shape[1] - 1)**0.5)
                    new_grid = int((model_state[k].shape[1] - 1)**0.5)
                    
                    pos_tokens = pos_tokens.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)
                    pos_tokens = F.interpolate(pos_tokens, size=(new_grid, new_grid), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid*new_grid, d)
                    
                    # Re-attach class token (from current model to preserve initialization)
                    new_v = torch.cat([model_state[k][:, :1, :], pos_tokens], dim=1)
                    updated_state[k] = new_v
                elif len(v.shape) == 2: # [N, D] (BioCLIP)
                    pos_tokens = v[1:, :]
                    d = v.shape[-1]
                    old_grid = int((v.shape[0] - 1)**0.5)
                    new_grid = int((model_state[k].shape[0] - 1)**0.5)
                    
                    pos_tokens = pos_tokens.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)
                    pos_tokens = F.interpolate(pos_tokens, size=(new_grid, new_grid), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(new_grid*new_grid, d)
                    
                    new_v = torch.cat([model_state[k][:1, :], pos_tokens], dim=0)
                    updated_state[k] = new_v
                else:
                    updated_state[k] = v
            else:
                # Other mismatch (e.g. classification head) -> handled by strict=False later
                updated_state[k] = v
        else:
            updated_state[k] = v
    return updated_state


def robust_load_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], 
                           strict: bool = False) -> Any:
    """
    Loads state dict with automatic positional embedding resampling and 
    torch.compile prefix handling.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load the state dict into.
    state_dict : Dict[str, torch.Tensor]
        The state dict to load.
    strict : bool, default False
        Whether to strictly enforce that the keys in state_dict match model keys.

    Returns
    -------
    Any
        Result of model.load_state_dict (NamedTuple with missing_keys and unexpected_keys).
    """
    raw_model = get_raw_model(model)
    current_state = raw_model.state_dict()
    
    # 1. Clean prefixes (e.g. from torch.compile)
    cleaned_state = {}
    for k, v in state_dict.items():
        # Strip '_orig_mod.' prefix if it exists in the checkpoint keys
        new_k = k.replace('_orig_mod.', '')
        cleaned_state[new_k] = v

    # 2. Perform Surgery on mismatched shapes (positional embeddings)
    processed_state = resample_pos_embed(cleaned_state, current_state)
    
    # ORACLE: Filter out parameters with size mismatches (e.g. upgraded projection layers)
    # This prevents RuntimeError when changing total_input_dim.
    final_state = {}
    for k, v in processed_state.items():
        if k in current_state:
            if v.shape == current_state[k].shape:
                final_state[k] = v
            else:
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f"[Surgery] Dropping mismatched param: {k} (Checkpoint: {list(v.shape)} | Model: {list(current_state[k].shape)})")
        else:
            final_state[k] = v

    # 3. Load into the base nn.Module
    result = raw_model.load_state_dict(final_state, strict=strict)
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        missing = result.missing_keys if result else []
        unexpected = result.unexpected_keys if result else []
        
        # Filter expected noise
        expected_missing = ['lora_', '_extra_state', 'positional_embedding', 'backbone', 'proj_linear', 'gating_network']
        missing = [k for k in missing if not any(x in k for x in expected_missing)]
        
        expected_unexpected = ['_orig_mod', 'fused_', 'tracked_']
        unexpected = [k for k in unexpected if not any(x in k for x in expected_unexpected)]
        
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"[Resume] Load complete with {len(missing)} unhandled missing keys and {len(unexpected)} unexpected keys.")
    return result


def load_phase1_checkpoint(model_engine: torch.nn.Module, device: torch.device) -> Tuple[int, float]:
    """
    Load Phase 1 checkpoint.

    Parameters
    ----------
    model_engine : torch.nn.Module
        The model engine (base or wrapped).
    device : torch.device
        Device to map tensors to.

    Returns
    -------
    Tuple[int, float]
        A tuple of (start_epoch, best_val_acc).
    """
    if os.path.exists(P1_CKPT_PATH):
        ckpt = torch.load(P1_CKPT_PATH, map_location=device, weights_only=False)
        robust_load_state_dict(model_engine, ckpt['model_state'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"[Phase1] Resumed from epoch {start_epoch} (val acc: {best_val_acc:.2f}%)")
        return start_epoch, best_val_acc
    return 0, 0.0


def load_phase1_heads_for_phase2(model: torch.nn.Module, device: torch.device) -> float:
    """
    Load only the classifier weights from the Phase 1 checkpoint into the Phase 2 model.
    Prioritizes the 'best' checkpoint if available.
    """
    # ORACLE: Prioritize the best-recorded weights for fine-tuning
    path = P1_BEST_CKPT_PATH if os.path.exists(P1_BEST_CKPT_PATH) else P1_CKPT_PATH
    
    if not os.path.exists(path):
        print("[Phase2] No Phase 1 checkpoint found -- starting with random heads.")
        return 0.0

    ckpt        = torch.load(path, map_location=device, weights_only=False)
    saved_state = ckpt['model_state']

    # Only load classifier -- skip backbones and old projection heads
    head_prefixes = ('classifier.',)
    head_state    = {k: v for k, v in saved_state.items()
                     if any(k.startswith(p) for p in head_prefixes)}

    robust_load_state_dict(model, head_state, strict=False)
    best_val_acc = ckpt.get('best_val_acc', 0.0)
    print(f"[Phase2] Loaded Phase 1 weights from {os.path.basename(path)} (val acc: {best_val_acc:.2f}%)")
    return best_val_acc


def load_phase2_checkpoint(model_engine: torch.nn.Module, device: torch.device, 
                           tag: Optional[str] = None) -> Tuple[Optional[int], float, int]:
    """
    Load Phase 2 checkpoint. Auto-discovers the latest unique checkpoint if tag is not provided.

    Parameters
    ----------
    model_engine : torch.nn.Module
        The model engine (base or wrapped).
    device : torch.device
        Device to map tensors to.
    tag : str, optional
        DeepSpeed checkpoint tag.

    Returns
    -------
    Tuple[Optional[int], float, int]
        A tuple of (start_epoch, best_val_acc, start_step).
    """
    # 1. Try DeepSpeed auto-load if it is a DeepSpeed engine and tag/latest exists
    if hasattr(model_engine, 'load_checkpoint') and os.path.exists(P2_CKPT_DIR):
        try:
            _, client_state = model_engine.load_checkpoint(P2_CKPT_DIR, tag=tag)
            if client_state is not None:
                start_epoch  = client_state['epoch'] + 1
                best_val_acc = client_state.get('best_val_acc', 0.0)
                print(f"[Phase2] Resumed from DeepSpeed checkpoint epoch {start_epoch}")
                return start_epoch, best_val_acc, 0
        except Exception as e:
            print(f"[Checkpoint] DeepSpeed load failed: {e}")

    # 2. Auto-discover unique mid-epoch or per-epoch checkpoints
    import glob
    # Use the imported config object directly to avoid NameError issues in some environments
    ckpts = glob.glob(f"{_cfg.BASE_MODEL_DIR}/phase2_checkpoint_ep*_step*.pth")
    if not ckpts: # legacy fallback
        ckpts = glob.glob("models/blackwell_v2/phase2_checkpoint_ep*_step*.pth")
        
    if ckpts:
        def sort_key(x):
            parts = x.replace(".pth", "").split("_step")
            epoch_part = parts[0].split("_ep")[-1]
            epoch = int(epoch_part)
            step_str = parts[1].replace("_", "")
            step = 999999999 if step_str == "final" else int(step_str)
            return [epoch, step]
            
        latest_ckpt = sorted(ckpts, key=sort_key)[-1]
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        
        robust_load_state_dict(model_engine, ckpt['model_state'], strict=False)
        
        epoch = ckpt.get('epoch', 0)
        step  = ckpt.get('step', 0)
        best_acc = ckpt.get('best_val_acc', 0.0)
        
        print(f"[Phase2] Resumed from unique checkpoint: {os.path.basename(latest_ckpt)}")
        if step >= ckpt.get('total_steps', 1e9) or step == 0: 
            return epoch + 1, best_acc, 0
        return epoch, best_acc, step

    return None, 0.0, 0


def _safe_torch_save(obj: Any, path: str) -> None:
    """
    Atomic and distributed-safe save.
    Only rank 0 writes. Writes to a temporary file then renames to avoid corruption.
    """
    # 1. Distributed Check
    is_master = True
    if torch.distributed.is_initialized():
        is_master = (torch.distributed.get_rank() == 0)
    
    if not is_master:
        return

    # 2. Path preparation
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    
    # 3. Atomic Write
    try:
        torch.save(obj, temp_path)
        os.rename(temp_path, path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def save_phase1_checkpoint(model_engine: torch.nn.Module, epoch: int, best_val_acc: float, is_best: bool = False) -> None:
    """
    Save lightweight Phase 1 checkpoint (model weights + metadata).
    Saves 'latest' and optionally 'best' state.
    """
    raw_model = get_raw_model(model_engine)
    data = {
        'epoch':        epoch,
        'model_state':  raw_model.state_dict(),
        'best_val_acc': best_val_acc,
    }
    
    # ORACLE: Atomic Save Latest
    _safe_torch_save(data, P1_CKPT_PATH)
    
    # ORACLE: Atomic Save Best
    if is_best:
        _safe_torch_save(data, P1_BEST_CKPT_PATH)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[Phase1] Best weights updated at epoch {epoch} (Acc: {best_val_acc:.2f}%)")

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[Phase1] Latest checkpoint saved (epoch {epoch})")


def save_epoch_checkpoint(model_engine: torch.nn.Module, epoch: int, best_val_acc: float) -> None:
    """
    Save unique per-epoch checkpoint and mirror to backup vault.
    """
    path = f"{_cfg.BASE_MODEL_DIR}/phase2_checkpoint_ep{epoch}_step_final.pth"
    raw_model = get_raw_model(model_engine)
    _safe_torch_save({
        'epoch':        epoch,
        'step':         0, # mark as finished
        'total_steps':  0,
        'model_state':  raw_model.state_dict(),
        'best_val_acc': best_val_acc,
    }, path)
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[Checkpoint] Saved epoch {epoch} to {path}")
        
        # ORACLE: Recovery Backup (Phase 2)
        try:
            backup_dir = os.path.join(os.path.dirname(_cfg.BASE_MODEL_DIR), "backups")
            os.makedirs(backup_dir, exist_ok=True)
            import shutil
            shutil.copy2(path, os.path.join(backup_dir, f"epoch_{epoch}_backup.pth"))
            print(f"[Backup] Mirrored epoch {epoch} to vault.")
        except: pass


def save_progress_checkpoint(model_engine: torch.nn.Module, epoch: int, step: int, total_steps: int) -> None:
    """
    Save unique mid-epoch progress checkpoint and auto-clean previous step.
    """
    path = f"{_cfg.BASE_MODEL_DIR}/phase2_checkpoint_ep{epoch}_step{step}.pth"
    raw_model = get_raw_model(model_engine)
    _safe_torch_save({
        'epoch':        epoch,
        'step':         step,
        'total_steps':  total_steps,
        'model_state':  raw_model.state_dict(),
    }, path)
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"\n[Progress-v2] Saved mid-epoch progress to {path}")
        
        # ORACLE: Storage Auto-Clean (Phase 2)
        # Delete previous mid-epoch step to prevent disk overflow
        import glob
        # Find other mid-epoch steps for THIS epoch only
        others = glob.glob(f"{_cfg.BASE_MODEL_DIR}/phase2_checkpoint_ep{epoch}_step*.pth")
        for old_path in others:
            # Don't delete the one we just saved or the 'final' epoch markers
            if old_path != path and "step_final" not in old_path:
                try:
                    os.remove(old_path)
                    print(f"[Cleanup] Reclaimed space: deleted {os.path.basename(old_path)}")
                except: pass


def save_deepspeed_checkpoint(model_engine: Any, directory: str, tag: str, 
                              epoch: int, best_val_acc: float) -> None:
    """
    Full DeepSpeed checkpoint -- model + optimizer + scheduler states.

    Parameters
    ----------
    model_engine : Any
        DeepSpeed engine.
    directory : str
        Directory to save the checkpoint into.
    tag : str
        Checkpoint tag.
    epoch : int
        Current epoch number.
    best_val_acc : float
        Best validation accuracy so far.
    """
    model_engine.save_checkpoint(
        directory, tag=tag,
        client_state={'epoch': epoch, 'best_val_acc': best_val_acc}
    )

