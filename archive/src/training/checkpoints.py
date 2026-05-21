import os
import torch
import config as _cfg
P1_CKPT_PATH  = getattr(_cfg, "P1_CKPT_PATH",  "models/phase1_checkpoint.pth")
P2_CKPT_DIR   = getattr(_cfg, "P2_CKPT_DIR",   "models/phase2_checkpoint")
P2_EPOCH_CKPT = getattr(_cfg, "P2_EPOCH_CKPT", "models/phase2_epoch_checkpoint.pth")
EPOCHS_PHASE1 = getattr(_cfg, "EPOCHS_PHASE1", 10)


def phase1_is_complete():
    """
    Returns True if a Phase 1 checkpoint exists covering all epochs.
    Used to auto-skip Phase 1 on restarts when the head is already trained.
    """
    if not os.path.exists(P1_CKPT_PATH):
        return False
    try:
        meta = torch.load(P1_CKPT_PATH, map_location='cpu', weights_only=False)
        return meta.get('epoch', -1) >= EPOCHS_PHASE1 - 1
    except Exception:
        return False


def load_phase1_checkpoint(model_engine, device):
    """
    Load Phase 1 checkpoint into a DeepSpeed engine.
    Strips _orig_mod. prefix if checkpoint was saved while torch.compile was active.
    Returns (start_epoch, best_val_acc).
    """
    if not os.path.exists(P1_CKPT_PATH):
        return 0, 0.0

    ckpt        = torch.load(P1_CKPT_PATH, map_location=device, weights_only=False)
    saved_state = ckpt['model_state']
    if any('_orig_mod.' in k for k in saved_state.keys()):
        saved_state = {k.replace('._orig_mod.', '.'): v for k, v in saved_state.items()}
    model_engine.module.load_state_dict(saved_state)
    start_epoch  = ckpt['epoch'] + 1
    best_val_acc = ckpt.get('best_val_acc', 0.0)
    print(f"[Phase1] Resumed from epoch {start_epoch} "
          f"(best val acc: {best_val_acc:.2f}%)")
    return start_epoch, best_val_acc


def save_phase1_checkpoint(model_engine, epoch, best_val_acc):
    """Save lightweight Phase 1 checkpoint (model weights + metadata)."""
    raw_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    torch.save({
        'epoch':        epoch,
        'model_state':  raw_model.state_dict(),
        'best_val_acc': best_val_acc,
    }, P1_CKPT_PATH)
    print(f"[Phase1] Checkpoint saved (epoch {epoch})")


def load_phase1_heads_for_phase2(raw_model, device):
    """
    Load only the projection head + classifier weights from the Phase 1 checkpoint
    into the Phase 2 model.  Backbone keys are intentionally skipped because Phase 2
    applies LoRA which changes the backbone key structure.
    Returns best_val_acc from Phase 1 for logging.
    """
    if not os.path.exists(P1_CKPT_PATH):
        print("[Phase2] No Phase 1 checkpoint found -- starting from scratch.")
        return 0.0

    ckpt        = torch.load(P1_CKPT_PATH, map_location=device, weights_only=False)
    saved_state = ckpt['model_state']

    # Only load classifier -- skip backbones and old projection heads
    # (LoRA and Grouped Projection changed the structure; classifier is still compatible)
    head_prefixes = ('classifier.',)
    head_state    = {k: v for k, v in saved_state.items()
                     if any(k.startswith(p) for p in head_prefixes)}

    raw_model.load_state_dict(head_state, strict=False)
    best_val_acc = ckpt.get('best_val_acc', 0.0)
    print(f"[Phase2] Loaded Phase 1 head weights "
          f"(epoch {ckpt['epoch']}, val acc: {best_val_acc:.2f}%)")
    return best_val_acc


def load_phase2_checkpoint(model_engine, device, tag=None):
    """
    Load Phase 2 checkpoint. Returns (start_epoch, best_val_acc, start_step).
    Auto-discovers the latest unique checkpoint if tag is not provided.
    """
    is_deepspeed = hasattr(model_engine, 'load_checkpoint')
    
    # 1. Try DeepSpeed auto-load if it is a DeepSpeed engine and tag/latest exists
    if is_deepspeed and os.path.exists(P2_CKPT_DIR):
        try:
            _, client_state = model_engine.load_checkpoint(P2_CKPT_DIR, tag=tag)
            if client_state is not None:
                start_epoch  = client_state['epoch'] + 1
                best_val_acc = client_state.get('best_val_acc', 0.0)
                print(f"[Phase2] Resumed from DeepSpeed checkpoint epoch {start_epoch}")
                return start_epoch, best_val_acc, 0
        except Exception as e:
            print(f"[Checkpoint] DeepSpeed load failed (expected if fresh): {e}")

    # 2. Auto-discover unique mid-epoch or per-epoch checkpoints
    import glob
    ckpts = glob.glob("models/phase2_checkpoint_ep*_step*.pth")
    if ckpts:
        # Sort by epoch then step to find the absolute latest
        # Handle '_final' by replacing it with a large number for sorting
        def sort_key(x):
            parts = x.replace("models/phase2_checkpoint_ep", "").replace(".pth", "").split("_step")
            epoch = int(parts[0])
            step_str = parts[1].replace("_", "")
            step = 999999999 if step_str == "final" else int(step_str)
            return [epoch, step]
            
        latest_ckpt = sorted(ckpts, key=sort_key)[-1]
        
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        raw_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
        
        # Use strict=False because Warmup checkpoints don't have LoRA keys, 
        # and DeepSpeed checkpoints have different wrappers. We only need the heads.
        raw_model.load_state_dict(ckpt['model_state'], strict=False)
        
        epoch = ckpt.get('epoch', 0)
        step  = ckpt.get('step', 0)
        best_acc = ckpt.get('best_val_acc', 0.0)
        
        print(f"[Phase2] Resumed from unique checkpoint: {os.path.basename(latest_ckpt)}")
        # If it was a full epoch checkpoint (step == total_steps), start next epoch
        if step >= ckpt.get('total_steps', 1e9):
            return epoch + 1, best_acc, 0
        return epoch, best_acc, step

    # 3. Legacy fallbacks
    for path in [P2_EPOCH_CKPT, "models/phase2_progress.pth"]:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            raw_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
            raw_model.load_state_dict(ckpt['model_state'], strict=False)
            print(f"[Phase2] Resumed from legacy path: {path}")
            return ckpt['epoch'], ckpt.get('best_val_acc', 0.0), ckpt.get('step', 0)

    return None, 0.0, 0


def save_epoch_checkpoint(model_engine, epoch, best_val_acc):
    """Save unique per-epoch checkpoint."""
    path = f"models/phase2_checkpoint_ep{epoch}_step_final.pth"
    raw_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    torch.save({
        'epoch':        epoch,
        'step':         0, # mark as finished
        'total_steps':  0,
        'model_state':  raw_model.state_dict(),
        'best_val_acc': best_val_acc,
    }, path)
    print(f"[Checkpoint] Saved epoch {epoch} to {path}")


def save_progress_checkpoint(model_engine, epoch, step, total_steps):
    """Save unique mid-epoch progress checkpoint."""
    path = f"models/phase2_checkpoint_ep{epoch}_step{step}.pth"
    raw_model = model_engine.module if hasattr(model_engine, 'module') else model_engine
    torch.save({
        'epoch':        epoch,
        'step':         step,
        'total_steps':  total_steps,
        'model_state':  raw_model.state_dict(),
    }, path)
    print(f"\n[Progress-v2] Saved mid-epoch progress to {path}")


def save_deepspeed_checkpoint(model_engine, directory, tag, epoch, best_val_acc):
    """Full DeepSpeed checkpoint -- model + optimizer + scheduler states."""
    model_engine.save_checkpoint(
        directory, tag=tag,
        client_state={'epoch': epoch, 'best_val_acc': best_val_acc}
    )
