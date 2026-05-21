import os
import sys
import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.config import USE_FP8
from src.training.accelerator import accelerator
from src.training.telemetry import telemetry

def run_epoch(model, loader, val_loader, criterion, trait_criterion, epoch, num_classes, device):
    """Universal standard training loop for raw images."""
    model.train()
    is_ds = hasattr(model, 'backward')
    rank = int(os.environ.get("RANK", 0))
    
    # ORACLE: Local-only counters
    local_correct = 0
    local_total = 0

    # ORACLE: Rank-aware progress bars for 4-GPU clusters
    pbar = tqdm(loader, desc=f"[Rank {rank}] Epoch {epoch}", position=rank, leave=True, disable=False)
    
    for i, data in enumerate(pbar):
        # 1. Data Loading
        if isinstance(data[0], dict):
            images = data[0]['data'].to(memory_format=torch.channels_last)
            labels = data[0]['label'].squeeze().long().to(device)
        else:
            images, labels = data
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device).squeeze().long()

        # 2. Precision Alignment
        model_dtype = next(model.module.parameters()).dtype if is_ds else next(model.parameters()).dtype
        images = images.to(dtype=model_dtype)

        # 3. Taxonomic CutMix (Final 1% Regularization)
        if model.training and np.random.random() < 0.4:
            lam = np.random.beta(1.0, 1.0)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).to(images.device)
            
            # Simplified CutMix for speed during Phase 2B
            H, W = images.shape[2], images.shape[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
            cx = np.random.randint(W); cy = np.random.randint(H)
            bbx1 = np.clip(cx - cut_w // 2, 0, W); bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W); bby2 = np.clip(cy + cut_h // 2, 0, H)
            images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        # 4. Forward Pass
        accel = accelerator()
        with accel.autocast(dtype=torch.bfloat16):
            outputs, trait_logits = model(images)
            species_logits = outputs.float()
            loss = criterion(species_logits, labels)

        # 5. Backward Pass
        if is_ds:
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            accel.optimizer_step(model.optimizer)
            model.optimizer.zero_grad()
            accel.mark_step()  # no-op on CUDA; flushes XLA graph on TPU

        # 6. UI Update (ORACLE: Per-step updates during compilation phase)
        with torch.no_grad():
            preds = species_logits.max(1)[1]
            local_correct += (preds == labels).sum().item()
            local_total += labels.size(0)

        # Force per-step update for first 50 steps to show machine is alive
        if i < 50 or (i+1) % 20 == 0:
            local_acc = (local_correct / local_total) * 100.0
            status = "Compiling..." if i < 15 else "Active"
            pbar.set_postfix({"L": f"{loss.item():.3f}", "Acc": f"{local_acc:.1f}%", "Status": status})

        # Structured telemetry: emit every 50 steps so the dashboard + liveness
        # probe + (future) sweep coordinator have an authoritative event stream.
        # The per-step progress bar above stays for the TTY-watching human.
        if (i + 1) % 50 == 0:
            telemetry.emit(
                "step",
                epoch=int(epoch) if isinstance(epoch, int) else str(epoch),
                step=i + 1,
                loss=float(loss.item()),
                local_acc=float((local_correct / local_total) * 100.0) if local_total > 0 else 0.0,
            )
            
        # ORACLE: Periodic HBM Defragmentation
        if (i+1) % 500 == 0:
            accel.empty_cache()

    # ORACLE: End-of-Epoch Deep Purge
    final_acc = (local_correct / local_total) * 100.0 if local_total > 0 else 0.0
    telemetry.emit("epoch.end", epoch=str(epoch), local_acc=float(final_acc), steps=i + 1)
    del outputs, species_logits, loss
    gc.collect()
    accel.empty_cache()
    return final_acc

def run_phase1_cached(model, loader, criterion, epoch, num_classes, device):
    """Phase 1: Fast MLP warmup on cached features."""
    import torch.distributed as dist
    rank = int(os.environ.get("RANK", 0))
    model.train()
    is_ds = hasattr(model, 'backward')

    try:
        import transformer_engine.pytorch as te
        has_te = True
    except ImportError:
        has_te = False

    local_correct = 0
    local_total = 0
    pbar = tqdm(loader, desc=f"[Rank {rank}] Phase 1 Epoch {epoch}", position=rank, disable=False)

    accel = accelerator()
    for i, (x, y) in enumerate(pbar):
        model_dtype = next(model.module.parameters()).dtype if is_ds else next(model.parameters()).dtype
        x, y = x.to(device, dtype=model_dtype), y.to(device)

        # Transformer-Engine FP8 is CUDA-only; on non-CUDA accelerators we
        # fall through to the accelerator's autocast (or no-op).
        use_te_fp8 = USE_FP8 and has_te and accel.is_cuda
        ctx = te.fp8_autocast(enabled=True) if use_te_fp8 else accel.autocast(enabled=False)
        with ctx:
            outputs = model(x)
            loss = criterion(outputs, y)

        if is_ds:
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            if hasattr(model, 'optimizer'):
                accel.optimizer_step(model.optimizer)
                model.optimizer.zero_grad()
                accel.mark_step()
        
        with torch.no_grad():
            preds = outputs.max(1)[1]
            local_correct += (preds == y).sum().item()
            local_total += y.size(0)
            
        if i < 50 or (i+1) % 20 == 0:
            local_acc = (local_correct / local_total) * 100.0
            pbar.set_postfix({"L": f"{loss.item():.3f}", "Acc": f"{local_acc:.1f}%"})
    
    return (local_correct / local_total) * 100.0 if local_total > 0 else 0.0

def validate(model, loader, criterion, num_classes, device):
    """Universal validation loop with distributed aggregation."""
    model.eval()
    import torch.distributed as dist
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    local_correct = torch.tensor(0.0).to(device)
    local_total = torch.tensor(0.0).to(device)
    local_loss = torch.tensor(0.0).to(device)
    num_batches = 0

    with torch.no_grad():
        for i, data in enumerate(loader):
            if isinstance(data[0], dict):
                images = data[0]['data'].to(memory_format=torch.channels_last)
                labels = data[0]['label'].squeeze().long().to(device)
            else:
                images, labels = data
                images = images.to(device, memory_format=torch.channels_last)
                labels = labels.to(device).squeeze().long()
            
            with accelerator().autocast(dtype=torch.bfloat16):
                outputs = model(images)
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss = criterion(outputs, labels)
            
            local_loss += loss.item()
            preds = outputs.max(1)[1]
            local_correct += (preds == labels).sum()
            local_total += labels.size(0)
            num_batches += 1
            
    # ORACLE: Cluster-Wide Aggregation
    if world_size > 1:
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        global_num_batches = torch.tensor(num_batches * world_size).to(device)
        dist.all_reduce(global_num_batches, op=dist.ReduceOp.SUM) # Handled by count
        
    avg_acc = (local_correct / local_total).item() * 100.0
    avg_loss = (local_loss / (num_batches * world_size)).item() if world_size > 1 else (local_loss / num_batches).item()

    if rank == 0:
        print(f"\n--- Validation Results: Acc: {avg_acc:.2f}% | Loss: {avg_loss:.4f} ---")
    telemetry.emit("validation.end", acc=float(avg_acc), loss=float(avg_loss), batches=num_batches)

    # ORACLE: Deep Purge
    import gc
    gc.collect()
    accelerator().empty_cache()

    return avg_acc, avg_loss
