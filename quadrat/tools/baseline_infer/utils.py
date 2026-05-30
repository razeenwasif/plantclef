"""
Shared utilities: logging, device, checkpointing, metrics, DDP helpers.
"""
from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Optional[str] = None, rank: int = 0) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if output_dir and rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(output_dir) / "run.log"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# AMP context
# ---------------------------------------------------------------------------

def amp_autocast(device: str, enabled: bool, dtype: torch.dtype):
    """Return an autocast context manager (or nullcontext when disabled)."""
    if not enabled or not device.startswith("cuda"):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


# ---------------------------------------------------------------------------
# Scheduler: linear warmup + cosine decay (step-based)
# ---------------------------------------------------------------------------

def build_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_fraction: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Step-based LR schedule: linear warmup then cosine decay.
    Call scheduler.step() once per optimizer step.
    """
    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_fraction, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict,
    path: str,
    is_best: bool = False,
    best_path: Optional[str] = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)
    logger.info(f"Checkpoint saved: {p}  (epoch {state.get('epoch', '?')})")
    if is_best and best_path:
        shutil.copyfile(p, best_path)
        logger.info(f"Best checkpoint: {best_path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
) -> tuple[int, dict]:
    """Load checkpoint, restore model + optionals. Returns (start_epoch, metrics)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logger.info(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    state_dict = ckpt["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(state_dict)

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    metrics     = ckpt.get("metrics", {})
    logger.info(f"Resumed from epoch {ckpt.get('epoch', '?')}")
    return start_epoch, metrics


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

TAXONOMY_WEIGHTS = {
    "genus":  0.30,
    "family": 0.15,
    "order":  0.05,
    "class":  0.02,
}


def compute_multitask_loss(
    outputs: tuple,
    labels: tuple,
    criterion: nn.Module,
) -> tuple[torch.Tensor, dict]:
    """
    Compute weighted multi-task classification loss.

    outputs : (sp_logits, gen_logits, fam_logits, ord_logits, cls_logits)
              Taxonomy logits can be None when that head is disabled.
    labels  : (sp_labels, gen_labels, fam_labels, ord_labels, cls_labels)
              Taxonomy labels == -1 for samples missing that annotation.

    Returns (total_loss, loss_components_dict).
    """
    sp_log, gen_log, fam_log, ord_log, cls_log = outputs
    sp_lbl, gen_lbl, fam_lbl, ord_lbl, cls_lbl = labels

    total = criterion(sp_log, sp_lbl)
    parts = {"species_loss": total.item()}

    aux = [
        (gen_log, gen_lbl, "genus_loss",  TAXONOMY_WEIGHTS["genus"]),
        (fam_log, fam_lbl, "family_loss", TAXONOMY_WEIGHTS["family"]),
        (ord_log, ord_lbl, "order_loss",  TAXONOMY_WEIGHTS["order"]),
        (cls_log, cls_lbl, "class_loss",  TAXONOMY_WEIGHTS["class"]),
    ]
    for logits, lbls, key, w in aux:
        if logits is None:
            continue
        mask = lbls != -1
        if not mask.any():
            continue
        aux_loss = criterion(logits[mask], lbls[mask])
        total    = total + w * aux_loss
        parts[key] = aux_loss.item()

    parts["total_loss"] = total.item()
    return total, parts


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Fraction of samples where the true label is in the top-k predictions."""
    if logits.numel() == 0:
        return 0.0
    k = min(k, logits.size(1))
    _, top_indices = logits.topk(k, dim=1)
    correct = top_indices.eq(labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def taxonomy_accuracy(logits: Optional[torch.Tensor], labels: torch.Tensor) -> Optional[float]:
    """Top-1 accuracy for a taxonomy head, ignoring -1 labels."""
    if logits is None:
        return None
    mask = labels != -1
    if not mask.any():
        return None
    pred = logits[mask].argmax(dim=1)
    return pred.eq(labels[mask]).float().mean().item()


# ---------------------------------------------------------------------------
# Metrics CSV / JSON
# ---------------------------------------------------------------------------

def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def append_metrics_csv(row: dict, path: str) -> None:
    """Append one row to a CSV file (create with header if new)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.exists()
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_dist_initialized() -> bool:
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_dist_initialized():
        import torch.distributed as dist
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if is_dist_initialized():
        import torch.distributed as dist
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_initialized():
        import torch.distributed as dist
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not is_dist_initialized():
        return tensor
    import torch.distributed as dist
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / get_world_size()
