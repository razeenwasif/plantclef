"""
Shared utilities: logging, device resolution, checkpointing, and evaluation metrics.

Adapted from 005_dinov3_vegetation_adapt/utils.py with minor additions.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    output_dir: Optional[str] = None,
    rank: int = 0,
    level: int = logging.INFO,
) -> None:
    """
    Configure root logger for console output (and optionally a file).

    When rank > 0 (DDP non-main process), sets level to WARNING to suppress
    duplicate messages.
    """
    effective_level = level if rank == 0 else logging.WARNING
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if output_dir and rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(output_dir) / "run.log"))
    logging.basicConfig(
        level=effective_level,
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
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: dict,
    path: str,
    is_best: bool = False,
    best_path: Optional[str] = None,
) -> None:
    """
    Save training state to a .pt file.

    Recommended state keys:
        epoch, model_state_dict, optimizer_state_dict,
        scheduler_state_dict, metrics, config, idx_to_species
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)
    logger.info(f"Checkpoint saved: {p}  (epoch {state.get('epoch', '?')})")
    if is_best and best_path:
        shutil.copyfile(p, best_path)
        logger.info(f"Best checkpoint updated: {best_path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    device: str = "cpu",
) -> tuple[int, dict]:
    """
    Load training checkpoint and restore model (+ optionally optimizer/scheduler) state.

    Handles DDP-wrapped state dicts (strips 'module.' prefix).

    Returns (start_epoch, metrics_at_checkpoint).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logger.info(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    state_dict = ckpt["model_state_dict"]
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # Unwrap DDP for loading
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state_dict)

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    metrics = ckpt.get("metrics", {})
    logger.info(f"Resumed from epoch {ckpt.get('epoch', '?')}")
    return start_epoch, metrics


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    results: list[dict],
    k_values: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """
    Compute Recall@K from per-image prediction dicts.

    Each dict must have:
        gt_species_id  (str)       : ground-truth species_id
        pred_ids       (list[str]) : predictions ranked best-first

    Returns {'recall_at_1': 0.42, 'recall_at_5': 0.71, ...}
    """
    if not results:
        return {}
    n_total = len(results)
    hits: dict[int, int] = defaultdict(int)
    for res in results:
        gt = res["gt_species_id"]
        preds = res["pred_ids"]
        for k in k_values:
            if gt in preds[:k]:
                hits[k] += 1
    return {f"recall_at_{k}": round(hits[k] / n_total, 4) for k in k_values}


def topk_predictions(
    logits: torch.Tensor,
    idx_to_species: list[str],
    top_n: int = 20,
) -> tuple[list[str], list[float]]:
    """
    Extract top-N species predictions from a (num_classes,) logit tensor.

    Returns (pred_species_ids_best_first, pred_scores).
    """
    top_n = min(top_n, logits.size(-1))
    scores, indices = logits.topk(top_n)
    pred_species = [idx_to_species[i.item()] for i in indices]
    pred_scores  = [round(s.item(), 5) for s in scores]
    return pred_species, pred_scores


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info(f"Saved: {path}")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


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
    """Average a scalar tensor across all DDP ranks."""
    if not is_dist_initialized():
        return tensor
    import torch.distributed as dist
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / get_world_size()
