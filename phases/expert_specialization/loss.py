"""Teacher training loss — owned exclusively by expert_specialization."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from src.training.losses import AsymmetricLoss   # stable shared dependency


def build_criterion(num_classes: int, class_counts: torch.Tensor | None,
                    device: torch.device) -> AsymmetricLoss:
    criterion = AsymmetricLoss(use_fused=True, class_counts=class_counts).to(device)
    if hasattr(torch, "compile"):
        criterion = torch.compile(criterion)
    return criterion
