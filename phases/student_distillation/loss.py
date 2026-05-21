"""Student training loss — owned exclusively by student_distillation."""
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


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: AsymmetricLoss,
    alpha: float = 0.3,
    beta:  float = 0.7,
    temp:  float = 2.0,
) -> torch.Tensor:
    """Weighted sum of hard-label CE and soft-label KL divergence."""
    ce_loss = criterion(student_logits.float(), labels)
    soft_s  = F.log_softmax(student_logits / temp, dim=1)
    soft_t  = F.softmax(teacher_logits   / temp, dim=1)
    kl_loss = F.kl_div(soft_s, soft_t, reduction="batchmean") * (temp ** 2)
    return alpha * ce_loss + beta * kl_loss
