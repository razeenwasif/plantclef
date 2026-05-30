import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, List, Union, Any

try:
    import plantclef_ext
except ImportError:
    plantclef_ext = None


class FusedASLFunction(Function):
    """
    Fused Asymmetric Loss function using custom CUDA kernels for performance.
    Supports Taxonomic Hierarchy Smoothing and dynamic genus-level priors.
    """
    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, targets: torch.Tensor, 
                logit_adjustments: torch.Tensor, genus_ids: torch.Tensor,
                gamma_neg_tensor: torch.Tensor, target_genus_ids: torch.Tensor,
                gamma_pos: float, clip: float, eps: float, taxon_smoothing: float) -> torch.Tensor:
        """
        Forward pass for advanced fused ASL.
        """
        dtype = logits.dtype
        # plantclef: Enforce contiguity and proper alignment for CUDA kernel
        l_c = logits.contiguous()
        t_c = targets.contiguous() # Now guaranteed to be dense [B, NumClasses]
        a_c = logit_adjustments.to(dtype).contiguous()
        g_c = genus_ids.int().contiguous()
        gn_c = gamma_neg_tensor.to(dtype).contiguous()
        tg_c = target_genus_ids.int().contiguous()

        # Invoke 10-argument CUDA kernel
        losses = plantclef_ext.fused_asl_forward(
            l_c, t_c, a_c, g_c, gn_c,
            gamma_pos, clip, eps, taxon_smoothing, tg_c
        )[0]
        
        ctx.save_for_backward(l_c, t_c, a_c, g_c, gn_c, tg_c)
        ctx.gamma_pos = gamma_pos
        ctx.clip = clip
        ctx.eps = eps
        ctx.taxon_smoothing = taxon_smoothing
        ctx.original_dtype = dtype
        
        return losses.mean()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass for advanced fused ASL.
        """
        logits, targets, logit_adjustments, genus_ids, gamma_neg_tensor, target_genus_ids = ctx.saved_tensors
        dtype = logits.dtype
        
        # plantclef: Ensure grad_output is contiguous and consistent
        go_c = grad_output.contiguous()
        
        grad_logits = plantclef_ext.fused_asl_backward(
            go_c, logits, targets, logit_adjustments,
            genus_ids, gamma_neg_tensor,
            ctx.gamma_pos, ctx.clip, ctx.eps, ctx.taxon_smoothing,
            target_genus_ids
        )
        
        return grad_logits, None, None, None, None, None, None, None, None, None


class LogitAdjustmentLoss(nn.Module):
    """
    Shifts logits by log(prior) to demand larger margins for common classes.
    """
    def __init__(self, class_counts: Union[List[int], torch.Tensor], tau: float = 1.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        priors = (counts + 1) / (counts.sum() + len(counts))
        self.adjustment = (tau * torch.log(priors)).to('cuda')
        self.criterion  = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(x + self.adjustment, y)


class AsymmetricLoss(nn.Module):
    """
    Advanced Asymmetric Loss with Fused CUDA support and Taxonomic Hierarchy.
    PLANTCLEF: Enhanced with Duality-Derived per-class focusing parameters.
    """
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1, clip: float = 0.05, 
                 eps: float = 1e-8, logit_adjustments: Optional[torch.Tensor] = None, 
                 genus_ids: Optional[torch.Tensor] = None,
                 class_counts: Optional[Union[List[int], torch.Tensor]] = None,
                 taxon_smoothing: float = 0.15,
                 use_fused: bool = True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps  = eps
        self.taxon_smoothing = taxon_smoothing
        self.use_fused = use_fused and (plantclef_ext is not None)
        
        num_classes = 7808
        if logit_adjustments is not None:
            num_classes = logit_adjustments.size(0)
            self.register_buffer('logit_adjustments', logit_adjustments)
        else:
            self.register_buffer('logit_adjustments', None)

        if genus_ids is not None:
            self.register_buffer('genus_ids', genus_ids)
        else:
            self.register_buffer('genus_ids', None)

        # plantclef: Duality-Derived Focusing (Fenchel Duality)
        # Optimal gamma ratio satisfies: gamma_neg/gamma_pos = log(N_neg/N_pos) / log(N_neg)
        if class_counts is not None:
            import numpy as np
            counts = torch.tensor(class_counts, dtype=torch.float32)
            total = counts.sum()
            # Handle potential padding in counts (zeros)
            mask = counts > 0
            safe_counts = counts.clamp(min=1)
            
            # Theoretical gamma_neg derivation
            # High for rare classes (low N_pos), lower for head classes
            neg_counts = total - safe_counts
            theory_gamma_neg = torch.log(neg_counts / safe_counts) / torch.log(neg_counts.clamp(min=2))
            
            # Rescale to reasonable ASL range (e.g. 2.0 to 6.0)
            # Default to 4.0 for padded/missing classes
            gamma_neg_tensor = 2.0 + (theory_gamma_neg * 4.0).clamp(0, 4.0)
            gamma_neg_tensor[~mask] = float(gamma_neg)
            
            self.register_buffer('gamma_neg_tensor', gamma_neg_tensor.float())
            print(f"[Loss] Duality-Derived ASL: Per-class Gamma_Neg range [{gamma_neg_tensor.min():.2f} - {gamma_neg_tensor.max():.2f}]")
        else:
            # Pre-compute uniform gamma_neg per class
            self.register_buffer('gamma_neg_tensor', torch.full((num_classes,), float(gamma_neg)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Fused path only if all requirements met
        if self.use_fused and self.logit_adjustments is not None:
            # If genus_ids missing, use identity mapping to prevent crash
            g_ids = self.genus_ids if self.genus_ids is not None else torch.arange(x.size(1), device=x.device).int()
            
            # Efficiently calculate target genus IDs once per batch
            with torch.no_grad():
                # y can be one-hot (float) or hard labels (long)
                if y.dim() == 1:
                    target_species = y.to(x.device)
                    # plantclef: The fused kernel expects a dense target matrix (one-hot or soft labels)
                    # matched to the shape of logits [B, NumClasses].
                    y_dense = torch.zeros_like(x)
                    y_dense.scatter_(1, target_species.unsqueeze(1), 1.0)
                else:
                    target_species = y.argmax(dim=1).to(x.device)
                    y_dense = y
                
                # Ensure g_ids is on the same device as target_species
                g_ids_local = g_ids.to(x.device)
                target_genus_ids = g_ids_local[target_species]

            return FusedASLFunction.apply(
                x, y_dense, self.logit_adjustments, g_ids, self.gamma_neg_tensor,
                target_genus_ids, self.gamma_pos, self.clip, self.eps, self.taxon_smoothing
            )
            
        # Standard Fallback
        x_adj = x + self.logit_adjustments.to(x.device) if self.logit_adjustments is not None else x
        xs_pos = torch.sigmoid(x_adj)
        xs_neg = xs_pos.clone()
        xs_neg.sub_(1.0).neg_() # Memory-efficient 1 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # plantclef: Auto-convert hard labels to one-hot for shape matching
        if y.dim() == 1:
            y_soft = torch.zeros_like(xs_pos)
            # plantclef: Guard against out-of-bounds labels when the dataloader encounters corrupted/missing class IDs
            y_valid = y.clamp(0, xs_pos.size(1) - 1)
            y_soft.scatter_(1, y_valid.unsqueeze(1), 1.0)
            
            # Mask out any invalid labels if they were out of bounds
            invalid_mask = (y < 0) | (y >= xs_pos.size(1))
            if invalid_mask.any():
                y_soft[invalid_mask] = 0.0
        else:
            y_soft = y

        # Apply Taxon Smoothing in Python if needed
        if self.taxon_smoothing > 0.0:
            y_soft = y_soft * (1 - self.taxon_smoothing) + (self.taxon_smoothing / x.size(1))

        los_pos = y_soft * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y_soft) * torch.log(xs_neg.clamp(min=self.eps))
        loss    = los_pos + los_neg
        with torch.no_grad():
            pt      = xs_pos * y_soft + xs_neg * (1 - y_soft)
            weights = (1 - pt).pow(self.gamma_pos * y_soft + self.gamma_neg * (1 - y_soft))
        loss *= weights
        return -loss.sum(dim=1).mean()

class EarlyStopping:
    """
    Advanced early stopping for PlantCLEF 2026.
    Triggers as soon as:
    1. Accuracy >= 85%
    2. In the next evaluation, the improvement is < 5%.
    """
    def __init__(self, patience: int = 7, target_threshold: float = 85.0, 
                 target_delta: float = 5.0, mode: str = 'max'):
        self.patience = patience
        self.target_threshold = target_threshold
        self.target_delta = target_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.reached_target = False
        self.prev_acc = 0.0

        if mode != 'max':
            raise ValueError("Competition early stopping requires 'max' mode (accuracy).")

    def __call__(self, val_acc: float):
        """
        Processes validation accuracy and updates stopping state.
        """
        # 1. Standard Best Tracking
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc > self.best_score:
            self.best_score = val_acc
            self.counter = 0
        else:
            self.counter += 1

        # 2. 'Target-then-Delta' Logic
        # A. Detect if we hit the high-performance threshold
        if not self.reached_target and val_acc >= self.target_threshold:
            self.reached_target = True
            print(f"\n[EarlyStopping] Target threshold ({self.target_threshold}%) reached. Monitoring next step delta...")
        
        # B. If target was reached in the PREVIOUS step, check the current improvement
        elif self.reached_target:
            improvement = val_acc - self.prev_acc
            print(f"[EarlyStopping] Current Step Delta: {improvement:+.2f}% (Limit: {self.target_delta}%)")
            
            if improvement < self.target_delta:
                print(f"[EarlyStopping] Convergence criteria met (Delta < {self.target_delta}%). Stopping.")
                self.early_stop = True

        self.prev_acc = val_acc
        
        # C. Standard Patience Fallback
        if self.counter >= self.patience:
            print(f"[EarlyStopping] Patience ({self.patience}) exhausted. Stopping.")
            self.early_stop = True
