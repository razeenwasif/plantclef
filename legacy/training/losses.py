import torch
import torch.nn as nn
from torch.autograd import Function

try:
    import plantclef_ext
except ImportError:
    plantclef_ext = None


class FusedASLFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, logit_adjustments, gamma_pos, gamma_neg, clip, eps):
        # Force to float32 for CUDA kernel compatibility and numerical precision
        logits_f32 = logits.float()
        targets_f32 = targets.float()
        adj_f32 = logit_adjustments.float()

        # fused_asl_forward returns a tensor of losses (batch_size,)
        losses = plantclef_ext.fused_asl_forward(
            logits_f32, targets_f32, adj_f32, gamma_pos, gamma_neg, clip, eps
        )[0]
        
        ctx.save_for_backward(logits_f32, targets_f32, adj_f32)
        ctx.gamma_pos = gamma_pos
        ctx.gamma_neg = gamma_neg
        ctx.clip = clip
        ctx.eps = eps
        ctx.original_dtype = logits.dtype
        
        return losses.mean().to(logits.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, logit_adjustments = ctx.saved_tensors
        
        grad_logits = plantclef_ext.fused_asl_backward(
            grad_output.float(), logits, targets, logit_adjustments,
            ctx.gamma_pos, ctx.gamma_neg, ctx.clip, ctx.eps
        )
        
        return grad_logits.to(ctx.original_dtype), None, None, None, None, None, None


class LogitAdjustmentLoss(nn.Module):
    """
    Shifts logits by log(prior) to demand larger margins for common classes
    and ease requirements for rare ones -- critical for 7,800-class long-tail.
    """
    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        priors = (counts + 1) / (counts.sum() + len(counts)) # Laplacian smoothing
        self.adjustment = (tau * torch.log(priors)).to('cuda')
        self.criterion  = nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.criterion(x + self.adjustment, y)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss: aggressively down-weights easy negatives across 7,800 classes.
    Optionally uses a custom CUDA kernel to reduce VRAM overhead by 3x.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, 
                 logit_adjustments=None, use_fused=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps  = eps
        self.use_fused = use_fused and (plantclef_ext is not None)
        
        if logit_adjustments is not None:
            self.register_buffer('logit_adjustments', logit_adjustments)
        else:
            self.register_buffer('logit_adjustments', None)

    def forward(self, x, y):
        if self.use_fused and self.logit_adjustments is not None:
            return FusedASLFunction.apply(
                x, y, self.logit_adjustments, 
                self.gamma_pos, self.gamma_neg, self.clip, self.eps
            )
            
        # Fallback to standard PyTorch implementation
        x_adj = x + self.logit_adjustments if self.logit_adjustments is not None else x
        xs_pos = torch.sigmoid(x_adj)
        xs_neg = 1 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y       * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss    = los_pos + los_neg
        with torch.no_grad():
            pt      = xs_pos * y + xs_neg * (1 - y)
            weights = (1 - pt).pow(self.gamma_pos * y + self.gamma_neg * (1 - y))
        loss *= weights
        return -loss.sum(dim=1).mean()


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter    = 0
