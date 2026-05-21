import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FusedLoRAFunction(torch.autograd.Function):
    """
    SOTA Fused LoRA Kernel logic.
    Forward: y = W_fused * x (1 GEMM, 1 HBM read)
    Backward: Calculate grad_A and grad_B using x, A, B (Standard LoRA backprop)
    """
    @staticmethod
    def forward(ctx, x, W_fused, A, B, scaling):
        ctx.save_for_backward(x, A, B)
        ctx.scaling = scaling
        # Theoretical Limit: Single HBM pass for the combined weight
        return F.linear(x, W_fused)

    @staticmethod
    def backward(ctx, grad_output):
        x, A, B = ctx.saved_tensors
        scaling = ctx.scaling
        
        # grad_W_fused is not needed (W is frozen)
        # We only calculate grad_A and grad_B
        
        # grad_output: [B, N, D_out]
        # x: [B, N, D_in]
        # A: [R, D_in], B: [D_out, R]
        
        # Flatten if necessary
        if grad_output.dim() == 3:
            grad_output = grad_output.reshape(-1, grad_output.size(-1))
            x = x.reshape(-1, x.size(-1))
            
        # Standard LoRA Backward (Sparse)
        # grad_B = grad_output.T @ (x @ A.T)
        grad_B = torch.matmul(grad_output.t(), torch.matmul(x, A.t())) * scaling
        # grad_A = (grad_output @ B).T @ x
        grad_A = torch.matmul(torch.matmul(grad_output, B).t(), x) * scaling
        
        return None, None, grad_A, grad_B, None

class FusedLoRALinear(nn.Module):
    """
    A Blackwell-optimized Linear layer with fused LoRA weights.
    """
    def __init__(self, base_layer: nn.Linear, r: int, lora_alpha: int, lora_dropout: float):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r
        
        # Parameters to be trained
        self.lora_A = nn.Parameter(base_layer.weight.new_zeros((r, base_layer.in_features)))
        self.lora_B = nn.Parameter(base_layer.weight.new_zeros((base_layer.out_features, r)))
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # The Fused Weight Buffer (Shadow copy of W + B@A)
        # Initialized with base weights
        self.register_buffer("W_fused", base_layer.weight.data.detach().clone())
        
        # Init LoRA
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def update_fusion(self):
        """Perform the in-place fusion: W_fused = W + (B @ A) * scaling"""
        # Optimized for Blackwell: Single GEMM for fusion update
        # This is called once per step, not once per forward.
        self.W_fused.copy_(self.base_layer.weight.data)
        self.W_fused.addmm_(self.lora_B.data, self.lora_A.data, alpha=self.scaling)

    def forward(self, x):
        return FusedLoRAFunction.apply(
            self.lora_dropout(x), 
            self.W_fused, 
            self.lora_A, 
            self.lora_B, 
            self.scaling
        )

import math
