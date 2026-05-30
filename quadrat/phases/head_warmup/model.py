"""Phase 2a model — lightweight MLP head only.

Owned exclusively by p2a_warmup. No backbone. No LoRA. No image pipeline.
Input is the pre-computed [B, 3328] feature vector from Phase 1.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class WarmupHead(nn.Module):
    """Residual MLP trained on cached backbone features.

    Identical architecture to the ResidualMLP in the legacy ensemble but defined
    here independently so p2a changes cannot affect any other phase.
    """

    def __init__(self, in_features: int = 3328, hidden_features: int = 2048,
                 out_features: int = 7808, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1      = nn.Linear(in_features, hidden_features)
        self.fc_final = nn.Linear(hidden_features, out_features)
        self.ln1      = nn.LayerNorm(hidden_features)
        self.gelu     = nn.GELU()
        self.res_block = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
        )
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.fc1.weight.dtype)
        x = self.gelu(self.ln1(self.fc1(x)))
        x = x + self.res_block(x)
        return self.fc_final(self.dropout(x))
