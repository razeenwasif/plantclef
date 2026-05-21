"""
Botanical Trait Distillation Head.

This module implements a multi-task head that predicts structural 
botanical traits (leaf shape, color, etc.) to guide taxonomic alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class BotanicalTraitHead(nn.Module):
    """
    Predicts a vector of botanical traits from visual features.

    Parameters
    ----------
    in_features : int
        Dimension of the fused ensemble features (e.g., 1536).
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        
        # We define 5 categories of traits based on our extraction script
        self.categories = {
            "leaf_shape": 12, # ovate, lanceolate, cordate, etc.
            "phyllotaxy": 4,  # alternate, opposite, whorled, basal
            "flower_color": 8, # yellow, white, blue, red, etc.
            "inflorescence": 6, # spike, umbel, cyme, etc.
            "stem_type": 5    # woody, herbaceous, succulent, etc.
        }
        
        self.total_traits = sum(self.categories.values())
        
        # Lightweight bottleneck head with stabilized output
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.total_traits),
            nn.LayerNorm(self.total_traits) # Final stabilization
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predicts traits.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, in_features).

        Returns
        -------
        results : Dict[str, torch.Tensor]
            A dictionary of logits per category.
        """
        logits = self.head(x)
        
        # Split logits into categories
        results = {}
        start = 0
        for name, size in self.categories.items():
            results[name] = logits[:, start:start+size]
            start += size
            
        return results

class TraitDistillationLoss(nn.Module):
    """
    Multi-task loss for botanical trait alignment.

    Parameters
    ----------
    alpha : float, optional
        Weight of the distillation loss vs classification loss (default is 0.1).
    """
    
    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred_traits: Dict[str, torch.Tensor], target_traits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes Cross-Entropy for each trait category.

        Parameters
        ----------
        pred_traits : Dict[str, torch.Tensor]
            Dictionary of logits from BotanicalTraitHead.
        target_traits : Dict[str, torch.Tensor]
            Dictionary of target label indices per category.

        Returns
        -------
        loss : torch.Tensor
            The weighted distillation loss.
        """
        loss = 0.0
        count = 0
        # For this prototype, we assume target_traits is a dictionary of label indices
        for category, logits in pred_traits.items():
            if category in target_traits:
                loss += F.cross_entropy(logits, target_traits[category], ignore_index=-1)
                count += 1
                
        if count == 0:
            return torch.tensor(0.0, device=next(iter(pred_traits.values())).device)
            
        return (loss / count) * self.alpha
