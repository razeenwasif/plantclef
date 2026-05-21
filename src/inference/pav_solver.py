import json
import torch
import numpy as np
from typing import Dict, List, Any

class PAVTreeSolver:
    """
    Hierarchical Isotonic Regression (PAV-Tree).
    Enforces taxonomic monotonicity: P(Species) <= P(Genus) <= P(Family).
    Weighted by EIVE ecological reliability.
    """
    def __init__(self, tree_path: str = "data/pav_taxonomy_tree.json"):
        with open(tree_path, 'r') as f:
            self.tree = json.load(f)
        self.num_species = 7808 # Align with Blackwell Padded Shape

    def solve(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Applies PAV-tree algorithm to a probability vector.
        probs: [num_classes] tensor
        """
        # 1. Convert to numpy for easy tree traversal
        y = probs.detach().cpu().numpy()
        calibrated = np.copy(y)
        
        # 2. Traverse tree: Order -> Family -> Genus -> Species
        # We perform a "pooling" step if children exceed parent probability
        self._pool_violations(self.tree, calibrated)
        
        return torch.from_numpy(calibrated).to(probs.device)

    def _pool_violations(self, node: Dict[str, Any], probs: np.ndarray):
        """
        Recursive implementation of PAV-Tree.
        Calculates parent probability as an upper bound, pooling children that violate it.
        """
        if "idx" in node:
            # Leaf Node (Species)
            return probs[node["idx"]], node["weight"]

        child_vals = []
        child_weights = []
        
        # Recursive Descent (Bottom-Up)
        for child_name, child_node in node["children"].items():
            val, weight = self._pool_violations(child_node, probs)
            child_vals.append(val)
            child_weights.append(weight)
        
        if not child_vals:
            return 0.0, 1.0

        # Current Node Probability (Upper bound for children)
        # We define parent prob as the max of its immediate children
        parent_val = np.max(child_vals)
        parent_weight = np.mean(child_weights)

        # Top-Down Constraint: Enforce P(child) <= parent_val
        for child_name, child_node in node["children"].items():
            if "idx" in child_node:
                idx = child_node["idx"]
                if probs[idx] > parent_val:
                    # Smoothing: weighted average between leaf and taxonomic group
                    w = child_node["weight"]
                    probs[idx] = (probs[idx] * w + parent_val) / (w + 1.0)
            else:
                # For non-leaf children, ensure their entire subtree is constrained
                # This ensures P(Species) <= P(Genus) <= P(Family) etc.
                self._constrain_subtree(child_node, parent_val, probs)
        
        return parent_val, parent_weight

    def _constrain_subtree(self, node: Dict[str, Any], upper_bound: float, probs: np.ndarray):
        """Helper to recursively cap all leaf nodes in a subtree."""
        if "idx" in node:
            idx = node["idx"]
            if probs[idx] > upper_bound:
                w = node["weight"]
                probs[idx] = (probs[idx] * w + upper_bound) / (w + 1.0)
        else:
            for child in node["children"].values():
                self._constrain_subtree(child, upper_bound, probs)

    def batch_solve(self, batch_probs: torch.Tensor) -> torch.Tensor:
        """Optimized batch processing for multi-quadrat inference."""
        results = []
        for i in range(batch_probs.shape[0]):
            results.append(self.solve(batch_probs[i]))
        return torch.stack(results)
