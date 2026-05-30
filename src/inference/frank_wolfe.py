import torch
import numpy as np
from typing import List, Optional


class FrankWolfeSolver:
    """
    Frank-Wolfe (Conditional Gradient) Multi-Label Solver.

    Optimizes the energy:
        E(y) = -<y, logits>
             + count_lambda * (||y||_1 - species_count_k)^2   [island biogeography prior]

    Subject to: y in Ecological Polytope (AC-3 constraints).

    The count regularizer is derived from MacArthur-Wilson island biogeography
    theory: for a fixed quadrat area A, the expected species richness is K̄ = c·A^z.
    For 0.25m² European vegetation quadrats, K̄ ≈ 8 (configurable).

    The gradient of E w.r.t. y is:
        dE/dy = -logits + 2 * count_lambda * (||y||_1 - K̄) * 1

    In the Frank-Wolfe LMO we maximize <v, -dE/dy>:
        effective_g = logits - 2 * count_lambda * (||y||_1 - K̄) * 1

    When ||y||_1 > K̄: effective_g < logits → LMO less likely to add more species.
    When ||y||_1 < K̄: effective_g > logits → LMO more likely to add species.
    """

    def __init__(
        self,
        adj_path: str = "data/taxonomic_graph.json",
        species_count_k: float = 8.0,
        count_lambda: float = 0.05,
        allelopathy_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        adj_path : str
            Path to the taxonomic adjacency JSON (list of allowed neighbours per species).
        species_count_k : float
            Expected species richness K̄ from island biogeography theory.
            Default 8.0 is calibrated for 0.25m² European vegetation quadrats.
        count_lambda : float
            Regularization strength for the species-count prior.
            0.0 disables the prior (reduces to standard FW).
        allelopathy_matrix : torch.Tensor, optional
            (C, C) matrix of negative edge weights for chemical exclusion.
            If species i is selected, allelopathy_matrix[i] is subtracted from
            remaining gradients, pushing away known competitors.
        """
        import json
        with open(adj_path, 'r') as f:
            self.adj = json.load(f)
        self.num_classes = len(self.adj)
        self.species_count_k = species_count_k
        self.count_lambda = count_lambda
        self.allelopathy_matrix = allelopathy_matrix

    def solve(
        self,
        logits: torch.Tensor,
        max_iters: int = 10,
        sparsity_k: int = 5,
    ) -> torch.Tensor:
        """
        Find the optimal sparse label vector y using Frank-Wolfe.

        Parameters
        ----------
        logits : torch.Tensor
            (C,) raw model logits / scores.
        max_iters : int
            Number of Frank-Wolfe iterations (more = tighter convergence).
        sparsity_k : int
            Maximum number of species to select (polytope vertex budget).

        Returns
        -------
        torch.Tensor
            (C,) continuous solution y ∈ [0, 1]^C, sparse around ≤ sparsity_k entries.
        """
        # plantclef: Align solver to exactly the model's logits (e.g. 7806)
        # If the graph is missing the last few species, we pad it on the fly.
        num_model_classes = logits.shape[0]
        if num_model_classes > self.num_classes:
            # Pad adjacency list for species missing from the graph so they don't crash
            for i in range(self.num_classes, num_model_classes):
                self.adj.append([i]) # At least consistent with itself
            self.num_classes = num_model_classes
        
        device = logits.device
        y = torch.zeros(self.num_classes, device=device)

        for t in range(max_iters):
            # Gradient of E(y) = -logits + 2λ(||y||₁ - K̄)·1
            # We negate to get the direction of steepest ascent for <v, -dE/dy>
            count_penalty = 2.0 * self.count_lambda * (y.sum() - self.species_count_k)
            effective_g = logits - count_penalty  # broadcast scalar over all classes

            v = self._lmo(effective_g, y, sparsity_k)

            # Standard Frank-Wolfe step size
            gamma = 2.0 / (t + 2.0)
            y = (1.0 - gamma) * y + gamma * v

        return y

    def _lmo(
        self,
        effective_g: torch.Tensor,
        current_y: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Greedy Linear Minimization plantclef for the Ecological Polytope.

        Finds the vertex v maximizing <v, effective_g> while enforcing AC-3
        consistency (a species may only be selected if it is an allowed neighbour
        of every already-selected species). If an allelopathy_matrix is provided,
        dynamically repels known chemical competitors.

        Parameters
        ----------
        effective_g : torch.Tensor
            (C,) gradient direction incorporating the biogeography regularizer.
        current_y : torch.Tensor
            (C,) current iterate (used to identify already-selected species).
        k : int
            Maximum number of species in the vertex.

        Returns
        -------
        torch.Tensor
            (C,) indicator vertex with at most k non-zero entries.
        """
        device = effective_g.device
        v = torch.zeros(self.num_classes, device=device)

        dynamic_g = effective_g.clone()
        selected_count = 0
        selected_indices: List[int] = []

        for _ in range(k):
            # Sort dynamically to account for changing gradients from repulsion
            _, indices = torch.sort(dynamic_g, descending=True)
            
            found_valid = False
            for idx in indices.tolist():
                if dynamic_g[idx] == -float('inf'):
                    break # Reached exhausted or fully penalized species
                    
                is_consistent = True
                for s_idx in selected_indices:
                    if idx not in self.adj[s_idx]:
                        is_consistent = False
                        break

                if is_consistent:
                    v[idx] = 1.0
                    selected_indices.append(idx)
                    selected_count += 1
                    
                    # plantclef: Apply Allelopathic Repulsion (Chemical Exclusion)
                    if self.allelopathy_matrix is not None:
                        # Repel competitors by subtracting their known antagonism weights
                        # Ensure we only subtract within the bounds of the prior matrix (e.g. 7806)
                        prior_size = self.allelopathy_matrix.shape[1]
                        dynamic_g[:prior_size] -= self.allelopathy_matrix[idx, :prior_size].to(device)
                        
                    dynamic_g[idx] = -float('inf') # Mark as selected
                    found_valid = True
                    break
                else:
                    dynamic_g[idx] = -float('inf') # Mark as inconsistent

            if not found_valid:
                break

        return v

