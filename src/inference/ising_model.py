import torch
import numpy as np
from typing import Optional

class SpinGlassSolver:
    """
    Statistical Mechanics / Ising Model Solver for Multi-Label Classification.
    
    Treats species presence as binary spins (0 or 1).
    External Magnetic Field (h): Neural network logits.
    Interaction Energy (J): Ecological co-occurrence matrix.
    
    Finds the lowest energy configuration (Ground State) using Fast Simulated Annealing.
    """
    def __init__(self,
                 adj_path: str = "data/taxonomic_graph.json",
                 num_classes: int = 7806,
                 j_strength: float = 0.5,
                 j_matrix_path: str | None = None):
        """
        Two ways to populate the interaction matrix J:

        1. **j_matrix_path** (preferred when offline priors exist) — load a pre-built
           [C, C] tensor from disk. Negative entries = repulsion (e.g., taxonomically
           distant or ecologically incompatible pairs). This path is what the new
           offline-prior pipeline uses; it carries far richer signal than the sparse
           graph-edge coupling.

        2. **adj_path** (legacy fallback) — build sparse symmetric J from a
           taxonomic adjacency list, +j_strength on every edge (rewards co-occurrence
           between genus/family neighbors).

        The energy is E = -h^T s - s^T J s, so positive J ⇒ attraction, negative J ⇒
        repulsion — both signs work in the same Metropolis loop below.
        """
        import json, os
        self.num_classes = num_classes
        self.j_strength = j_strength

        if j_matrix_path is not None and os.path.exists(j_matrix_path):
            self.J = torch.load(j_matrix_path, map_location="cpu", weights_only=True).float()
            # Symmetrize defensively — energy formula assumes J = J^T.
            if not torch.allclose(self.J, self.J.T, atol=1e-5):
                self.J = 0.5 * (self.J + self.J.T)
            print(f"[Ising] Loaded precomputed J from {j_matrix_path} "
                  f"(shape={tuple(self.J.shape)}, range=[{self.J.min().item():.3f}, {self.J.max().item():.3f}]).")
            return

        # Legacy path: build from taxonomic adjacency.
        with open(adj_path, 'r') as f:
            adj = json.load(f)
        self.J = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(min(len(adj), num_classes)):
            for j in adj[i]:
                if j < num_classes:
                    # Positive interaction energy for allowed co-occurrence
                    self.J[i, j] = j_strength
                    self.J[j, i] = j_strength

    def solve(self, logits: torch.Tensor, init_k: int = 10, max_iters: int = 1000, initial_temp: float = 1.0, cooling_rate: float = 0.99) -> torch.Tensor:
        """
        Runs Simulated Annealing to find the ground state of the quadrat.
        
        Parameters
        ----------
        logits : torch.Tensor
            (C,) raw model logits / scores (acts as external magnetic field).
        init_k : int
            Number of initial 'spins' set to 1 based on top logits.
        max_iters : int
            Maximum iterations for the simulated annealing process.
            
        Returns
        -------
        torch.Tensor
            (C,) binary vector representing the lowest energy configuration.
        """
        # ORACLE: Slice logits to exactly self.num_classes to match J
        if logits.shape[0] > self.num_classes:
            h = logits[:self.num_classes].clone()
        else:
            h = logits.clone()
            
        device = h.device
        J = self.J.to(device)
        
        # Initialize spins (s in {0, 1}) using Top-K
        s = torch.zeros(self.num_classes, device=device)
        _, top_idx = torch.topk(h, min(init_k, self.num_classes))
        s[top_idx] = 1.0
        
        T = initial_temp
        
        # Fast simulated annealing
        for i in range(max_iters):
            # Pick a random spin to flip
            idx = torch.randint(0, self.num_classes, (1,), device=device).item()
            
            current_spin = s[idx].item()
            new_spin = 1.0 - current_spin
            
            # Calculate Delta E
            # E = -h^T s - s^T J s
            # If s_i goes 0 -> 1: Delta E = -h_i - 2 * sum_j J_{ij} s_j
            # If s_i goes 1 -> 0: Delta E = h_i + 2 * sum_j J_{ij} s_j
            interaction_term = 2.0 * torch.dot(J[idx], s).item()
            
            if current_spin == 0.0:
                delta_e = -h[idx].item() - interaction_term
            else:
                delta_e = h[idx].item() + interaction_term
                
            # Metropolis-Hastings acceptance criterion
            if delta_e < 0:
                s[idx] = new_spin
            else:
                prob = np.exp(-delta_e / T)
                if np.random.rand() < prob:
                    s[idx] = new_spin
                    
            T *= cooling_rate
            if T < 1e-4:
                break
                
        return s
