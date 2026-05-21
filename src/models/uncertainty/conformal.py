"""Hierarchical Conformal Prediction for Taxonomic Classification.

This module provides statistically guaranteed uncertainty sets for species 
identification, with automatic roll-up to Genus when uncertainty is high.
"""

import torch
import numpy as np
import os

class HierarchicalConformalPredictor:
    """Implements APS (Adaptive Prediction Sets) with Hierarchical Aggregation."""

    def __init__(self, alpha=0.05, genus_map_path=None):
        """Initializes the predictor.
        
        Args:
            alpha: Error rate (e.g., 0.05 for 95% coverage).
            genus_map_path: Path to genus_ids.pt mapping.
        """
        self.alpha = alpha
        self.q_hat = None # Calibrated quantile
        
        if genus_map_path and os.path.exists(genus_map_path):
            self.genus_ids = torch.load(genus_map_path)
        else:
            self.genus_ids = None

    def calibrate(self, logits, labels):
        """Calibrates the model using a hold-out set.
        
        Args:
            logits: (N, 7806) tensor of raw model outputs.
            labels: (N,) tensor of ground truth indices.
        """
        print(f"[Conformal] Calibrating on {len(labels)} samples (alpha={self.alpha})...")
        
        probs = torch.softmax(logits, dim=1)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
        
        # Find the rank of the true label
        # (Where in the sorted list does the true label appear?)
        # For each sample, find where sorted_indices == label
        ranks = (sorted_indices == labels.unsqueeze(1)).nonzero()[:, 1]
        
        # Compute the cumulative sum of probabilities up to the true rank
        # This is the 'non-conformity score' for APS
        cum_probs = torch.cumsum(sorted_probs, dim=1)
        scores = cum_probs[torch.arange(len(labels)), ranks]
        
        # Compute the (1-alpha) quantile
        n = len(labels)
        self.q_hat = np.quantile(scores.float().cpu().numpy(), (1 - self.alpha) * (n + 1) / n, method='higher')
        
        print(f"[Conformal] Calibration complete. Quantile (q_hat): {self.q_hat:.4f}")

    def predict_set(self, logits):
        """Returns the set of species that satisfy the 1-alpha coverage.
        
        Returns:
            A list of lists, where each sublist contains the indices in the prediction set.
        """
        if self.q_hat is None:
            raise RuntimeError("Predictor must be calibrated before inference.")
            
        probs = torch.softmax(logits, dim=1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=1)
        
        # Prediction sets: all indices where cumulative probability <= q_hat
        # Plus the first index that exceeds q_hat (to ensure coverage)
        masks = cum_probs <= self.q_hat
        
        # Bit of trickery to include the first element that pushes it over q_hat
        # We shift the mask by 1
        masks = torch.cat([torch.ones((logits.size(0), 1), dtype=torch.bool, device=logits.device), masks[:, :-1]], dim=1)
        
        prediction_sets = []
        for i in range(logits.size(0)):
            set_indices = sorted_indices[i, masks[i]].tolist()
            prediction_sets.append(set_indices)
            
        return prediction_sets

    def predict_hierarchical(self, logits, max_set_size=5):
        """Aggregates to Genus if the prediction set is too large or diverse.
        
        Args:
            logits: Model logits.
            max_set_size: Maximum allowed species in a set before rolling up.
            
        Returns:
            A list of tuples: (Type, ID, Set)
            Type: 'SpeciesSet' or 'Genus'
        """
        species_sets = self.predict_set(logits)
        results = []
        
        for i, s_set in enumerate(species_sets):
            if len(s_set) <= max_set_size:
                results.append(('SpeciesSet', s_set))
            elif self.genus_ids is not None:
                # Find the dominant Genus in the set
                set_genera = self.genus_ids[torch.tensor(s_set)]
                unique_genera, counts = torch.unique(set_genera, return_counts=True)
                dominant_genus = unique_genera[torch.argmax(counts)].item()
                results.append(('Genus', dominant_genus, s_set))
            else:
                results.append(('LargeSpeciesSet', s_set))
                
        return results

    def save(self, path):
        """Saves the calibrated quantile."""
        state = {'q_hat': self.q_hat, 'alpha': self.alpha}
        torch.save(state, path)
        
    def load(self, path):
        """Loads a calibrated quantile."""
        state = torch.load(path)
        self.q_hat = state['q_hat']
        self.alpha = state['alpha']
