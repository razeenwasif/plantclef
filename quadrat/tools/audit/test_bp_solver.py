import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_loopy_belief_propagation():
    """Verify that the Loopy BP solver produces coherent marginals via soft consensus."""
    try:
        from data_auditor import loopy_belief_propagation
    except ImportError:
        print("Skipping BP test: data_auditor extension not installed.")
        return

    # Setup: 2 tiles, 3 species
    # Tile 0: Species 0 is strong (0.9)
    # Tile 1: Uncertain, but Species 1 has a tiny bit of signal (0.1)
    # Species 0 and 1 are "friends" (strong co-occurrence)
    
    num_tiles = 2
    num_species = 3
    
    # Unary potentials (probabilities)
    unary = np.array([
        [0.9, 0.05, 0.05], # Tile 0
        [0.05, 0.1, 0.85]  # Tile 1 (Species 2 is dominant initially)
    ], dtype=np.float32)
    
    # Pairwise: Species 0 co-occurs with Species 1
    indices = [(0, 1)]
    weights = [0.9] # Strong co-occurrence probability

    # Solve
    # We run 5 iterations
    refined_marginals = loopy_belief_propagation(unary, indices, weights, 5)

    assert refined_marginals.shape == (num_tiles, num_species)
    
    # Ensure they sum to 1 (normalized by the solver)
    sums = refined_marginals.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)
    
    # Verify Soft Consensus:
    # Tile 0's strong Species 0 should have boosted Tile 1's Species 1
    # because they are "friends" (co-occur).
    initial_tile1_s1 = unary[1, 1]
    refined_tile1_s1 = refined_marginals[1, 1]
    
    print(f"Tile 1, Species 1: Initial={initial_tile1_s1:.4f}, Refined={refined_tile1_s1:.4f}")
    
    # In a proper BP, refined should be significantly higher than initial
    assert refined_tile1_s1 > initial_tile1_s1
    print("✅ Loopy BP solver verified: Soft consensus successfully boosted related species.")

if __name__ == "__main__":
    test_loopy_belief_propagation()
