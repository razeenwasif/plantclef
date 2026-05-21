"""Unified Health Check for PlantCLEF 2026 Consolidated Branch.

Performs a multi-layered validation:
1. Foundation: PyTorch, CUDA, and DALI availability.
2. Architecture: PlantEnsemble and model component integrity.
3. Research Modules: Conformal Prediction and Depth Estimation.
4. Pipeline: Inference orchestration and config validation.
"""

import os
import sys
import torch
import numpy as np
from typing import Optional

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if os.path.join(root_dir, "src") not in sys.path:
    sys.path.insert(0, os.path.join(root_dir, "src"))

def run_diagnostic() -> None:
    """
    Executes a multi-layered validation of the project's foundation, architecture, research modules, and pipeline.

    Checks for PyTorch/CUDA/DALI availability, model ensemble integrity,
    research module accessibility, and inference pipeline configuration.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    print("="*60)
    print(" PLANTCLEF 2026: UNIFIED BRANCH HEALTH CHECK")
    print("="*60)

    # --- 1. Foundation Check ---
    print("\n[1/4] FOUNDATION (PyTorch & CUDA)")
    try:
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA Available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - Device:          {torch.cuda.get_device_name(0)}")
            print(f"  - VRAM Total:      {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        
        import nvidia.dali
        print(f"  - NVIDIA DALI:     OK")
    except Exception as e:
        print(f"  !! Foundation check FAILED: {e}")

    # --- 2. Architecture Check ---
    print("\n[2/4] ARCHITECTURE (7,806 Class Ensemble)")
    try:
        from src.models.ensemble import PlantEnsemble
        model = PlantEnsemble(num_classes=7806)
        print(f"  - Ensemble Init:   OK")
        
        from src.models.layers.gcn import EcologicalGCNHead
        from src.models.layers.distillation import BotanicalTraitHead
        print(f"  - Symbolic Heads:  OK")
    except Exception as e:
        print(f"  !! Architecture check FAILED: {e}")

    # --- 3. Research Modules ---
    print("\n[3/4] RESEARCH MODULES")
    try:
        from src.models.uncertainty.conformal import HierarchicalConformalPredictor
        print(f"  - Conformal APS:   OK")
        
        from src.models.biomass.depth_pipeline import MonocularBiomassEstimator
        print(f"  - Biomass Depth:   OK")
        
        from src.data.sam_extractor import main as sam_main
        print(f"  - Generative SAM:  OK")
    except Exception as e:
        print(f"  !! Research module check FAILED: {e}")

    # --- 4. Inference Pipeline ---
    print("\n[4/4] INFERENCE PIPELINE")
    try:
        from src.inference.pipeline import InferencePipeline
        from src.inference.config import ModelConfig
        print(f"  - Pipeline Logic:  OK")
        
        adj_path = "/workspace/plantclef/processed/logit_adj.npy"
        if os.path.exists(adj_path):
            print(f"  - Calibration Map: OK")
        else:
            print(f"  -- Warning: logit_adj.npy not found (needed for F1 fix)")
    except Exception as e:
        print(f"  !! Pipeline check FAILED: {e}")

    print("\n" + "="*60)
    print(" DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_diagnostic()
