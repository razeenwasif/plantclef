
import torch
import os
import sys
from pathlib import Path

def audit_checkpoint(path):
    print(f"\n--- Auditing: {path} ---")
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist.")
        return

    try:
        # Use weights_only=False to allow full architectural audit if needed, 
        # but map to CPU to avoid VRAM fragmentation.
        state = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle different save formats
        if isinstance(state, dict):
            if 'model_state' in state:
                state = state['model_state']
            elif 'state_dict' in state:
                state = state['state_dict']
            elif 'module' in state:
                state = state['module']
        
        print(f"Loaded successfully. Found {len(state)} parameter tensors.")

        # 1. Structural Integrity (Check for expected PLANTCLEF components)
        expected_keys = [
            "bioclip.backbone", 
            "dinov3.backbone", 
            "convnext.backbone",
            "gating_network", 
            "species_classifier.theta"
        ]
        
        found_components = []
        for component in expected_keys:
            if any(k.startswith(component) or component in k for k in state.keys()):
                found_components.append(component)
        
        print(f"Architectural Components Found: {', '.join(found_components)}")

        # 2. Numerical Integrity (Check for NaNs/Infs)
        nan_count = 0
        inf_count = 0
        zero_count = 0
        total_elements = 0
        
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            
            total_elements += tensor.numel()
            if torch.isnan(tensor).any():
                nan_count += 1
                print(f"  [CRITICAL] NaN detected in: {name}")
            if torch.isinf(tensor).any():
                inf_count += 1
                print(f"  [CRITICAL] Inf detected in: {name}")
            
            # Check for dead layers (all zeros)
            if torch.sum(torch.abs(tensor)) < 1e-12:
                zero_count += 1
                # Only print if it's a large weight matrix, ignore small biases or masks
                if tensor.numel() > 1000:
                    print(f"  [WARNING] Dead layer detected (all zeros): {name}")

        if nan_count == 0 and inf_count == 0:
            print("Numerical Health: PASS (No NaNs or Infs detected).")
        else:
            print(f"Numerical Health: FAIL ({nan_count} NaNs, {inf_count} Infs found).")

        if zero_count > 0:
            print(f"Sparsity Note: {zero_count} layers are near-zero (Dead).")
        
        # 3. Check for LoRA presence
        lora_keys = [k for k in state.keys() if "lora_" in k]
        if lora_keys:
            print(f"LoRA Status: ACTIVE ({len(lora_keys)} adapter tensors found).")
        else:
            print("LoRA Status: INACTIVE (Full fine-tuning or frozen backbones detected).")

    except Exception as e:
        print(f"Error during audit: {e}")

if __name__ == "__main__":
    expert_paths = [
        "models/cuda_deep_sat/expert_bioclip_512/phase2_checkpoint_ep7_step_final.pth",
        "models/cuda_deep_sat/expert_dinov3_512/phase2_checkpoint_ep7_step_final.pth"
    ]
    
    for p in expert_paths:
        audit_checkpoint(p)
