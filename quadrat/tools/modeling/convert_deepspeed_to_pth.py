import torch
import os
import sys
from collections import OrderedDict

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

def convert():
    # 1. Paths
    ds_path = "models/cuda_v3/phase2_checkpoint/checkpoint_latest/mp_rank_00_model_states.pt"
    out_path = "models/cuda_v3/phase2_final_recovered.pth"
    
    if not os.path.exists(ds_path):
        print(f"Error: DeepSpeed checkpoint not found at {ds_path}")
        return

    print(f"Converting {ds_path} to standard .pth format...")
    
    # 2. Load
    checkpoint = torch.load(ds_path, map_location='cpu', weights_only=True)
    print(f"Loaded keys: {checkpoint.keys()}")
    
    # DeepSpeed usually wraps state_dict in 'module' or returns it directly
    state_dict = checkpoint['module'] if 'module' in checkpoint else checkpoint
    
    # 3. Strip 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v.float() # Convert to FP32 for inference stability
    
    # 4. Save as a standard PyTorch state_dict
    save_obj = {
        'model_state': new_state_dict,
        'epoch': 24,
        'best_val_acc': 57.81
    }
    
    torch.save(save_obj, out_path)
    print(f"Success! Model saved to {out_path}")
    print(f"Weight count: {len(new_state_dict)}")

if __name__ == "__main__":
    convert()
