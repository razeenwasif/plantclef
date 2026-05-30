import torch
import os

def recover_model():
    input_path = 'models/archive/final/mp_rank_00_model_states.pt'
    output_path = 'models/archive/recovered_75pct_accuracy.pth'
    
    print(f"Loading {input_path}...")
    ckpt = torch.load(input_path, map_location='cpu')
    
    if 'module' in ckpt:
        state_dict = ckpt['module']
        # Clean up DeepSpeed prefixes if they exist
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith('_orig_mod.'):
                new_k = new_k.replace('_orig_mod.', '')
            new_state_dict[new_k] = v
            
        torch.save(new_state_dict, output_path)
        print(f"Successfully recovered model weights to {output_path}")
        print(f"Metadata - Epoch: {ckpt.get('epoch')}, Best Acc: {ckpt.get('best_val_acc')}")
    else:
        print("Error: 'module' key not found in checkpoint.")

if __name__ == "__main__":
    recover_model()
