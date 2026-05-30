import torch
import os

BASE_DIR = "/workspace/PlantCLEF2026"
MODEL_PATH = os.path.join(BASE_DIR, "models/best_calibrated/mp_rank_00_model_states.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models/final/mp_rank_00_model_states.pt")

if os.path.exists(MODEL_PATH):
    print(f"Loading {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    state_dict = ckpt['module'] if isinstance(ckpt, dict) and 'module' in ckpt else ckpt
    
    keys = list(state_dict.keys())
    print(f"Total keys: {len(keys)}")
    print("First 20 keys:")
    for k in keys[:20]:
        print(f"  {k}")
    
    print("\nKeys containing 'classifier':")
    classifier_keys = [k for k in keys if 'classifier' in k]
    for k in classifier_keys:
        print(f"  {k} : {state_dict[k].shape}")
        
    print("\nKeys containing 'phase1_head':")
    p1_keys = [k for k in keys if 'phase1_head' in k]
    for k in p1_keys:
        print(f"  {k} : {state_dict[k].shape}")
else:
    print(f"File not found: {MODEL_PATH}")
