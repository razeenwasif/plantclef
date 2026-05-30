import torch
import os
import argparse
from collections import OrderedDict

def merge_lora_to_base(checkpoint_path, output_path, r=512, alpha=1024):
    print(f"[Merge] Loading checkpoint: {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location="cpu")
    
    # Unwrap
    is_wrapped = False
    if isinstance(sd, dict) and "model_state" in sd:
        is_wrapped = True
        weights = sd["model_state"]
    else:
        weights = sd

    # 1. Clean prefixes
    clean_sd = OrderedDict()
    for k, v in weights.items():
        nk = k.replace("_orig_mod.", "").replace("module.", "")
        clean_sd[nk] = v

    # 2. Identify LoRA pairs
    # Logic: find keys ending in .lora_A, then look for corresponding .lora_B and .base_layer.weight
    lora_keys = [k for k in clean_sd.keys() if k.endswith(".lora_A")]
    print(f"[Merge] Found {len(lora_keys)} LoRA layers to merge.")
    
    scaling = alpha / r
    merged_count = 0
    
    final_sd = OrderedDict()
    # First, copy non-lora keys
    for k, v in clean_sd.items():
        if ".lora_" not in k and ".base_layer." not in k:
            final_sd[k] = v

    # Now merge LoRA
    for k_a in lora_keys:
        prefix = k_a.rsplit(".lora_A", 1)[0]
        k_b = prefix + ".lora_B"
        # The base weight can be either 'prefix.weight' or 'prefix.base_layer.weight'
        k_base = prefix + ".base_layer.weight"
        if k_base not in clean_sd:
             k_base = prefix + ".weight"
        
        if k_b in clean_sd and k_base in clean_sd:
            W_base = clean_sd[k_base]
            A = clean_sd[k_a]
            B = clean_sd[k_b]
            
            # W_new = W_base + (B @ A) * scaling
            # Note: A is [r, in], B is [out, r]
            dW = (B @ A) * scaling
            
            # Handle potential dtypes (bf16 vs float32)
            W_merged = W_base.float() + dW.float()
            final_sd[prefix + ".weight"] = W_merged.to(W_base.dtype)
            
            # Also handle bias if it was in a base_layer
            k_base_bias = prefix + ".base_layer.bias"
            if k_base_bias in clean_sd:
                final_sd[prefix + ".bias"] = clean_sd[k_base_bias]
                
            merged_count += 1
        else:
            print(f"  [Warn] Missing base or B for {prefix}")

    print(f"[Merge] Successfully merged {merged_count} LoRA layers into base weights.")
    
    if is_wrapped:
        sd["model_state"] = final_sd
        torch.save(sd, output_path)
    else:
        torch.save(final_sd, output_path)
    print(f"[Merge] Saved clean model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--r", type=int, default=512)
    parser.add_argument("--alpha", type=int, default=1024)
    args = parser.parse_args()
    merge_lora_to_base(args.input, args.output, args.r, args.alpha)
