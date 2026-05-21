path = 'phases/inference/pipeline.py'
with open(path, 'r') as f:
    content = f.read()

# 1. Add Logit Adjustment function
la_func = """
def _logit_adjustment(logits: torch.Tensor, counts_path: str, mapping_path: str, tau: float = 1.0) -> torch.Tensor:
    import pandas as pd
    try:
        counts_df = pd.read_csv(counts_path)
        mapping_df = pd.read_csv(mapping_path, header=None)
        
        mapping_ids = mapping_df[0].tolist()
        counts_dict = dict(zip(counts_df['species_id'], counts_df['count']))
        
        # Build prior vector in mapping order
        priors = []
        for sid in mapping_ids:
            # Missing IDs get count 1 (neutral)
            count = counts_dict.get(int(sid), 1)
            priors.append(float(count))
            
        priors = torch.tensor(priors, device=logits.device, dtype=torch.float32)
        priors = priors / priors.sum()
        
        # Logit adjustment: z - tau * log(pi)
        # Note: we use log(pi + epsilon)
        adjusted = logits - tau * torch.log(priors + 1e-9)
        return adjusted
    except Exception as e:
        print(f"[Inference] Logit Adjustment failed: {e}")
        return logits
"""

# Insert before _postprocess
content = content.replace("def _postprocess", la_func + "\ndef _postprocess")

# 2. Integrate into _postprocess
old_probs = "    probs = torch.sigmoid(logits)"
new_probs = """    # ORACLE Day 1: Logit Adjustment
    # Counts path from config
    counts_path = "/workspace/plantclef/processed/species_train_counts.csv"
    mapping_path = cfg.species_mapping
    logits = _logit_adjustment(logits, counts_path, mapping_path, tau=1.0)
    
    probs = torch.sigmoid(logits)"""
content = content.replace(old_probs, new_probs)

with open(path, 'w') as f:
    f.write(content)
