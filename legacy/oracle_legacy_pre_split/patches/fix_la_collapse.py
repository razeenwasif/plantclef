path = 'phases/inference/pipeline.py'
with open(path, 'r') as f:
    content = f.read()

# 1. Update Logit Adjustment to be MUCH more gentle
content = content.replace("logits = _logit_adjustment(logits, counts_path, mapping_path, tau=1.0)", "logits = _logit_adjustment(logits, counts_path, mapping_path, tau=0.1)")

# 2. Fix the species_ids vs eval() issue - ensure submission has quotes
# We will do this in the post-processing script later

# 3. Add a check for NaNs just in case
content = content.replace("probs = torch.sigmoid(logits)", "logits = torch.nan_to_num(logits, nan=0.0)\n    probs = torch.sigmoid(logits)")

with open(path, 'w') as f:
    f.write(content)
