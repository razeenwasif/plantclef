import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. Loading and Exploring the Data
# =============================================================================

def inspect_batch(images, labels):
    """
    Prints the input shape, target shape, dtypes, devices, value ranges, 
    and unique labels for a batch.
    """
    print("--- Batch Inspection ---")
    print(f"Images Shape:  {images.shape}")
    print(f"Images Dtype:  {images.dtype}")
    print(f"Images Device: {images.device}")
    print(f"Images Range:  Min: {images.min().item():.3f}, Max: {images.max().item():.3f}")
    
    print(f"Labels Shape:  {labels.shape}")
    print(f"Labels Dtype:  {labels.dtype}")
    print(f"Labels Device: {labels.device}")
    
    unique_labels = torch.unique(labels)
    print(f"Unique Labels in Batch: {unique_labels.tolist()}")
    print("-" * 24)

def validate_class_labels(labels, num_classes):
    """
    Verifies that labels are integer class indices in the range [0, num_classes - 1].
    Catches wrong dtypes, negative labels, and out-of-range labels.
    """
    if not torch.is_floating_point(labels) and labels.dtype not in [torch.int64, torch.int32]:
        raise TypeError(f"Labels should be integers (int64 or int32), got {labels.dtype}")
    
    if (labels < 0).any():
        raise ValueError(f"Found negative labels! Min label: {labels.min().item()}")
        
    if (labels >= num_classes).any():
        raise ValueError(f"Found labels out of range! Max label: {labels.max().item()}, Expected max: {num_classes - 1}")
    
    print(f"Labels Validated: All {labels.numel()} labels are within [0, {num_classes-1}].")

def detect_missing_labels(all_labels, expected_num_classes):
    """
    Flags missing classes or suspicious label collapse in a full dataset pass.
    """
    unique_classes = torch.unique(all_labels)
    num_unique = len(unique_classes)
    
    print(f"Detected {num_unique} unique classes out of {expected_num_classes} expected.")
    
    if num_unique < expected_num_classes:
        missing_count = expected_num_classes - num_unique
        print(f"[Warning] Missing {missing_count} classes in this split!")
        # Optional: Print missing IDs if the count isn't too huge
        if missing_count < 50:
            expected_set = set(range(expected_num_classes))
            actual_set = set(unique_classes.tolist())
            missing_set = expected_set - actual_set
            print(f"Missing Class IDs: {list(missing_set)}")
    elif num_unique > expected_num_classes:
        print("[Error] Found MORE unique classes than expected. Label corruption likely.")

def plot_label_distribution(all_labels, num_classes, save_path="debug/label_dist.png"):
    """
    Shows how many examples belong to each class to spot imbalance or corruption.
    """
    counts = torch.bincount(all_labels, minlength=num_classes).cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(sorted(counts, reverse=True))
    plt.title("Label Distribution (Sorted by Frequency)")
    plt.xlabel("Class Rank")
    plt.ylabel("Number of Samples")
    plt.yscale('log') # Log scale is best for long-tail datasets like PlantCLEF
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Label distribution plot saved to {save_path}")

# =============================================================================
# 2. Mathematical Correctness vs Numerical Stability
# =============================================================================

def compare_naive_and_stable_softmax(logits):
    """
    Runs both naive and stable softmax on logits, reporting if outputs are finite and sum to 1.
    """
    print("--- Softmax Comparison ---")
    # Naive
    naive_exp = torch.exp(logits)
    naive_softmax = naive_exp / torch.sum(naive_exp, dim=-1, keepdim=True)
    
    # Stable (subtract max logit for numerical stability)
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    stable_exp = torch.exp(logits - max_logits)
    stable_softmax = stable_exp / torch.sum(stable_exp, dim=-1, keepdim=True)
    
    print(f"Naive  - All Finite: {torch.isfinite(naive_softmax).all().item()} | Sums to 1: {torch.allclose(naive_softmax.sum(dim=-1), torch.ones_like(naive_softmax.sum(dim=-1)))}")
    print(f"Stable - All Finite: {torch.isfinite(stable_softmax).all().item()} | Sums to 1: {torch.allclose(stable_softmax.sum(dim=-1), torch.ones_like(stable_softmax.sum(dim=-1)))}")
    print("-" * 26)

def check_finite_tensor(tensor, name="Tensor"):
    """
    Raises an error when a tensor contains NaN or Inf.
    """
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf values!")

# =============================================================================
# 3. Model Outputs and Loss Checks
# =============================================================================

def inspect_logits(logits):
    """
    Reports the logits shape, value range, mean, std dev, and checks for finite values.
    """
    print("--- Logits Inspection ---")
    print(f"Shape: {logits.shape}")
    print(f"Mean:  {logits.mean().item():.4f}")
    print(f"Std:   {logits.std().item():.4f}")
    print(f"Min:   {logits.min().item():.4f}, Max: {logits.max().item():.4f}")
    check_finite_tensor(logits, "Logits")
    print("Status: All values are finite.")
    print("-" * 25)

def check_loss_inputs(logits, targets, num_classes):
    """
    Verifies logits and targets are shaped correctly for nn.CrossEntropyLoss.
    """
    if logits.dim() != 2:
        raise ValueError(f"Logits should be 2D (Batch, Classes), got {logits.dim()}D")
    if targets.dim() != 1:
        raise ValueError(f"Targets should be 1D (Batch,), got {targets.dim()}D")
    if logits.size(0) != targets.size(0):
        raise ValueError(f"Batch size mismatch: Logits {logits.size(0)} vs Targets {targets.size(0)}")
    if logits.size(1) != num_classes:
        raise ValueError(f"Logits class count ({logits.size(1)}) does not match expected ({num_classes})")
    if targets.dtype != torch.long:
        raise TypeError(f"Targets for CrossEntropyLoss must be torch.long, got {targets.dtype}")
    print("Loss Inputs Validated: Shapes and Dtypes are correct for CrossEntropyLoss.")

# =============================================================================
# 4. Gradients and Optimization
# =============================================================================

def gradient_summary(model):
    """
    Iterates over model parameters and reports if gradients exist and their norm.
    """
    print("--- Gradient Summary ---")
    total_norm = 0.0
    none_count = 0
    valid_count = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                none_count += 1
            else:
                valid_count += 1
                param_norm = param.grad.detach().data.norm(2).item()
                total_norm += param_norm ** 2
                
    total_norm = total_norm ** 0.5
    print(f"Parameters with gradients: {valid_count}")
    print(f"Parameters missing gradients (None): {none_count}")
    print(f"Total Gradient L2 Norm: {total_norm:.4f}")
    
    if total_norm < 1e-6 and valid_count > 0:
        print("[Warning] Gradients are extremely small (Vanishing Gradients).")
    print("-" * 24)

def parameter_update_norm(model_before, model_after):
    """
    Compares model parameters before and after optimizer.step() to measure change.
    Note: Requires saving a deepcopy of the state_dict before the step.
    """
    total_change = 0.0
    for (name1, p1), (name2, p2) in zip(model_before.items(), model_after.named_parameters()):
        if p2.requires_grad:
            change = (p1 - p2).norm(2).item()
            total_change += change ** 2
    
    total_change = total_change ** 0.5
    print(f"Total Parameter Update L2 Norm: {total_change:.6f}")
    if total_change == 0.0:
        print("[Warning] Parameters did not change! Optimizer step may have failed or LR is 0.")

def overfit_single_batch(model, optimizer, criterion, images, labels, steps=50):
    """
    Trains the model on one mini-batch to ensure it can memorize it.
    """
    print(f"--- Overfitting Single Batch (Steps: {steps}) ---")
    model.train()
    initial_loss = None
    
    for i in range(steps):
        optimizer.zero_grad()
        outputs = model(images)
        # Handle tuple returns from complex models
        if isinstance(outputs, tuple): outputs = outputs[0] 
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i == 0: initial_loss = loss.item()
        if (i+1) % 10 == 0 or i == steps - 1:
            print(f"Step {i+1:02d} | Loss: {loss.item():.4f}")
            
    print(f"Initial Loss: {initial_loss:.4f} -> Final Loss: {loss.item():.4f}")
    if loss.item() > initial_loss * 0.5:
        print("[Warning] Model failed to significantly overfit a single batch. Architecture or LR issue likely.")
    print("-" * 47)

# =============================================================================
# 5. Model Modes and Autograd Pitfalls
# =============================================================================

def check_model_mode(model):
    """
    Reports model mode and lists mode-sensitive submodules.
    """
    mode = "Training" if model.training else "Evaluation"
    print(f"--- Model Mode Check: {mode} ---")
    
    mode_sensitive_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)
    
    affected_modules = []
    for name, module in model.named_modules():
        if isinstance(module, mode_sensitive_types):
            affected_modules.append((name, type(module).__name__))
            
    print(f"Found {len(affected_modules)} mode-sensitive submodules.")
    if len(affected_modules) > 0:
        print("First 5 sensitive modules:")
        for name, mod_type in affected_modules[:5]:
            print(f"  - {name} ({mod_type})")
    print("-" * 33)

def run_inplace_autograd():
    """
    Triggers an expected autograd error due to an in-place operation.
    """
    print("--- Testing In-Place Autograd Error ---")
    x = torch.tensor([2.0], requires_grad=True)
    y = x * 3        # y is computed from x — PyTorch saves x for the backward pass
    
    # In-place modification of x after the graph was built
    x += 1           
    loss = y.sum()
    
    try:
        loss.backward()  
        return "Success (Unexpected)"
    except RuntimeError as e:
        print(f"Caught expected RuntimeError: {e}")
        return str(e)

def fix_inplace_autograd_issue():
    """
    Rewrites the computation using an out-of-place operation so backward() succeeds.
    """
    print("--- Fixing In-Place Autograd Error ---")
    x = torch.tensor([2.0], requires_grad=True)
    y = x * 3
    
    # FIX: Out-of-place operation creates a new tensor 'x_new' without corrupting the graph history of 'x'
    x_new = x + 1 
    loss = y.sum()
    
    try:
        loss.backward()
        print(f"Success! Gradient of x: {x.grad.item()}")
        return True
    except RuntimeError as e:
        print(f"Failed: {e}")
        return False

# =============================================================================
# Run Suite (Example Usage)
# =============================================================================
if __name__ == "__main__":
    print("PlantCLEF 2026 Master Diagnostic Suite Initialized.\n")
    run_inplace_autograd()
    print()
    fix_inplace_autograd_issue()
