import torch
import gc

from src.training.accelerator import accelerator

def vram_purity_cache_clear() -> None:
    """
    Conditional empty_cache that only triggers if memory pressure is high (>90%).
    Avoids frequent, slow OS-level re-allocations and stop-the-world latency.
    Routed through the accelerator so TPU runs are a no-op (XLA manages HBM
    via its own runtime).
    """
    accel = accelerator()
    if accel.memory_pressure_high(0.9):
        accel.empty_cache()

def aggressive_deep_scavenge() -> None:
    """
    Absolute memory purge. Forces a sync on the active accelerator, collects
    all Python generations, and releases reserved memory where applicable.
    Best for Phase transitions and high-res batch intervals.
    """
    import gc
    accel = accelerator()
    # 1. Block until the device is idle so no pending allocations linger
    accel.synchronize()

    # 2. Multi-generation Python GC
    gc.collect()
    gc.collect(1)
    gc.collect(2)

    # 3. Purge device cache where applicable
    accel.empty_cache()
    accel.reset_peak_memory_stats()

def plantclef_batch_scavenge() -> None:
    """
    Micro-cleanup for per-batch execution.
    Prevents fragmentation without the latency of a full sync.
    """
    import gc
    gc.collect(0)
    accel = accelerator()
    if accel.memory_pressure_high(0.85):
        accel.empty_cache()

def coordinated_gc() -> None:
    """
    Triggers a manual garbage collection followed by a VRAM purity check.
    Best used after heavy operations like validation or epoch transitions.
    """
    gc.collect()
    vram_purity_cache_clear()

def centralize_gradients(model: torch.nn.Module) -> None:
    """
    Applies Gradient Centralization (GC) to all convolutional and linear layers.
    GC centers the gradients to have zero-mean, which flattens the loss landscape.
    """
    for p in model.parameters():
        if p.grad is not None and p.dim() > 1:
            # Center the gradients for each output channel/neuron
            p.grad.data.add_(-p.grad.data.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True))
