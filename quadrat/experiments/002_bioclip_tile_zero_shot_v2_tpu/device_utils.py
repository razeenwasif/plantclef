"""
Device resolution utilities supporting CPU, CUDA, and TPU (PyTorch/XLA).

Centralises all device-selection logic so that run_inference.py and utils.py
need no direct torch_xla imports and remain importable without XLA installed.

Device modes
------------
auto   Try TPU (XLA) first, then CUDA, then CPU.
tpu    Require PyTorch/XLA; raise a clear error if unavailable.
cuda   Require CUDA; raise a clear error if unavailable.
cpu    Always use CPU regardless of available hardware.
       Anything else (e.g. "cuda:0") is passed through as-is with backend="cuda".
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_device(mode: str) -> tuple:
    """
    Translate a device-mode string into (device, backend).

    Parameters
    ----------
    mode : str
        One of "auto", "tpu", "cuda", "cpu", or a raw torch device string
        such as "cuda:0".

    Returns
    -------
    device : str | torch.device
        Suitable for passing to tensor.to(device) and model.to(device).
    backend : str
        One of "xla", "cuda", "cpu".  Used by mark_step() and device_str().
    """
    if mode == "tpu":
        return _require_xla()
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda specified but CUDA is not available."
            )
        return "cuda", "cuda"
    if mode == "cpu":
        return "cpu", "cpu"
    if mode == "auto":
        # Prefer TPU > CUDA > CPU
        try:
            return _try_xla()
        except Exception:
            pass
        if torch.cuda.is_available():
            return "cuda", "cuda"
        return "cpu", "cpu"
    # Raw device string (e.g. "cuda:0") — pass through; treat as CUDA backend.
    return mode, "cuda"


def mark_step(backend: str) -> None:
    """
    Flush the XLA execution graph when running on TPU.

    On non-XLA backends this is a no-op, so callers can always call it
    unconditionally after encoding batches.

    Calling this periodically (e.g. after each encode batch) prevents the XLA
    graph from growing unboundedly and avoids silent OOM on the device.
    """
    if backend == "xla":
        import torch_xla.core.xla_model as xm  # noqa: PLC0415
        xm.mark_step()


def device_str(device) -> str:
    """Return a JSON-serialisable string for a device (str or torch.device)."""
    return str(device)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_xla():
    """Attempt to obtain an XLA device; propagate any ImportError/RuntimeError."""
    import torch_xla.core.xla_model as xm  # noqa: PLC0415
    device = xm.xla_device()
    return device, "xla"


def _require_xla():
    try:
        return _try_xla()
    except ImportError as exc:
        raise RuntimeError(
            "--device tpu specified but torch_xla is not installed.\n"
            "Install it with: pip install torch-xla\n"
            f"Original error: {exc}"
        ) from exc
