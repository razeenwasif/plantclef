"""
PLANTCLEF Accelerator Abstraction
==============================

Single interface over CUDA / TPU (and a CPU dev fallback) so the trainer
and training loops don't care which silicon they're running on. Switch
via `plantclef.py train --mode {cuda,tpu,auto}` or the ``CLUSTER_MODE``
environment variable.

Design constraints
------------------
* Pure-CUDA path stays byte-equivalent to the previous code (no regression).
* ``torch_xla`` is imported lazily — this module loads on machines that
  don't have XLA installed, and only blows up if you actually request the
  TPU path.
* Every method is a thin wrapper: it does the right CUDA thing when
  ``mode == 'cuda'``, the right XLA thing when ``mode == 'tpu'``, and a
  no-op (or CPU-safe fallback) otherwise.

Usage sketch
------------
    from src.training.accelerator import make_accelerator
    accel = make_accelerator(os.environ.get("CLUSTER_MODE", "auto"))
    accel.init_distributed()
    device = accel.device          # cuda:0 / xla:0 / cpu
    model = model.to(device)
    with accel.autocast():
        out = model(batch)
    loss.backward()
    accel.optimizer_step(optimizer)
    accel.mark_step()              # no-op on CUDA; flushes XLA graph on TPU

What this module deliberately does NOT do
-----------------------------------------
* It does not abstract DALI. DALI is CUDA-only; the TPU path will need a
  native PyTorch DataLoader (or WebDataset) replacement. See docs/TPU.md
  for the open work item.
* It does not abstract ``torchrun`` vs ``xmp.spawn``. Launching the
  correct multiprocess scaffold is the job of launch.sh; once
  inside a worker, this module handles the rest.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Iterator, Literal, Optional

import torch

Mode = Literal["cuda", "tpu", "cpu"]


# ── XLA lazy loader ─────────────────────────────────────────────────────────
def _try_import_xla():
    """Returns (xm, xmp, pl) on success, raises ImportError on failure."""
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        import torch_xla.distributed.xla_multiprocessing as xmp  # type: ignore
        import torch_xla.distributed.parallel_loader as pl  # type: ignore
    except ImportError as e:
        raise ImportError(
            "torch_xla is not installed. Install with one of:\n"
            "  pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html\n"
            "  pip install torch~=2.4.0 torch_xla~=2.4.0   # match torch major.minor exactly\n"
            "Then re-run with --mode tpu."
        ) from e
    return xm, xmp, pl


def _detect_auto() -> Mode:
    """Pick the best available accelerator. Order: TPU → CUDA → CPU."""
    if os.environ.get("TPU_NAME") or os.environ.get("PJRT_DEVICE") == "TPU":
        try:
            _try_import_xla()
            return "tpu"
        except ImportError:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Base interface ──────────────────────────────────────────────────────────
class Accelerator:
    """Base class. Subclasses override the device-specific bits."""

    mode: Mode = "cpu"

    # — device / rank ——————————————————————————————————————————————————————
    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def world_size(self) -> int:
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return int(os.environ.get("WORLD_SIZE", 1))

    @property
    def rank(self) -> int:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return int(os.environ.get("RANK", 0))

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_cuda(self) -> bool:
        return self.mode == "cuda"

    @property
    def is_tpu(self) -> bool:
        return self.mode == "tpu"

    # — lifecycle ——————————————————————————————————————————————————————————
    def init_distributed(self) -> None:
        """Init torch.distributed if WORLD_SIZE > 1. No-op on single device."""
        pass

    def destroy_distributed(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    # — autocast / mixed precision ————————————————————————————————————————
    @contextmanager
    def autocast(self, dtype: torch.dtype = torch.bfloat16, enabled: bool = True) -> Iterator[None]:
        with nullcontext():
            yield

    # — memory / sync ——————————————————————————————————————————————————————
    def empty_cache(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def reset_peak_memory_stats(self) -> None:
        pass

    def memory_pressure_high(self, threshold: float = 0.9) -> bool:
        """Best-effort: returns True if we should consider freeing memory."""
        return False

    # — step bookkeeping ——————————————————————————————————————————————————
    def mark_step(self) -> None:
        """Flush pending operations. No-op on CUDA; required on TPU after
        each optimizer step so XLA emits and executes the compiled graph."""
        pass

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step the optimizer with cross-replica gradient reduction where
        appropriate. CUDA: plain optimizer.step(); TPU: xm.optimizer_step()."""
        optimizer.step()

    # — dataloader wrapping ————————————————————————————————————————————————
    def wrap_loader(self, loader):
        """Optionally wrap a DataLoader for accelerator-specific prefetching.
        CUDA: returns the loader unchanged. TPU: wraps in MpDeviceLoader."""
        return loader


# ── CUDA implementation ─────────────────────────────────────────────────────
class CudaAccelerator(Accelerator):
    mode: Mode = "cuda"

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CudaAccelerator requested but CUDA is not available.")

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.local_rank}")

    def init_distributed(self) -> None:
        torch.cuda.set_device(self.local_rank)
        if self.world_size > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
            )

    @contextmanager
    def autocast(self, dtype: torch.dtype = torch.bfloat16, enabled: bool = True) -> Iterator[None]:
        with torch.amp.autocast("cuda", dtype=dtype, enabled=enabled):
            yield

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def memory_pressure_high(self, threshold: float = 0.9) -> bool:
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            return torch.cuda.memory_reserved() > threshold * total
        except Exception:
            # Defensive: return False so we don't repeatedly empty cache on
            # quirky drivers that fail get_device_properties.
            return False


# ── TPU implementation ──────────────────────────────────────────────────────
class TpuAccelerator(Accelerator):
    mode: Mode = "tpu"

    def __init__(self) -> None:
        self._xm, self._xmp, self._pl = _try_import_xla()
        # XLA's preferred way to get bf16: set XLA_USE_BF16=1 before any tensor
        # ops. We don't enforce here because some users prefer explicit casts.

    @property
    def device(self) -> torch.device:
        return self._xm.xla_device()

    @property
    def local_rank(self) -> int:
        # In an xmp.spawn worker, rank comes from the spawned process index.
        # Fall back to the env var when launched directly.
        return self._xm.get_local_ordinal()

    @property
    def world_size(self) -> int:
        return self._xm.xrt_world_size()

    @property
    def rank(self) -> int:
        return self._xm.get_ordinal()

    def init_distributed(self) -> None:
        # XLA discovers the topology automatically; no explicit
        # init_process_group call is needed (it uses the 'xla' backend
        # internally). torch.distributed *can* be initialised with backend=xla
        # if the user wants cross-replica collectives, but xm.* primitives
        # already cover that.
        if self.world_size > 1 and not torch.distributed.is_initialized():
            try:
                torch.distributed.init_process_group(backend="xla", init_method="xla://")
            except Exception:
                # PJRT runtime may not need it; ignore silently.
                pass

    @contextmanager
    def autocast(self, dtype: torch.dtype = torch.bfloat16, enabled: bool = True) -> Iterator[None]:
        # XLA recommends explicit bf16 casts at model boundary instead of
        # the CUDA-style autocast context. We provide an autocast for API
        # parity but make it a no-op by default; if the user wants bf16
        # they should set XLA_USE_BF16=1 or cast tensors directly.
        try:
            with torch.amp.autocast("xla", dtype=dtype, enabled=enabled):
                yield
        except (RuntimeError, ValueError, TypeError):
            # Older torch_xla versions lack the autocast key; fall back.
            yield

    def synchronize(self) -> None:
        # rendezvous all replicas
        self._xm.rendezvous("plantclef.sync")

    def mark_step(self) -> None:
        self._xm.mark_step()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        # xm.optimizer_step also performs the all-reduce for gradients.
        self._xm.optimizer_step(optimizer)

    def wrap_loader(self, loader):
        return self._pl.MpDeviceLoader(loader, self.device)


# ── CPU dev fallback ────────────────────────────────────────────────────────
class CpuAccelerator(Accelerator):
    mode: Mode = "cpu"

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


# ── Factory ─────────────────────────────────────────────────────────────────
def make_accelerator(mode: str | None = None) -> Accelerator:
    """Build an Accelerator for the requested mode.

    Args
    ----
    mode
        ``"cuda"``, ``"tpu"``, ``"cpu"``, ``"auto"``, or ``None`` (treated
        as ``"auto"``). Reads ``CLUSTER_MODE`` env var as a fallback if
        ``mode`` is None.

    Returns
    -------
    Accelerator
        Configured instance ready to call ``.init_distributed()`` on.
    """
    requested = (mode or os.environ.get("CLUSTER_MODE") or "auto").lower()
    if requested == "auto":
        requested = _detect_auto()
    if requested == "cuda":
        return CudaAccelerator()
    if requested == "tpu":
        return TpuAccelerator()
    if requested == "cpu":
        return CpuAccelerator()
    raise ValueError(f"Unknown CLUSTER_MODE: {requested!r}. Expected cuda | tpu | cpu | auto.")


# Convenience: a process-wide singleton, lazily initialised on first read.
# Keeps call sites short (`from src.training.accelerator import accelerator`)
# but doesn't force a particular mode on import.
_singleton: Optional[Accelerator] = None


def accelerator() -> Accelerator:
    global _singleton
    if _singleton is None:
        _singleton = make_accelerator()
    return _singleton


def set_accelerator(accel: Accelerator) -> None:
    """Override the process singleton (test hook + explicit-init path)."""
    global _singleton
    _singleton = accel
