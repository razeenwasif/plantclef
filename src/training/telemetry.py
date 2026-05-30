"""
PLANTCLEF structured telemetry
===========================

A single event bus that every training-pipeline component can emit into:
training step metrics, heartbeats, checkpoint writes, NCCL errors, rank
init, GC pressure. Each event is a typed dict with a stable schema:

    {
      "ts":        "2026-05-17T08:14:22.135Z",
      "run_id":    "plantclef_s42_p2a_20260517T081422Z",
      "rank":      2,
      "world":     8,
      "phase":     "p2a",
      "host_id":   "b2",
      "name":      "step",
      "fields":    {"step": 1234, "loss": 0.43, "lr": 6.1e-5, ...}
    }

Sinks are pluggable; the default is a JSON-Lines file under
``reports/telemetry/<run_id>.jsonl``. Stdout, /dev/null, and W&B are
also wired. Choose with ``PLANTCLEF_TELEMETRY=file|stdout|wandb|none``
(default ``file``).

Design constraints
------------------
* **Thread-safe.** Emits from worker threads, signal handlers, or the
  optional heartbeat daemon must not corrupt the file.
* **No blocking.** Emits are best-effort. A failing sink logs to stderr
  and is dropped; training never stops because telemetry hiccups.
* **Cheap.** ``emit()`` is < 50 µs in the file-sink hot path. Hot loops
  (per-step) should still rate-limit themselves at the call site.
* **Foundation for liveness/idempotency/sweep-coordinator.** The
  ``heartbeat()`` daemon and the structured event schema are the
  contracts those features will be built against.

Quick start
-----------
.. code-block:: python

    from src.training.telemetry import telemetry, start_heartbeat

    telemetry.bind(run_id="plantclef_s42_p2a_…", rank=0, world=8,
                   phase="p2a", host_id="b1")
    telemetry.emit("run.start", config_hash="abc123")
    start_heartbeat(interval_sec=15)  # daemon thread, marks alive

    for step, batch in enumerate(loader):
        ...
        if step % 50 == 0:
            telemetry.emit("step", step=step, loss=loss.item())

    telemetry.emit("run.end", status="ok")
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ── Event schema ───────────────────────────────────────────────────────────
@dataclass
class Event:
    name: str
    fields: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"))

    def to_dict(self, ctx: "Context") -> Dict[str, Any]:
        return {
            "ts":      self.ts,
            "run_id":  ctx.run_id,
            "rank":    ctx.rank,
            "world":   ctx.world,
            "phase":   ctx.phase,
            "host_id": ctx.host_id,
            "name":    self.name,
            "fields":  self.fields,
        }


@dataclass
class Context:
    """Process-wide telemetry context. Bound once near the start of
    ``trainer.train()`` and inherited by all subsequent emits."""
    run_id: str = ""
    rank: int = 0
    world: int = 1
    phase: str = ""
    host_id: str = field(default_factory=lambda: socket.gethostname())


# ── Sinks ──────────────────────────────────────────────────────────────────
class Sink:
    """A thread-safe destination for serialized event lines."""
    name = "abstract"

    def write(self, line: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        return None


class NoopSink(Sink):
    name = "none"
    def write(self, line: str) -> None:
        return None


class StdoutSink(Sink):
    name = "stdout"
    def __init__(self) -> None:
        self._lock = threading.Lock()
    def write(self, line: str) -> None:
        with self._lock:
            print(line, flush=True)


class FileSink(Sink):
    name = "file"
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)  # line-buffered
        self._lock = threading.Lock()

    def write(self, line: str) -> None:
        with self._lock:
            try:
                self._fh.write(line + "\n")
            except (OSError, ValueError) as e:
                print(f"[telemetry] FileSink write failed: {e}", file=sys.stderr)

    def close(self) -> None:
        with self._lock:
            try: self._fh.close()
            except Exception: pass


class WandbSink(Sink):
    """Forward step/epoch events to a W&B run. Other events fall through
    to a parallel FileSink so the JSONL log stays complete."""
    name = "wandb"

    def __init__(self, parallel_file: Optional[FileSink] = None) -> None:
        try:
            import wandb  # type: ignore
        except ImportError as e:
            raise ImportError(
                "wandb not installed; use PLANTCLEF_TELEMETRY=file or pip install wandb"
            ) from e
        self._wandb = wandb
        self._parallel = parallel_file
        self._lock = threading.Lock()

    def write(self, line: str) -> None:
        if self._parallel is not None:
            self._parallel.write(line)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return
        if payload.get("name") in ("step", "epoch.end", "validation.end") and self._wandb.run is not None:
            with self._lock:
                self._wandb.log({**payload["fields"], "rank": payload["rank"]}, commit=False)

    def close(self) -> None:
        if self._parallel is not None: self._parallel.close()


# ── Telemetry singleton ───────────────────────────────────────────────────
class _Telemetry:
    """Process-wide singleton. Use ``telemetry`` exported below."""

    def __init__(self) -> None:
        self._ctx: Context = Context()
        self._sinks: List[Sink] = []
        self._lock = threading.Lock()
        self._configured = False
        self._heartbeat_stop: Optional[threading.Event] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

    # ── configuration ────────────────────────────────────────────────────
    def bind(self, *, run_id: Optional[str] = None,
             rank: int = 0, world: int = 1, phase: str = "",
             host_id: Optional[str] = None) -> None:
        """Set / update the process context (typically called once near
        trainer entry, after the accelerator has reported rank/world).

        Run-id resolution, in priority order:

        1. Explicit ``run_id=`` argument (rare; usually used in tests).
        2. ``$PLANTCLEF_RUN_ID_TEMPLATE`` env var, which may contain the
           literal placeholder ``{rank}`` — substituted with this rank.
           Used by the pod-agent spawner to pin every rank of a run to
           the same stamp+short while keeping per-rank filenames distinct.
        3. Auto-generated via :func:`_generate_run_id`.
        """
        if not run_id:
            tmpl = os.environ.get("PLANTCLEF_RUN_ID_TEMPLATE")
            if tmpl:
                run_id = tmpl.replace("{rank}", str(rank))
            else:
                run_id = _generate_run_id(phase=phase, rank=rank)
        with self._lock:
            self._ctx = Context(
                run_id=run_id, rank=rank, world=world, phase=phase,
                host_id=host_id or socket.gethostname(),
            )
            if not self._configured:
                self._configure_sinks_from_env()
                atexit.register(self.shutdown)
                self._configured = True

    def _configure_sinks_from_env(self) -> None:
        choice = os.environ.get("PLANTCLEF_TELEMETRY", "file").lower()
        if choice == "none":
            self._sinks = [NoopSink()]
            return
        if choice == "stdout":
            self._sinks = [StdoutSink()]
            return
        # default: JSONL file under reports/telemetry/<run_id>.jsonl
        log_root = Path(os.environ.get("PLANTCLEF_TELEMETRY_DIR",
                                       "reports/telemetry")).resolve()
        log_path = log_root / f"{self._ctx.run_id}.jsonl"
        file_sink = FileSink(log_path)
        if choice == "wandb":
            try:
                self._sinks = [WandbSink(parallel_file=file_sink)]
            except ImportError as e:
                print(f"[telemetry] {e}; falling back to file sink at {log_path}", file=sys.stderr)
                self._sinks = [file_sink]
        else:
            self._sinks = [file_sink]

    # ── emit ──────────────────────────────────────────────────────────────
    def emit(self, name: str, **fields: Any) -> None:
        """Record a single event. Cheap; never raises."""
        if not self._sinks:
            # Lazy default if someone forgot to bind() — at least give
            # them stdout instead of silently dropping the event.
            self._sinks = [StdoutSink()]
        try:
            event = Event(name=name, fields=fields)
            payload = event.to_dict(self._ctx)
            line = json.dumps(payload, default=_json_default)
        except Exception as e:  # pragma: no cover — best-effort
            print(f"[telemetry] emit({name}) serialize failed: {e}", file=sys.stderr)
            return
        for sink in self._sinks:
            try:
                sink.write(line)
            except Exception as e:  # pragma: no cover — best-effort
                print(f"[telemetry] sink {sink.name!r} failed: {e}", file=sys.stderr)

    # ── heartbeat daemon ─────────────────────────────────────────────────
    def start_heartbeat(self, interval_sec: float = 15.0) -> None:
        """Spin up a daemon thread that emits a `heartbeat` event every
        ``interval_sec`` seconds. The liveness probe consumes these."""
        if self._heartbeat_thread is not None:
            return  # idempotent
        stop = threading.Event()
        def _loop():
            t0 = time.monotonic()
            while not stop.wait(interval_sec):
                self.emit("heartbeat", uptime_sec=round(time.monotonic() - t0, 2))
        t = threading.Thread(target=_loop, name="plantclef-heartbeat", daemon=True)
        t.start()
        self._heartbeat_stop = stop
        self._heartbeat_thread = t

    def stop_heartbeat(self) -> None:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    # ── teardown ─────────────────────────────────────────────────────────
    def shutdown(self) -> None:
        self.stop_heartbeat()
        for sink in self._sinks:
            try: sink.close()
            except Exception: pass


# ── module-level convenience ──────────────────────────────────────────────
telemetry = _Telemetry()


def start_heartbeat(interval_sec: float = 15.0) -> None:
    """Convenience re-export."""
    telemetry.start_heartbeat(interval_sec)


# ── internals ──────────────────────────────────────────────────────────────
def _generate_run_id(phase: str, rank: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    return f"plantclef_{phase or 'run'}_r{rank}_{stamp}_{short}"


def _json_default(o: Any) -> Any:
    """Make common non-JSON types serialisable."""
    try:
        import torch
        if isinstance(o, torch.Tensor):
            if o.numel() == 1:
                return o.item()
            return o.tolist()
    except Exception:
        pass
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
