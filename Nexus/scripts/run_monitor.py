"""
Run monitor — slice 2 of the control plane.
==========================================

Watches the local telemetry directory (where ``src/training/telemetry.py``
writes one JSONL per rank) and projects what it sees into the
``/runs/{run_id}`` Firestore collection.

The agent uses this to **re-attach** to any training process that is
already running on the pod, without that process having gone through
the slice-3 Start-Run modal. As long as the trainer is emitting
telemetry — which it does unconditionally — the dashboard's Mission
tab will pick it up.

Conventions
-----------
Telemetry filenames follow ``src.training.telemetry._generate_run_id``::

    oracle_<phase>_r<rank>_<UTC-stamp>_<short-hex>.jsonl

We group files by the **stem with the rank token stripped out**, e.g.
``oracle_p2a_20260517T081422Z_ab12``. That stem identifies one logical
multi-rank run; we elect a canonical run_id by formatting the same
stem with ``_r0_`` reinserted at the rank position, and write *one*
``/runs/{run_id}`` document per logical run.

Per-rank state (last heartbeat, last step, stale flag) is rolled into
``runs.ranks[r]`` from every sibling file in the group, so the Mission
card can render the full rank fleet later (slice 4) without changing
schema.

Slice 2 deliberately does not detect run completion. ``status='running'``
stays until the file goes idle past ``IDLE_AFTER_SEC`` (10 minutes by
default), at which point it flips to ``failed``. Slice 4 layers proper
completion / cancellation detection on top of this.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

# Soft import: callers may pre-validate firebase_admin themselves.
try:
    from firebase_admin import firestore  # type: ignore
    _HAS_FIRESTORE = True
except ImportError:
    _HAS_FIRESTORE = False


# Tunables (CLI-overridable from pod_agent.py).
METRIC_WINDOW = 200                 # how many step points to keep per run
IDLE_AFTER_SEC = 600.0              # flip status='failed' if no new line for this long
RANK_STALE_AFTER_SEC = 120.0        # rank flagged stale in /runs.ranks[r].stale

_RANK_RE = re.compile(r"_r(\d+)_")


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry parsing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _MetricsRing:
    """Bounded rolling buffer for per-step metrics. We push downsampled
    Firestore writes from this; the local buffer is the source of truth."""
    step: Deque[int] = field(default_factory=lambda: deque(maxlen=METRIC_WINDOW))
    loss: Deque[float] = field(default_factory=lambda: deque(maxlen=METRIC_WINDOW))
    local_acc: Deque[float] = field(default_factory=lambda: deque(maxlen=METRIC_WINDOW))
    val_acc: Deque[float] = field(default_factory=lambda: deque(maxlen=METRIC_WINDOW))
    val_step: Deque[int] = field(default_factory=lambda: deque(maxlen=METRIC_WINDOW))

    def push_step(self, step: int, loss: float, local_acc: Optional[float]) -> None:
        self.step.append(step)
        self.loss.append(loss)
        if local_acc is not None:
            self.local_acc.append(local_acc)

    def push_val(self, step: int, acc: float) -> None:
        self.val_step.append(step)
        self.val_acc.append(acc)

    def as_dict(self) -> Dict[str, List[Any]]:
        return {
            "step":     list(self.step),
            "loss":     list(self.loss),
            "local_acc": list(self.local_acc),
            "val_step": list(self.val_step),
            "val_acc":  list(self.val_acc),
        }


@dataclass
class _FileCursor:
    """Per-file tail state. The agent persists nothing — on restart we
    re-scan from byte 0 and let idempotent writes overwrite."""
    path: Path
    rank: int
    offset: int = 0
    last_event_ts: Optional[datetime] = None
    last_heartbeat_ts: Optional[datetime] = None
    last_step: Optional[int] = None
    last_mtime: float = 0.0


@dataclass
class _Run:
    """Aggregated per-logical-run state held in memory between polls."""
    run_id: str                        # canonical: rank-0 file stem
    phase: str
    started_at: datetime
    status: str = "running"
    metrics: _MetricsRing = field(default_factory=_MetricsRing)
    rank_cursors: Dict[int, _FileCursor] = field(default_factory=dict)
    last_pushed_step: int = -1         # rate-limit Firestore writes
    firestore_initialised: bool = False
    finished_at: Optional[datetime] = None


def _strip_rank(stem: str) -> Tuple[str, Optional[int]]:
    """``oracle_p2a_r3_20260517_ab12`` → (``oracle_p2a_20260517_ab12``, 3)."""
    m = _RANK_RE.search(stem)
    if not m:
        return stem, None
    rank = int(m.group(1))
    canonical = stem[: m.start()] + "_" + stem[m.end():]
    return canonical, rank


def _canonical_run_id(canonical_stem: str) -> str:
    """Insert ``_r0_`` so the stored run_id reads as the rank-0 file's id.
    Cosmetic — it's just a stable, recognisable key in Firestore."""
    parts = canonical_stem.split("_", 2)
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}_r0_" + "_".join(parts[2:])
    return canonical_stem + "_r0"


def _parse_ts(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RunMonitor
# ─────────────────────────────────────────────────────────────────────────────


class RunMonitor:
    """Polls a telemetry directory and projects new events into Firestore.

    Construction is cheap; call :meth:`poll` from the agent's main loop
    every ``poll_interval`` seconds. The class is not thread-safe with
    respect to itself — use one instance per directory.
    """

    def __init__(self,
                 telemetry_dir: Path | str,
                 pod_id: str,
                 firestore_client: Any,                     # firestore.Client (kept untyped)
                 idle_after_sec: float = IDLE_AFTER_SEC,
                 rank_stale_after_sec: float = RANK_STALE_AFTER_SEC) -> None:
        if not _HAS_FIRESTORE:
            raise RuntimeError("firebase-admin not installed; run_monitor needs Firestore.")
        self._dir = Path(telemetry_dir)
        self._pod_id = pod_id
        self._fs = firestore_client
        self._idle_after = idle_after_sec
        self._rank_stale_after = rank_stale_after_sec
        self._runs: Dict[str, _Run] = {}

    # ── public surface ───────────────────────────────────────────────────
    def poll(self) -> None:
        """One reconcile sweep. Idempotent. Cheap if nothing changed."""
        if not self._dir.exists():
            return
        self._discover_new_files()
        self._tail_active_files()
        self._flush_to_firestore()
        self._reap_idle_runs()

    def snapshot(self) -> Dict[str, str]:
        """Per-run status (for the agent's --once flag / debugging)."""
        return {r.run_id: r.status for r in self._runs.values()}

    # ── internals: file discovery ────────────────────────────────────────
    def _discover_new_files(self) -> None:
        for p in self._dir.glob("oracle_*.jsonl"):
            stem = p.stem
            canonical_stem, rank = _strip_rank(stem)
            if rank is None:
                continue
            run_id = _canonical_run_id(canonical_stem)
            run = self._runs.get(run_id)
            if run is None:
                # New logical run — bootstrap.
                phase = self._phase_from_stem(stem)
                run = _Run(
                    run_id=run_id,
                    phase=phase,
                    started_at=datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc),
                )
                self._runs[run_id] = run
            if rank not in run.rank_cursors:
                run.rank_cursors[rank] = _FileCursor(path=p, rank=rank)

    @staticmethod
    def _phase_from_stem(stem: str) -> str:
        # `oracle_<phase>_r<rank>_…` — phase is the second underscore-separated token.
        parts = stem.split("_", 2)
        return parts[1] if len(parts) >= 2 else "unknown"

    # ── internals: tail ─────────────────────────────────────────────────
    def _tail_active_files(self) -> None:
        for run in self._runs.values():
            for cursor in run.rank_cursors.values():
                self._tail_file(run, cursor)

    def _tail_file(self, run: _Run, cursor: _FileCursor) -> None:
        try:
            stat = cursor.path.stat()
        except FileNotFoundError:
            return
        if stat.st_mtime == cursor.last_mtime and stat.st_size == cursor.offset:
            return
        cursor.last_mtime = stat.st_mtime
        try:
            with cursor.path.open("rb") as fh:
                fh.seek(cursor.offset)
                chunk = fh.read()
                cursor.offset = fh.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            if not line.strip():
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._apply_event(run, cursor, evt)

    def _apply_event(self, run: _Run, cursor: _FileCursor, evt: Dict[str, Any]) -> None:
        ts = _parse_ts(evt.get("ts"))
        if ts is not None:
            cursor.last_event_ts = ts
        name = evt.get("name", "")
        fields = evt.get("fields") or {}
        if name == "heartbeat":
            cursor.last_heartbeat_ts = ts or cursor.last_event_ts
            return
        if name == "step":
            step = fields.get("step")
            loss = fields.get("loss")
            local_acc = fields.get("local_acc")
            if isinstance(step, int) and isinstance(loss, (int, float)):
                cursor.last_step = step
                if cursor.rank == 0:
                    run.metrics.push_step(step, float(loss),
                                          float(local_acc) if isinstance(local_acc, (int, float)) else None)
            return
        if name == "epoch.end" and cursor.rank == 0:
            local_acc = fields.get("local_acc")
            if isinstance(local_acc, (int, float)) and cursor.last_step is not None:
                run.metrics.push_step(cursor.last_step, run.metrics.loss[-1] if run.metrics.loss else 0.0,
                                      float(local_acc))
            return
        if name == "validation.end" and cursor.rank == 0:
            acc = fields.get("acc")
            if isinstance(acc, (int, float)) and cursor.last_step is not None:
                run.metrics.push_val(cursor.last_step, float(acc))
            return
        if name == "run.end":
            status_raw = fields.get("status", "ok")
            run.status = "completed" if status_raw == "ok" else "failed"
            run.finished_at = ts or datetime.now(timezone.utc)

    # ── internals: Firestore writes ─────────────────────────────────────
    def _flush_to_firestore(self) -> None:
        for run in self._runs.values():
            doc_ref = self._fs.collection("runs").document(run.run_id)
            if not run.firestore_initialised:
                # Distinguish two paths:
                #   • Spawner pre-created the doc (slice 3) — its spec /
                #     source / request_id are authoritative. We only fill
                #     metrics + ranks and (later) status.
                #   • No-one's seen this run before (slice 2 re-attach) —
                #     we write the full doc with source='reattach'.
                try:
                    existing = doc_ref.get()
                except Exception:
                    existing = None
                if existing is not None and getattr(existing, "exists", False):
                    # Non-destructive: bump 'starting' to 'running' once we
                    # see real telemetry, then leave the rest to the periodic
                    # update path below.
                    doc_ref.update({"status": "running"})
                else:
                    doc_ref.set(self._initial_doc(run), merge=True)
                run.firestore_initialised = True
            # Rate-limit metric updates to "new step since last push" or
            # "status change".
            cursor0 = run.rank_cursors.get(0)
            current_step = cursor0.last_step if cursor0 else None
            payload: Dict[str, Any] = {
                "ranks": self._ranks_payload(run),
            }
            if current_step is not None and current_step != run.last_pushed_step:
                payload["metrics"] = run.metrics.as_dict()
                run.last_pushed_step = current_step
            if run.status != "running":
                payload["status"] = run.status
                if run.finished_at is not None:
                    payload["finished_at"] = run.finished_at
            doc_ref.update(payload)

    def _initial_doc(self, run: _Run) -> Dict[str, Any]:
        return {
            "run_id": run.run_id,
            "pod_id": self._pod_id,
            "request_id": None,         # slice 2 = re-attached, not requested
            "spec": {
                "name": run.run_id,
                "phase": run.phase,
                "seed": None,
                "cluster_manifest_ref": None,
            },
            "started_at": run.started_at,
            "finished_at": None,
            "status": run.status,
            "exit_code": None,
            "ranks": self._ranks_payload(run),
            "metrics": run.metrics.as_dict(),
            "cancel_requested": False,
            "log_url": None,
            "source": "reattach",       # slice 2 marker; slice 3 will write "request"
        }

    def _ranks_payload(self, run: _Run) -> Dict[str, Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        out: Dict[str, Dict[str, Any]] = {}
        for rank, cursor in run.rank_cursors.items():
            hb = cursor.last_heartbeat_ts or cursor.last_event_ts
            stale = (hb is None) or ((now - hb).total_seconds() > self._rank_stale_after)
            out[str(rank)] = {
                "host_id": self._pod_id,
                "last_heartbeat": hb,
                "last_step": cursor.last_step,
                "stale": stale,
            }
        return out

    # ── internals: idle-run reaping ──────────────────────────────────────
    def _reap_idle_runs(self) -> None:
        now = datetime.now(timezone.utc)
        for run in self._runs.values():
            if run.status != "running":
                continue
            newest = max(
                (c.last_event_ts for c in run.rank_cursors.values() if c.last_event_ts is not None),
                default=None,
            )
            if newest is None:
                continue
            if (now - newest).total_seconds() > self._idle_after:
                run.status = "failed"
                run.finished_at = newest
                # The next flush will propagate this to Firestore.

    # ── for tests / smoke ────────────────────────────────────────────────
    def runs(self) -> Iterable[_Run]:
        return self._runs.values()
