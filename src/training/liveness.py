"""
NCCL / XLA rank liveness probe
==============================

A watchdog that watches each rank's last heartbeat (emitted by the
telemetry daemon) and surfaces a clear diagnostic when one stalls
beyond a configurable threshold. Pre-empts the usual "training silently
hangs on `nccl-allreduce`" failure mode.

Design
------
* The **producer** is ``src.training.telemetry``'s heartbeat thread:
  every ``PLANTCLEF_HEARTBEAT_INTERVAL`` seconds (default 15), each rank
  appends ``{name: "heartbeat", rank: N, ts: ...}`` to its sink.
* The **consumer** is this module's :class:`LivenessMonitor`. It
  reads every rank's most recent JSONL line and computes the time
  since the last heartbeat. Anything older than ``stale_after_sec``
  is flagged.
* The probe **does not kill** ranks itself. PyTorch's
  ``TORCH_NCCL_ASYNC_ERROR_HANDLING=1`` already terminates a process
  on watchdog timeout; this monitor's job is to give us **human**
  visibility before that fires.

Usage
-----
.. code-block:: python

    # On the master (rank 0):
    from src.training.liveness import LivenessMonitor
    monitor = LivenessMonitor(world_size=8, stale_after_sec=120)
    monitor.start()
    ...
    monitor.snapshot()   # returns {rank: seconds_since_heartbeat}
    monitor.stop()

CLI usage::

    python -m src.training.liveness --watch reports/telemetry/

Tails the most recent run's per-rank heartbeats and prints stalled
ranks to stderr every 30 s.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_STALE_SEC = 120.0
DEFAULT_POLL_SEC = 10.0


@dataclass
class RankHealth:
    rank: int
    last_heartbeat_ts: Optional[datetime] = None
    seconds_since: float = float("inf")
    is_stale: bool = True
    last_step: Optional[int] = None


class LivenessMonitor:
    """Polls a directory of JSONL telemetry files, computes per-rank
    last-heartbeat staleness, and surfaces a snapshot on demand. Also
    runs as a background thread that calls a user-supplied callback
    whenever a rank crosses the staleness threshold."""

    def __init__(self,
                 telemetry_dir: Path | str = "reports/telemetry",
                 world_size: Optional[int] = None,
                 stale_after_sec: float = DEFAULT_STALE_SEC,
                 poll_sec: float = DEFAULT_POLL_SEC,
                 on_stale=None) -> None:
        self._dir = Path(telemetry_dir)
        self._world_size = world_size
        self._stale_after = stale_after_sec
        self._poll = poll_sec
        self._on_stale = on_stale
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_run_files: List[Path] = []
        self._snapshot_cache: Dict[int, RankHealth] = {}

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        t = threading.Thread(target=self._loop, name="plantclef-liveness", daemon=True)
        t.start()
        self._thread = t

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    # ── public surface ───────────────────────────────────────────────────
    def snapshot(self) -> Dict[int, RankHealth]:
        with self._lock:
            return dict(self._snapshot_cache)

    def stale_ranks(self) -> List[int]:
        return sorted(r for r, h in self.snapshot().items() if h.is_stale)

    # ── internals ────────────────────────────────────────────────────────
    def _loop(self) -> None:
        last_reported: Dict[int, bool] = {}
        while not self._stop.wait(self._poll):
            try:
                snap = self._refresh()
                if self._on_stale:
                    for rank, h in snap.items():
                        was_stale = last_reported.get(rank, False)
                        if h.is_stale and not was_stale:
                            try: self._on_stale(rank, h)
                            except Exception: pass
                        last_reported[rank] = h.is_stale
            except Exception as e:  # pragma: no cover — best-effort
                print(f"[liveness] poll failed: {e}", file=sys.stderr)

    def _refresh(self) -> Dict[int, RankHealth]:
        # Pick the most recent run group. Telemetry filenames are
        # `plantclef_<phase>_r<rank>_<timestamp>_<short>.jsonl`. Group by
        # everything except the rank suffix.
        all_files = sorted(self._dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not all_files:
            self._latest_run_files = []
            self._snapshot_cache = {}
            return {}
        # Newest file's run stem (strip "_r<N>_…")
        latest = all_files[0]
        run_prefix = _strip_rank(latest.stem) + "_"
        run_files = [p for p in all_files if p.name.startswith(run_prefix)]
        self._latest_run_files = run_files

        snap: Dict[int, RankHealth] = {}
        now = datetime.now(timezone.utc)
        for p in run_files:
            rank, hb_ts, last_step = _read_latest_heartbeat(p)
            if rank is None:
                continue
            seconds_since = (now - hb_ts).total_seconds() if hb_ts else float("inf")
            snap[rank] = RankHealth(
                rank=rank,
                last_heartbeat_ts=hb_ts,
                seconds_since=seconds_since,
                is_stale=seconds_since > self._stale_after,
                last_step=last_step,
            )

        # Fill in missing ranks (no file yet → stale)
        if self._world_size is not None:
            for r in range(self._world_size):
                snap.setdefault(r, RankHealth(rank=r, is_stale=True))

        with self._lock:
            self._snapshot_cache = snap
        return snap


# ── parsing helpers ─────────────────────────────────────────────────────────
def _strip_rank(stem: str) -> str:
    """``plantclef_p2a_r3_2026…_ab12`` → ``plantclef_p2a``"""
    parts = stem.split("_")
    out = []
    for tok in parts:
        if tok.startswith("r") and tok[1:].isdigit():
            break
        out.append(tok)
    return "_".join(out)


def _read_latest_heartbeat(p: Path) -> Tuple[Optional[int], Optional[datetime], Optional[int]]:
    """Read the file's tail and pull the rank + most recent heartbeat ts.
    Also surfaces the latest ``step`` field if a ``step`` event is the
    most recent thing the rank emitted — useful as a forward-progress
    signal alongside the heartbeat itself."""
    try:
        # Cheap tail: read last 8 KB
        with p.open("rb") as fh:
            try: fh.seek(-8192, 2)
            except OSError: fh.seek(0)
            chunk = fh.read().decode("utf-8", errors="ignore")
    except OSError:
        return None, None, None

    rank: Optional[int] = None
    last_hb: Optional[datetime] = None
    last_step: Optional[int] = None
    for line in chunk.splitlines():
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rank is None and "rank" in evt:
            rank = evt["rank"]
        name = evt.get("name", "")
        ts_raw = evt.get("ts")
        if name == "heartbeat" and ts_raw:
            try:
                last_hb = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
        if name == "step":
            step = (evt.get("fields") or {}).get("step")
            if isinstance(step, int):
                last_step = step
    return rank, last_hb, last_step


# ── CLI ─────────────────────────────────────────────────────────────────────
def _cli() -> int:
    p = argparse.ArgumentParser(description="PLANTCLEF rank-liveness watchdog")
    p.add_argument("--watch", default="reports/telemetry",
                   help="Directory of telemetry JSONL files (default: reports/telemetry)")
    p.add_argument("--world", type=int, default=None,
                   help="Expected world size (so missing ranks are reported as stale)")
    p.add_argument("--stale-after", type=float, default=DEFAULT_STALE_SEC,
                   help=f"Seconds without a heartbeat before a rank is stale (default: {DEFAULT_STALE_SEC})")
    p.add_argument("--interval", type=float, default=30.0,
                   help="Seconds between watchdog reports (default: 30)")
    args = p.parse_args()

    def _on_stale(rank: int, h: RankHealth):
        step_info = f" last_step={h.last_step}" if h.last_step is not None else ""
        print(f"[liveness] STALE  rank={rank}  silent_for={h.seconds_since:.1f}s{step_info}",
              file=sys.stderr, flush=True)

    monitor = LivenessMonitor(
        telemetry_dir=args.watch, world_size=args.world,
        stale_after_sec=args.stale_after, poll_sec=args.interval,
        on_stale=_on_stale,
    )
    monitor.start()
    print(f"[liveness] watching {args.watch}/  stale_after={args.stale_after}s  interval={args.interval}s")
    try:
        while True:
            time.sleep(args.interval)
            snap = monitor.snapshot()
            healthy = sum(1 for h in snap.values() if not h.is_stale)
            stale = [r for r, h in snap.items() if h.is_stale]
            print(f"[liveness] {datetime.now(timezone.utc).strftime('%H:%M:%S')}  "
                  f"healthy={healthy}/{len(snap)}  stale={stale}", flush=True)
    except KeyboardInterrupt:
        monitor.stop()
        print("[liveness] bye")
        return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
