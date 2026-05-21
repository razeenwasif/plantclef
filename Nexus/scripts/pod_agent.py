#!/usr/bin/env python3
"""
ORACLE pod agent — Slices 1 + 2 + 3 (inventory, heartbeat, runs, spawn)
======================================================================

A long-lived daemon, one per pod, that:

1. Loads its identity from a ``cluster.yaml`` entry (selected by ``--pod-id``).
2. Authenticates to Firestore via a service-account JSON file.
3. Registers itself at ``/pods/{pod_id}`` with status ``online``.
4. Every ``--heartbeat-interval`` seconds, writes a fresh
   ``last_heartbeat`` + utilisation snapshot.
5. **Slice 2:** every ``--run-poll-interval`` seconds, scans
   ``--telemetry-dir`` for in-flight training runs, projects them into
   ``/runs/{run_id}`` Firestore docs, and tails new events (step, loss,
   epoch.end, validation.end, run.end) so the Mission tab can render
   live training cards without going through the Start-Run modal.
6. **Slice 3:** on the same loop, also reconciles ``/run_requests``
   targeted at this pod. If the pod is free, claims the oldest pending
   request, pre-creates ``/runs/{run_id}``, spawns ``oracle.py train``
   via :mod:`subprocess`, and watches the process to update
   ``/runs/{run_id}.status`` on exit. Run-monitor and spawner cooperate
   on the same ``/runs/`` documents (spawner owns ``spec``/``source``;
   monitor owns ``metrics``/``ranks``).
7. On SIGTERM / SIGINT, sets ``status='offline'`` and exits cleanly.

This file is intentionally **independent of the training stack** — it
does not import torch, DALI, DeepSpeed, or anything else from
``src/``. The only third-party dep is ``firebase-admin``; everything
else (PyYAML, subprocess, threading) is in the std lib path the rest
of ORACLE already requires. That keeps the agent installable on a
TPU VM or a fresh GCP image without dragging in the full training
environment.

Future slices (Start-Run modal, cancellation, per-rank liveness badges)
extend this; for now the agent proves "the pod is alive" + "here's what
it's working on, derived from the existing telemetry stream".

Usage
-----
.. code-block:: bash

    python scripts/pod_agent.py \\
        --pod-id pod-5090 \\
        --service-account /etc/oracle/pod_agent.json \\
        --cluster configs/cluster.yaml \\
        [--heartbeat-interval 15] \\
        [--telemetry-dir reports/telemetry] \\
        [--run-poll-interval 10] \\
        [--no-runs]                         # disable slice-2 monitor
        [--once]                            # write one heartbeat then exit (smoke test)

See ``docs/CONTROL_PLANE.md`` for the full Firestore schema, security
rules, and the slice-3 / slice-4 work this skeleton unlocks.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# ── third-party (kept optional so a `pod_agent --help` works on a fresh box) ─
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    _HAS_FIREBASE = True
except ImportError:
    _HAS_FIREBASE = False

# ── ORACLE imports (only the manifest loader, which depends only on PyYAML) ──
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(_PROJECT_ROOT), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.cluster.manifest import load_manifest, HostEntry  # noqa: E402

AGENT_VERSION = "pod_agent/0.1.0"

# ─────────────────────────────────────────────────────────────────────────────
# Inventory + utilisation snapshots
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AcceleratorInfo:
    kind: str            # 'cuda' | 'tpu' | 'cpu'
    device_count: int
    sku: str             # 'RTX 5090', 'RTX PRO 6000 96GB', 'TPU v5p', 'CPU only'
    memory_gb: float     # per-device, best effort; 0.0 if unknown


@dataclass
class Utilisation:
    gpu_util_pct: float
    vram_used_gb: float
    temperature_c: float
    power_w: float


def _detect_cuda_sku_and_memory() -> tuple[str, float]:
    """Use ``nvidia-smi`` to read the first GPU's product name + total memory.

    Returns ``("", 0.0)`` if ``nvidia-smi`` is missing or fails.
    """
    if shutil.which("nvidia-smi") is None:
        return "", 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits", "--id=0"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        first = out.stdout.strip().splitlines()[0]
        name, mem_mb = [s.strip() for s in first.split(",", 1)]
        return name, round(int(mem_mb) / 1024.0, 1)
    except (subprocess.SubprocessError, ValueError, IndexError):
        return "", 0.0


def _collect_cuda_utilisation() -> Optional[Utilisation]:
    """Aggregate utilisation across all visible CUDA devices."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        rows = [r for r in out.stdout.strip().splitlines() if r.strip()]
        if not rows:
            return None
        util = mem = temp = pwr = 0.0
        for r in rows:
            u, m, t, p = [s.strip() for s in r.split(",")]
            util += float(u); mem += float(m); temp += float(t)
            try: pwr += float(p)
            except ValueError: pass  # [N/A] from cards without telemetry
        n = len(rows)
        return Utilisation(
            gpu_util_pct=round(util / n, 1),
            vram_used_gb=round(mem / 1024.0, 2),
            temperature_c=round(temp / n, 1),
            power_w=round(pwr, 1),
        )
    except (subprocess.SubprocessError, ValueError):
        return None


def _capabilities(kind: str) -> list[str]:
    """Best-effort list of capability tags. Used by the dashboard for badges."""
    caps: list[str] = [platform.system().lower()]
    if kind == "cuda":
        # Best-effort CUDA version sniff
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=3, check=True)
            drv = out.stdout.strip().splitlines()[0]
            if drv:
                caps.append(f"nvidia-driver-{drv}")
        except (subprocess.SubprocessError, IndexError):
            pass
        if shutil.which("nvcc"):
            caps.append("nvcc")
    elif kind == "tpu":
        caps.append("xla")
    return caps


def build_inventory(host: HostEntry, display_name: Optional[str] = None) -> tuple[AcceleratorInfo, list[str], str]:
    """Translate a cluster.yaml HostEntry into the dashboard's pod inventory."""
    sku = ""
    mem_gb = 0.0
    if host.type == "cuda":
        sku, mem_gb = _detect_cuda_sku_and_memory()
        if not sku:
            sku = "CUDA (unknown SKU)"
    elif host.type == "tpu":
        sku = "TPU"  # cluster.yaml doesn't carry the variant; can refine later
        mem_gb = 0.0
    else:
        sku = "CPU only"

    accel = AcceleratorInfo(
        kind=host.type,
        device_count=host.device_count,
        sku=sku,
        memory_gb=mem_gb,
    )
    caps = _capabilities(host.type)
    name = display_name or _default_display_name(host, sku)
    return accel, caps, name


def _default_display_name(host: HostEntry, sku: str) -> str:
    """Human-friendly fallback if the operator didn't pass --display-name."""
    if host.device_count > 1:
        return f"{host.id} · {host.device_count}× {sku}"
    return f"{host.id} · {sku}"


# ─────────────────────────────────────────────────────────────────────────────
# The agent itself
# ─────────────────────────────────────────────────────────────────────────────


class PodAgent:
    """Slices 1 + 2: register the pod, heartbeat, and project local
    training runs into /runs/{run_id} via the telemetry-tailing RunMonitor."""

    def __init__(self,
                 pod_id: str,
                 service_account_path: str,
                 cluster_path: str,
                 heartbeat_interval: float = 15.0,
                 display_name: Optional[str] = None,
                 telemetry_dir: Optional[str] = "reports/telemetry",
                 run_poll_interval: float = 10.0,
                 enable_run_monitor: bool = True) -> None:
        if not _HAS_FIREBASE:
            raise SystemExit(
                "[pod_agent] firebase-admin is not installed. Install with:\n"
                "    pip install firebase-admin\n"
                "(or add it to your pod's lightweight requirements.)"
            )
        self.pod_id = pod_id
        self.heartbeat_interval = heartbeat_interval
        self.run_poll_interval = run_poll_interval

        # Locate this pod in the cluster manifest.
        manifest = load_manifest(cluster_path)
        try:
            self.host = manifest.resolve(pod_id)
        except Exception as e:
            raise SystemExit(f"[pod_agent] {e}")

        # Build inventory once; refreshed only if utilisation snapshot changes.
        self.accelerator, self.capabilities, self.display_name = build_inventory(
            self.host, display_name=display_name
        )

        # Init Firebase Admin once (idempotent guard for module reload).
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.pod_ref = self.db.collection("pods").document(self.pod_id)

        # Slice 2: run monitor (telemetry tail → /runs/{run_id}).
        self._run_monitor = None
        self._run_spawner = None
        if enable_run_monitor and telemetry_dir:
            from run_monitor import RunMonitor   # local sibling import (scripts/ on sys.path)
            self._run_monitor = RunMonitor(
                telemetry_dir=telemetry_dir,
                pod_id=self.pod_id,
                firestore_client=self.db,
            )
            self._telemetry_dir = telemetry_dir
            # Slice 3: spawner — reconciles /run_requests targeting this pod.
            from run_spawner import RunSpawner
            self._run_spawner = RunSpawner(
                pod_id=self.pod_id,
                project_root=_PROJECT_ROOT,
                telemetry_dir=telemetry_dir,
                firestore_client=self.db,
            )

        self._stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._run_thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────
    def register(self) -> None:
        """Initial write: status=online + full inventory + first heartbeat."""
        doc = {
            "pod_id": self.pod_id,
            "display_name": self.display_name,
            "accelerator": asdict(self.accelerator),
            "status": "online",
            "current_run_id": None,
            "last_heartbeat": firestore.SERVER_TIMESTAMP,
            "agent_version": AGENT_VERSION,
            "capabilities": self.capabilities,
            "host": {
                "hostname": socket.gethostname(),
                "role": self.host.role,
            },
            "utilisation": self._utilisation_snapshot(),
        }
        self.pod_ref.set(doc, merge=True)
        print(f"[pod_agent] registered pod {self.pod_id!r} "
              f"({self.accelerator.device_count}× {self.accelerator.sku}, "
              f"{self.accelerator.kind})")

    def heartbeat_once(self) -> None:
        """Single heartbeat update — also used for `--once` smoke tests."""
        self.pod_ref.update({
            "last_heartbeat": firestore.SERVER_TIMESTAMP,
            "utilisation": self._utilisation_snapshot(),
            "status": "online",   # in slice 2+, this flips to 'busy' during runs
        })

    def run(self) -> None:
        """Blocking entry point: register, heartbeat-loop until SIGTERM.
        Also spins up the run-monitor poll loop if slice 2 is enabled."""
        self.register()
        self._install_signal_handlers()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, name="pod-agent-hb", daemon=True)
        self._hb_thread.start()
        if self._run_monitor is not None:
            self._run_thread = threading.Thread(target=self._run_monitor_loop, name="pod-agent-runs", daemon=True)
            self._run_thread.start()
            print(f"[pod_agent] run monitor watching {self._telemetry_dir!r} every {self.run_poll_interval}s")
        # Park on the main thread so signals are delivered here.
        try:
            while not self._stop.wait(60.0):
                pass
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self, *_: object) -> None:
        if self._stop.is_set():
            return
        print(f"[pod_agent] shutdown requested — marking pod {self.pod_id!r} offline")
        self._stop.set()
        try:
            self.pod_ref.update({
                "status": "offline",
                "last_heartbeat": firestore.SERVER_TIMESTAMP,
            })
        except Exception as e:
            print(f"[pod_agent] could not write offline status: {e}", file=sys.stderr)

    # ── internals ────────────────────────────────────────────────────────
    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            try:
                self.heartbeat_once()
            except Exception as e:
                # Network blips are expected; log and keep going.
                print(f"[pod_agent] heartbeat failed: {e}", file=sys.stderr)

    def _run_monitor_loop(self) -> None:
        assert self._run_monitor is not None
        while not self._stop.wait(self.run_poll_interval):
            # Spawn reconcile runs first so a freshly-claimed request gets
            # its /runs/ doc in place before the monitor's next pass tries
            # to enrich it.
            if self._run_spawner is not None:
                try:
                    self._run_spawner.poll()
                except Exception as e:
                    print(f"[pod_agent] run spawner poll failed: {e}", file=sys.stderr)
            try:
                self._run_monitor.poll()
            except Exception as e:
                # Network blips / file-races are expected; keep going.
                print(f"[pod_agent] run monitor poll failed: {e}", file=sys.stderr)

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

    def _utilisation_snapshot(self) -> Optional[dict]:
        if self.accelerator.kind == "cuda":
            u = _collect_cuda_utilisation()
            return asdict(u) if u else None
        # TPU / CPU: no cheap polling channel yet.
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ORACLE pod agent — registers /pods/{pod_id}, heartbeats, "
                    "and projects local training runs into /runs/.",
        epilog="See docs/CONTROL_PLANE.md for the full slice plan.",
    )
    p.add_argument("--pod-id", required=True, help="Matches a host id in --cluster")
    p.add_argument("--service-account", required=True,
                   help="Path to the Firebase service-account JSON file")
    p.add_argument("--cluster", default="configs/cluster.yaml",
                   help="Path to cluster.yaml (default: configs/cluster.yaml)")
    p.add_argument("--heartbeat-interval", type=float, default=15.0,
                   help="Seconds between heartbeats (default: 15)")
    p.add_argument("--display-name", default=None,
                   help="Human-friendly label for the Fleet card (optional)")
    # Slice 2: run-monitor controls.
    p.add_argument("--telemetry-dir", default="reports/telemetry",
                   help="Directory the trainer writes JSONL events to (default: reports/telemetry)")
    p.add_argument("--run-poll-interval", type=float, default=10.0,
                   help="Seconds between run-monitor polls (default: 10)")
    p.add_argument("--no-runs", action="store_true",
                   help="Disable the slice-2 run monitor (slice-1 heartbeat only)")
    p.add_argument("--once", action="store_true",
                   help="Write one heartbeat + one run-monitor poll then exit (smoke test)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    agent = PodAgent(
        pod_id=args.pod_id,
        service_account_path=args.service_account,
        cluster_path=args.cluster,
        heartbeat_interval=args.heartbeat_interval,
        telemetry_dir=args.telemetry_dir,
        run_poll_interval=args.run_poll_interval,
        enable_run_monitor=not args.no_runs,
        display_name=args.display_name,
    )
    if args.once:
        agent.register()
        agent.heartbeat_once()
        if agent._run_spawner is not None:
            try:
                agent._run_spawner.poll()
            except Exception as e:
                print(f"[pod_agent] run spawner poll failed: {e}", file=sys.stderr)
        if agent._run_monitor is not None:
            try:
                agent._run_monitor.poll()
                snap = agent._run_monitor.snapshot()
                if snap:
                    print(f"[pod_agent] run monitor saw: {snap}")
                else:
                    print(f"[pod_agent] run monitor saw no telemetry under {args.telemetry_dir!r}")
            except Exception as e:
                print(f"[pod_agent] run monitor poll failed: {e}", file=sys.stderr)
        print("[pod_agent] one-shot complete — exiting (--once)")
        return 0
    agent.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
