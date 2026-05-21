"""
Run spawner — slice 3 of the control plane.
==========================================

Reconciles ``/run_requests`` against this pod's actual state:

1. Polls Firestore for ``/run_requests`` where
   ``target_pod_id == self.pod_id AND status == 'pending'``.
2. If the pod is busy (already running a process), flips the request
   to ``rejected`` with a reason and moves on.
3. Otherwise, claims the request: pre-creates ``/runs/{run_id}``,
   marks the pod ``busy``, updates the request to ``claimed`` with a
   ``claimed_run_id`` back-pointer, then spawns ``oracle.py train``
   via :mod:`subprocess` with a deterministic ``ORACLE_RUN_ID_TEMPLATE``
   so the existing :class:`RunMonitor` can stream telemetry into the
   pre-created doc.
4. A watcher thread waits on the spawned process; on exit it flips
   ``/runs/{run_id}.status`` based on the exit code and frees the pod.

The :class:`RunSpawner` is independent of the :class:`RunMonitor` but
they cooperate: the spawner *creates* the ``/runs/`` doc with
``source='request'``, the monitor *enriches* it with metrics and
per-rank state by tailing the same telemetry files the trainer is
writing. See ``docs/CONTROL_PLANE.md``.
"""

from __future__ import annotations

import os
import re
import signal as _signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Soft import so unit tests can monkey-patch.
try:
    from firebase_admin import firestore  # type: ignore
    from google.cloud.firestore_v1.base_query import FieldFilter  # type: ignore
    _HAS_FIRESTORE = True
except ImportError:
    _HAS_FIRESTORE = False


VALID_PHASES = {"p1", "p2a", "p2b", "all"}
_DANGEROUS_ARG = re.compile(r"[;|&`\n\r]")
CANCEL_GRACE_SEC = 60.0     # SIGTERM → SIGKILL window


@dataclass
class _ActiveProc:
    request_id: str
    run_id: str
    proc: subprocess.Popen
    cancel_started: bool = False    # latched once the cancel thread fires


class RunSpawner:
    """One spawner per agent process. Polls /run_requests and shells out
    to ``oracle.py train`` when a request targeting this pod arrives.

    The spawner deliberately accepts only one in-flight process at a
    time — slice-3 scope is "the pod has one job to do". A queue lives
    in /run_requests itself (status='pending' is the queue); the agent
    drains it one request at a time. New requests arriving while the
    pod is busy are rejected, not queued internally."""

    def __init__(self,
                 pod_id: str,
                 project_root: Path | str,
                 telemetry_dir: Path | str,
                 firestore_client: Any,
                 default_cluster_manifest: str = "configs/cluster.yaml") -> None:
        if not _HAS_FIRESTORE:
            raise RuntimeError("firebase-admin not installed; run_spawner needs Firestore.")
        self.pod_id = pod_id
        self.project_root = Path(project_root)
        self.telemetry_dir = Path(telemetry_dir)
        self.default_cluster_manifest = default_cluster_manifest
        self._fs = firestore_client
        self._active: Optional[_ActiveProc] = None
        self._lock = threading.Lock()
        self._watchers: List[threading.Thread] = []

    # ── public surface ───────────────────────────────────────────────────
    def is_busy(self) -> bool:
        with self._lock:
            return self._active is not None and self._active.proc.poll() is None

    def poll(self) -> None:
        """Reconcile one tick. Idempotent. Called from the agent's loop."""
        # Reap any process whose watcher already cleared.
        with self._lock:
            if self._active and self._active.proc.poll() is not None:
                # The watcher already handled status update; just drop the ref.
                self._active = None

        # Slice 4: surface dashboard-initiated cancellation requests. Cheap
        # 1-doc get(); we only run it while a process is alive on this pod.
        self._check_cancel()

        if self.is_busy():
            # Slice 3 scope: one job at a time per pod. Reject anything else.
            self._reject_overflow()
            return

        req_doc = self._claim_next_pending()
        if req_doc is None:
            return
        try:
            self._spawn(req_doc)
        except Exception as e:
            print(f"[spawner] spawn failed: {e}", file=sys.stderr)
            self._mark_request_failed(req_doc, f"spawn error: {e}")

    # ── internals: Firestore query ───────────────────────────────────────
    def _pending_query(self):
        return (self._fs.collection("run_requests")
                .where(filter=FieldFilter("target_pod_id", "==", self.pod_id))
                .where(filter=FieldFilter("status", "==", "pending"))
                .order_by("requested_at")
                .limit(1))

    def _claim_next_pending(self):
        """Return the oldest pending request targeted at this pod, or None.
        Does not yet mutate Firestore — the caller drives the transition."""
        try:
            docs = list(self._pending_query().get())
        except Exception as e:
            print(f"[spawner] /run_requests query failed: {e}", file=sys.stderr)
            return None
        return docs[0] if docs else None

    def _reject_overflow(self) -> None:
        """Whenever the pod is busy, drain any other pending requests that
        targeted us with a clear rejection. Keeps the inbox tidy."""
        try:
            docs = list(self._pending_query().limit(3).get())
        except Exception:
            return
        for d in docs:
            try:
                d.reference.update({
                    "status": "rejected",
                    "rejection_reason": "pod is busy with another run",
                })
            except Exception as e:
                print(f"[spawner] reject failed: {e}", file=sys.stderr)

    # ── internals: spawn ─────────────────────────────────────────────────
    def _spawn(self, req_doc) -> None:
        req = req_doc.to_dict() or {}
        request_id = req_doc.id
        spec = (req.get("spec") or {})
        phase = spec.get("phase", "p2a")
        if phase not in VALID_PHASES:
            self._mark_request_failed(req_doc, f"invalid phase: {phase!r}")
            return
        seed = int(spec.get("seed") or 42)
        name = (spec.get("name") or f"{phase}-{self.pod_id}").strip()[:80] or f"{phase}-{self.pod_id}"
        cluster_ref = spec.get("cluster_manifest_ref") or self.default_cluster_manifest
        extra_args = _sanitise_args(spec.get("extra_args") or [])
        env_overrides = {str(k): str(v) for k, v in (spec.get("env_overrides") or {}).items()}
        requester_uid = req.get("requested_by_uid")  # propagated onto /runs/ for ownership checks

        # Deterministic per-run identifiers. The template is per-rank; the
        # rank-0 substitution is the canonical /runs/{run_id} key the
        # dashboard sees.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        short = uuid.uuid4().hex[:6]
        template = f"oracle_{phase}_r{{rank}}_{stamp}_{short}"
        run_id = template.replace("{rank}", "0")

        # Pre-create /runs/{run_id} before spawning — the monitor will
        # add metrics on top. Using merge=True so a race with the monitor
        # is non-destructive.
        run_ref = self._fs.collection("runs").document(run_id)
        run_ref.set({
            "run_id": run_id,
            "pod_id": self.pod_id,
            "request_id": request_id,
            "created_by_uid": requester_uid,   # slice 4: ownership check for cancel toggle
            "spec": {
                "name": name,
                "phase": phase,
                "seed": seed,
                "cluster_manifest_ref": cluster_ref,
            },
            "started_at": firestore.SERVER_TIMESTAMP,
            "finished_at": None,
            "status": "starting",
            "exit_code": None,
            "ranks": {},
            "metrics": {"step": [], "loss": [], "local_acc": [], "val_step": [], "val_acc": []},
            "cancel_requested": False,
            "log_url": None,
            "source": "request",
        }, merge=True)

        # Flip pod → busy.
        self._fs.collection("pods").document(self.pod_id).update({
            "status": "busy",
            "current_run_id": run_id,
        })

        # Flip request → claimed.
        req_doc.reference.update({
            "status": "claimed",
            "claimed_run_id": run_id,
            "claimed_at": firestore.SERVER_TIMESTAMP,
        })

        proc = self._popen(phase=phase, seed=seed, cluster_ref=cluster_ref,
                           extra_args=extra_args, env_overrides=env_overrides,
                           run_id_template=template, name=name)

        with self._lock:
            self._active = _ActiveProc(request_id=request_id, run_id=run_id, proc=proc)
        watcher = threading.Thread(
            target=self._watch, args=(request_id, run_id, proc),
            name=f"spawn-watcher-{run_id[-12:]}", daemon=True,
        )
        watcher.start()
        self._watchers.append(watcher)
        print(f"[spawner] spawned run {run_id!r} for request {request_id!r}  pid={proc.pid}")

    def _popen(self, *, phase: str, seed: int, cluster_ref: str,
               extra_args: List[str], env_overrides: Dict[str, str],
               run_id_template: str, name: str) -> subprocess.Popen:
        env = os.environ.copy()
        env["ORACLE_RUN_ID_TEMPLATE"] = run_id_template
        env["ORACLE_NAME"] = name
        env["ORACLE_HOST_ID"] = self.pod_id
        env["ORACLE_SEED"] = str(seed)
        env["ORACLE_TELEMETRY_DIR"] = str(self.telemetry_dir.resolve())
        env.update(env_overrides)

        cmd = [
            sys.executable, str(self.project_root / "oracle.py"), "train",
            "--phase", phase,
            "--seed", str(seed),
            "--cluster", cluster_ref,
            "--host-id", self.pod_id,
            *extra_args,
        ]
        # shell=False; ORACLE_RUN_ID_TEMPLATE and friends propagate to torchrun
        # workers via the standard fork-and-exec env inheritance.
        # start_new_session=True puts the child in its own process group so
        # ``os.killpg`` on cancellation reaches every descendent (torchrun
        # spawns DDP workers; SIGTERM to oracle.py alone would orphan them).
        return subprocess.Popen(
            cmd, env=env, cwd=str(self.project_root),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _watch(self, request_id: str, run_id: str, proc: subprocess.Popen) -> None:
        rc = proc.wait()
        # Decide terminal status. The cancel_started flag is set BEFORE we
        # signal the process, so this read here always reflects whether we
        # initiated this termination.
        with self._lock:
            was_cancelled = bool(self._active and self._active.run_id == run_id and self._active.cancel_started)
        if was_cancelled:
            status = "cancelled"
        else:
            status = "completed" if rc == 0 else "failed"
        try:
            self._fs.collection("runs").document(run_id).update({
                "status": status,
                "exit_code": rc,
                "finished_at": firestore.SERVER_TIMESTAMP,
            })
        except Exception as e:
            print(f"[spawner] could not update /runs/{run_id} status: {e}", file=sys.stderr)
        try:
            self._fs.collection("pods").document(self.pod_id).update({
                "status": "online",
                "current_run_id": None,
            })
        except Exception as e:
            print(f"[spawner] could not clear pod busy state: {e}", file=sys.stderr)
        print(f"[spawner] run {run_id!r} exited rc={rc}  status={status}")

    # ── internals: cancellation ──────────────────────────────────────────
    def _check_cancel(self) -> None:
        """If the dashboard set cancel_requested on the active run, spin up
        a cancel thread. The check is cheap (one get()) and only runs while
        a process is alive on this pod."""
        with self._lock:
            active = self._active
        if active is None or active.cancel_started:
            return
        if active.proc.poll() is not None:
            return  # already exited; watcher handled it
        try:
            snap = self._fs.collection("runs").document(active.run_id).get()
        except Exception as e:
            print(f"[spawner] cancel check failed: {e}", file=sys.stderr)
            return
        data = snap.to_dict() if (snap and getattr(snap, "exists", False)) else {}
        if not (data or {}).get("cancel_requested"):
            return
        with self._lock:
            # Re-check under lock — another poll might have already latched.
            if self._active is not active or self._active.cancel_started:
                return
            self._active.cancel_started = True
        print(f"[spawner] cancel requested for run {active.run_id!r}; initiating SIGTERM")
        threading.Thread(target=self._cancel_run, args=(active,),
                         name=f"spawn-cancel-{active.run_id[-12:]}", daemon=True).start()

    def _cancel_run(self, active: _ActiveProc) -> None:
        """Graceful: SIGTERM the process group, wait up to CANCEL_GRACE_SEC,
        then SIGKILL. The watcher thread reads cancel_started and writes
        status='cancelled' once the process actually exits."""
        pid = active.proc.pid
        try:
            os.killpg(os.getpgid(pid), _signal.SIGTERM)
        except (ProcessLookupError, PermissionError) as e:
            print(f"[spawner] SIGTERM to pgid({pid}) failed: {e}", file=sys.stderr)
            return
        deadline = time.monotonic() + CANCEL_GRACE_SEC
        while time.monotonic() < deadline:
            if active.proc.poll() is not None:
                return
            time.sleep(1.0)
        # Grace expired — escalate.
        if active.proc.poll() is None:
            print(f"[spawner] cancel grace expired for run {active.run_id!r}; SIGKILL", file=sys.stderr)
            try:
                os.killpg(os.getpgid(pid), _signal.SIGKILL)
            except (ProcessLookupError, PermissionError) as e:
                print(f"[spawner] SIGKILL to pgid({pid}) failed: {e}", file=sys.stderr)

    # ── internals: error paths ───────────────────────────────────────────
    def _mark_request_failed(self, req_doc, reason: str) -> None:
        try:
            req_doc.reference.update({"status": "rejected", "rejection_reason": reason})
        except Exception as e:
            print(f"[spawner] reject write failed: {e}", file=sys.stderr)


# ── helpers ─────────────────────────────────────────────────────────────────
def _sanitise_args(raw: List[Any]) -> List[str]:
    """Reject shell metacharacters even though we use shell=False — keeps
    the audit log readable and the contract sharp."""
    out: List[str] = []
    for r in raw:
        s = str(r).strip()
        if not s:
            continue
        if _DANGEROUS_ARG.search(s):
            raise ValueError(f"extra_args entry contains forbidden character(s): {s!r}")
        out.append(s)
    return out
