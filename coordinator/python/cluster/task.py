"""
TrainingTask — typed unit of work for distributed training, cache builds, and inference.

A `TrainingTask` bundles everything that distinguishes one run from
another (phase, role, seed, config, accelerator mode, cluster manifest).
The coordinator CLI builds a task from CLI args and hands it to the launcher
via env vars. The trainer can also accept a task object directly when
imported as a library.

Why this exists
---------------
Inspired by the per-device ``DeviceTask`` pattern from
``princesoin/java_netconf_automation_engine`` — explicit work units
beat implicit (phase, role) tuples scattered through bash + argparse.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manifest import Manifest, HostEntry


@dataclass(frozen=True)
class TrainingTask:
    phase: str                              # 'p1' | 'p2a' | 'p2b-student' | …
    role: str                               # 'master' | 'worker' | 'sprint'
    seed: int = 42
    name: Optional[str] = None              # CLUSTER_NAME
    config: Optional[str] = None            # path to YAML config
    dataset: Optional[str] = None
    mode: str = "auto"                      # cuda | tpu | auto
    force: bool = False
    extra_args: List[str] = field(default_factory=list)
    # Resolved at task-build time when --cluster is supplied:
    manifest: Optional[Manifest] = None
    host: Optional[HostEntry] = None        # this machine's entry, if any

    # ── env-var projection ────────────────────────────────────────────────
    def to_env(self) -> Dict[str, str]:
        """Build the env dict the launcher expects. Manifest-derived
        values (NNODES, MASTER_IP, NODE_RANK, GPUs/TPU cores, mode) win
        over the plain task fields when a manifest is supplied."""
        env: Dict[str, str] = {
            "CLUSTER_SEED": str(self.seed),
            "CLUSTER_NAME": self.name or f"cluster_s{self.seed}",
            "CLUSTER_MODE": self.mode,
        }
        if self.manifest and self.host:
            env.update(self.manifest.env_for(self.host))
        return env

    # ── argv projection ───────────────────────────────────────────────────
    def to_launcher_argv(self) -> List[str]:
        """Build the launcher argv (after the phase / role positional)."""
        argv: List[str] = ["--seed", str(self.seed)]
        if self.config:
            argv += ["--config", self.config]
        if self.dataset:
            argv += ["--dataset", self.dataset]
        if self.force:
            argv.append("--force")
        argv += list(self.extra_args)
        return argv

    # ── ergonomic helpers ────────────────────────────────────────────────
    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Manifest is a dataclass with nested lists — keep it lightweight here
        d["manifest"] = self.manifest.name if self.manifest else None
        d["host"] = self.host.id if self.host else None
        return d


def build_task_from_args(args: Any, unknown: List[str]) -> TrainingTask:
    """Convert the argparse Namespace from the coordinator CLI's `train` subparser
    into a TrainingTask. Resolves the manifest + this host if --cluster
    is set."""
    manifest = None
    host = None
    cluster_path = getattr(args, "cluster", None)
    if cluster_path:
        from .manifest import load_manifest
        manifest = load_manifest(cluster_path)
        host_id = getattr(args, "host_id", None)
        host = manifest.resolve(host_id) if host_id else manifest.resolve_self()

    # Mode: explicit --mode wins over manifest.mode.
    mode = getattr(args, "mode", None) or (manifest.mode if manifest else "auto")
    if mode == "auto" and host:
        mode = host.type  # if we know the host's accelerator, use it.

    # Role: explicit --role wins over manifest's host.role.
    role = getattr(args, "role", None) or (host.role if host else "sprint")

    return TrainingTask(
        phase=args.phase,
        role=role,
        seed=getattr(args, "seed", 42),
        name=getattr(args, "name", None),
        config=getattr(args, "config", None),
        dataset=getattr(args, "dataset", None),
        mode=mode,
        force=getattr(args, "force", False),
        extra_args=list(unknown or []),
        manifest=manifest,
        host=host,
    )
