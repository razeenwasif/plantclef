"""Cluster manifest + per-host topology resolution.

A cluster.yaml file describes the cluster (master, workers, accelerator
type, device counts). The loader returns a typed :class:`Manifest` you
can query for "what env vars should this host export to ``the launcher``?"

Public surface:

    from src.cluster import load_manifest, Manifest, HostEntry, TrainingTask

    m = load_manifest("configs/cluster.yaml")
    me = m.resolve_self()              # auto-detect by hostname
    me = m.resolve("b2")               # or explicit
    env = m.env_for(me)                # dict[str, str] for the launcher
"""

from .manifest import load_manifest, Manifest, HostEntry, ClusterValidationError
from .task import TrainingTask, build_task_from_args

__all__ = [
    "load_manifest",
    "Manifest",
    "HostEntry",
    "ClusterValidationError",
    "TrainingTask",
    "build_task_from_args",
]
