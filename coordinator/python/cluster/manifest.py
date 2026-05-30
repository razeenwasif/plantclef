"""
Cluster manifest schema + loader.

The manifest is a single source of truth for cluster topology, replacing
the implicit "discover the master via a file on shared FS, atomically
increment a rank counter" dance that ``the launcher`` does today.
The launcher still works without a manifest (back-compat), but when
``--cluster cluster.yaml`` is passed to ``the coordinator CLI``, the env vars it
would otherwise sniff are derived deterministically from the YAML.

YAML schema
-----------
.. code-block:: yaml

    name: cluster-blackwell-pod
    mode: cuda          # cuda | tpu | auto
    master:
      host: b1          # references an entry in hosts[].id
      port: 29505       # default 29505 if omitted

    hosts:
      - id: b1
        hostname: node-b1.local   # resolvable name OR IP
        type: cuda                  # cuda | tpu (must match `mode` or auto)
        device_count: 8             # GPUs (CUDA) or TPU cores (TPU)
        role: master                # master | worker
        aliases: [10.0.0.1]         # extra hostnames/IPs for self-detect

      - id: b2
        hostname: node-b2.local
        type: cuda
        device_count: 8
        role: worker

Notes
-----
* ``role: master`` is required on exactly one host. All others must be ``worker``.
* If the user omits ``--host-id``, ``Manifest.resolve_self()`` picks the
  entry whose hostname matches ``socket.gethostname()`` (or one of the
  aliases). If nothing matches, it raises a clear error.
* ``mode`` here is the cluster's preferred accelerator. Individual hosts
  *can* still override via ``CLUSTER_MODE`` env, but most clusters are
  homogeneous so this lets the user set it once.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


class ClusterValidationError(ValueError):
    """Raised when a cluster.yaml is malformed or internally inconsistent."""


@dataclass(frozen=True)
class HostEntry:
    id: str
    hostname: str
    type: str               # 'cuda' | 'tpu'
    device_count: int
    role: str               # 'master' | 'worker'
    aliases: List[str] = field(default_factory=list)

    def names(self) -> List[str]:
        """Every name this host answers to — used for self-detect."""
        return [self.id, self.hostname, *self.aliases]


@dataclass(frozen=True)
class Manifest:
    name: str
    mode: str               # 'cuda' | 'tpu' | 'auto'
    hosts: List[HostEntry]
    master_id: str
    master_port: int = 29505

    # ── lookups ─────────────────────────────────────────────────────────
    def master(self) -> HostEntry:
        return self.resolve(self.master_id)

    def workers(self) -> List[HostEntry]:
        return [h for h in self.hosts if h.role == "worker"]

    def resolve(self, host_id: str) -> HostEntry:
        for h in self.hosts:
            if h.id == host_id:
                return h
        raise ClusterValidationError(
            f"Host id {host_id!r} not found in manifest {self.name!r}. "
            f"Known ids: {[h.id for h in self.hosts]}"
        )

    def resolve_self(self) -> HostEntry:
        """Pick the entry matching this machine's hostname / FQDN."""
        candidates = {
            socket.gethostname().lower(),
            _safe_fqdn().lower(),
        }
        for h in self.hosts:
            if any(n.lower() in candidates for n in h.names()):
                return h
        raise ClusterValidationError(
            f"Could not auto-detect a host entry for {socket.gethostname()!r} "
            f"(also tried FQDN). Pass --host-id explicitly. Known hosts: "
            f"{[h.id + ' (' + h.hostname + ')' for h in self.hosts]}"
        )

    # ── env vars for the launcher ───────────────────────────────────
    def env_for(self, host: HostEntry) -> Dict[str, str]:
        """Derive the launcher env vars from the manifest, for *this* host."""
        env: Dict[str, str] = {
            "CLUSTER_MODE": self.mode,
            "CLUSTER_NNODES": str(len(self.hosts)),
            "CLUSTER_MASTER_IP": self.master().hostname,
            "CLUSTER_MASTER_PORT": str(self.master_port),
        }
        # Per-host accelerator capacity
        if host.type == "tpu":
            env["CLUSTER_TPU_CORES"] = str(host.device_count)
        else:
            env["CLUSTER_GPUS"] = str(host.device_count)
        # Rank: master is rank 0, workers ordered by their list position
        ranks = {h.id: 0 if h.id == self.master_id else i + 1
                 for i, h in enumerate([w for w in self.hosts if w.id != self.master_id])}
        # That dict comp is a touch awkward — simpler:
        workers_sorted = [h for h in self.hosts if h.id != self.master_id]
        rank = 0 if host.id == self.master_id else (
            next(i for i, h in enumerate(workers_sorted) if h.id == host.id) + 1
        )
        env["CLUSTER_NODE_RANK"] = str(rank)
        return env


# ── loader ─────────────────────────────────────────────────────────────────
def load_manifest(path: Union[str, Path]) -> Manifest:
    """Parse + validate a cluster.yaml. Raises ``ClusterValidationError``
    on any structural problem."""
    p = Path(path)
    if not p.exists():
        raise ClusterValidationError(f"Manifest not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ClusterValidationError(f"Manifest {p} did not parse to a mapping.")
    return _validate(raw, source=str(p))


# ── internals ──────────────────────────────────────────────────────────────
def _validate(raw: dict, source: str) -> Manifest:
    required_top = {"name", "mode", "hosts", "master"}
    missing = required_top - set(raw)
    if missing:
        raise ClusterValidationError(f"{source}: missing top-level keys {sorted(missing)}")

    name = str(raw["name"])
    mode = str(raw["mode"]).lower()
    if mode not in ("cuda", "tpu", "auto"):
        raise ClusterValidationError(f"{source}: mode must be cuda | tpu | auto, got {mode!r}")

    master_raw = raw["master"]
    if not isinstance(master_raw, dict) or "host" not in master_raw:
        raise ClusterValidationError(f"{source}: master must be a mapping with at least `host`")
    master_id = str(master_raw["host"])
    master_port = int(master_raw.get("port", 29505))

    hosts_raw = raw["hosts"]
    if not isinstance(hosts_raw, list) or not hosts_raw:
        raise ClusterValidationError(f"{source}: hosts must be a non-empty list")

    hosts: List[HostEntry] = []
    seen_ids: set[str] = set()
    masters: List[str] = []
    for i, h in enumerate(hosts_raw):
        if not isinstance(h, dict):
            raise ClusterValidationError(f"{source}: hosts[{i}] must be a mapping")
        for k in ("id", "hostname", "type", "device_count", "role"):
            if k not in h:
                raise ClusterValidationError(f"{source}: hosts[{i}] missing key {k!r}")
        if h["id"] in seen_ids:
            raise ClusterValidationError(f"{source}: duplicate host id {h['id']!r}")
        seen_ids.add(h["id"])
        if h["type"] not in ("cuda", "tpu"):
            raise ClusterValidationError(
                f"{source}: hosts[{i}].type must be cuda | tpu, got {h['type']!r}"
            )
        if h["role"] not in ("master", "worker"):
            raise ClusterValidationError(
                f"{source}: hosts[{i}].role must be master | worker, got {h['role']!r}"
            )
        if h["role"] == "master":
            masters.append(h["id"])
        if int(h["device_count"]) <= 0:
            raise ClusterValidationError(
                f"{source}: hosts[{i}].device_count must be positive, got {h['device_count']}"
            )
        hosts.append(HostEntry(
            id=str(h["id"]),
            hostname=str(h["hostname"]),
            type=str(h["type"]),
            device_count=int(h["device_count"]),
            role=str(h["role"]),
            aliases=[str(a) for a in (h.get("aliases") or [])],
        ))

    if len(masters) != 1:
        raise ClusterValidationError(
            f"{source}: exactly one host must have role=master (found {masters or 'none'})"
        )
    if masters[0] != master_id:
        raise ClusterValidationError(
            f"{source}: master.host = {master_id!r} but the master role is assigned to "
            f"{masters[0]!r}. Pick one source of truth."
        )

    # If mode is cuda or tpu, all hosts should agree; warn if mismatched.
    if mode in ("cuda", "tpu"):
        wrong = [h.id for h in hosts if h.type != mode]
        if wrong:
            raise ClusterValidationError(
                f"{source}: cluster mode={mode!r} but these hosts have a different type: {wrong}. "
                f"Either flip the cluster mode to 'auto' or fix the host types."
            )

    return Manifest(
        name=name, mode=mode, hosts=hosts,
        master_id=master_id, master_port=master_port,
    )


def _safe_fqdn() -> str:
    """Avoid hangs / DNS exceptions on certain configurations."""
    try:
        return socket.getfqdn()
    except Exception:
        return socket.gethostname()
