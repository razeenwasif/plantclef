# ORACLE Cluster Manifest

A `cluster.yaml` file is the single source of truth for cluster topology — what hosts exist, what accelerator they run, which one is the master, how many devices each carries. Pass it with `--cluster` and `oracle.py` derives every env var `launch_oracle.sh` needs deterministically.

Without a manifest, the launcher still works in its legacy mode (file-lock rank assignment + `ORACLE_MASTER_IP` env). The manifest just removes the implicit coordination.

---

## Quick start

```bash
# On every host of the cluster:
./oracle.py train --phase p2a --cluster configs/cluster.yaml

# That single line resolves:
#   • Which host this is (by hostname / FQDN / aliases)
#   • What role it plays (master / worker — from the manifest)
#   • What its NODE_RANK is (master=0, workers ordered by manifest position)
#   • How many GPUs / TPU cores it owns
#   • Where the master is reachable
#   • Which accelerator mode the cluster runs (cuda / tpu / auto)
```

If auto-detect can't pick a host (e.g. you renamed the box), pass `--host-id b2` explicitly.

---

## YAML schema

```yaml
name: oracle-blackwell-pod   # arbitrary; used in logs
mode: cuda                   # cuda | tpu | auto

master:
  host: b1                   # references one of hosts[].id
  port: 29505                # default

hosts:
  - id: b1
    hostname: oracle-b1.local
    type: cuda               # cuda | tpu
    device_count: 8          # GPUs (cuda) or TPU cores (tpu)
    role: master
    aliases:                 # optional: extra names this host answers to
      - 10.0.0.1

  - id: b2
    hostname: oracle-b2.local
    type: cuda
    device_count: 8
    role: worker

  - id: b3
    hostname: oracle-b3.local
    type: cuda
    device_count: 8
    role: worker
```

Validation enforced by the loader:

| Rule | Why |
|---|---|
| Exactly one host has `role: master` | The launcher needs a deterministic rank 0. |
| `master.host` matches the `id` of that host | Prevents the two sources of truth drifting. |
| All `device_count` values are positive integers | Caught at parse time, not at NCCL init. |
| If `mode` is `cuda` or `tpu`, every host's `type` matches | Heterogeneous clusters need `mode: auto`. |
| Host ids are unique | Self-detect would be ambiguous otherwise. |

A copy-pasteable example lives at `configs/cluster.example.yaml`.

---

## Env vars the manifest emits

When `--cluster` is passed, oracle.py exports these to `launch_oracle.sh` before invoking it:

| Variable | Value (per-host) |
|---|---|
| `ORACLE_MODE` | `manifest.mode` (or `--mode` if the user overrides) |
| `ORACLE_NNODES` | `len(hosts)` |
| `ORACLE_MASTER_IP` | hostname of the master entry |
| `ORACLE_MASTER_PORT` | `master.port` (default `29505`) |
| `ORACLE_NODE_RANK` | `0` for the master, sequential `1..N-1` for workers in manifest order |
| `ORACLE_GPUS` | `host.device_count` (CUDA hosts) |
| `ORACLE_TPU_CORES` | `host.device_count` (TPU hosts) |

Explicit `--rank` still wins over the manifest-derived `NODE_RANK` if you need to override.

---

## Worked examples

### Single-host sprint (no manifest needed)

```bash
./oracle.py train --phase p2a --role sprint --mode cuda
# launch_oracle.sh sees NNODES=1, NODE_RANK=0, master=127.0.0.1
```

### 3-node CUDA pod via manifest

On `b1` (master), `b2`, `b3` — same command everywhere:

```bash
./oracle.py train --phase p2a --cluster configs/cluster.yaml
```

The launcher fans out via `torchrun` with the correct `--nnodes`, `--node_rank`, `--nproc_per_node`, `--master_addr`, `--master_port` derived from the manifest.

### TPU pod slice (v4-32, four v4-8 VMs)

```yaml
name: oracle-v4-32
mode: tpu
master:
  host: tpu-v4-32-0
hosts:
  - id: tpu-v4-32-0
    hostname: 10.130.0.10
    type: tpu
    device_count: 8
    role: master
  - id: tpu-v4-32-1
    hostname: 10.130.0.11
    type: tpu
    device_count: 8
    role: worker
  - id: tpu-v4-32-2
    hostname: 10.130.0.12
    type: tpu
    device_count: 8
    role: worker
  - id: tpu-v4-32-3
    hostname: 10.130.0.13
    type: tpu
    device_count: 8
    role: worker
```

Then on every TPU VM:

```bash
./oracle.py train --phase p2a --cluster configs/cluster.yaml
```

The standard pattern for fanning the same command across all four VMs is `gcloud compute tpus tpu-vm ssh ... --worker=all -- <command>` — that's outside `launch_oracle.sh`'s scope, but the manifest makes the per-VM command identical.

---

## TrainingTask

Under the hood, `--cluster` flows through a `TrainingTask` dataclass in `src/cluster/task.py`:

```python
from src.cluster import build_task_from_args, TrainingTask

task = build_task_from_args(args, unknown)
# task.phase, task.role, task.seed, task.mode, task.manifest, task.host, ...
env = task.to_env()       # dict for launch_oracle.sh
argv = task.to_launcher_argv()  # ['--seed', '42', '--config', ...]
```

This is the unit-of-work abstraction the previous bash + argparse plumbing implicitly had. Code that wants to embed training inside a larger workflow (e.g. a sweep coordinator) can construct `TrainingTask` directly without going through the CLI.

---

## What this doesn't (yet) do

The manifest is a **topology description**, not an **orchestrator**:

- It does not SSH into each host and run the command for you — you still log into each box (or use `gcloud compute tpus tpu-vm ssh --worker=all`, or your favourite multi-host tool).
- It does not detect a dead worker and reassign rank. That's the **NCCL liveness probe** stretch goal.
- It does not coordinate seed sweeps across the cluster. That's the **leader-elected coordinator** stretch goal.

Those are tracked separately and will land in follow-up changes.
