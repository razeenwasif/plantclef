# Distributed training coordinator

A two-language control plane used by the project to coordinate
multi-node training and inference runs. The paper's i002 model can be
trained end-to-end on a single 2-GPU host with plain
`torchrun --nproc_per_node=2 src/train.py [args]`, and **none of this
code is required to reproduce the published numbers**. The coordinator
is preserved here because the development trace used it for the
multi-node experiments that fed the paper's appendix (within-backbone
saturation diagnostics across two pods, the i003 GBIF tail-augmentation
run, and the cross-checkpoint ensembling sweep).

## Layout

```
coordinator/
├── README.md
├── python/cluster/                     Python cluster-manifest module
│   ├── __init__.py
│   ├── manifest.py                     load_manifest, Manifest, HostEntry
│   └── task.py                         TrainingTask + env-var projection
├── go/                                 Go control plane (the "hub")
│   ├── go.mod
│   ├── main.go                         TCP master/worker handshake (NK/GO/HB)
│   ├── main_test.go                    Unit tests for the handshake
│   ├── telemetry.go                    UDP heartbeat receiver + telemetry HTTP API
│   ├── expert/main.go                  Standalone expert launcher (single-node)
│   └── inf/                            Standalone inference launcher
│       ├── go.mod
│       └── main.go
└── configs/
    └── cluster.example.yaml            Multi-host topology example
```

## What it does

* **Python cluster module** (`python/cluster/`). Parses the
  `cluster.yaml` topology file, validates it, and turns it into a
  typed `Manifest` of `HostEntry` objects. The `Manifest.env_for(host)`
  method projects the manifest into the env vars the launcher needs
  (`CLUSTER_MASTER_IP`, `CLUSTER_NODE_RANK`, `CLUSTER_NNODES`,
  `CLUSTER_GPUS` / `CLUSTER_TPU_CORES`). `TrainingTask` is the typed
  unit of work; a project CLI builds one from argparse and hands it to
  the launcher script via `to_env()` / `to_launcher_argv()`.
* **Go hub** (`go/main.go` + `go/telemetry.go`). The hub listens on
  `:9999` (TCP) for the worker handshake and on `:9001` (UDP) for
  per-pod heartbeats. The handshake is a two-byte protocol: master
  sends `NK` ("nuke" = clean up any leftover `torchrun`), then `GO`
  (workers may start), then `HB` every ten seconds for the duration
  of the run. The HTTP telemetry API on `:9000` exposes a JSON view
  of cluster state, the model registry, log tails, and an optional
  Anthropic-API crash-analysis endpoint that the dashboard consumes.
* **Standalone launchers** (`go/expert/`, `go/inf/`). Lightweight
  wrappers that skip the network handshake and launch a local
  training/inference run directly. Useful for single-node sprints
  where you do not need the hub.

## Environment variables

The Go binaries and the Python launcher read the same set of env
vars, all prefixed `CLUSTER_*`:

| Variable                  | Meaning                                                          |
|---------------------------|------------------------------------------------------------------|
| `CLUSTER_MASTER_IP`       | IP / hostname of the rank-0 node                                 |
| `CLUSTER_MASTER_PORT`     | NCCL rendezvous port (default 29505)                             |
| `CLUSTER_NODE_RANK`       | Global rank of this host (0 for master)                          |
| `CLUSTER_NNODES`          | Total number of nodes in the cluster                             |
| `CLUSTER_TOTAL_WORKERS`   | Worker count = `CLUSTER_NNODES - 1` (used by the Go hub)         |
| `CLUSTER_GPUS`            | Per-host CUDA device count override                              |
| `CLUSTER_TPU_CORES`       | Per-host TPU core count override                                 |
| `CLUSTER_MODE`            | `cuda` / `tpu` / `auto`                                          |
| `CLUSTER_SEED`            | Training seed (passed through to the trainer)                    |
| `CLUSTER_NAME`            | Run name (passed through to the trainer)                         |
| `CLUSTER_API_PORT`        | HTTP port for the telemetry API (default 9000)                   |
| `CLUSTER_CLI_PATH`        | Path to the project's CLI used by `/api/launch` and `inf` (default `./coordinator.py`) |

## How the paper used (or did not use) it

The paper's i002 training run is single-node, 2x RTX 5090, launched
directly with `torchrun --nproc_per_node=2 src/train.py`. The
coordinator hub is not needed for that case. The hub became
load-bearing for the broader project when:

1. Two pods of 2x RTX 5090 each were used to run the i002 and i003
   training arms in parallel, with the hub coordinating start-time
   and providing the dashboard heartbeat that operators monitored.
2. The within-backbone saturation diagnostic (Section 5.3) required
   running four checkpoints of the i002 model side-by-side at
   matched inference resolution; the inference launcher (`go/inf/`)
   was used to fan-out those runs.
3. The cross-checkpoint ensembling sweep at the end of the project
   (i002 + 010 ensemble, public 0.389 / private 0.405) was launched
   through the hub so the two backbones could share the same shard
   set without copying it twice.

A reader who only cares about reproducing the headline 0.41826 public
Macro F1 can ignore everything in this directory and run
`python src/inf_script.py` directly against the i002 checkpoint, as
the top-level `README.md` describes.

## Building the Go binaries

```bash
cd coordinator/go
go build -o coord ./...           # the main hub
cd expert && go build -o expert_launcher
cd ../inf && go build -o inf_launcher
```

Tested with `go 1.22`. The hub has no external dependencies (uses
only the Go standard library); the inference and expert launchers
have a single transitive dependency on the standard library too.

## Using the Python cluster module

```python
from coordinator.python.cluster import load_manifest

m = load_manifest("coordinator/configs/cluster.yaml")
me = m.resolve_self()                      # auto-detect by hostname
env = m.env_for(me)                        # dict[str, str] for the launcher
print(env)
# {'CLUSTER_MODE': 'cuda', 'CLUSTER_NNODES': '3', 'CLUSTER_MASTER_IP': 'node-b1.local',
#  'CLUSTER_MASTER_PORT': '29505', 'CLUSTER_GPUS': '8', 'CLUSTER_NODE_RANK': '0'}
```

The only third-party dependency is `pyyaml`; everything else is the
standard library.

## What is *not* included

* The original launcher script is not ported because it routed
  through phase-specific Python modules
  (`phases.foundation_caching.run`, `phases.head_warmup.run`, etc.)
  that do not exist in this repository. Use
  `torchrun --nnodes=$CLUSTER_NNODES --node_rank=$CLUSTER_NODE_RANK
  --master_addr=$CLUSTER_MASTER_IP --master_port=$CLUSTER_MASTER_PORT
  --nproc_per_node=$CLUSTER_GPUS src/train.py [args]` as a drop-in
  replacement.
* The original project-side Python CLI is also not ported. The Go
  telemetry API expects a project-side CLI at `$CLUSTER_CLI_PATH`
  (default `./coordinator.py`) - implement that shim if you want
  to use the dashboard's remote-launch button.
* Pre-built Go binaries are not included. Rebuild from source with
  `go build` per the instructions above.
