# ORACLE Control Plane

How a click in the dashboard's **Mission** tab launches a real training run on a
real pod, and how the **Fleet** tab knows that the pod is alive without anyone
SSHing in. The architecture is deliberately small: Firestore is the message bus,
a tiny `pod_agent.py` daemon is the executor, and the existing telemetry stream
is the run-state feedback channel.

This document is the contract between three components:

1. **Dashboard** (React, hosted on Firebase) — proposes work and observes state.
2. **Firestore** — single source of truth for desired state and observed state.
3. **Pod agent** (`scripts/pod_agent.py`) — runs on every pod; long-lived,
   stateless, idempotent. Reconciles desired ↔ observed.

The accelerator-side machinery (`accelerator.py`, `telemetry.py`,
`liveness.py`) already exists; the control plane only adds a thin orchestrator
on top.

---

## Why Firestore instead of, say, gRPC?

* The dashboard is already authenticated against Firestore for user data
  (Journal, API keys). No new auth boundary.
* No public ingress to home/cluster pods. The agent **polls** Firestore — pods
  never need to be reachable from the internet.
* Firestore subscriptions push state changes to the dashboard for free; no
  WebSocket plumbing.
* Costs are negligible at our scale (a handful of pods, one or two human
  operators).

The trade-off is latency on the order of 1–3 seconds for state propagation,
which is acceptable for a "start a 20-hour training job" workflow. Sub-second
control belongs over a dedicated socket; nothing in ORACLE needs it.

---

## Topology

```
┌─────────────────────────┐                ┌──────────────────────────┐
│   Dashboard (browser)   │  read + write  │       Firestore          │
│   - Fleet (subscribe)   │ ◀────────────▶ │  /pods/{pod_id}          │
│   - Mission (subscribe) │                │  /run_requests/{id}      │
│   - Start-Run modal     │                │  /runs/{run_id}          │
└─────────────────────────┘                └──────────────────────────┘
                                                 ▲           ▲
                                            poll │           │ write
                                                 │           │
                                          ┌──────────────────────┐
                                          │  pod_agent.py         │  one per pod
                                          │   - heartbeat         │
                                          │   - reconcile         │
                                          │   - tail telemetry    │
                                          └──────────────────────┘
                                                 │
                                                 │ exec
                                                 ▼
                                          ┌──────────────────────┐
                                          │  oracle.py / torchrun │
                                          │  emits telemetry      │
                                          └──────────────────────┘
```

The four pod roster targets (these IDs are stable across config + dashboard +
agent):

| pod_id   | host                       | accelerator                  | mode |
|----------|----------------------------|------------------------------|------|
| `local`  | razeen's tower             | 1 × RTX 4090                 | cuda |
| `pod-5090` | rented pod                | 2 × RTX 5090                 | cuda |
| `pod-pro6k`| rented pod                | 1 × RTX PRO 6000 96 GB       | cuda |
| `tpu-v5p` | GCP TPU slice (on demand)  | TPU v5p-8                    | tpu  |

Each row maps one-to-one onto a `cluster.yaml` host entry. The pod_id is the
glue.

---

## Firestore collections

### `/pods/{pod_id}` — pod inventory and live state

Written by the agent every `heartbeat_interval` seconds (default 15) and on
every state transition. Read-only from the dashboard.

```ts
{
  pod_id: string,                 // primary key, matches cluster.yaml host id
  display_name: string,           // "Pod 5090 ×2", "Local 4090", …
  accelerator: {
    kind: 'cuda' | 'tpu' | 'cpu',
    device_count: number,
    sku: string,                  // "RTX 5090", "RTX PRO 6000 96GB", "TPU v5p"
    memory_gb: number,            // per-device
  },
  status: 'online' | 'busy' | 'offline' | 'degraded',
  current_run_id: string | null,  // when status === 'busy'
  last_heartbeat: Timestamp,
  agent_version: string,          // e.g. "pod_agent/0.1.0"
  capabilities: string[],         // ['cuda-12.4', 'fp8', 'flash-attn-4', …]
  utilisation: {                  // optional snapshot for the Fleet card
    gpu_util_pct: number,
    vram_used_gb: number,
    temperature_c: number,
    power_w: number,
  } | null,
}
```

The Fleet tab subscribes to this collection. A pod is rendered **offline** if
`last_heartbeat` is older than 60 seconds (3 × default heartbeat). The agent
doesn't need to write `offline` explicitly; the dashboard makes that decision
from the timestamp.

### `/run_requests/{request_id}` — desired state

Written by the dashboard (via the Start-Run modal). Polled by the agent of the
target pod. Deleted by the agent once it has been claimed and a `/runs/{id}`
document exists.

```ts
{
  request_id: string,             // client-generated UUID v4
  target_pod_id: string,
  requested_by_uid: string,       // Firebase auth uid
  requested_at: Timestamp,
  spec: {
    name: string,                 // human-readable, e.g. "plantclef-p2b-resume"
    phase: 'p1' | 'p2a' | 'p2b' | 'all',
    cluster_manifest_ref: string, // 'configs/cluster.yaml' or inline yaml
    seed: number,
    resume_from: string | null,   // checkpoint path on the pod, optional
    env_overrides: Record<string, string>,  // e.g. ORACLE_NAME, ORACLE_RUN_ID
    extra_args: string[],         // forwarded to oracle.py
  },
  status: 'pending' | 'claimed' | 'rejected',
  rejection_reason?: string,      // set if the agent refuses (busy / mismatch)
}
```

A request transitions through `pending → claimed`. Once claimed, a `/runs/`
document is created and the request is deleted within a few seconds.

### `/runs/{run_id}` — observed state of an in-flight run

Created by the agent under one of two conditions:

* It accepted a `/run_requests/` document (slice 3, not yet shipped); or
* It detected a local telemetry file under `--telemetry-dir` that has no
  matching `/runs/` doc yet — the **re-attach** path that ships in slice 2.

Updated continuously by the agent as it tails the run's telemetry JSONL
files. Read-only from the dashboard. The `source` field distinguishes the
two paths (`"request"` vs `"reattach"`).

```ts
{
  run_id: string,                 // matches ORACLE_RUN_ID exported to the process
  pod_id: string,
  request_id: string,             // back-pointer to the originating request
  created_by_uid: string | null,  // copy of /run_requests/.requested_by_uid; null for reattached runs
  spec: {                         // immutable snapshot of the request spec
    name: string,
    phase: string,
    seed: number,
    cluster_manifest_ref: string,
  },
  started_at: Timestamp,
  finished_at: Timestamp | null,
  status: 'starting' | 'running' | 'completed' | 'failed' | 'cancelled',
  exit_code: number | null,
  ranks: {                        // live per-rank summary, from telemetry/liveness
    [rank: string]: {
      host_id: string,
      last_heartbeat: Timestamp,
      last_step: number | null,
      stale: boolean,
    }
  },
  metrics: {                      // bounded rolling window — last 200 points
    step: number[],
    loss: number[],
    local_acc: number[],
    val_acc: number[],
  },
  cancel_requested: boolean,      // dashboard sets to request graceful stop
  log_url: string | null,         // optional Cloud Storage link to full logs
}
```

The Mission tab subscribes to `/runs` filtered to `status in ('starting',
'running')`. Completed/failed runs are listed under a "Recent" section.

---

## Security rules (delta from current `firestore.rules`)

Append the following blocks to the existing rules. They preserve the existing
`isAdmin()` predicate (`razeen.wasif66@gmail.com`).

```cel
match /databases/{database}/documents {

  // … existing /users/{uid}/journal and /users/{uid}/apiKeys rules …

  match /pods/{podId} {
    // Anyone signed in can read pod inventory (Fleet tab is universal).
    allow read: if request.auth != null;
    // Only the pod agent writes. We enforce this with a custom-claim
    // 'pod_agent' set on a per-pod service account. No human writes.
    allow write: if request.auth != null
                 && request.auth.token.pod_agent == true
                 && request.auth.token.pod_id == podId;
  }

  match /run_requests/{requestId} {
    // Owner reads their own request; admin reads all.
    allow read: if request.auth != null
                && (resource.data.requested_by_uid == request.auth.uid
                    || isAdmin());
    // Authenticated users create requests; cannot mutate after creation.
    allow create: if request.auth != null
                  && request.resource.data.requested_by_uid == request.auth.uid;
    // Only the targeted pod agent flips status / deletes.
    allow update, delete: if request.auth.token.pod_agent == true
                          && request.auth.token.pod_id == resource.data.target_pod_id;
  }

  match /runs/{runId} {
    // Anyone signed in can observe runs (operations transparency).
    allow read: if request.auth != null;
    // Only the owning pod agent writes telemetry / status.
    allow create, update: if request.auth.token.pod_agent == true
                          && request.resource.data.pod_id == request.auth.token.pod_id;
    // Owner of the originating request (or admin) can flip cancel_requested.
    allow update: if request.auth != null
                  && request.resource.data.diff(resource.data).affectedKeys()
                        .hasOnly(['cancel_requested'])
                  && (isAdmin() || ownerOfRequest(resource.data.request_id));
    // No one deletes runs through the API; they're archival.
    allow delete: if false;
  }
}

function ownerOfRequest(requestId) {
  return get(/databases/$(database)/documents/run_requests/$(requestId))
           .data.requested_by_uid == request.auth.uid;
}
```

The `pod_agent` custom claim is minted out-of-band: see "Bootstrapping a pod"
below.

---

## `scripts/pod_agent.py` contract

Long-lived daemon, one per pod. Stateless across restarts (all state lives in
Firestore). Idempotent: if it crashes mid-run, the next start picks up the
in-flight run from `/runs/` and resumes tailing telemetry.

```text
USAGE:
  python scripts/pod_agent.py \
      --pod-id pod-5090 \
      --service-account /etc/oracle/pod_agent.json \
      --cluster configs/cluster.yaml \
      [--heartbeat-interval 15] \
      [--poll-interval 5] \
      [--telemetry-dir reports/telemetry/]
```

### Lifecycle

```
start
 ├─▶ load service-account credentials
 ├─▶ register self in /pods/{pod_id}      (status='online')
 ├─▶ start heartbeat loop (writes /pods/{pod_id}.last_heartbeat)
 ├─▶ on startup, scan /runs/ for {pod_id, status in ('starting','running')}
 │     and if one exists with the local oracle.py process still alive,
 │     re-attach (resume telemetry-tail)
 ├─▶ enter reconcile loop:
 │     while True:
 │       req = first /run_requests/ where target_pod_id == self
 │       if req:
 │         if self.status == 'busy': reject(req, 'pod is busy')
 │         else: claim(req) → spawn oracle.py → create /runs/{id}
 │       sleep(poll_interval)
shutdown (SIGTERM):
 ├─▶ set /pods/{pod_id}.status = 'offline'
 ├─▶ leave any in-flight run untouched (operator decides what to do)
```

### Spawning a run

The agent shells out to `./oracle.py` (already the canonical entry point for
training) with the request's spec translated into the existing flags:

```bash
ORACLE_RUN_ID=$run_id \
ORACLE_NAME=$spec.name \
ORACLE_HOST_ID=$pod_id \
ORACLE_SEED=$spec.seed \
./oracle.py train \
    --phase $spec.phase \
    --cluster $spec.cluster_manifest_ref \
    --host-id $pod_id \
    $spec.extra_args
```

`ORACLE_RUN_ID` flows through `telemetry.bind()` (already supported in
`trainer.py`) so the JSONL files end up under
`reports/telemetry/oracle_{phase}_r{rank}_{ts}_{run_id}.jsonl`. The agent
watches that directory, parses each new line, and pushes a downsampled
summary into `/runs/{run_id}.metrics`.

The same logic the existing `LivenessMonitor` uses to detect stale ranks is
reused — the agent imports it directly:

```python
from src.training.liveness import LivenessMonitor
monitor = LivenessMonitor(telemetry_dir=args.telemetry_dir,
                          world_size=cluster_world_size,
                          stale_after_sec=120,
                          on_stale=lambda rank, h: mark_rank_stale(run_id, rank))
monitor.start()
```

### Cancellation

The dashboard sets `/runs/{run_id}.cancel_requested = true`. The agent's
reconcile loop notices, sends SIGTERM to the run's process group, waits up
to 60 s for graceful shutdown (DeepSpeed/NCCL teardown), then SIGKILL. The
run document is updated to `status='cancelled'` and `finished_at=now`.

### Failure modes

| Failure                              | Agent behaviour                                    |
|--------------------------------------|----------------------------------------------------|
| Network blip mid-run                 | Heartbeat resumes; run keeps running locally       |
| Agent crash mid-run                  | On restart, re-attaches via /runs/ lookup          |
| oracle.py exits non-zero             | Run flipped to 'failed' with `exit_code` set       |
| Telemetry directory disappears       | Run flipped to 'degraded'; agent keeps reconciling |
| Rank goes silent > 120 s             | LivenessMonitor flags rank; run not killed         |
| Pod power-cycles                     | Run shows 'failed' (no heartbeat); admin restarts  |

The agent never silently retries a failed run. Re-runs are explicit dashboard
clicks.

---

## Dashboard wiring

### Fleet tab (`dashboard/src/App.tsx`, cluster view)

Replace the hard-coded `GPU_FLEET` array with a Firestore subscription. The
hook should live in `dashboard/src/lib/fleet.ts`:

```ts
export function useFleet(): Pod[] {
  const [pods, setPods] = useState<Pod[]>([]);
  useEffect(() => {
    const q = query(collection(db, 'pods'), orderBy('display_name'));
    return onSnapshot(q, snap => {
      const now = Date.now();
      setPods(snap.docs.map(d => {
        const data = d.data();
        const last = data.last_heartbeat?.toMillis() ?? 0;
        const stale = now - last > 60_000;
        return { ...data, id: d.id, derivedStatus: stale ? 'offline' : data.status };
      }));
    });
  }, []);
  return pods;
}
```

Each fleet card maps onto a `Pod`:

```
┌─ Local 4090 ──────────────── ●online ─┐
│ RTX 4090 · 24 GB · cuda               │
│ GPU 0% · 38°C · 28 W                  │
│ [ Start Run ▶ ]                       │
└───────────────────────────────────────┘
```

The `Start Run` button on each card opens the Start-Run modal pre-filled with
that pod's `pod_id`.

### Mission tab

Replace the hard-coded `JOB_TABLE` with a Firestore subscription on `/runs/`
filtered to `status in ('starting', 'running')`, plus a paginated "Recent
runs" list for completed runs. Each card shows:

* `spec.name` and `pod_id` (e.g. *"plantclef-p2b-resume on pod-pro6k"*)
* `started_at` / runtime
* The latest `metrics.step` / `metrics.loss` value
* Per-rank liveness badges (green/yellow) derived from `ranks[r].stale`
* `[ Cancel ]` button that flips `cancel_requested`

A small sparkline of `metrics.loss` is rendered with Recharts (we already
depend on it for the Analytics tab).

### Start-Run modal

Triggered from any Fleet card's `Start Run` button or the global command
palette (`> Start Run …`). Lives at
`dashboard/src/components/StartRunModal.tsx`. Flow:

1. **Target pod** — pre-filled from the launching context; user can switch.
2. **Phase** — segmented control: `p1 · p2a · p2b · all`.
3. **Name** — free-text, defaults to `${phase}-${pod_id}-${YYYYMMDD-HHMM}`.
4. **Seed** — number input, default 42.
5. **Cluster manifest** — dropdown of paths under `configs/`, default
   `configs/cluster.yaml`.
6. **Resume from** — optional checkpoint path on the pod's filesystem.
7. **Extra env** — key/value rows; merged into `env_overrides`.
8. **Submit**:

   ```ts
   await addDoc(collection(db, 'run_requests'), {
     request_id: crypto.randomUUID(),
     target_pod_id: pod.id,
     requested_by_uid: user.uid,
     requested_at: serverTimestamp(),
     spec: { name, phase, seed, cluster_manifest_ref, resume_from, env_overrides, extra_args },
     status: 'pending',
   });
   ```

   The modal stays open, subscribed to the request doc, showing one of:
   *queued → claimed → running* (then closes and routes the user to the new
   `/runs/{id}` card in Mission), *rejected* (with reason), or *timed out*
   after 30 s of pending.

Validation: phase + seed + cluster_manifest_ref are required; `extra_args`
strings are trimmed and rejected if they contain `;`, `|`, `&`, backticks, or
newlines (the agent runs them through `subprocess` with `shell=False`, so
shell metacharacters aren't *exploitable*, but we still reject them to keep
the audit log clean).

---

## Bootstrapping a pod

One-time setup, per pod, by the admin (razeen). All three scripts live under
`scripts/` and ship as part of slice 1.

1. **Create a service-account *user*** in Firebase Auth (Console → Authentication
   → Add user, with a synthetic email like `pod-5090@oracle.local`). Note its
   `uid`.

2. **Mint the pod-agent claims** from your operator machine, using a separate
   admin-SDK credential file you keep off the pod:

   ```bash
   python scripts/mint_pod_agent_claim.py \
       --admin-credentials ~/.config/oracle/firebase-admin.json \
       --service-account-uid <uid from step 1> \
       --pod-id pod-5090
   ```

   This sets the custom claims `{ pod_agent: true, pod_id: 'pod-5090' }` on
   that user. The firestore.rules block on `/pods/{podId}` cross-checks
   `pod_id == podId` so a JSON key for `pod-5090` cannot write `pod-pro6k`'s
   document.

3. **Download a service-account JSON** for that user (Firebase Console →
   Project Settings → Service Accounts) and copy it to the pod at
   `/etc/oracle/pod_agent.json` (`chmod 600`).

4. **Smoke-test on the pod** before enabling the systemd unit:

   ```bash
   python scripts/pod_agent.py \
       --pod-id pod-5090 \
       --service-account /etc/oracle/pod_agent.json \
       --cluster configs/cluster.yaml \
       --once
   ```

   The Fleet tab in the dashboard should show the pod within ~15 s. The
   `--once` flag writes a single heartbeat and exits so you can verify auth
   and rules without committing to a long-running daemon.

5. **Install the systemd unit** (`scripts/oracle-pod-agent.service`,
   ships in the repo — edit `POD_ID` and `User` to match this pod):

   ```bash
   sudo cp scripts/oracle-pod-agent.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now oracle-pod-agent
   sudo systemctl status oracle-pod-agent
   ```

   The unit's `KillSignal=SIGTERM` plus the agent's signal handler ensure
   that on `systemctl stop`, the pod's status is flipped to `offline`
   before the process exits, so the Fleet card doesn't sit in a stale
   `online` state until heartbeat expiry.

6. **Verify** — refresh the Fleet tab. The pod's card should appear as
   `online` with live `gpu_util_pct` / `temperature_c` / `power_w` (CUDA
   pods only — TPU/CPU pods show inventory without per-device telemetry).

The TPU "pod" is a slight special case: TPUs are ephemeral (created on demand
in GCP). The same agent runs on the TPU VM and writes `accelerator.kind='tpu'`
in its inventory document; the only difference is that its lifecycle is
bounded by the GCP slice's lifetime, and `utilisation` is null because
`nvidia-smi` doesn't apply (XLA equivalents land in a later slice).

---

## What's deliberately *not* in this design

* **No SSH from the dashboard.** Pods are never publicly reachable.
* **No streaming of full logs through Firestore.** Logs go to local disk;
  optionally rsynced to GCS, exposed via a signed URL in `runs.log_url`.
* **No multi-tenancy.** All pods belong to the admin; users propose runs but
  only the admin's pod_agents will execute them. Tier-gating happens
  upstream in the modal (free/pro users can't submit at all).
* **No scheduling / queuing.** A pod accepts one run at a time. If a request
  arrives while a pod is busy, it is *rejected*, not queued. The dashboard's
  Mission tab is responsible for surfacing this and letting the user retry
  on another pod.
* **No automatic resume on failure.** Failed runs stay failed until a human
  re-submits. The point of the control plane is visibility, not autonomy.

These omissions are not bugs; they are the line between an ergonomic personal
control plane and a multi-team scheduler. We do not need the latter.

---

## Implementation order

The work breaks into four independent slices, each shippable on its own:

1. **`/pods/` schema + agent heartbeat** — `pod_agent.py` registers and
   heartbeats. Fleet tab subscribes to live data. **Shipped.**
2. **`/runs/` schema + Mission live data** — agent re-attaches existing
   processes by tailing `--telemetry-dir`. Mission tab subscribes to live
   data with a sparkline. No Start-Run modal yet. **Shipped.**
3. **`/run_requests/` + Start-Run modal** — full Mission round-trip:
   dashboard → request → agent → run → telemetry stream. Pod agent's
   `run_spawner.py` polls `/run_requests`, claims pending ones targeting
   it, pre-creates `/runs/{run_id}` with `source='request'`, spawns
   `oracle.py train` via subprocess (passing `ORACLE_RUN_ID_TEMPLATE` so
   every rank shares a stamp+short while keeping per-rank filenames
   distinct), and watches the process for an exit-code-driven status
   flip. Modal subscribes to its request doc and routes to Mission on
   `claimed`. **Shipped.**
4. **Cancellation + LivenessMonitor integration** — Mission's Cancel
   button flips `cancel_requested=true`; the spawner's poll loop catches
   it, SIGTERMs the process group, escalates to SIGKILL after a 60 s
   grace, and the watcher writes `status='cancelled'`. Per-rank stale
   badges render from the existing `runs.ranks[*].stale` field. Only
   the user who submitted the originating request (via the new
   `created_by_uid` field) — or admin — can issue a cancel; firestore
   rules pin the toggle to that single field. **Shipped.**

Anything beyond slice 4 (auto-resume, log streaming, scheduling) is out of
scope and lives in `docs/NOVELTY_BACKLOG.md`.

---

## Cross-references

* [TPU support](./TPU.md) — accelerator mode plumbing
* [Cluster manifest](./CLUSTER.md) — `cluster.yaml` schema referenced by
  `spec.cluster_manifest_ref`
* `src/training/telemetry.py` — JSONL event format the agent tails
* `src/training/liveness.py` — rank-staleness probe reused by the agent
* `firestore.rules` — security rules this design extends
