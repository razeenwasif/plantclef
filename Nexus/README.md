# Nexus

**Distributed training console.** Real-time monitoring + control plane for
ML training clusters. Firestore-backed pod inventory, run lifecycle,
multi-rank telemetry, and a glass-aesthetic React shell wired to the
`oracle-pod-agent` service running on each training host.

## Stack

- **Frontend:** React 19 + Vite + Tailwind 4 + Firebase Auth/Firestore
- **Backend:** Firebase project `nexus-cluster` (Firestore + Hosting + Auth)
- **Pod side:** Python `scripts/pod_agent.py` (heartbeat + utilisation) and
  `scripts/run_monitor.py` (telemetry tail) running as a systemd unit on
  each training host

## Local development

```bash
bun install
bun run dev          # http://localhost:9000
```

The dev server uses the same Firebase project as production (`nexus-cluster`),
so a signed-in user sees real fleet state. To work against an empty / fake
fleet, stop your local pod agents or sign in as a different user.

## Deploy

```bash
./scripts/redeploy.sh    # builds + ships to https://nexus-cluster.web.app
```

## Documentation

- [`docs/CONTROL_PLANE.md`](docs/CONTROL_PLANE.md) — pod agent + Firestore
  schema + dashboard subscription model
- [`docs/CLUSTER.md`](docs/CLUSTER.md) — `cluster.yaml` manifest workflow,
  spawner / monitor responsibilities, NCCL liveness probe
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — what shipped, what's active, what's
  parked

## Lineage

Nexus began as a sibling shell to the [Oracle](https://oracle-neuro-sym.web.app)
identification platform (shared React codebase, two Firebase Hosting
targets). After the visual decoupling settled, it was extracted into this
standalone repo. The Oracle umbrella may reappear later as a registered
entry in `src/lib/umbrellas.ts` once per-product entitlements exist; today
the registry is empty and the shell stands alone.
