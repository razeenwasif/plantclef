# Roadmap

Working roadmap for Nexus. Updated as initiatives ship.

## Currently active

**Standalone extraction settling.** Just lifted out of the Oracle repo
(shared-codebase era). Verifying production deploy on the new Firebase
project `nexus-cluster`; re-minting pod-agent service-account keys for
the training hosts; updating cross-references (Oracle's `firebase.json`
no longer has the `nexus` Hosting target; Portfolio Nexus entry points
at the new URL).

## Recently shipped

### Visual identity
- **`LightTrails` background** — subtle diagonal light-streak canvas; pure-
  CSS keyframe motion; respects `prefers-reduced-motion`
- **CSS-variable theme tokens** — monochrome palette + one ice-blue accent;
  all components route through `--nexus-*` (with `oracle-*` aliases so
  files copied from the shared codebase keep compiling)
- **`NexusMark` glyph** — two diagonal trail strokes echoing the LightTrails
  canvas; monochrome brand chip in App.tsx header + SidePanel
- **`THEME_COLORS`** map (in `src/lib/site.ts`) — for places Tailwind
  classes can't reach (Recharts strokes, three.js scene, GlowBar tints)
- **Browser-chrome** — `nexus-favicon.svg` + page title

### Visual touches (concept-references pull)
- **KPI sparklines** on `AnalyticsView`'s `StatCards` — `Sparkline.tsx`
- **`RadialGauge`** — circular percentage indicator on the Silicon
  view's "Avg utilisation" tile
- **Utilisation-proportional row fill** on `PodCard` — Quantix-style

### Control plane
- Firestore-backed pod inventory (`/pods`), run lifecycle (`/runs`),
  command surface (`/run_requests`); slice-by-slice incremental rollout
- Python `pod_agent.py` for heartbeats + utilisation telemetry
- `run_monitor.py` projects local training telemetry into `/runs`
- Cancellation flow: owner can flip `cancel_requested` to true; spawner-
  side enforcement; per-rank stale badges
- Live tabs: Fleet, Mission, Analytics, Silicon, Console, Systems
- Adaptive event console derived from state deltas (no separate log
  stream — derived events are cheap)

### Architecture
- Single-tenant standalone Firebase project (`nexus-cluster`)
- Umbrella registry (`src/lib/umbrellas.ts`) — empty by default, ready to
  accept parent products via a single array entry when per-product
  entitlements ship

## Silicon tab — monitoring backlog

Today's Silicon tab shows a point-in-time snapshot of `pod.utilisation`
(GPU util %, temperature °C, power W, VRAM GB) aggregated across the
fleet, plus a decorative `CoreVitals3D` blob that flips colour between
training and idle. Everything below would push Silicon from "status
page" to "real ops tool". Ranked by impact-to-effort. None wired yet.

### Easy wins (1-2 hours each — extra fields in `scripts/pod_agent.py`)
- **Per-device breakdown** on multi-GPU pods. Stop averaging across
  devices; nvidia-smi already returns one row per device. Hidden hot
  GPU on an otherwise-cool host is invisible today.
- **Fan speed** — `nvidia-smi fan.speed`. First signal of cooling stress
  before the temperature spike.
- **ECC + Xid errors** — `nvidia-smi ecc.errors.{corrected,uncorrected}.aggregate`.
  Surfaces hardware degradation early; `uncorrected > 0` is replace-this-
  card territory.
- **Memory bandwidth** — `nvidia-smi dmon -c 1 -s u`. Common bottleneck
  for big-batch training and currently invisible.
- **Driver / firmware version skew badge** — every pod heartbeats its
  versions; surface a chip when not all pods match. Catches the "we
  just upgraded one host" gotcha that silently changes training
  behaviour. ~30 min of work.

### Medium effort (half a day each)
- **Sparklines on every Silicon tile**, last 15 minutes. Today every
  metric is an instantaneous read; a temp spike or util collapse is
  invisible until you happen to glance. `Sparkline.tsx` is already in
  the codebase from Analytics — needs a per-pod rolling-history ring
  buffer fed by the live `/pods` subscription.
- **Throttling indicator** — derived from `clocks.gr` vs `clocks.max.gr`.
  When current < max the GPU is downclocked; surface with an amber chip.
- **NCCL bandwidth** for multi-host training — `nccl-tests` style probes,
  or read from PyTorch's `torch.distributed` collective timings if the
  training code is instrumented. Probably belongs on Mission rather
  than Silicon.

### Bigger lifts (1-2 days each)
- **Thresholded alerts** — "hold temp > 85 °C for 2 min → flip tile to
  red + ping admin". No alerting infrastructure today; would need a
  Cloud Function (or pod-side check + Firestore write) and a
  notification surface.
- **Workload mix breakdown** — "60% training, 30% eval, 10% idle over
  the last hour". Requires per-second resource attribution + history
  retention beyond the current snapshot model.
- **Cost view** — translate `power_w × hours` into kWh and $ at known
  rates. Per-pod and per-run. Useful for budgeting.
- **Per-rank straggler attribution** — connect Mission's "rank 2 stale"
  badge to Silicon's per-device data so you can see which physical GPU
  the stalled rank lives on. Cross-tab plumbing.

### Recommended first move
**Per-device tiles + 15-minute rolling sparklines** (the first easy
win plus the first medium one) together turn Silicon from a status
page into an ops tool — the single change with the highest "feels like
a real product" payoff.

### Out of scope / probably never
- Real-time decoding the `CoreVitals3D` blob's animation parameters
  from live data. Its job is "alive / asleep / off" at a glance; making
  it quantitatively meaningful would compete with the actual metric
  tiles below it without adding precision.

## Console tab — what to wire it up to

The Console tab is technically already wired: `useConsoleEvents()` in
[`lib/console.ts`](../src/lib/console.ts) subscribes to `/pods` + `/runs`
and synthesises high-level lifecycle events ("pod-5090 went online",
"run xyz cancelled", "rank 2 stale"). It looks empty today because no
`pod_agent.py` is heartbeating against `nexus-cluster` yet — boot one
real agent and the state-delta stream fills in.

But that's thin — high-level bullet points, not the continuous stream
of "epoch 3 / 10 · loss 1.23" lines people instinctively expect from a
Terminal-icon tab. Four layers we could add, each independently
shippable, ranked by value-per-effort.

### 1. Real log tail from training runs *(recommended first move)*

`run_monitor.py` already tails the training process's telemetry JSONL.
Extend it to also tail the process's stdout/stderr (the script's main
log file), batch the last ~50 lines into a
`/runs/{runId}.recent_logs` array every 5 s, capped at 200 lines FIFO.

Console merges these with the state-delta events into a single stream,
renders with monospace + colour-coded severity (regex heuristic on
`ERROR | WARN | INFO`). Suddenly the tab shows real training output
crossing the screen, not just lifecycle ticks.

**Effort:** ~half a day. ~30 lines in `run_monitor.py`, ~50 lines in
`lib/console.ts` + `App.tsx`. Bounded write cost (1 write/run/5 s) and
bounded storage (capped array — Firestore handles up to 1 MiB / doc
easily for 200 short log lines). The single change with the highest
"this is a real ops tool" payoff.

### 2. Pod-side heartbeat events *(small but tactile)*

`pod_agent.py` already writes `last_heartbeat`. Add a
`/pods/{podId}.recent_events` array (same array-with-cap pattern as #1)
where the agent emits structured events:
- agent (re)started
- claimed run X
- rejected request Y, reason Z
- spawner exited with code N
- service-account token refreshed

Catches the things state-delta synthesis misses — *why* a request was
rejected, not just that its status flipped to `rejected`.

**Effort:** ~2 hours. ~15 lines on each side.

### 3. Ops command palette *(operational, no new UI shape)*

Generalise the `/run_requests` pattern. New collection
`/pod_commands/{cmdId}`:

```
{
  command_id, target_pod_id,
  command: 'drain' | 'restart-agent' | 'force-offline' | 'reset-current-run',
  issued_by_uid, issued_at,
  status: 'pending' | 'claimed' | 'rejected' | 'completed',
  result?: string
}
```

Pod-agent's existing request-watcher loop also watches `/pod_commands`
filtered to `target_pod_id == self`. Claim, execute, write back result.
Dashboard's ⌘K palette gains "Drain pod-5090", "Restart agent on
pod-pro6k" actions. Console shows command execution as just another
event class — the tab becomes the audit log of operator-issued cluster
actions.

**Effort:** ~half a day for schema + Firestore rules + agent watcher;
~half a day for palette UI + Console renderer for command events.

### 4. Actual interactive shell *(don't build this yet)*

WebSocket bridge to a tmux/bash session on a pod, xterm.js rendering,
session auth, recording. Real shell-through-the-browser.

**High-effort, high-blast-radius** — now we're routing shell access
through a web app, with all the auth, audit, and security-review work
that implies. Worth it only if there's a genuine need to debug stuck
training processes from anywhere with a browser. Even then, plain
`ssh + journalctl -u nexus-pod-agent -f` is usually faster.

**Effort:** 2-3 days minimum, plus ongoing maintenance cost (session
state, websocket scaling, attack-surface review). **Defer indefinitely
until a real reason appears.**

### Recommended sequence

1. Ship **#1** first — turns Console from "lifecycle bulletin" into
   "live training output". Single biggest perceptual win.
2. Ship **#2** alongside or shortly after — small change, fills in the
   "why did X happen" gaps that #1 alone misses.
3. Ship **#3** when there's a real ops need to drain / restart pods
   from the dashboard. Until then `systemctl restart nexus-pod-agent`
   over SSH covers it.
4. **#4** is YAGNI territory. If a paying customer's SLA ever
   demands it, revisit.

## Multi-product training coordination

Long-term direction: Nexus is the **distributed training coordinator**
for a portfolio of AI products. Today there's one product (Oracle) with
its own isolated Firebase project; tomorrow there'll be several (Prism
is next on the list, plus future tools like video gen) and every one
of them should be able to offload training jobs to Nexus with a single
button. This section captures the architecture decision and a concrete
rollout order.

### The shape: hub-and-spoke with Nexus as the identity hub

```
                    nexus-cluster (Firebase project)
                    ┌─────────────────────────────────────┐
                    │   Auth (single sign-on)             │
                    │   Firestore: /pods /runs /requests  │
                    │   Cloud Function: submitTrainingRun │
                    └────────┬────────┬────────┬──────────┘
                             │        │        │
              ┌──────────────┘        │        └──────────────┐
              ▼                       ▼                       ▼
        oracle-neuro-sym         prism-automl         video-gen-thing
        (own Firestore for       (own Firestore       (own Firestore
         user journals,           for AutoML jobs)     for video assets)
         identifications)
```

### Why this shape

- **Auth lives at the hub.** Every product trusts `nexus-cluster` Auth.
  A user signs in once and gets a Firebase ID token with custom claims
  like `{ products: ['oracle', 'prism'], tier_oracle: 'pro', tier_prism: 'free' }`.
- **Product data stays per-product.** Oracle's journals stay in
  `oracle-neuro-sym` Firestore; Prism's AutoML jobs stay in
  `prism-automl` Firestore. Only auth + training are centralised.
- **Training surface is one HTTP endpoint** —
  `POST /trainingRuns` on a Cloud Function in `nexus-cluster`. Each
  product calls it with the user's ID token + a `product_id`
  discriminator. The function validates the claim, writes
  `/run_requests/{id}`. The pod agent claims it as today.
- **Umbrella registry reads claims** —
  [`lib/umbrellas.ts`](../src/lib/umbrellas.ts) already exists with the
  empty-registry pattern. Once claims carry `products`, it surfaces
  the right breadcrumb automatically: "Oracle / Nexus" for an Oracle
  customer, "Prism / Nexus" for a Prism customer, no breadcrumb at all
  for direct Nexus users.

### The tradeoff

Nexus becomes a dependency of every product. If `nexus-cluster` goes
down for maintenance, every product loses sign-in until it's back.
Firebase Auth has a 99.99 % SLA, so the *availability* concern is
mostly theoretical, but the *coupling* is real — you can no longer
take Nexus down for "30 min while I poke at Firestore" without
coordination. Mitigation: the training surface degrades gracefully
(button greys out with "training service unavailable") rather than
breaking the main product UI; auth itself is on Firebase's
infrastructure so it ~never goes down with us.

### Rollout order

Each step is independently shippable; do not start step N+1 until N
is verified in production.

1. **Migrate Oracle's auth to `nexus-cluster`.** Oracle keeps its own
   Firestore for product data; only the Auth domain swaps. ~1 day.
   Validates the pattern with one real product before fanning out.
2. **Add `submitTrainingRun` Cloud Function** on `nexus-cluster` +
   register the Oracle entry in `lib/umbrellas.ts` with a real
   `hasAccess` check that reads `claims.products.includes('oracle')`.
   ~half a day.
3. **Add Oracle's "Train your own model" button.** Probably on the
   Research tab, gated on `claims.tier_oracle === 'pro'`. Modal POSTs
   to the Cloud Function. ~half a day.
4. **Document the pattern in `docs/INTEGRATION.md`.** Copy-paste recipe
   for the next product owner: how to set their app to trust
   `nexus-cluster` Auth, how to call `submitTrainingRun`, how to add
   their umbrella entry. ~half a day.
5. **Migrate Prism** using the recipe. ~1 day. Validates that adding a
   new product is actually cheap once the hub is built. If step 5
   costs more than a day, step 4's docs need revision.

**Total ~3-4 focused days** from today's isolated setup to "any product
in the portfolio can offload training to Nexus with one button".

### Out of scope for this initiative

- **Per-product entitlements** as a separate item below this section —
  largely superseded by this plan, but kept for the case where we want
  read-only "Oracle customers can see Oracle's training runs in
  Nexus's Mission tab" without giving them the Start-Run button.
- **Multi-tenant Nexus** — different question entirely (customer-org
  isolation, not product isolation). Defer until a paying customer
  asks.
- **Cross-project Firestore writes** without going through the Cloud
  Function. Considered briefly; rejected because direct cross-project
  writes would mean each product's frontend carries `nexus-cluster`
  Firestore credentials, which leaks blast radius.

## Future / parking lot

- **Per-product entitlements.** Register a parent-product umbrella (e.g.
  Oracle) in `lib/umbrellas.ts` once we have a way to know the visitor's
  product list (custom claims, shared identity, or a manual bridge).
- **Multi-tenant Nexus.** Each customer org gets isolated Firestore
  data. Today Nexus is single-tenant.
- **Distributed-training stretch tasks** (carried over from the shared
  codebase):
  - Checkpoint-resume idempotency
  - Leader-elected seed-sweep coordinator
- **Sellable Nexus.** Long-term direction. Prerequisites: per-product
  entitlements, separate billing path, customer-facing documentation,
  a real licensing model.
- **AccountMenu simplification.** Today the menu carries `pro`/`field`
  tier UI inherited from the Oracle codebase. Standalone Nexus might
  want a different pricing model (per-pod, per-customer-org) — replace
  the tier UI when that's decided.

## Recent milestones

- 2026-05-20 — Visual decoupling shipped in the Oracle repo
- 2026-05-20 — Standalone Nexus repo created; first commit; deploying
  to fresh `nexus-cluster` Firebase project
