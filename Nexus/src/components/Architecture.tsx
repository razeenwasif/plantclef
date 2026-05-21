import { motion } from 'framer-motion';
import {
  Server, Network, GitBranch, Database, Activity, Workflow,
  Code2, Radio, Box, ShieldCheck, Cpu,
} from 'lucide-react';

// ─── Shared scaffolding ─────────────────────────────────────────────────────
const Section = ({ id, title, eyebrow, icon: Icon, children }: any) => (
  <motion.section
    id={id}
    initial={{ opacity: 0, y: 16 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.4 }}
    className="mb-20 scroll-mt-24"
  >
    <div className="flex items-center gap-4 mb-6">
      <div className="p-3 rounded-xl bg-oracle-accent/10 border border-oracle-accent/20">
        <Icon size={20} className="text-oracle-accent" />
      </div>
      <div>
        {eyebrow && (
          <p className="text-[10px] font-semibold uppercase tracking-[0.25em] text-oracle-accent mb-1.5">{eyebrow}</p>
        )}
        <h2 className="text-2xl font-light text-white tracking-tight italic">{title}</h2>
      </div>
    </div>
    <div className="text-oracle-muted leading-relaxed space-y-5 text-[15px]">
      {children}
    </div>
  </motion.section>
);

const Card = ({ title, accent = '#a4c4dc', children }: any) => (
  <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-5 hover:border-white/15 transition-colors">
    <h3 className="text-sm font-semibold text-white tracking-tight flex items-center gap-2.5 mb-3">
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: accent, boxShadow: `0 0 8px ${accent}` }} />
      {title}
    </h3>
    <div className="text-[13px] text-oracle-muted leading-relaxed space-y-2">{children}</div>
  </div>
);

const Code = ({ children }: any) => (
  <pre className="bg-black/50 border border-white/5 rounded-xl p-4 overflow-x-auto text-[12px] font-mono leading-relaxed text-white/90 custom-scrollbar my-3">
    <code>{children}</code>
  </pre>
);

const Stat = ({ label, value, sub, accent = '#a4c4dc' }: any) => (
  <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-4">
    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-oracle-muted opacity-70 mb-2">{label}</p>
    <p className="text-2xl font-light tracking-tight tabular-nums" style={{ color: accent }}>{value}</p>
    {sub && <p className="text-[10px] text-oracle-muted mt-1 opacity-70">{sub}</p>}
  </div>
);

const Tag = ({ children, color = 'rgba(255,255,255,0.04)', text = '#cbd5e1' }: any) => (
  <span className="inline-flex items-center px-2.5 py-1 rounded-md text-[10px] font-mono"
        style={{ background: color, color: text, border: `1px solid ${text}22` }}>
    {children}
  </span>
);

// ─── Main ────────────────────────────────────────────────────────────────────
export const Architecture = () => {
  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 md:px-8 py-6 sm:py-12 space-y-10 sm:space-y-12">

        {/* ─── Hero ───────────────────────────────────────────────────── */}
        <header>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-oracle-accent/10 border border-oracle-accent/20 mb-5">
            <Server size={12} className="text-oracle-accent" />
            <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-oracle-accent">Engineering · Control Plane</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-extralight text-white tracking-tight leading-[1.1] italic mb-5 max-w-3xl">
            How NEXUS is <span className="text-oracle-accent font-light not-italic">actually built</span>.
          </h1>
          <p className="text-base text-oracle-muted max-w-2xl leading-relaxed">
            Three small components doing one thing each: a Firestore message bus,
            a Python <code className="font-mono text-white/80">pod_agent.py</code> daemon per training host,
            and the existing training-side telemetry stream as the run-state feedback channel. No
            queues, no orchestrators, no in-band auth tokens — Firebase custom claims do the
            mutual authentication, and every UI surface is a Firestore subscription.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-10">
            <Stat label="Message bus" value="Firestore" sub="three top-level collections" accent="#a4c4dc" />
            <Stat label="Agent" value="pod_agent.py" sub="systemd unit, ~600 LoC" accent="#cfd8e6" />
            <Stat label="Auth" value="Custom claims" sub="pod_agent + pod_id per host" accent="#94a3b8" />
            <Stat label="UI surface" value="Realtime" sub="onSnapshot subscriptions" accent="#bee3eb" />
          </div>
        </header>

        {/* ─── Architecture overview ───────────────────────────────────── */}
        <Section id="overview" title="Architecture at a glance" eyebrow="Map" icon={Workflow}>
          <p>
            Nexus's control plane is deliberately small. There are three Firestore collections,
            one per concern, and exactly one writer per collection per document. State flows
            one direction through each.
          </p>
          <Code>{`Dashboard          Firestore                    Training host
────────           ─────────                    ─────────────
                   /run_requests   ← user click → Start-Run modal
                       │
   Mission tab ← onSnapshot
                       │
                       ▼  claim
                                              pod_agent.py · spawner
                                                  │
                   /runs           ← run_monitor.py tails telemetry JSONL
                       │
   Mission tab + Analytics  ← onSnapshot
                                              pod_agent.py · heartbeat
                   /pods           ←─────────────────┘   every 15 s
                       │
   Fleet tab + Silicon tab ← onSnapshot`}</Code>
          <p>
            <strong className="text-white/90">No polling.</strong> The dashboard subscribes to
            each collection with <code className="font-mono text-white/80">onSnapshot</code>; Firestore
            pushes updates as soon as the writer commits. Same primitive powers Fleet, Mission,
            Analytics, Silicon, and Console — each is a different projection of the same three
            collections.
          </p>
        </Section>

        {/* ─── Pods collection ─────────────────────────────────────────── */}
        <Section id="pods" title="Pods · live fleet inventory" eyebrow="/pods/{podId}" icon={Cpu}>
          <p>
            One document per training host. Written exclusively by the host's own
            <code className="font-mono text-white/80"> pod_agent.py</code>, read by anyone signed in. The
            agent heartbeats every 15 s with current accelerator utilisation; if the heartbeat
            stops, the document remains but its age becomes visible in the Fleet card.
          </p>
          <div className="grid md:grid-cols-2 gap-3 mt-4">
            <Card title="Schema" accent="#a4c4dc">
              <Code>{`{
  pod_id, display_name,
  accelerator: { kind, device_count, sku, memory_gb },
  status:        'online' | 'busy' | 'degraded' | 'offline',
  current_run_id, last_heartbeat,
  utilisation:   { gpu_util_pct, temperature_c, power_w, vram_used_gb },
  agent_version, capabilities, host
}`}</Code>
            </Card>
            <Card title="Write authorisation" accent="#cfd8e6">
              <p>
                Firestore rule: the request must carry <Tag>pod_agent: true</Tag> in its custom-claims
                token, AND its <Tag>pod_id</Tag> claim must equal the document ID. A leaked
                service-account key for pod-5090 cannot forge writes to /pods/pod-pro6k.
              </p>
              <p>
                Claims are minted once per pod via{' '}
                <code className="font-mono text-white/80">scripts/mint_pod_agent_claim.py</code> using
                the Firebase Admin SDK.
              </p>
            </Card>
          </div>
        </Section>

        {/* ─── Run requests ────────────────────────────────────────────── */}
        <Section id="requests" title="Run requests · desired-state inbox" eyebrow="/run_requests/{id}" icon={Box}>
          <p>
            Created by the dashboard's <Tag>Start-Run</Tag> modal, claimed by the targeted pod's
            agent. Status walks from <Tag>pending</Tag> → <Tag>claimed</Tag> or <Tag>rejected</Tag>.
            Once claimed, the agent stamps <code className="font-mono text-white/80">claimed_run_id</code>{' '}
            so the dashboard can follow the link through to the resulting /runs document.
          </p>
          <Code>{`{
  request_id, target_pod_id,
  requested_by_uid, requested_at,
  spec: {
    name, phase,                     // p1 | p2a | p2b | all
    seed, cluster_manifest_ref,
    resume_from, env_overrides, extra_args
  },
  status: 'pending' | 'claimed' | 'rejected',
  claimed_run_id?, rejection_reason?
}`}</Code>
          <p>
            <strong className="text-white/90">Cross-checked target.</strong> The Firestore rule for
            transitioning <Tag>pending</Tag> → <Tag>claimed</Tag> requires the writer's pod_id claim
            to match the request's <code className="font-mono text-white/80">target_pod_id</code>.
            A pod can't poach a request that wasn't directed at it.
          </p>
        </Section>

        {/* ─── Runs ────────────────────────────────────────────────────── */}
        <Section id="runs" title="Runs · observed state" eyebrow="/runs/{runId}" icon={Activity}>
          <p>
            Live mirror of an in-flight training run. The owning host's{' '}
            <code className="font-mono text-white/80">run_monitor.py</code> tails the training process's
            telemetry JSONL and projects step / epoch / validation / heartbeat events into
            <code className="font-mono text-white/80"> runs.metrics</code> and{' '}
            <code className="font-mono text-white/80">runs.ranks</code>. The Mission card and the Analytics
            chart are both derived projections of this document.
          </p>
          <Code>{`{
  run_id, pod_id, request_id,
  spec, started_at, finished_at,
  status: 'starting' | 'running' | 'completed' | 'failed' | 'cancelled',
  exit_code, log_url, source,
  ranks: {                           // per-rank liveness
    "0": { host_id, last_heartbeat, last_step, stale },
    "1": { ... }
  },
  metrics: {                         // monotonic, append-only
    step: [...], loss: [...], local_acc: [...],
    val_step: [...], val_acc: [...]
  },
  cancel_requested,                  // owner-toggle
  created_by_uid                     // copied from /run_requests
}`}</Code>
          <div className="grid md:grid-cols-2 gap-3 mt-4">
            <Card title="Per-rank liveness" accent="#a4c4dc">
              <p>
                Each rank's NCCL collective tickets are observed by the agent. If a rank stops
                making progress for &gt;90 s, its dot in the Mission card flips amber. Doesn't
                kill the run — just surfaces the stall so an operator can decide.
              </p>
            </Card>
            <Card title="Cancellation" accent="#cfd8e6">
              <p>
                Only the user who originated the request (or admin) can flip{' '}
                <code className="font-mono text-white/80">cancel_requested</code> from false to true.
                The agent watches that field and sends SIGTERM (60 s grace) → SIGKILL to the
                training process. Status moves to <Tag>cancelled</Tag>.
              </p>
            </Card>
          </div>
        </Section>

        {/* ─── Pod agent ───────────────────────────────────────────────── */}
        <Section id="agent" title="Pod agent · the host-side executor" eyebrow="scripts/pod_agent.py" icon={Radio}>
          <p>
            A single Python daemon per host, run under systemd
            (<code className="font-mono text-white/80">nexus-pod-agent.service</code>). Three concurrent
            loops, no other threads:
          </p>
          <div className="grid md:grid-cols-3 gap-3 mt-2">
            <Card title="Heartbeat loop" accent="#a4c4dc">
              <p>
                Every 15 s: snapshot
                <Tag color="rgba(255,255,255,0.06)">nvidia-smi</Tag> (or TPU equivalent) →
                write <code className="font-mono text-white/80">/pods/{'{podId}'}</code> with
                fresh utilisation + heartbeat timestamp.
              </p>
            </Card>
            <Card title="Request watcher" accent="#cfd8e6">
              <p>
                Firestore listen on <code className="font-mono text-white/80">/run_requests</code> where
                <code className="font-mono text-white/80"> target_pod_id == self</code>. On a new pending
                request, claim it transactionally, fork the spawner, write{' '}
                <code className="font-mono text-white/80">/runs/{'{runId}'}</code>.
              </p>
            </Card>
            <Card title="Cancel watcher" accent="#94a3b8">
              <p>
                Firestore listen on each owned <code className="font-mono text-white/80">/runs</code>{' '}
                document. On <code className="font-mono text-white/80">cancel_requested == true</code>,
                deliver SIGTERM to the spawner; if still running after 60 s, escalate to SIGKILL.
              </p>
            </Card>
          </div>
          <p className="mt-4">
            <strong className="text-white/90">Stateless across restarts.</strong> The agent doesn't
            persist a local DB. On boot it lists <code className="font-mono text-white/80">/runs</code>{' '}
            with <code className="font-mono text-white/80">pod_id == self</code> and status not in
            terminal state, then attempts to re-attach to those local subprocess by PID. If the
            PID is dead, the run is marked failed with exit_code 137.
          </p>
        </Section>

        {/* ─── Auth ────────────────────────────────────────────────────── */}
        <Section id="auth" title="Auth · custom claims, not service-account email" eyebrow="Identity model" icon={ShieldCheck}>
          <p>
            Every Firestore rule predicates on three custom claims minted onto the agent's
            Firebase user:
          </p>
          <div className="grid md:grid-cols-3 gap-3">
            <Card title="pod_agent : true" accent="#a4c4dc">
              Boolean. Distinguishes agent traffic from human-user traffic in every rule.
            </Card>
            <Card title="pod_id : 'pod-5090'" accent="#cfd8e6">
              Pins the agent to a specific document ID in /pods. Cross-checked on writes
              to /pods and /runs.
            </Card>
            <Card title="capabilities : [...]" accent="#94a3b8">
              Optional. Future: gate which run phases a pod is allowed to claim.
            </Card>
          </div>
          <p className="mt-4">
            Claims are set once per agent via the Admin SDK:
          </p>
          <Code>{`python3 ~/Nexus/scripts/mint_pod_agent_claim.py \\
    --credentials nexus-credentials.json \\
    --uid <firebase-uid-for-bot-user> \\
    --pod-id pod-5090`}</Code>
        </Section>

        {/* ─── Tech stack ──────────────────────────────────────────────── */}
        <Section id="stack" title="Tech stack" eyebrow="Stack" icon={GitBranch}>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Card title="Frontend" accent="#a4c4dc">
              React 19 · Vite · Tailwind 4 · Framer Motion · Recharts · React-Three-Fiber for
              the Silicon vitals scene.
            </Card>
            <Card title="Backend" accent="#cfd8e6">
              Firebase project <code className="font-mono text-white/80">nexus-cluster</code>.
              Firestore for state, Auth for identity, Hosting for the React bundle. No Cloud
              Functions — every write is direct, gated by rules.
            </Card>
            <Card title="Pod side" accent="#94a3b8">
              Python 3.11 · firebase-admin SDK · systemd. The agent has zero non-stdlib deps
              beyond the Firebase SDK.
            </Card>
            <Card title="Observability" accent="#bee3eb">
              The dashboard's Console tab is derived from /pods + /runs deltas in the browser
              — no separate log stream needed. Server-side, just <code className="font-mono text-white/80">journalctl</code>.
            </Card>
            <Card title="Deploy" accent="#cbd5e1">
              <code className="font-mono text-white/80">./scripts/redeploy.sh</code> → Bun build →
              <code className="font-mono text-white/80"> firebase deploy --only hosting</code>. ~30 s
              end-to-end.
            </Card>
            <Card title="Reference docs" accent="#a4c4dc">
              <code className="font-mono text-white/80">docs/CONTROL_PLANE.md</code>,
              <code className="font-mono text-white/80"> docs/CLUSTER.md</code>,
              <code className="font-mono text-white/80"> docs/POD_MIGRATION.md</code> — the
              technical contracts in long-form text.
            </Card>
          </div>
        </Section>

        {/* ─── Footer ──────────────────────────────────────────────────── */}
        <footer className="pt-10 border-t border-white/5">
          <div className="flex items-center gap-2 text-[10px] text-oracle-muted uppercase tracking-[0.25em]">
            <Network size={12} />
            <span>Architecture · single source of truth in docs/CONTROL_PLANE.md</span>
          </div>
        </footer>

      </div>
    </div>
  );
};
