import { motion } from 'framer-motion';
import {
  BookOpen, Layers, GitBranch, Workflow, Wrench, Sparkles,
  ShieldCheck, AlertCircle, Rocket,
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

const Choice = ({ title, accent = '#a4c4dc', children }: any) => (
  <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-5 hover:border-white/15 transition-colors">
    <h3 className="text-sm font-semibold text-white tracking-tight flex items-center gap-2.5 mb-3">
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: accent, boxShadow: `0 0 8px ${accent}` }} />
      {title}
    </h3>
    <div className="text-[13px] text-oracle-muted leading-relaxed space-y-2">{children}</div>
  </div>
);

const Slice = ({ n, title, summary }: { n: string; title: string; summary: string }) => (
  <div className="flex gap-4 py-3 border-b border-white/5 last:border-b-0">
    <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-oracle-accent/10 border border-oracle-accent/20 flex items-center justify-center font-mono text-[11px] font-bold text-oracle-accent">
      {n}
    </div>
    <div className="min-w-0">
      <p className="text-sm font-semibold text-white tracking-tight mb-1">{title}</p>
      <p className="text-[12px] text-oracle-muted leading-relaxed">{summary}</p>
    </div>
  </div>
);

// ─── Main ────────────────────────────────────────────────────────────────────
export const Methodology = () => {
  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 md:px-8 py-6 sm:py-12 space-y-10 sm:space-y-12">

        {/* ─── Hero ───────────────────────────────────────────────────── */}
        <header>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-oracle-accent/10 border border-oracle-accent/20 mb-5">
            <BookOpen size={12} className="text-oracle-accent" />
            <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-oracle-accent">Methodology · Design Notes</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-extralight text-white tracking-tight leading-[1.1] italic mb-5 max-w-3xl">
            Why NEXUS looks <span className="text-oracle-accent font-light not-italic">like this</span>.
          </h1>
          <p className="text-base text-oracle-muted max-w-2xl leading-relaxed">
            The Architecture tab covers <em>what</em> the pieces are. This one covers <em>why</em> —
            the choices behind a Firestore-backed control plane instead of a queue, custom-claim
            auth instead of service-account email lookups, and a slice-by-slice rollout instead of
            a single big-bang launch. Every choice has an obvious-in-hindsight motivation and a
            less-obvious alternative we rejected.
          </p>
        </header>

        {/* ─── Slice-by-slice rollout ─────────────────────────────────── */}
        <Section id="slices" title="Slice-by-slice rollout" eyebrow="Process" icon={Layers}>
          <p>
            Nexus shipped as four slices, each a self-contained increment that left the system
            useful in its own right. The alternative — design the whole control plane up front,
            implement it, then flip the switch — would have hidden integration issues until the
            last day. The slice cadence let every commit run end-to-end against a real Firestore.
          </p>
          <div className="bg-white/[0.02] border border-white/5 rounded-2xl p-2 mt-4">
            <Slice n="1" title="/pods inventory + heartbeat + Fleet tab"
                   summary="Pod-agent skeleton, /pods schema, Firestore rules, Fleet card. End: a real pod registers + heartbeats; the dashboard shows it. No runs yet." />
            <Slice n="2" title="/runs + Mission live data + run_monitor.py"
                   summary="run_monitor.py tails training telemetry; Mission tab subscribes. End: a manually-launched training run shows up in the dashboard with live loss curves." />
            <Slice n="3" title="/run_requests + Start-Run modal + spawner"
                   summary="Dashboard can request runs; pod agents claim + spawn. End: clicking Start-Run on the dashboard actually launches training. The dashboard is now a control plane, not a viewer." />
            <Slice n="4" title="Cancellation + per-rank stale badges"
                   summary="Owner-toggle cancel_requested, spawner-side SIGTERM, multi-rank stale detection. End: full lifecycle, including bailing on a misbehaving run." />
          </div>
          <p className="mt-4">
            Each slice's Firestore rules went in first, so the dashboard could be developed
            against a real backend without bypassing security. The agents matched, not the
            other way around.
          </p>
        </Section>

        {/* ─── Firestore vs alternatives ──────────────────────────────── */}
        <Section id="firestore" title="Why Firestore as the message bus" eyebrow="Backend choice" icon={Workflow}>
          <p>
            The obvious alternatives were a relational DB with a websocket layer (Postgres +
            something), a job queue (Celery, RQ, Kafka), or a dedicated agent protocol
            (gRPC + Envoy). All of them required hosting infrastructure that Nexus didn't have.
            Firestore is fully managed, has push-based subscriptions in the SDK, and the security
            rules ARE the API contract. The downsides — write quotas, no joins, no transactions
            across multiple documents — turned out not to bite for this workload.
          </p>
          <div className="grid md:grid-cols-2 gap-3 mt-4">
            <Choice title="What Firestore got us" accent="#a4c4dc">
              <p>
                Push-based subscriptions out of the box (no websocket layer to build / scale).
                Rules-as-code mean the security model lives in version control, not in a wiki.
                Zero ops — no nodes to patch, no migrations to coordinate.
              </p>
            </Choice>
            <Choice title="What we gave up" accent="#cfd8e6">
              <p>
                No joins — every dashboard projection denormalises in the browser. Writes are
                metered (1 write / sec / document is a real ceiling), so the run-monitor batches
                metric appends every ~5 steps. No native event log — we synthesise one from
                /pods + /runs deltas in the Console tab.
              </p>
            </Choice>
          </div>
        </Section>

        {/* ─── Custom claims ──────────────────────────────────────────── */}
        <Section id="claims" title="Why custom-claim auth" eyebrow="Identity" icon={ShieldCheck}>
          <p>
            The simpler-sounding option was "agent runs as a service-account, rules check the
            email matches an allow-list". That works for a single fleet but fails the moment you
            want a pod to write its own document and only its own document. With a static
            service-account identity, every agent shares the same write capability across all
            pods.
          </p>
          <p>
            Custom claims (<code className="font-mono text-white/80">pod_agent: true</code>,{' '}
            <code className="font-mono text-white/80">pod_id: 'pod-5090'</code>) let the security rules
            require both "the writer is an agent" AND "the writer's pod_id matches this document's
            ID". A leaked credential for pod-5090 cannot forge writes to pod-pro6k's row.
          </p>
          <Choice title="The tradeoff we accepted" accent="#a4c4dc">
            Claims need to be minted once per pod via the Admin SDK. We documented this in{' '}
            <code className="font-mono text-white/80">docs/POD_MIGRATION.md</code> as a one-shot script.
            The alternative (rotating claims automatically) would have required a server-side
            component that we deliberately don't have. Manual minting is acceptable when adding
            a pod is a once-a-month operation.
          </Choice>
        </Section>

        {/* ─── Decoupling from Oracle ─────────────────────────────────── */}
        <Section id="decoupling" title="The decoupling story" eyebrow="Repo extraction" icon={GitBranch}>
          <p>
            Nexus started as a sibling shell inside the Oracle dashboard — same React build, two
            Firebase Hosting targets gated by a <code className="font-mono text-white/80">SITE</code>{' '}
            hostname check. That was the right shape for a month: zero infrastructure overhead,
            instant code sharing, and a quick path to two distinct URLs.
          </p>
          <p>
            What pushed us to extract: the visual redesign. Nexus's monochrome+ice-blue palette
            wanted to live somewhere different than Oracle's violet/cyan/pink rainbow, and the
            shared <code className="font-mono text-white/80">tailwind.config.js</code> was starting to
            grow per-SITE branches. The CSS-variable theme tokens kept it manageable, but the
            files were diverging anyway — Architecture tab content, brand glyph, favicon, page
            title.
          </p>
          <div className="grid md:grid-cols-2 gap-3 mt-2">
            <Choice title="Why not extract on day one" accent="#a4c4dc">
              <p>
                The shared codebase was load-bearing during slices 1-4 — every iteration on the
                control plane was instantly visible in both shells. Extracting too early would
                have meant two repos to update for every Firestore-schema tweak.
              </p>
            </Choice>
            <Choice title="Why extract eventually" accent="#cfd8e6">
              <p>
                Once the schema settled, the shared code shrank to a small lib/ (auth, firebase,
                fleet hooks, runs hooks). The cost of duplicating that across two repos was less
                than the cost of carrying the SITE conditional in every component file.
              </p>
            </Choice>
          </div>
        </Section>

        {/* ─── Visual identity ────────────────────────────────────────── */}
        <Section id="visual" title="Visual identity decisions" eyebrow="Design" icon={Sparkles}>
          <p>
            Five concrete decisions made Nexus visually distinct from Oracle without restructuring
            the underlying React shell.
          </p>
          <div className="grid md:grid-cols-2 gap-3">
            <Choice title="Pure-black canvas + LightTrails" accent="#a4c4dc">
              The cyber-grid + bg-orb chrome from Oracle reads as "consumer dashboard". Diagonal
              light streaks on solid black reads as "ops console". Pure CSS animation, no canvas
              or WebGL, so the cost is zero.
            </Choice>
            <Choice title="Monochrome palette + one accent" accent="#cfd8e6">
              CSS-variable tokens swap the palette: <code className="font-mono text-white/80">--oracle-bg</code>{' '}
              → black, <code className="font-mono text-white/80">--oracle-accent</code> → ice-blue. Existing
              <code className="font-mono text-white/80"> text-oracle-cyan</code> /
              <code className="font-mono text-white/80"> text-oracle-pink</code> classes collapse to a
              neutral zinc, so no per-class refactor was needed.
            </Choice>
            <Choice title="NexusMark glyph" accent="#94a3b8">
              Two diagonal trail strokes echoing the LightTrails canvas. Replaces the Oracle
              Zap+gradient chip in the header, sidebar, and favicon — visual continuity from
              the canvas to the brand mark.
            </Choice>
            <Choice title="Umbrella registry, empty by default" accent="#bee3eb">
              The <code className="font-mono text-white/80">lib/umbrellas.ts</code> module ships with
              an empty registry. Adding Oracle (or any future parent product) is a single entry;
              the shell automatically surfaces a breadcrumb when the visitor has access. Built
              for scale, not used today.
            </Choice>
          </div>
        </Section>

        {/* ─── What we'd do differently ───────────────────────────────── */}
        <Section id="hindsight" title="What we'd do differently" eyebrow="Postmortem" icon={AlertCircle}>
          <p>
            In honest retrospect, a few things that would have saved time if we'd known up front:
          </p>
          <ul className="list-disc pl-6 space-y-2 text-[14px]">
            <li>
              <strong className="text-white/90">Lift the visual layer earlier.</strong> The
              shared-codebase + SITE conditional approach was clever but tied up too much of
              every component file in two-shell concerns. A theme-tokens module from day one
              would have been simpler.
            </li>
            <li>
              <strong className="text-white/90">Skip the favicon</strong> until the brand mark
              was settled. The Nexus favicon got rewritten three times across the redesign;
              two of those were wasted work.
            </li>
            <li>
              <strong className="text-white/90">Mint claims during pod agent install</strong>,
              not as a separate manual step. Today the operator runs{' '}
              <code className="font-mono text-white/80">mint_pod_agent_claim.py</code> manually after
              dropping the service-account JSON; folding that into the setup script would
              eliminate a class of "pod is online but rules reject its writes" debugging
              sessions.
            </li>
          </ul>
        </Section>

        {/* ─── Where to go next ───────────────────────────────────────── */}
        <Section id="roadmap" title="What's parked" eyebrow="Future direction" icon={Rocket}>
          <p>
            See <code className="font-mono text-white/80">docs/ROADMAP.md</code> for the full list. The
            three items most likely to move soon:
          </p>
          <div className="grid md:grid-cols-3 gap-3">
            <Choice title="Per-product entitlements" accent="#a4c4dc">
              Wire the umbrella registry's <code className="font-mono text-white/80">hasAccess</code>{' '}
              callback to a real product-list check (custom claims or a shared identity bridge),
              so signed-in Oracle customers see the Oracle breadcrumb on Nexus.
            </Choice>
            <Choice title="Multi-tenant" accent="#cfd8e6">
              Today Nexus is single-tenant (one Firestore, one fleet). Per-customer-org isolation
              is the next architectural lift if a paying customer requires it.
            </Choice>
            <Choice title="Checkpoint promotion API" accent="#94a3b8">
              Today Nexus → Oracle weight hand-off is a manual <code className="font-mono text-white/80">scp</code>{' '}
              + service restart. A "promote this checkpoint" button + shared GCS bucket would
              close the loop.
            </Choice>
          </div>
        </Section>

        {/* ─── Footer ──────────────────────────────────────────────────── */}
        <footer className="pt-10 border-t border-white/5">
          <div className="flex items-center gap-2 text-[10px] text-oracle-muted uppercase tracking-[0.25em]">
            <Wrench size={12} />
            <span>Methodology · decisions tracked in docs/ROADMAP.md</span>
          </div>
        </footer>

      </div>
    </div>
  );
};
