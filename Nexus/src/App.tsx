import { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play, Activity, Loader2, Box, Server,
  Search, X, GitBranch, Radio, ChevronRight, Globe, ShieldCheck, Zap,
  Clock, Menu
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import { CoreVitals3D } from './components/CoreVitals';
import { AnalyticsChart } from './components/Analytics';
import { Architecture } from './components/Architecture';
import { Methodology } from './components/Methodology';
import { AccountMenu } from './components/AccountMenu';
import { CommandPalette } from './components/CommandPalette';
import type { PaletteAction } from './components/CommandPalette';
import { AuthModal } from './components/AuthModal';
import { StartRunModal } from './components/StartRunModal';
import { CancelSubscriptionDialog } from './components/CancelSubscriptionDialog';
import { SidePanel, findNavItem } from './components/SidePanel';
import { useAuth } from './lib/auth';
import { useFleet } from './lib/fleet';
import type { Pod } from './lib/fleet';
import { useRuns, lossSeries, formatRuntime, cancelRun } from './lib/runs';
import type { Run } from './lib/runs';
import { useConsoleEvents, fmtConsoleTs, severityClass } from './lib/console';
import { SITE_HOSTNAME, SITE_TAGLINE, THEME_COLORS } from './lib/site';
import { LightTrails } from './components/LightTrails';
import { NexusMark } from './components/NexusMark';
import { Sparkline } from './components/Sparkline';
import { RadialGauge } from './components/RadialGauge';
import { useActiveUmbrella } from './lib/umbrellas';

// ─── Types & Mock Data ────────────────────────────────────────────────────────
// Fleet and Mission are both live: see lib/fleet.ts and lib/runs.ts. The pod
// agent (scripts/pod_agent.py + scripts/run_monitor.py) projects state into
// Firestore at /pods and /runs respectively. See docs/CONTROL_PLANE.md.

// ─── Visual Components ────────────────────────────────────────────────────────

const AESTClock = () => {
  const [time, setTime] = useState(() => 
    new Date().toLocaleTimeString('en-AU', { timeZone: 'Australia/Sydney', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
  );
  useEffect(() => {
    const t = setInterval(() => {
      setTime(new Date().toLocaleTimeString('en-AU', { timeZone: 'Australia/Sydney', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    }, 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 glass-button rounded-lg text-xs font-mono text-oracle-muted">
      <Clock size={12} className="text-oracle-accent" />
      {time} AEST
    </div>
  );
};

// ── Fleet card — renders one /pods/{id} document ──────────────────────────
const formatHeartbeat = (ms: number) => {
  if (!ms) return 'never';
  const age = Math.max(0, Date.now() - ms);
  if (age < 60_000)     return `${Math.round(age / 1000)}s ago`;
  if (age < 3_600_000)  return `${Math.round(age / 60_000)}m ago`;
  return `${Math.round(age / 3_600_000)}h ago`;
};

const statusColor = (s: Pod['status']): { dot: string; shadow: string; label: string } => {
  switch (s) {
    case 'online':   return { dot: 'bg-emerald-400', shadow: 'shadow-[0_0_12px_#10b981]', label: 'Online' };
    case 'busy':     return { dot: 'bg-oracle-accent', shadow: 'shadow-[0_0_12px_rgb(var(--oracle-accent))]', label: 'Busy' };
    case 'degraded': return { dot: 'bg-amber-400',   shadow: 'shadow-[0_0_12px_#fbbf24]', label: 'Degraded' };
    case 'offline':  return { dot: 'bg-zinc-600',    shadow: '',                          label: 'Offline' };
  }
};

const PodCard = ({ pod, onStartRun }: { pod: Pod; onStartRun?: (podId: string) => void }) => {
  const sc = statusColor(pod.status);
  const u = pod.utilisation;
  const vramTotal = pod.accelerator.memory_gb * Math.max(1, pod.accelerator.device_count);
  const canStart = pod.status === 'online';
  return (
    <div className={`glass-panel p-5 sm:p-6 rounded-2xl sm:rounded-3xl hover:bg-white/[0.03] transition-colors relative group overflow-hidden border ${pod.status === 'offline' ? 'border-white/5 opacity-70' : 'border-white/5'}`}>
      <div className="absolute top-0 right-0 w-48 h-48 bg-oracle-accent/10 rounded-full blur-[80px] -translate-y-24 translate-x-24" />
      {/* Utilisation-proportional row fill — Quantix-style horizontal bar
       * sitting behind the card content, width tracks gpu_util_pct so the
       * card itself reads as a sparkline-of-one. Subtle by design; the
       * existing GPU Util text stat still carries the precise value. */}
      {pod.status !== 'offline' && u && (
        <div
          className="absolute inset-y-0 left-0 bg-gradient-to-r from-oracle-accent/[0.06] to-transparent pointer-events-none transition-[width] duration-700 ease-out"
          style={{ width: `${Math.min(100, u.gpu_util_pct)}%` }}
        />
      )}
      <div className="flex justify-between items-center mb-6 sm:mb-8 relative z-10 gap-3">
        <div className="flex items-center gap-3 sm:gap-4 min-w-0">
          <div className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${sc.dot} ${sc.shadow}`} />
          <div className="min-w-0">
            <h3 className="text-base font-bold text-white uppercase tracking-wider truncate">{pod.display_name}</h3>
            <p className="text-[10px] text-oracle-muted font-bold tracking-widest opacity-60 uppercase truncate">
              {pod.accelerator.device_count}× {pod.accelerator.sku} · {pod.accelerator.kind}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="px-2 sm:px-3 py-1 rounded-lg bg-white/5 border border-white/5 text-[9px] font-black uppercase tracking-[0.2em] text-oracle-muted truncate max-w-[140px]">
            {pod.current_run_id ? pod.current_run_id.slice(0, 12) : sc.label}
          </div>
          {onStartRun && canStart && (
            <button
              onClick={() => onStartRun(pod.id)}
              className="px-2.5 py-1 rounded-lg bg-oracle-accent/15 hover:bg-oracle-accent/30 border border-oracle-accent/30 text-[9px] font-black uppercase tracking-[0.2em] text-oracle-accent transition-colors flex items-center gap-1"
              title={`Start a run on ${pod.display_name}`}
            >
              <Play size={9} fill="currentColor" />
              Run
            </button>
          )}
        </div>
      </div>

      {pod.status === 'offline' ? (
        <div className="relative z-10 text-[11px] text-oracle-muted leading-relaxed">
          Last heartbeat: <span className="font-mono text-white/80">{formatHeartbeat(pod.last_heartbeat_ms)}</span>
          {pod.agent_version && <span className="opacity-60"> · {pod.agent_version}</span>}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-4 sm:gap-8 mb-6 sm:mb-8 relative z-10">
            <div>
              <p className="text-[10px] font-black text-oracle-muted uppercase mb-2 tracking-widest opacity-40">GPU Util</p>
              <p className="text-2xl font-light text-white italic">{u ? Math.round(u.gpu_util_pct) : 0}%</p>
            </div>
            <div>
              <p className="text-[10px] font-black text-oracle-muted uppercase mb-2 tracking-widest opacity-40">Temp</p>
              <p className={`text-2xl font-light italic ${u && u.temperature_c > 75 ? 'text-oracle-pink' : 'text-oracle-cyan'}`}>
                {u ? Math.round(u.temperature_c) : 0}°C
              </p>
            </div>
            <div>
              <p className="text-[10px] font-black text-oracle-muted uppercase mb-2 tracking-widest opacity-40">Power</p>
              <p className="text-2xl font-light text-white italic">{u ? Math.round(u.power_w) : 0}W</p>
            </div>
          </div>
          {vramTotal > 0 && (
            <div className="relative z-10">
              <GlowBar
                value={u ? (u.vram_used_gb / vramTotal) * 100 : 0}
                label={`VRAM ALLOCATION (${u ? u.vram_used_gb.toFixed(1) : '0.0'}/${vramTotal.toFixed(0)}GB)`}
                color={THEME_COLORS.vramBar}
              />
            </div>
          )}
          <p className="text-[9px] font-mono text-oracle-muted opacity-50 mt-4">
            heartbeat {formatHeartbeat(pod.last_heartbeat_ms)} · {pod.agent_version || 'agent ?'}
          </p>
        </>
      )}
    </div>
  );
};

// ── Mission card — renders one /runs/{run_id} document ──────────────────────
const runStatusStyles = (s: Run['status']): { bar: string; pill: string } => {
  switch (s) {
    case 'starting': return { bar: 'bg-amber-400 shadow-[0_0_20px_#fbbf24]',     pill: 'bg-amber-500/10 text-amber-400 border-amber-500/20' };
    case 'running':  return { bar: 'bg-oracle-cyan shadow-[0_0_20px_rgb(var(--oracle-cyan))]',   pill: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' };
    case 'completed':return { bar: 'bg-emerald-400 shadow-[0_0_20px_#10b981]',   pill: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' };
    case 'failed':   return { bar: 'bg-red-500 shadow-[0_0_20px_#ef4444]',       pill: 'bg-red-500/10 text-red-400 border-red-500/20' };
    case 'cancelled':return { bar: 'bg-zinc-500 shadow-[0_0_12px_#71717a]',      pill: 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20' };
  }
};

const RunCard = ({ run, canCancel }: { run: Run; canCancel: boolean }) => {
  const styles = runStatusStyles(run.status);
  const series = lossSeries(run.metrics);
  const latestStep = run.metrics.step.length ? run.metrics.step[run.metrics.step.length - 1] : null;
  const latestLoss = run.metrics.loss.length ? run.metrics.loss[run.metrics.loss.length - 1] : null;
  const latestAcc  = run.metrics.local_acc.length ? run.metrics.local_acc[run.metrics.local_acc.length - 1] : null;
  const sortedRanks = Object.entries(run.ranks).sort(([a], [b]) => Number(a) - Number(b));
  const rankCount = sortedRanks.length;
  const staleRanks = sortedRanks.filter(([, r]) => r.stale).length;
  const isInFlight = run.status === 'starting' || run.status === 'running';
  const [cancelBusy, setCancelBusy] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const cancelling = isInFlight && run.cancel_requested;
  return (
    <div className="glass-panel p-5 sm:p-8 rounded-2xl sm:rounded-3xl flex flex-col gap-5 relative overflow-hidden group hover:bg-white/[0.03] transition-colors">
      <div className={`w-1.5 h-full absolute left-0 top-0 ${styles.bar}`} />
      <div className="flex flex-col md:flex-row md:items-center gap-4 md:gap-8 pl-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2 flex-wrap">
            <h3 className="text-lg sm:text-xl font-black text-white uppercase tracking-tighter italic truncate">{run.spec.name}</h3>
            <span className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-widest border ${styles.pill}`}>{run.status}</span>
            {cancelling && (
              <span className="px-2 py-0.5 rounded-full text-[8px] font-bold uppercase tracking-widest text-amber-300 border border-amber-500/30 bg-amber-500/10">
                cancelling
              </span>
            )}
            {run.source === 'reattach' && (
              <span className="px-2 py-0.5 rounded-full text-[8px] font-bold uppercase tracking-widest text-oracle-muted border border-white/10">
                re-attached
              </span>
            )}
          </div>
          <p className="text-[10px] text-oracle-muted uppercase font-bold tracking-[0.2em] opacity-60 truncate">
            {run.pod_id} · phase {run.spec.phase} · {rankCount} rank{rankCount === 1 ? '' : 's'}
            {staleRanks > 0 && <span className="text-amber-400"> · {staleRanks} stale</span>}
            <span className="opacity-70"> · {formatRuntime(run)}</span>
          </p>
        </div>
        <div className="flex items-center gap-4 md:gap-6 shrink-0">
          <div className="grid grid-cols-3 gap-6 md:gap-10 text-left md:text-right">
            <div>
              <p className="text-[10px] font-black text-oracle-muted uppercase mb-1 tracking-widest opacity-40">Step</p>
              <p className="text-base sm:text-lg font-mono text-white italic">{latestStep ?? '—'}</p>
            </div>
            <div>
              <p className="text-[10px] font-black text-oracle-muted uppercase mb-1 tracking-widest opacity-40">Loss</p>
              <p className="text-base sm:text-lg font-mono text-oracle-cyan italic">{latestLoss !== null ? latestLoss.toFixed(3) : '—'}</p>
            </div>
            <div>
              <p className="text-[10px] font-black text-oracle-muted uppercase mb-1 tracking-widest opacity-40">Acc</p>
              <p className="text-base sm:text-lg font-mono text-oracle-pink italic">{latestAcc !== null ? `${latestAcc.toFixed(1)}%` : '—'}</p>
            </div>
          </div>
          {isInFlight && canCancel && (
            <button
              onClick={async () => {
                if (cancelBusy || cancelling) return;
                setCancelBusy(true);
                setCancelError(null);
                try { await cancelRun(run.id); }
                catch (e: any) { setCancelError(e?.message || String(e)); }
                finally { setCancelBusy(false); }
              }}
              disabled={cancelBusy || cancelling}
              className="px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-widest border transition-colors flex items-center gap-1.5 disabled:opacity-60 disabled:cursor-not-allowed bg-red-500/10 hover:bg-red-500/20 text-red-300 border-red-500/30"
              title={cancelling ? 'Cancel already requested' : 'Send SIGTERM (60s grace before SIGKILL)'}
            >
              <X size={11} />
              {cancelling ? 'Cancelling…' : cancelBusy ? 'Sending…' : 'Cancel'}
            </button>
          )}
        </div>
      </div>

      {rankCount > 0 && (
        <div className="flex items-center gap-2 -mt-1 pl-2">
          <span className="text-[9px] font-semibold uppercase tracking-[0.18em] text-oracle-muted opacity-60">Ranks</span>
          <div className="flex flex-wrap gap-1">
            {sortedRanks.map(([rank, snap]) => (
              <div
                key={rank}
                title={`rank ${rank}${snap.last_step !== null ? ` · step ${snap.last_step}` : ''}${snap.stale ? ' · stale' : ''}`}
                className={`px-1.5 py-0.5 rounded-md text-[9px] font-mono leading-none flex items-center gap-1 border ${
                  snap.stale
                    ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
                    : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                }`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${snap.stale ? 'bg-amber-400' : 'bg-emerald-400 shadow-[0_0_6px_#10b981]'}`} />
                r{rank}
              </div>
            ))}
          </div>
        </div>
      )}

      {cancelError && (
        <div className="text-[10px] text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2 mx-2">
          Cancel failed: <span className="font-mono">{cancelError}</span>
        </div>
      )}

      {series.length > 1 && (
        <div className="h-16 -mx-2 sm:mx-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
              <Line type="monotone" dataKey="loss" stroke={THEME_COLORS.chartPrimary} strokeWidth={1.5} dot={false} isAnimationActive={false} />
              <XAxis dataKey="step" hide />
              <YAxis hide domain={['dataMin', 'dataMax']} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

// ── Analytics view — derives live aggregates from /runs ──────────────────
const AnalyticsView = ({ runs }: { runs: ReturnType<typeof useRuns> }) => {
  const all = [...runs.active, ...runs.recent];
  const chartable = all.filter((r) => r.metrics.step.length > 1);

  // Latest loss across actively-training runs (only ones with fresh points)
  const latestLosses = runs.active
    .map((r) => r.metrics.loss[r.metrics.loss.length - 1])
    .filter((v): v is number => typeof v === 'number');
  const avgLatestLoss = latestLosses.length
    ? latestLosses.reduce((a, b) => a + b, 0) / latestLosses.length
    : null;

  // Best val_acc across all observed runs (most useful single-number summary)
  const bestValAcc = all.reduce<number | null>((best, r) => {
    const v = r.metrics.val_acc.length ? Math.max(...r.metrics.val_acc) : null;
    if (v === null) return best;
    return best === null ? v : Math.max(best, v);
  }, null);

  // Cumulative step count across active runs — rough "compute work in flight"
  const totalActiveSteps = runs.active.reduce(
    (sum, r) => sum + (r.metrics.step.length ? r.metrics.step[r.metrics.step.length - 1] : 0),
    0,
  );

  const completedToday = runs.recent.filter((r) => {
    if (r.status !== 'completed') return false;
    if (!r.finished_at_ms) return false;
    return Date.now() - r.finished_at_ms < 86_400_000;
  }).length;

  return (
    <>
      <div className="flex justify-between items-start flex-wrap gap-3">
        <div>
          <h2 className="text-2xl font-extralight text-white mb-2 italic tracking-tight">Training Telemetry</h2>
          <p className="text-[10px] text-oracle-muted uppercase tracking-[0.3em] font-black opacity-40">
            Live · {chartable.length} run{chartable.length === 1 ? '' : 's'} charted
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-oracle-accent/10 border border-oracle-accent/20">
          <Radio size={12} className={`text-oracle-accent ${runs.active.length > 0 ? 'animate-pulse' : 'opacity-40'}`} />
          <span className="text-[10px] font-black text-oracle-accent uppercase tracking-widest">
            {runs.active.length > 0 ? `${runs.active.length} ACTIVE` : 'IDLE'}
          </span>
        </div>
      </div>

      <div className="flex-1 min-h-[320px] sm:min-h-[450px] glass-panel rounded-2xl sm:rounded-[40px] p-3 sm:p-6 md:p-8 relative overflow-hidden group">
        <div className="absolute inset-0 bg-gradient-to-b from-oracle-accent/5 to-transparent pointer-events-none" />
        <AnalyticsChart runs={all} metricKey="loss" />
      </div>

      {/* For the sparklines: pick the longest active-run loss series as the
       * "avg loss trend" representative (smoothest fallback when there are
       * multiple runs with different step counts), and the val_acc curve
       * from whichever run owns the current best val_acc. Both are read
       * straight off the existing /runs subscription — no extra storage. */}
      {(() => {
        const lossTrend = runs.active
          .map((r) => r.metrics.loss)
          .filter((arr) => arr.length > 1)
          .sort((a, b) => b.length - a.length)[0]
          ?.slice(-30) ?? [];

        const bestRun = all.reduce<{ acc: number; series: number[] } | null>((best, r) => {
          if (!r.metrics.val_acc.length) return best;
          const peak = Math.max(...r.metrics.val_acc);
          if (best === null || peak > best.acc) return { acc: peak, series: r.metrics.val_acc };
          return best;
        }, null);
        const valTrend = bestRun?.series ?? [];

        return (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-6">
            <StatCard label="Active runs" value={String(runs.active.length).padStart(2, '0')} accent="white" />
            <StatCard
              label="Avg latest loss"
              value={avgLatestLoss !== null ? avgLatestLoss.toFixed(3) : '—'}
              accent="oracle-cyan"
              series={lossTrend}
              sparkColor={THEME_COLORS.chartPrimary}
            />
            <StatCard
              label="Best val acc"
              value={bestValAcc !== null ? `${bestValAcc.toFixed(1)}%` : '—'}
              accent="emerald-400"
              series={valTrend}
              sparkColor="#10b981"
            />
            <StatCard
              label="Completed · 24h"
              value={String(completedToday).padStart(2, '0')}
              accent="oracle-pink"
              delta={runs.active.length > 0 ? `+${totalActiveSteps.toLocaleString()} steps in flight` : undefined}
            />
          </div>
        );
      })()}

      {/* Latest validation results across runs — concise table-ish list */}
      {all.some((r) => r.metrics.val_acc.length > 0) && (
        <div className="glass-panel rounded-2xl p-5 sm:p-6 border border-white/5">
          <p className="text-[10px] font-black text-oracle-muted uppercase tracking-[0.2em] opacity-60 mb-4">
            Latest validation
          </p>
          <div className="space-y-2">
            {all
              .filter((r) => r.metrics.val_acc.length > 0)
              .slice(0, 6)
              .map((r) => {
                const acc = r.metrics.val_acc[r.metrics.val_acc.length - 1];
                const step = r.metrics.val_step[r.metrics.val_step.length - 1];
                return (
                  <div key={r.id} className="flex items-center justify-between gap-3 text-[12px]">
                    <span className="text-white font-mono truncate flex-1 min-w-0">{r.spec.name}</span>
                    <span className="text-oracle-muted text-[10px] font-mono opacity-60 shrink-0">step {step ?? '—'}</span>
                    <span className="text-oracle-cyan font-mono font-bold shrink-0 w-16 text-right">{acc.toFixed(1)}%</span>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </>
  );
};

// ── Silicon view — live /pods utilisation aggregates + per-pod strip ─────
const SiliconView = ({
  fleet,
  runsActive,
}: {
  fleet: ReturnType<typeof useFleet>;
  runsActive: number;
}) => {
  const online = fleet.pods.filter((p) => p.status !== 'offline');
  const totalDevices = online.reduce((s, p) => s + (p.accelerator.device_count || 0), 0);
  const totalVramGb = online.reduce(
    (s, p) => s + (p.accelerator.memory_gb || 0) * Math.max(1, p.accelerator.device_count || 0),
    0,
  );
  const utilsList = online.map((p) => p.utilisation).filter((u): u is NonNullable<typeof u> => !!u);
  const avgUtil = utilsList.length ? utilsList.reduce((s, u) => s + u.gpu_util_pct, 0) / utilsList.length : 0;
  const vramUsed = utilsList.reduce((s, u) => s + u.vram_used_gb, 0);
  const hottestTemp = utilsList.reduce<number | null>((m, u) => (m === null ? u.temperature_c : Math.max(m, u.temperature_c)), null);
  const totalPower = utilsList.reduce((s, u) => s + u.power_w, 0);

  const vitalsStatus = runsActive > 0 ? 'training' : online.length > 0 ? 'idle' : 'offline';

  // The "Avg utilisation" tile is special-cased downstream into a radial
  // gauge; the remaining five remain plain stat tiles. Keeping `gauge: true`
  // on the entry lets us drive both treatments from the same source list.
  const stats: { n: string; v: string; u: string; gauge?: boolean }[] = [
    { n: 'Online devices',   v: totalDevices ? String(totalDevices) : '—',                            u: 'GPU' },
    { n: 'VRAM allocated',   v: totalVramGb ? `${vramUsed.toFixed(1)} / ${totalVramGb.toFixed(0)}`  : '—', u: 'GB' },
    { n: 'Avg utilisation',  v: utilsList.length ? `${avgUtil.toFixed(0)}` : '—',                     u: '%',    gauge: true },
    { n: 'Hottest core',     v: hottestTemp !== null ? `${hottestTemp.toFixed(0)}` : '—',             u: '°C' },
    { n: 'Total draw',       v: utilsList.length ? `${totalPower.toFixed(0)}` : '—',                  u: 'W' },
    { n: 'Pods reporting',   v: `${utilsList.length}/${online.length}`,                               u: 'live' },
  ];

  return (
    <>
      <div className="flex-1 min-h-0 relative">
        <CoreVitals3D status={vitalsStatus} />
        <div className="absolute top-10 left-10">
          <h2 className="text-2xl font-extralight text-white mb-2 italic tracking-tight">Silicon Vitals Matrix</h2>
          <p className="text-[10px] text-oracle-muted uppercase tracking-[0.4em] font-black opacity-40">
            {online.length === 0
              ? 'No pods reporting · start scripts/pod_agent.py'
              : `${online.length} pod${online.length === 1 ? '' : 's'} · ${runsActive} active run${runsActive === 1 ? '' : 's'}`}
          </p>
        </div>
      </div>

      <div className="border-t border-white/5 p-4 sm:p-6 md:p-8 bg-black/40 backdrop-blur-xl relative z-10 space-y-4 md:space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3 sm:gap-4">
          {stats.map((metric) => (
            <div
              key={metric.n}
              className="glass-panel p-4 sm:p-5 rounded-xl sm:rounded-2xl flex flex-col gap-3 border border-white/5 group hover:border-oracle-accent/30 transition-colors"
            >
              <span className="text-[9px] font-black text-oracle-muted uppercase tracking-widest opacity-40 group-hover:opacity-80 transition-opacity">
                {metric.n}
              </span>
              {metric.gauge && utilsList.length ? (
                <div className="flex items-center justify-center py-1">
                  <RadialGauge value={avgUtil} size={86} strokeWidth={7} unit={metric.u} />
                </div>
              ) : (
                <div className="flex items-baseline gap-2">
                  <span className="text-xl sm:text-2xl font-light text-white tracking-tighter">{metric.v}</span>
                  <span className="text-[9px] font-black text-oracle-cyan uppercase opacity-60">{metric.u}</span>
                </div>
              )}
            </div>
          ))}
        </div>

        {online.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 sm:gap-4">
            {online.map((pod) => {
              const u = pod.utilisation;
              const cap = pod.accelerator.memory_gb * Math.max(1, pod.accelerator.device_count);
              return (
                <div
                  key={pod.id}
                  className="glass-panel p-4 rounded-xl border border-white/5 flex flex-col gap-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-[11px] font-bold text-white uppercase tracking-wider truncate">{pod.display_name}</p>
                    <span className={`px-2 py-0.5 rounded-md text-[8px] font-black uppercase tracking-widest border ${
                      pod.status === 'busy'
                        ? 'border-oracle-accent/40 text-oracle-accent bg-oracle-accent/10'
                        : 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10'
                    }`}>
                      {pod.status === 'busy' ? 'Busy' : 'Online'}
                    </span>
                  </div>
                  <p className="text-[9px] text-oracle-muted font-mono uppercase tracking-wider opacity-60 truncate">
                    {pod.accelerator.device_count}× {pod.accelerator.sku}
                  </p>
                  {u ? (
                    <div className="grid grid-cols-3 gap-2 mt-1">
                      <Micro stat label="util" value={`${u.gpu_util_pct.toFixed(0)}%`} accent="white" />
                      <Micro stat label="temp" value={`${u.temperature_c.toFixed(0)}°`} accent={u.temperature_c > 75 ? 'oracle-pink' : 'oracle-cyan'} />
                      <Micro stat label="draw" value={`${u.power_w.toFixed(0)}W`} accent="white" />
                    </div>
                  ) : (
                    <p className="text-[9px] text-oracle-muted font-mono opacity-60">{pod.accelerator.kind === 'tpu' ? 'TPU · no per-device poll' : 'Awaiting heartbeat…'}</p>
                  )}
                  {u && cap > 0 && (
                    <GlowBar value={(u.vram_used_gb / cap) * 100} label={`VRAM (${u.vram_used_gb.toFixed(1)}/${cap.toFixed(0)} GB)`} color={THEME_COLORS.vramBar} />
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </>
  );
};

// ── Console view — live event stream derived from /pods + /runs ──────────
const ConsoleView = ({
  pods,
  runsActive,
  runsRecent,
  logEndRef,
}: {
  pods: Pod[];
  runsActive: Run[];
  runsRecent: Run[];
  logEndRef: React.RefObject<HTMLDivElement | null>;
}) => {
  const allRuns = useMemo(() => [...runsActive, ...runsRecent], [runsActive, runsRecent]);
  const events = useConsoleEvents(pods, allRuns);
  useEffect(() => {
    // Scroll the log pane to its bottom by setting scrollTop directly on the
    // container. We deliberately don't call scrollIntoView on the sentinel —
    // if any ancestor isn't strictly scroll-constrained, scrollIntoView walks
    // up the DOM and silently shifts an overflow-hidden parent's scrollTop,
    // which the user then can't scroll back (no scrollbar to drag).
    const container = logEndRef.current?.parentElement;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [events.length, logEndRef]);
  return (
    <>
      <div className="flex justify-between items-center px-4 sm:px-8 py-4 sm:py-5 border-b border-white/5 bg-black/60 backdrop-blur-xl">
        <div className="flex gap-2 sm:gap-2.5">
          <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/40" />
          <div className="w-3 h-3 rounded-full bg-amber-500/20 border border-amber-500/40" />
          <div className="w-3 h-3 rounded-full bg-emerald-500/20 border border-emerald-400/40" />
        </div>
        <span className="hidden sm:inline text-[10px] font-black text-oracle-muted uppercase tracking-[0.4em] opacity-40">
          Cluster Event Stream · {events.length}/{200} entries
        </span>
        <Activity size={14} className={`${runsActive.length > 0 ? 'text-oracle-cyan animate-pulse' : 'text-oracle-muted/30'}`} />
      </div>
      <div className="flex-1 min-h-0 p-3 sm:p-6 md:p-8 overflow-y-auto custom-scrollbar font-mono text-[10px] sm:text-[11px] leading-relaxed text-zinc-500 selection:bg-oracle-accent/40 antialiased">
        {events.length === 0 ? (
          <div className="px-2 sm:px-4 py-2 text-zinc-600 italic">Waiting for the first state delta…</div>
        ) : (
          events.map((e, i) => (
            <div
              key={e.id}
              className={`flex gap-3 sm:gap-6 hover:bg-white/[0.02] px-2 sm:px-4 py-1.5 rounded-lg transition-colors ${e.severity === 'error' ? 'bg-red-500/5' : ''} ${severityClass(e.severity)}`}
            >
              <span className="w-6 sm:w-10 text-right text-zinc-800 select-none font-black italic opacity-20 flex-shrink-0">{i + 1}</span>
              <span className="text-zinc-600 shrink-0 font-mono opacity-70">{fmtConsoleTs(e.ts)}</span>
              <span className="text-zinc-500 shrink-0 uppercase tracking-widest text-[9px] font-black opacity-60 w-12">[{e.source}]</span>
              <span className="break-all tracking-tighter">{e.message}</span>
            </div>
          ))
        )}
        <div ref={logEndRef} />
      </div>
    </>
  );
};

const Micro = ({ label, value, accent }: { label: string; value: string; accent: string; stat?: boolean }) => (
  <div>
    <p className="text-[8px] font-black text-oracle-muted uppercase tracking-widest opacity-50">{label}</p>
    <p className={`text-sm font-mono text-${accent}`}>{value}</p>
  </div>
);

const StatCard = ({ label, value, accent, delta, series, sparkColor }: { label: string; value: string; accent: string; delta?: string; series?: number[]; sparkColor?: string }) => (
  <div className="glass-panel p-5 sm:p-6 rounded-2xl border border-white/5">
    <p className="text-[9px] font-black text-oracle-muted uppercase mb-2 sm:mb-3 tracking-widest opacity-40">{label}</p>
    <p className={`text-2xl sm:text-3xl font-light tracking-tighter text-${accent}`}>{value}</p>
    {delta && <p className="text-[9px] font-bold text-oracle-muted opacity-60 mt-1 tracking-wide">{delta}</p>}
    {series && series.length > 1 && (
      <div className="mt-3 opacity-80">
        <Sparkline series={series} height={28} color={sparkColor} />
      </div>
    )}
  </div>
);

const GlowBar = ({ value, color = THEME_COLORS.accent, label, suffix = '%' }: any) => (
  <div className="w-full">
    <div className="flex justify-between text-[9px] font-bold uppercase tracking-wider text-oracle-muted mb-1.5">
      <span>{label}</span>
      <span className="text-white">{Math.round(value)}{suffix}</span>
    </div>
    <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden border border-white/5 shadow-inner">
      <motion.div 
        initial={{ width: 0 }}
        animate={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        className="h-full rounded-full"
        style={{ 
          background: `linear-gradient(90deg, transparent, ${color})`,
          boxShadow: `0 0 12px ${color}` 
        }}
      />
    </div>
  </div>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState<string>('cluster');
  const logEndRef = useRef<HTMLDivElement>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [authPromptOpen, setAuthPromptOpen] = useState(false);
  const [sidePanelOpen, setSidePanelOpen] = useState(false);
  const [startRunOpen, setStartRunOpen] = useState(false);
  const [startRunPodId, setStartRunPodId] = useState<string | null>(null);
  const [cancelSubOpen, setCancelSubOpen] = useState(false);
  const { signOut, viewAsTier, upgradeTier, user, isAdmin } = useAuth();
  const umbrella = useActiveUmbrella();
  const openStartRun = (podId: string | null) => {
    if (!user) { setAuthPromptOpen(true); return; }
    setStartRunPodId(podId);
    setStartRunOpen(true);
  };

  // Global ⌘K / Ctrl+K listener
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const handlePaletteAction = (action: PaletteAction) => {
    switch (action.kind) {
      case 'tab':      setActiveTab(action.tab); break;
      case 'signin':   setAuthPromptOpen(true); break;
      case 'signout':  signOut(); break;
      case 'viewAs':   viewAsTier(action.tier); break;
      case 'upgrade':  upgradeTier(action.tier); break;
      case 'startRun': openStartRun(action.podId); break;
      case 'cancelSubscription':
        if (!user) { setAuthPromptOpen(true); break; }
        setCancelSubOpen(true);
        break;
    }
  };

  const fleet = useFleet();
  const runs = useRuns(10);
  const activeNavItem = findNavItem(activeTab);
  const onlinePods = fleet.pods.filter((p) => p.status !== 'offline').length;
  const totalDevices = fleet.pods.reduce((sum, p) => sum + (p.accelerator.device_count || 0), 0);

  return (
    <>
      <LightTrails />

      <div className="relative z-10 h-[100dvh] flex flex-col p-2 sm:p-4 md:p-6 lg:p-8 gap-3 sm:gap-6 max-w-[1800px] mx-auto overflow-hidden font-sans antialiased">
        
        {/* ── Header ── */}
        <header className="flex items-center gap-2 sm:gap-4 glass-panel px-3 sm:px-5 py-2.5 sm:py-4 rounded-xl sm:rounded-2xl relative z-50">
          {/* Left: Menu + logo + breadcrumb */}
          <div className="flex items-center gap-2 sm:gap-4 flex-shrink min-w-0">
            <button
              onClick={() => setSidePanelOpen(true)}
              className="w-10 h-10 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 flex items-center justify-center transition-colors flex-shrink-0"
              title="Open navigation"
              aria-label="Open navigation"
            >
              <Menu size={16} className="text-white" />
            </button>

            <div className="flex items-center gap-2 sm:gap-3 flex-shrink min-w-0">
              <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-black border border-white/15 shadow-[0_0_18px_rgba(255,255,255,0.10)] flex items-center justify-center flex-shrink-0 text-white">
                <NexusMark size={16} />
              </div>
              <div className="hidden sm:block min-w-0">
                <h1 className="text-sm md:text-base font-semibold tracking-tight text-white leading-tight italic uppercase truncate">
                  {umbrella && (
                    <>
                      <a
                        href={umbrella.marketingUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-extralight text-oracle-muted not-italic opacity-70 hover:opacity-100 hover:text-white transition-opacity"
                        title={`Open ${umbrella.name}`}
                      >
                        {umbrella.name}
                      </a>
                      <span className="font-extralight text-oracle-muted not-italic opacity-40 mx-1.5">/</span>
                    </>
                  )}
                  Nexus
                </h1>
                <p className="text-[8px] text-oracle-accent font-semibold tracking-[0.3em] uppercase opacity-60 mt-0.5 truncate">{SITE_TAGLINE}</p>
              </div>
            </div>

            {/* Current tab breadcrumb */}
            {activeNavItem && (
              <div className="flex items-center gap-1.5 sm:gap-2 min-w-0 sm:ml-1 sm:pl-4 sm:border-l sm:border-white/10">
                <ChevronRight size={11} className="hidden sm:flex text-oracle-muted opacity-50 flex-shrink-0" />
                <activeNavItem.icon size={12} className="text-oracle-accent flex-shrink-0" />
                <span className="text-[10px] sm:text-[11px] font-semibold uppercase tracking-[0.15em] sm:tracking-[0.18em] text-white truncate">
                  {activeNavItem.label}
                </span>
              </div>
            )}
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Right: Search, Clock, Account */}
          <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
            <button
              onClick={() => setPaletteOpen(true)}
              className="hidden md:flex items-center gap-2 pl-3 pr-2 py-1.5 rounded-lg bg-white/[0.03] hover:bg-white/[0.06] border border-white/10 transition-colors"
              title="Open command palette"
            >
              <Search size={12} className="text-oracle-muted" />
              <span className="text-[11px] text-oracle-muted">Search</span>
              <kbd className="text-[9px] font-mono text-oracle-muted bg-white/5 border border-white/10 rounded px-1.5 py-0.5">
                ⌘K
              </kbd>
            </button>
            <button
              onClick={() => setPaletteOpen(true)}
              className="md:hidden w-10 h-10 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 flex items-center justify-center transition-colors flex-shrink-0"
              aria-label="Search"
            >
              <Search size={14} className="text-white" />
            </button>
            <div className="hidden lg:block">
              <AESTClock />
            </div>
            <AccountMenu onCancelSubscription={() => setCancelSubOpen(true)} />
          </div>
        </header>

        {/* ── Main Grid ── */}
        <main className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">
          
          {/* Sidebar */}
          <aside className="hidden lg:flex lg:col-span-3 flex-col gap-6 overflow-y-auto no-scrollbar">
            <div className="glass-panel p-6 rounded-2xl border-t border-white/10 shadow-2xl">
               <div className="flex justify-between items-center mb-8">
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] text-oracle-muted opacity-60">Total Throughput</span>
                  <Activity size={14} className="text-oracle-cyan animate-pulse" />
               </div>
               <div className="space-y-6">
                  <div className="flex items-baseline gap-2">
                    <span className="text-4xl font-light tracking-tighter text-white">12.4</span>
                    <span className="text-[10px] text-oracle-muted uppercase font-black opacity-40">images/sec</span>
                  </div>
                  <GlowBar value={84} label="Hardware Saturation" color={THEME_COLORS.saturationBar} />
                  <GlowBar value={96} label="HBM3e Allocation" color={THEME_COLORS.hbmBar} />
               </div>
            </div>

            <div className="glass-panel p-6 rounded-2xl flex-1 flex flex-col border-b border-white/10 shadow-2xl overflow-hidden">
              <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-oracle-muted mb-6 flex items-center gap-2 opacity-60">
                <GitBranch size={14} className="text-oracle-accent" />
                Market Insight
              </h3>
              <div className="space-y-3 flex-1 overflow-y-auto no-scrollbar">
                {[
                  { label: 'API Request Volume', value: '1.2M/day', icon: Globe },
                  { label: 'Sovereign Nodes', value: 'Online', icon: ShieldCheck },
                  { label: 'Audit Velocity', value: 'Fast', icon: Zap },
                ].map((item, i) => (
                  <div key={i} className="flex flex-col gap-1 bg-white/[0.03] border border-white/5 p-4 rounded-xl hover:bg-white/[0.06] transition-all group cursor-default">
                    <div className="flex justify-between items-center">
                       <span className="text-[10px] text-oracle-muted font-black uppercase tracking-widest opacity-60">{item.label}</span>
                       <item.icon size={10} className="text-oracle-accent" />
                    </div>
                    <span className="text-sm text-white font-mono font-bold tracking-tight">{item.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </aside>

          {/* Viewport */}
          <section className="col-span-12 lg:col-span-9 h-full glass-panel rounded-2xl sm:rounded-3xl flex flex-col overflow-hidden border border-white/10 shadow-[0_0_60px_rgba(0,0,0,0.5)] sm:shadow-[0_0_100px_rgba(0,0,0,0.5)]">
            <AnimatePresence mode="wait">
              
              {/* FLEET VIEW */}
              {activeTab === 'cluster' && (
                <motion.div key="fleet" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="flex-1 p-4 sm:p-6 md:p-8 overflow-y-auto custom-scrollbar">
                   <div className="flex justify-between items-end mb-6 sm:mb-10 flex-wrap gap-4">
                      <div>
                        <h2 className="text-2xl sm:text-3xl font-extralight text-white mb-1 tracking-tight">Silicon Distributed Fleet</h2>
                        <p className="text-[10px] text-oracle-muted uppercase tracking-[0.3em] font-black opacity-40">Live · /pods Firestore subscription</p>
                      </div>
                      <div className="flex gap-6 sm:gap-10 sm:border-l sm:border-white/5 sm:pl-10">
                         <div>
                            <p className="text-[9px] font-black text-oracle-muted uppercase tracking-widest mb-1">Pods Online</p>
                            <p className="text-2xl font-bold text-oracle-cyan tracking-tighter">{String(onlinePods).padStart(2,'0')}<span className="text-oracle-muted opacity-50 text-base font-light"> / {String(fleet.pods.length).padStart(2,'0')}</span></p>
                         </div>
                         <div>
                            <p className="text-[9px] font-black text-oracle-muted uppercase tracking-widest mb-1">Total Devices</p>
                            <p className="text-2xl font-bold text-white tracking-tighter">{String(totalDevices).padStart(2,'0')}</p>
                         </div>
                      </div>
                   </div>

                   {fleet.loading && fleet.pods.length === 0 && (
                     <div className="glass-panel p-10 rounded-3xl flex flex-col items-center justify-center gap-3 text-center">
                        <Loader2 size={20} className="text-oracle-accent animate-spin" />
                        <p className="text-sm text-oracle-muted">Subscribing to /pods…</p>
                     </div>
                   )}
                   {!fleet.loading && fleet.pods.length === 0 && (
                     <div className="glass-panel p-10 rounded-3xl flex flex-col items-center justify-center gap-3 text-center">
                        <Server size={20} className="text-oracle-muted opacity-60" />
                        <p className="text-sm text-white font-medium">No pods registered yet</p>
                        <p className="text-[11px] text-oracle-muted max-w-md leading-relaxed">
                          A pod appears here once <span className="font-mono text-oracle-cyan">scripts/pod_agent.py</span> is running and authenticated against this project. See <span className="font-mono text-oracle-accent">docs/CONTROL_PLANE.md</span> for the bootstrap steps.
                        </p>
                     </div>
                   )}
                   {fleet.error && (
                     <div className="glass-panel p-5 rounded-2xl border border-red-500/20 bg-red-500/5 text-[11px] text-red-300 mb-6">
                       Fleet subscription error: <span className="font-mono">{fleet.error}</span>
                     </div>
                   )}

                   <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                      {fleet.pods.map((pod) => (
                        <PodCard key={pod.id} pod={pod} onStartRun={openStartRun} />
                      ))}
                   </div>
                </motion.div>
              )}

              {/* MISSION VIEW */}
              {activeTab === 'mission' && (
                <motion.div key="mission" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 p-4 sm:p-6 md:p-10 overflow-y-auto custom-scrollbar">
                   <div className="flex items-center justify-between mb-8 flex-wrap gap-3">
                     <div className="flex items-center gap-3">
                        <Box size={24} className="text-oracle-pink" />
                        <h2 className="text-2xl font-extralight text-white tracking-tight">Mission Operations Control</h2>
                     </div>
                     <p className="text-[10px] text-oracle-muted uppercase tracking-[0.3em] font-black opacity-40">
                       Live · /runs Firestore subscription
                     </p>
                   </div>

                   {runs.error && (
                     <div className="glass-panel p-5 rounded-2xl border border-red-500/20 bg-red-500/5 text-[11px] text-red-300 mb-6">
                       Runs subscription error: <span className="font-mono">{runs.error}</span>
                     </div>
                   )}

                   {/* Active runs */}
                   <section className="mb-10">
                     <div className="flex items-center justify-between mb-4">
                       <h3 className="text-[10px] font-black text-oracle-muted uppercase tracking-[0.25em] opacity-70">
                         In flight {runs.active.length > 0 && <span className="text-white">· {runs.active.length}</span>}
                       </h3>
                     </div>
                     {runs.loading && runs.active.length === 0 && runs.recent.length === 0 && (
                       <div className="glass-panel p-10 rounded-3xl flex flex-col items-center justify-center gap-3 text-center">
                         <Loader2 size={20} className="text-oracle-accent animate-spin" />
                         <p className="text-sm text-oracle-muted">Subscribing to /runs…</p>
                       </div>
                     )}
                     {!runs.loading && runs.active.length === 0 && (
                       <div className="glass-panel p-10 rounded-3xl flex flex-col items-center justify-center gap-4 text-center">
                         <Activity size={20} className="text-oracle-muted opacity-60" />
                         <p className="text-sm text-white font-medium">No active runs</p>
                         <p className="text-[11px] text-oracle-muted max-w-md leading-relaxed">
                           Training runs appear here as soon as a pod claims a request, or any time <span className="font-mono text-oracle-cyan">pod_agent.py</span> re-attaches to an existing local telemetry stream.
                         </p>
                         <button
                           onClick={() => openStartRun(null)}
                           className="mt-2 px-4 py-2 rounded-lg bg-oracle-accent text-white text-[11px] font-bold uppercase tracking-widest hover:bg-oracle-accent/90 transition-colors flex items-center gap-2"
                         >
                           <Play size={11} fill="currentColor" />
                           Start a Run
                         </button>
                       </div>
                     )}
                     <div className="grid grid-cols-1 gap-6">
                       {runs.active.map((run) => (
                         <RunCard
                           key={run.id}
                           run={run}
                           canCancel={isAdmin || (!!user && run.created_by_uid === user.uid)}
                         />
                       ))}
                     </div>
                   </section>

                   {/* Recent (completed/failed/cancelled) */}
                   {runs.recent.length > 0 && (
                     <section>
                       <h3 className="text-[10px] font-black text-oracle-muted uppercase tracking-[0.25em] opacity-70 mb-4">
                         Recent · {runs.recent.length}
                       </h3>
                       <div className="grid grid-cols-1 gap-6">
                         {runs.recent.map((run) => (
                           <RunCard key={run.id} run={run} canCancel={false} />
                         ))}
                       </div>
                     </section>
                   )}
                </motion.div>
              )}

              {/* CONSOLE VIEW */}
              {activeTab === 'terminal' && (
                <motion.div key="console" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 flex flex-col bg-[#020204]">
                   <ConsoleView pods={fleet.pods} runsActive={runs.active} runsRecent={runs.recent} logEndRef={logEndRef} />
                </motion.div>
              )}

              {/* ANALYTICS VIEW — live /runs metrics */}
              {activeTab === 'analytics' && (
                <motion.div key="analytics" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 p-4 sm:p-6 md:p-10 flex flex-col gap-6 md:gap-10 overflow-y-auto custom-scrollbar">
                   <AnalyticsView runs={runs} />
                </motion.div>
              )}

              {/* SILICON VIEW — live /pods utilisation */}
              {activeTab === 'hardware' && (
                <motion.div key="silicon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 flex flex-col overflow-hidden">
                   <SiliconView fleet={fleet} runsActive={runs.active.length} />
                </motion.div>
              )}

              {/* ARCHITECTURE VIEW — Reference: how it's built */}
              {activeTab === 'architecture' && (
                <motion.div key="architecture" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 flex flex-col overflow-hidden">
                   <Architecture />
                </motion.div>
              )}

              {/* METHODOLOGY VIEW — Reference: why those choices */}
              {activeTab === 'methodology' && (
                <motion.div key="methodology" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex-1 flex flex-col overflow-hidden">
                   <Methodology />
                </motion.div>
              )}

            </AnimatePresence>
          </section>
        </main>

        {/* Footer */}
        <footer className="hidden sm:flex flex-wrap justify-between items-center gap-2 text-[10px] font-medium text-zinc-700 uppercase tracking-[0.25em] mt-auto pb-3 sm:pb-4 border-t border-white/5 pt-4 sm:pt-6 mx-2">
          <div className="flex items-center gap-3">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]" />
            <span>All systems operational</span>
          </div>
          <span className="text-oracle-muted opacity-50">{SITE_HOSTNAME}</span>
        </footer>
      </div>

      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onAction={handlePaletteAction}
      />
      <AuthModal
        open={authPromptOpen}
        onClose={() => setAuthPromptOpen(false)}
      />
      <StartRunModal
        open={startRunOpen}
        onClose={() => setStartRunOpen(false)}
        pods={fleet.pods}
        initialPodId={startRunPodId}
        onNavigateToMission={() => setActiveTab('mission')}
      />
      <CancelSubscriptionDialog
        open={cancelSubOpen}
        onClose={() => setCancelSubOpen(false)}
      />
      <SidePanel
        open={sidePanelOpen}
        onClose={() => setSidePanelOpen(false)}
        activeTab={activeTab}
        onSelect={setActiveTab}
      />
    </>
  );
}
