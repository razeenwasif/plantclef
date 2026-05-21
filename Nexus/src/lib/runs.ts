// Runs — live training-run subscription.
// Backed by /runs Firestore collection written by scripts/run_monitor.py.
// See docs/CONTROL_PLANE.md (slice 2) for the schema this hook expects.

import { useEffect, useState } from 'react';
import { collection, doc, onSnapshot, orderBy, query, Timestamp, updateDoc } from 'firebase/firestore';
import { db } from './firebase';

export type RunStatus = 'starting' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface RankSnapshot {
  host_id: string;
  last_heartbeat_ms: number;
  last_step: number | null;
  stale: boolean;
}

export interface RunMetrics {
  step: number[];
  loss: number[];
  local_acc: number[];
  val_step: number[];
  val_acc: number[];
}

export interface Run {
  id: string;                  // == run_id
  run_id: string;
  pod_id: string;
  request_id: string | null;
  created_by_uid: string | null;   // requester's uid (request-spawned runs only)
  spec: {
    name: string;
    phase: string;
    seed: number | null;
    cluster_manifest_ref: string | null;
  };
  started_at_ms: number;
  finished_at_ms: number | null;
  status: RunStatus;
  exit_code: number | null;
  ranks: Record<string, RankSnapshot>;
  metrics: RunMetrics;
  cancel_requested: boolean;
  log_url: string | null;
  source?: 'request' | 'reattach';
}

const EMPTY_METRICS: RunMetrics = {
  step: [], loss: [], local_acc: [], val_step: [], val_acc: [],
};

const tsToMs = (v: unknown): number => {
  if (!v) return 0;
  if (v instanceof Timestamp) return v.toMillis();
  if (typeof v === 'object' && v && 'toMillis' in v) {
    try { return (v as Timestamp).toMillis(); } catch { return 0; }
  }
  if (typeof v === 'number') return v;
  return 0;
};

const toRanks = (raw: any): Record<string, RankSnapshot> => {
  const out: Record<string, RankSnapshot> = {};
  if (!raw || typeof raw !== 'object') return out;
  for (const [rank, snap] of Object.entries(raw as Record<string, any>)) {
    out[rank] = {
      host_id: snap?.host_id ?? '',
      last_heartbeat_ms: tsToMs(snap?.last_heartbeat),
      last_step: typeof snap?.last_step === 'number' ? snap.last_step : null,
      stale: Boolean(snap?.stale),
    };
  }
  return out;
};

const toRun = (id: string, raw: any): Run => ({
  id,
  run_id: raw.run_id ?? id,
  pod_id: raw.pod_id ?? 'unknown',
  request_id: raw.request_id ?? null,
  created_by_uid: raw.created_by_uid ?? null,
  spec: {
    name: raw.spec?.name ?? id,
    phase: raw.spec?.phase ?? 'unknown',
    seed: raw.spec?.seed ?? null,
    cluster_manifest_ref: raw.spec?.cluster_manifest_ref ?? null,
  },
  started_at_ms: tsToMs(raw.started_at),
  finished_at_ms: raw.finished_at ? tsToMs(raw.finished_at) : null,
  status: (raw.status as RunStatus) ?? 'running',
  exit_code: raw.exit_code ?? null,
  ranks: toRanks(raw.ranks),
  metrics: {
    step:      raw.metrics?.step      ?? [],
    loss:      raw.metrics?.loss      ?? [],
    local_acc: raw.metrics?.local_acc ?? [],
    val_step:  raw.metrics?.val_step  ?? [],
    val_acc:   raw.metrics?.val_acc   ?? [],
  } as RunMetrics,
  cancel_requested: Boolean(raw.cancel_requested),
  log_url: raw.log_url ?? null,
  source: raw.source,
});

export interface RunsState {
  active: Run[];      // status in ('starting', 'running')
  recent: Run[];      // status in ('completed', 'failed', 'cancelled'), newest first
  loading: boolean;
  error: string | null;
}

const ACTIVE_STATUSES: ReadonlySet<RunStatus> = new Set(['starting', 'running']);

/** Subscribe to /runs, partitioned into in-flight vs recent. */
export const useRuns = (recentLimit = 10): RunsState => {
  const [state, setState] = useState<RunsState>({
    active: [], recent: [], loading: true, error: null,
  });

  useEffect(() => {
    const q = query(collection(db, 'runs'), orderBy('started_at', 'desc'));
    const unsub = onSnapshot(
      q,
      (snap) => {
        const runs = snap.docs.map((d) => toRun(d.id, d.data()));
        const active = runs.filter((r) => ACTIVE_STATUSES.has(r.status));
        const recent = runs.filter((r) => !ACTIVE_STATUSES.has(r.status)).slice(0, recentLimit);
        setState({ active, recent, loading: false, error: null });
      },
      (err) => setState({ active: [], recent: [], loading: false, error: err.message }),
    );
    return unsub;
  }, [recentLimit]);

  return state;
};

/** Recharts-ready loss series: [{step, loss}, ...]. */
export const lossSeries = (m: RunMetrics): { step: number; loss: number }[] =>
  m.step.map((s, i) => ({ step: s, loss: m.loss[i] ?? 0 }));

/** Request the agent gracefully cancel a run. Sets cancel_requested=true
 *  on the run doc; the firestore.rules block restricts this single-field
 *  toggle to the user whose uid matches run.created_by_uid (or admin).
 *  The spawner's reconcile loop picks it up within ~10 s and SIGTERMs
 *  the process; if it doesn't exit within 60 s, SIGKILL escalates. */
export const cancelRun = async (runId: string): Promise<void> => {
  await updateDoc(doc(db, 'runs', runId), { cancel_requested: true });
};

/** Human-friendly elapsed runtime. */
export const formatRuntime = (run: Run): string => {
  const end = run.finished_at_ms ?? Date.now();
  const ms = Math.max(0, end - run.started_at_ms);
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  return `${h}h ${String(m).padStart(2, '0')}m`;
};

export { EMPTY_METRICS };
