// Console event stream — derived from live /pods and /runs state.
// We don't have raw log-line streaming (training logs live on the pod's
// disk, never in Firestore). What we *do* have is high-signal state
// changes — pods coming online, runs starting/completing, step batches
// landing in metrics, cancellations propagating. This hook diffs each
// snapshot against the previous one and appends the deltas as
// human-readable log lines, capped at a fixed ring-buffer size.
//
// Use it at the top of a component and feed its `events` straight to the
// Console tab's render. No subscriptions of its own; it is purely
// derived from the data already flowing through useFleet + useRuns.

import { useEffect, useRef, useState } from 'react';
import type { Pod, PodStatus } from './fleet';
import type { Run, RunStatus } from './runs';

export type ConsoleSeverity = 'info' | 'success' | 'warn' | 'error';

export interface ConsoleEvent {
  id: string;          // monotonic
  ts: number;          // epoch ms
  source: string;      // 'pod' | 'run' | 'agent' | 'system'
  severity: ConsoleSeverity;
  message: string;
}

const RING_SIZE = 200;
const STEP_LOG_GAP = 1000;       // log at most one step-progress line per 1k steps per run

let _seq = 0;
const _id = (): string => `e${_seq++}`;

const fmtAge = (ms: number): string => {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60)     return `${s}s`;
  if (s < 3600)   return `${Math.round(s / 60)}m`;
  return `${Math.round(s / 3600)}h`;
};

// Snapshots we hold across renders to compute deltas. Map keyed by id so
// adds and removes are diffable in O(n).
interface PodSnap {
  status: PodStatus;
  current_run_id: string | null;
  last_heartbeat_ms: number;
}
interface RunSnap {
  status: RunStatus;
  cancel_requested: boolean;
  last_logged_step: number;        // we log step progress every STEP_LOG_GAP
  latest_step: number;
}

export const useConsoleEvents = (pods: Pod[], runs: Run[]): ConsoleEvent[] => {
  const [events, setEvents] = useState<ConsoleEvent[]>([]);
  const podsRef = useRef<Map<string, PodSnap>>(new Map());
  const runsRef = useRef<Map<string, RunSnap>>(new Map());
  // Suppress the deluge on first render: just seed the snapshots without emitting.
  const seededRef = useRef(false);
  // Emit the seeding line once so the user knows the stream is live.
  const startupRef = useRef(false);

  useEffect(() => {
    if (!startupRef.current) {
      startupRef.current = true;
      setEvents((prev) => append(prev, {
        id: _id(), ts: Date.now(), source: 'system', severity: 'info',
        message: 'Console attached · subscribing to /pods and /runs',
      }));
    }

    const next: ConsoleEvent[] = [];
    const now = Date.now();

    // ── pods diff ──────────────────────────────────────────────────────
    const podMap = new Map<string, PodSnap>();
    for (const p of pods) {
      podMap.set(p.id, {
        status: p.status,
        current_run_id: p.current_run_id,
        last_heartbeat_ms: p.last_heartbeat_ms,
      });
    }
    if (seededRef.current) {
      for (const [id, cur] of podMap) {
        const prev = podsRef.current.get(id);
        if (!prev) {
          next.push(line('pod', 'success', `pod ${id} registered`));
          continue;
        }
        if (prev.status !== cur.status) {
          if (cur.status === 'offline') {
            const age = cur.last_heartbeat_ms ? fmtAge(now - cur.last_heartbeat_ms) : 'never';
            next.push(line('pod', 'warn', `pod ${id} went offline (last heartbeat ${age} ago)`));
          } else if (prev.status === 'offline') {
            next.push(line('pod', 'success', `pod ${id} came back online`));
          } else if (cur.status === 'busy' && prev.status !== 'busy') {
            const ref = cur.current_run_id ? cur.current_run_id.slice(-12) : '?';
            next.push(line('pod', 'info', `pod ${id} accepted run ${ref}`));
          } else if (prev.status === 'busy' && cur.status === 'online') {
            next.push(line('pod', 'info', `pod ${id} freed`));
          } else {
            next.push(line('pod', 'info', `pod ${id} status → ${cur.status}`));
          }
        } else if (cur.current_run_id !== prev.current_run_id) {
          const ref = cur.current_run_id ? cur.current_run_id.slice(-12) : 'idle';
          next.push(line('pod', 'info', `pod ${id} now on run ${ref}`));
        }
      }
      for (const id of podsRef.current.keys()) {
        if (!podMap.has(id)) {
          next.push(line('pod', 'warn', `pod ${id} removed from fleet`));
        }
      }
    }
    podsRef.current = podMap;

    // ── runs diff ──────────────────────────────────────────────────────
    const runMap = new Map<string, RunSnap>();
    for (const r of runs) {
      const latestStep = r.metrics.step.length ? r.metrics.step[r.metrics.step.length - 1] : 0;
      const prev = runsRef.current.get(r.id);
      runMap.set(r.id, {
        status: r.status,
        cancel_requested: r.cancel_requested,
        last_logged_step: prev ? prev.last_logged_step : latestStep,
        latest_step: latestStep,
      });
    }
    if (seededRef.current) {
      for (const r of runs) {
        const cur = runMap.get(r.id)!;
        const prev = runsRef.current.get(r.id);
        if (!prev) {
          next.push(line(
            'run', sevForRun(cur.status),
            `${cur.status === 'completed' ? 'observed completed run' : cur.status === 'failed' ? 'observed failed run' : 'run started'} ${r.spec.name} on ${r.pod_id} (phase ${r.spec.phase})`,
          ));
          continue;
        }
        if (prev.status !== cur.status) {
          next.push(line('run', sevForRun(cur.status), `run ${r.spec.name} → ${cur.status}`));
        }
        if (cur.cancel_requested && !prev.cancel_requested) {
          next.push(line('run', 'warn', `cancel requested for run ${r.spec.name}`));
        }
        // Step progress: emit when the latest step crosses the next STEP_LOG_GAP boundary
        if (cur.latest_step >= prev.last_logged_step + STEP_LOG_GAP) {
          const lossArr = r.metrics.loss;
          const loss = lossArr.length ? lossArr[lossArr.length - 1] : null;
          next.push(line('run', 'info',
            `run ${r.spec.name} step ${cur.latest_step.toLocaleString()}` +
            (loss !== null ? ` · loss=${loss.toFixed(3)}` : ''),
          ));
          runMap.get(r.id)!.last_logged_step = cur.latest_step;
        } else {
          // carry the previous last_logged_step forward so we don't reset on every render
          runMap.get(r.id)!.last_logged_step = prev.last_logged_step;
        }
      }
      // No remove-emit for runs — completed runs stay in /runs as archive.
    }
    runsRef.current = runMap;

    if (!seededRef.current) {
      seededRef.current = true;
      // First render seeded the snapshots; nothing to append from the diff.
    } else if (next.length > 0) {
      setEvents((prev) => {
        let out = prev;
        for (const e of next) out = append(out, e);
        return out;
      });
    }
  }, [pods, runs]);

  return events;
};

const line = (source: ConsoleEvent['source'], severity: ConsoleSeverity, message: string): ConsoleEvent => ({
  id: _id(),
  ts: Date.now(),
  source,
  severity,
  message,
});

const append = (buf: ConsoleEvent[], e: ConsoleEvent): ConsoleEvent[] => {
  const next = buf.length >= RING_SIZE ? buf.slice(buf.length - RING_SIZE + 1) : buf.slice();
  next.push(e);
  return next;
};

const sevForRun = (s: RunStatus): ConsoleSeverity => {
  switch (s) {
    case 'completed': return 'success';
    case 'failed':    return 'error';
    case 'cancelled': return 'warn';
    case 'starting':  return 'info';
    case 'running':   return 'info';
  }
};

/** Pretty timestamp HH:MM:SS for the log line gutter. */
export const fmtConsoleTs = (ms: number): string => {
  const d = new Date(ms);
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  return `${h}:${m}:${s}`;
};

/** Tailwind class for a given severity. */
export const severityClass = (sev: ConsoleSeverity): string => {
  switch (sev) {
    case 'success': return 'text-emerald-300';
    case 'warn':    return 'text-amber-300';
    case 'error':   return 'text-red-400';
    case 'info':    return 'text-zinc-400';
  }
};
