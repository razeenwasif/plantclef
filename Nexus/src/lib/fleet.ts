// Fleet — live pod inventory.
// Subscribes to the top-level `/pods` Firestore collection populated by
// scripts/pod_agent.py. The agent writes its `last_heartbeat` every ~15 s;
// we derive an `offline` status client-side if the heartbeat is older than
// `STALE_AFTER_MS`. See docs/CONTROL_PLANE.md (slice 1) for the schema.

import { useEffect, useState } from 'react';
import { collection, onSnapshot, orderBy, query, Timestamp } from 'firebase/firestore';
import { db } from './firebase';

export type AcceleratorKind = 'cuda' | 'tpu' | 'cpu';
export type PodStatus = 'online' | 'busy' | 'offline' | 'degraded';

export interface PodUtilisation {
  gpu_util_pct: number;
  vram_used_gb: number;
  temperature_c: number;
  power_w: number;
}

export interface Pod {
  id: string;                       // == pod_id
  display_name: string;
  accelerator: {
    kind: AcceleratorKind;
    device_count: number;
    sku: string;
    memory_gb: number;
  };
  status: PodStatus;                // derived: 'offline' if heartbeat stale
  reportedStatus: PodStatus;        // what the agent wrote, before staleness override
  current_run_id: string | null;
  last_heartbeat_ms: number;        // epoch ms; 0 if never written
  agent_version: string;
  capabilities: string[];
  utilisation: PodUtilisation | null;
  host?: { hostname?: string; role?: string };
}

/** Heartbeat freshness budget. The agent's default cadence is 15 s, so 60 s
 *  tolerates three missed beats before we call it offline. */
export const STALE_AFTER_MS = 60_000;

const tsToMs = (v: unknown): number => {
  if (!v) return 0;
  if (v instanceof Timestamp) return v.toMillis();
  if (typeof v === 'object' && v && 'toMillis' in v) {
    try { return (v as Timestamp).toMillis(); } catch { return 0; }
  }
  if (typeof v === 'number') return v;
  return 0;
};

const toPod = (id: string, raw: any, now: number): Pod => {
  const heartbeat = tsToMs(raw.last_heartbeat);
  const reported: PodStatus = (raw.status as PodStatus) ?? 'offline';
  const stale = heartbeat === 0 || (now - heartbeat) > STALE_AFTER_MS;
  return {
    id,
    display_name: raw.display_name ?? id,
    accelerator: {
      kind: raw.accelerator?.kind ?? 'cpu',
      device_count: raw.accelerator?.device_count ?? 0,
      sku: raw.accelerator?.sku ?? 'Unknown',
      memory_gb: raw.accelerator?.memory_gb ?? 0,
    },
    status: stale ? 'offline' : reported,
    reportedStatus: reported,
    current_run_id: raw.current_run_id ?? null,
    last_heartbeat_ms: heartbeat,
    agent_version: raw.agent_version ?? '',
    capabilities: raw.capabilities ?? [],
    utilisation: raw.utilisation ?? null,
    host: raw.host,
  };
};

export interface FleetState {
  pods: Pod[];
  loading: boolean;
  error: string | null;
}

/** Live subscription to /pods. Re-derives `status` from heartbeat freshness
 *  on every snapshot AND on a 15 s interval (so a pod's card flips to
 *  'offline' even without a fresh write). */
export const useFleet = (): FleetState => {
  const [pods, setPods] = useState<Pod[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Firestore subscription
  useEffect(() => {
    const q = query(collection(db, 'pods'), orderBy('display_name'));
    const unsub = onSnapshot(
      q,
      (snap) => {
        const now = Date.now();
        setPods(snap.docs.map((d) => toPod(d.id, d.data(), now)));
        setLoading(false);
        setError(null);
      },
      (err) => {
        // Permission-denied is the common shape if the user is signed out
        // before the rules permit reads — surface it but don't throw.
        setError(err.message);
        setLoading(false);
      },
    );
    return unsub;
  }, []);

  // Tick: re-derive staleness without waiting for the next Firestore write.
  useEffect(() => {
    if (pods.length === 0) return;
    const interval = setInterval(() => {
      setPods((prev) => {
        const now = Date.now();
        let changed = false;
        const next = prev.map((p) => {
          const stale = p.last_heartbeat_ms === 0 || (now - p.last_heartbeat_ms) > STALE_AFTER_MS;
          const newStatus: PodStatus = stale ? 'offline' : p.reportedStatus;
          if (newStatus !== p.status) { changed = true; return { ...p, status: newStatus }; }
          return p;
        });
        return changed ? next : prev;
      });
    }, 15_000);
    return () => clearInterval(interval);
  }, [pods.length]);

  return { pods, loading, error };
};
