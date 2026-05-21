// Run requests — submit + observe Start-Run flow.
// Backed by /run_requests Firestore collection. The targeted pod's
// agent (scripts/pod_agent.py → run_spawner.py) reconciles these into
// /runs/{run_id} documents. See docs/CONTROL_PLANE.md (slice 3).

import { useEffect, useState } from 'react';
import {
  addDoc, collection, doc, onSnapshot, serverTimestamp, Timestamp,
} from 'firebase/firestore';
import { db, auth } from './firebase';

export type RunPhase = 'p1' | 'p2a' | 'p2b' | 'all';
export type RunRequestStatus = 'pending' | 'claimed' | 'rejected';

export interface RunRequestSpec {
  name: string;
  phase: RunPhase;
  seed: number;
  cluster_manifest_ref?: string | null;
  resume_from?: string | null;
  env_overrides?: Record<string, string>;
  extra_args?: string[];
}

export interface RunRequest {
  id: string;
  target_pod_id: string;
  requested_by_uid: string;
  requested_at_ms: number;
  spec: RunRequestSpec;
  status: RunRequestStatus;
  claimed_run_id?: string;
  claimed_at_ms?: number;
  rejection_reason?: string;
}

const tsToMs = (v: unknown): number => {
  if (!v) return 0;
  if (v instanceof Timestamp) return v.toMillis();
  if (typeof v === 'object' && v && 'toMillis' in v) {
    try { return (v as Timestamp).toMillis(); } catch { return 0; }
  }
  return typeof v === 'number' ? v : 0;
};

const toRunRequest = (id: string, raw: any): RunRequest => ({
  id,
  target_pod_id: raw.target_pod_id ?? '',
  requested_by_uid: raw.requested_by_uid ?? '',
  requested_at_ms: tsToMs(raw.requested_at),
  spec: {
    name: raw.spec?.name ?? '',
    phase: (raw.spec?.phase as RunPhase) ?? 'p2a',
    seed: raw.spec?.seed ?? 42,
    cluster_manifest_ref: raw.spec?.cluster_manifest_ref ?? null,
    resume_from: raw.spec?.resume_from ?? null,
    env_overrides: raw.spec?.env_overrides ?? {},
    extra_args: raw.spec?.extra_args ?? [],
  },
  status: (raw.status as RunRequestStatus) ?? 'pending',
  claimed_run_id: raw.claimed_run_id,
  claimed_at_ms: raw.claimed_at ? tsToMs(raw.claimed_at) : undefined,
  rejection_reason: raw.rejection_reason,
});

/** Submit a new run request. Returns the request id (Firestore doc id). */
export const submitRunRequest = async (
  targetPodId: string,
  spec: RunRequestSpec,
): Promise<string> => {
  const user = auth.currentUser;
  if (!user) throw new Error('Sign in to start a run.');
  const docRef = await addDoc(collection(db, 'run_requests'), {
    target_pod_id: targetPodId,
    requested_by_uid: user.uid,
    requested_at: serverTimestamp(),
    spec: {
      name: spec.name,
      phase: spec.phase,
      seed: spec.seed,
      cluster_manifest_ref: spec.cluster_manifest_ref ?? null,
      resume_from: spec.resume_from ?? null,
      env_overrides: spec.env_overrides ?? {},
      extra_args: spec.extra_args ?? [],
    },
    status: 'pending',
  });
  return docRef.id;
};

export interface RunRequestSubscription {
  request: RunRequest | null;
  loading: boolean;
  error: string | null;
}

/** Subscribe to a single /run_requests/{id} doc. Used by the modal to
 *  surface pending → claimed → rejected transitions. */
export const useRunRequest = (requestId: string | null): RunRequestSubscription => {
  const [state, setState] = useState<RunRequestSubscription>({
    request: null, loading: !!requestId, error: null,
  });

  useEffect(() => {
    if (!requestId) { setState({ request: null, loading: false, error: null }); return; }
    setState((s) => ({ ...s, loading: true }));
    const unsub = onSnapshot(
      doc(db, 'run_requests', requestId),
      (snap) => {
        if (!snap.exists()) { setState({ request: null, loading: false, error: null }); return; }
        setState({ request: toRunRequest(snap.id, snap.data()), loading: false, error: null });
      },
      (err) => setState({ request: null, loading: false, error: err.message }),
    );
    return unsub;
  }, [requestId]);

  return state;
};

/** Make a sensible default name for a fresh request:
 *  ``{phase}-{pod}-YYYYMMDD-HHmm``. */
export const defaultRunName = (phase: RunPhase, podId: string): string => {
  const now = new Date();
  const yyyy = now.getFullYear();
  const mm = String(now.getMonth() + 1).padStart(2, '0');
  const dd = String(now.getDate()).padStart(2, '0');
  const hh = String(now.getHours()).padStart(2, '0');
  const mi = String(now.getMinutes()).padStart(2, '0');
  return `${phase}-${podId}-${yyyy}${mm}${dd}-${hh}${mi}`;
};
