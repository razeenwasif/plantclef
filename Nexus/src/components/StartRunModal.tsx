import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, Play, Loader2, CheckCircle2, AlertTriangle, Cpu, GitBranch, Hash,
} from 'lucide-react';
import type { Pod } from '../lib/fleet';
import {
  defaultRunName,
  submitRunRequest,
  useRunRequest,
} from '../lib/runRequests';
import type { RunPhase } from '../lib/runRequests';
import { useAuth } from '../lib/auth';

const PHASES: { id: RunPhase; label: string; description: string }[] = [
  { id: 'p1',  label: 'P1',  description: 'Phase-1 feature extract' },
  { id: 'p2a', label: 'P2A', description: 'Warmup on cached features' },
  { id: 'p2b', label: 'P2B', description: 'Full ensemble fine-tune' },
  { id: 'all', label: 'All', description: 'P1 → P2A → P2B end-to-end' },
];

interface Props {
  open: boolean;
  onClose: () => void;
  pods: Pod[];
  initialPodId?: string | null;
  onNavigateToMission?: () => void;
}

type Stage = 'compose' | 'submitting' | 'awaiting' | 'claimed' | 'rejected' | 'error';

export const StartRunModal = ({ open, onClose, pods, initialPodId, onNavigateToMission }: Props) => {
  const { user } = useAuth();

  // Form state
  const [podId,  setPodId]  = useState<string>('');
  const [phase,  setPhase]  = useState<RunPhase>('p2a');
  const [name,   setName]   = useState<string>('');
  const [seed,   setSeed]   = useState<number>(42);
  const [touchedName, setTouchedName] = useState(false);

  // Submission state
  const [stage,         setStage]         = useState<Stage>('compose');
  const [requestId,     setRequestId]     = useState<string | null>(null);
  const [errorMessage,  setErrorMessage]  = useState<string>('');
  const [autoCloseTimer, setAutoCloseTimer] = useState(0);

  const podOptions = useMemo(
    () => pods.filter((p) => p.status !== 'offline'),
    [pods],
  );

  // Reset state on open. Pre-fill targeted pod from launching context.
  useEffect(() => {
    if (!open) return;
    setStage('compose');
    setRequestId(null);
    setErrorMessage('');
    setAutoCloseTimer(0);
    setTouchedName(false);
    const targetPod = initialPodId
      || podOptions[0]?.id
      || pods[0]?.id
      || '';
    setPodId(targetPod);
    setPhase('p2a');
    setName(targetPod ? defaultRunName('p2a', targetPod) : '');
    setSeed(42);
  }, [open, initialPodId, pods, podOptions]);

  // Re-default name when phase/pod change unless the user has typed.
  useEffect(() => {
    if (touchedName) return;
    if (!podId) return;
    setName(defaultRunName(phase, podId));
  }, [phase, podId, touchedName]);

  // Subscribe to the submitted request so we can transition stages.
  const sub = useRunRequest(stage === 'awaiting' && requestId ? requestId : null);
  useEffect(() => {
    if (stage !== 'awaiting') return;
    const r = sub.request;
    if (!r) return;
    if (r.status === 'claimed') {
      setStage('claimed');
    } else if (r.status === 'rejected') {
      setStage('rejected');
      setErrorMessage(r.rejection_reason || 'Pod refused the request.');
    }
  }, [sub.request, stage]);

  // 30 s timeout from awaiting → fall back to error.
  useEffect(() => {
    if (stage !== 'awaiting') return;
    const t = setTimeout(() => {
      setStage('error');
      setErrorMessage('Pod did not claim the request within 30 s. Is the pod_agent running?');
    }, 30_000);
    return () => clearTimeout(t);
  }, [stage]);

  // After claimed, auto-close after a short countdown and route to Mission.
  useEffect(() => {
    if (stage !== 'claimed') return;
    setAutoCloseTimer(3);
    const tick = setInterval(() => setAutoCloseTimer((t) => t - 1), 1000);
    const close = setTimeout(() => {
      clearInterval(tick);
      onNavigateToMission?.();
      onClose();
    }, 3000);
    return () => { clearInterval(tick); clearTimeout(close); };
  }, [stage, onClose, onNavigateToMission]);

  const submit = async () => {
    if (!user) {
      setStage('error');
      setErrorMessage('Sign in required to start a run.');
      return;
    }
    if (!podId) {
      setStage('error');
      setErrorMessage('No pod selected.');
      return;
    }
    setStage('submitting');
    setErrorMessage('');
    try {
      const id = await submitRunRequest(podId, {
        name: name.trim() || defaultRunName(phase, podId),
        phase,
        seed,
      });
      setRequestId(id);
      setStage('awaiting');
    } catch (e: any) {
      setStage('error');
      setErrorMessage(e?.message || String(e));
    }
  };

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="fixed inset-0 z-[300] flex items-center justify-center p-4 bg-black/70 backdrop-blur-xl"
        onClick={onClose}
        aria-label="Start a training run"
      >
        <motion.div
          initial={{ opacity: 0, y: 10, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 10, scale: 0.98 }}
          transition={{ type: 'spring', stiffness: 400, damping: 36 }}
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-[560px] glass-panel rounded-2xl overflow-hidden shadow-[0_25px_80px_rgba(0,0,0,0.7)]"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-5 border-b border-white/5">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-oracle-accent/20 flex items-center justify-center">
                <Play size={14} className="text-oracle-accent" fill="currentColor" />
              </div>
              <div className="leading-tight">
                <h2 className="text-sm font-semibold text-white tracking-tight italic">Start a Training Run</h2>
                <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-oracle-muted opacity-70 mt-0.5">
                  /run_requests · slice 3
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 flex items-center justify-center transition-colors"
              aria-label="Close"
            >
              <X size={14} className="text-white/80" />
            </button>
          </div>

          {/* Body — branches on stage */}
          <div className="p-6 space-y-5">
            {(stage === 'compose' || stage === 'submitting') && (
              <>
                {/* Target pod */}
                <Field icon={Cpu} label="Target pod">
                  {podOptions.length === 0 ? (
                    <div className="text-[12px] text-amber-300 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                      No online pods. Start <span className="font-mono">scripts/pod_agent.py</span> on at least one host.
                    </div>
                  ) : (
                    <select
                      value={podId}
                      onChange={(e) => setPodId(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg bg-white/[0.04] border border-white/10 text-sm text-white outline-none focus:border-oracle-accent/60 transition-colors"
                      disabled={stage === 'submitting'}
                    >
                      {podOptions.map((p) => (
                        <option key={p.id} value={p.id} className="bg-oracle-bg">
                          {p.display_name} ({p.accelerator.device_count}× {p.accelerator.sku})
                        </option>
                      ))}
                    </select>
                  )}
                </Field>

                {/* Phase segmented */}
                <Field icon={GitBranch} label="Phase">
                  <div className="grid grid-cols-4 gap-2">
                    {PHASES.map((p) => {
                      const active = phase === p.id;
                      return (
                        <button
                          key={p.id}
                          type="button"
                          onClick={() => setPhase(p.id)}
                          disabled={stage === 'submitting'}
                          className={`px-3 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest transition-colors border ${
                            active
                              ? 'bg-oracle-accent/15 border-oracle-accent/40 text-white'
                              : 'bg-white/[0.03] border-white/10 text-oracle-muted hover:text-white hover:border-white/20'
                          }`}
                          title={p.description}
                        >
                          {p.label}
                        </button>
                      );
                    })}
                  </div>
                  <p className="text-[10px] text-oracle-muted opacity-70 mt-2">
                    {PHASES.find((p) => p.id === phase)?.description}
                  </p>
                </Field>

                {/* Name */}
                <Field icon={Hash} label="Run name">
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => { setName(e.target.value); setTouchedName(true); }}
                    disabled={stage === 'submitting'}
                    className="w-full px-3 py-2 rounded-lg bg-white/[0.04] border border-white/10 text-sm text-white outline-none focus:border-oracle-accent/60 font-mono transition-colors"
                    spellCheck={false}
                    maxLength={80}
                  />
                </Field>

                {/* Seed */}
                <Field icon={Hash} label="Seed">
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
                    disabled={stage === 'submitting'}
                    className="w-full px-3 py-2 rounded-lg bg-white/[0.04] border border-white/10 text-sm text-white outline-none focus:border-oracle-accent/60 font-mono transition-colors"
                    min={0}
                    max={2147483647}
                  />
                </Field>
              </>
            )}

            {stage === 'awaiting' && (
              <Status
                icon={<Loader2 size={18} className="text-oracle-accent animate-spin" />}
                title="Waiting for pod to claim the request"
                body={`Targeting ${podId}. The agent should pick this up within ~10 s. The Mission tab opens automatically once the run starts.`}
              />
            )}

            {stage === 'claimed' && (
              <Status
                icon={<CheckCircle2 size={18} className="text-emerald-400" />}
                title={`Claimed by ${podId}`}
                body={`Run ${sub.request?.claimed_run_id ?? ''} is starting. Routing to Mission in ${autoCloseTimer}s…`}
                accent="emerald"
              />
            )}

            {stage === 'rejected' && (
              <Status
                icon={<AlertTriangle size={18} className="text-amber-400" />}
                title="Request rejected"
                body={errorMessage}
                accent="amber"
              />
            )}

            {stage === 'error' && (
              <Status
                icon={<AlertTriangle size={18} className="text-red-400" />}
                title="Something went wrong"
                body={errorMessage}
                accent="red"
              />
            )}
          </div>

          {/* Footer actions */}
          <div className="flex items-center justify-end gap-2 px-6 py-4 border-t border-white/5 bg-black/30">
            {(stage === 'rejected' || stage === 'error') && (
              <button
                onClick={() => { setStage('compose'); setRequestId(null); setErrorMessage(''); }}
                className="px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-white/[0.04] hover:bg-white/[0.08] text-white border border-white/10 transition-colors"
              >
                Try again
              </button>
            )}
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-white/[0.04] hover:bg-white/[0.08] text-oracle-muted hover:text-white border border-white/10 transition-colors"
            >
              {stage === 'claimed' ? 'Close' : 'Cancel'}
            </button>
            {stage === 'compose' && (
              <button
                onClick={submit}
                disabled={!podId || !name.trim() || podOptions.length === 0}
                className="px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-oracle-accent text-white hover:bg-oracle-accent/90 disabled:bg-white/[0.05] disabled:text-oracle-muted disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                <Play size={11} fill="currentColor" />
                Submit
              </button>
            )}
            {stage === 'submitting' && (
              <button disabled className="px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-oracle-accent/60 text-white flex items-center gap-2">
                <Loader2 size={11} className="animate-spin" />
                Submitting…
              </button>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body,
  );
};

// ── tiny presentation helpers ────────────────────────────────────────────
const Field = ({ icon: Icon, label, children }: { icon: any; label: string; children: React.ReactNode }) => (
  <div>
    <div className="flex items-center gap-2 mb-2">
      <Icon size={12} className="text-oracle-accent opacity-80" />
      <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-oracle-muted">{label}</span>
    </div>
    {children}
  </div>
);

const Status = ({ icon, title, body, accent }: { icon: React.ReactNode; title: string; body: string; accent?: 'emerald' | 'amber' | 'red' }) => {
  const wrapClass =
    accent === 'emerald' ? 'border-emerald-500/20 bg-emerald-500/5' :
    accent === 'amber'   ? 'border-amber-500/20 bg-amber-500/5' :
    accent === 'red'     ? 'border-red-500/20 bg-red-500/5' :
    'border-white/10 bg-white/[0.03]';
  return (
    <div className={`p-5 rounded-xl border ${wrapClass} flex gap-4`}>
      <div className="shrink-0 mt-0.5">{icon}</div>
      <div className="min-w-0">
        <p className="text-sm text-white font-medium mb-1">{title}</p>
        <p className="text-[12px] text-oracle-muted leading-relaxed break-words">{body}</p>
      </div>
    </div>
  );
};
