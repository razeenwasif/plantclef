import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertTriangle, Crown, Check, Loader2 } from 'lucide-react';
import { useAuth, tierMeta } from '../lib/auth';

interface Props {
  open: boolean;
  onClose: () => void;
}

type Stage = 'confirm' | 'submitting' | 'done' | 'error';

const WHAT_YOU_LOSE: Record<'pro' | 'field', string[]> = {
  pro:   [
    'Unlimited identifications (back to 5/day)',
    'API key issuance and the developer console',
    'Agentic ecological report generation',
    'Rare-species alerts',
  ],
  field: [
    'Offline field mode',
    'Unlimited identifications (back to 5/day)',
    'Agentic ecological report generation',
    'Rare-species alerts',
  ],
};

export const CancelSubscriptionDialog = ({ open, onClose }: Props) => {
  const { realTier, cancelSubscription } = useAuth();
  const [stage, setStage] = useState<Stage>('confirm');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const cancellableTier: 'pro' | 'field' | null =
    realTier === 'pro' ? 'pro' : realTier === 'field' ? 'field' : null;

  // Reset whenever opened (or whenever the user's actual tier changes
  // mid-flight to e.g. 'free' via the dev quick-switcher).
  useEffect(() => {
    if (!open) return;
    setStage('confirm');
    setErrorMessage('');
  }, [open, realTier]);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const confirm = async () => {
    setStage('submitting');
    setErrorMessage('');
    try {
      await cancelSubscription();
      setStage('done');
      // Close the dialog after a brief acknowledgement so the menu can
      // re-render against the new tier.
      setTimeout(() => onClose(), 1600);
    } catch (e: any) {
      setStage('error');
      setErrorMessage(e?.message || String(e));
    }
  };

  if (!open) return null;

  return createPortal(
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="fixed inset-0 z-[320] flex items-center justify-center p-4 bg-black/70 backdrop-blur-xl"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, y: 10, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 10, scale: 0.98 }}
          transition={{ type: 'spring', stiffness: 400, damping: 36 }}
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-[480px] glass-panel rounded-2xl overflow-hidden shadow-[0_25px_80px_rgba(0,0,0,0.7)]"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-5 border-b border-white/5">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-amber-500/15 flex items-center justify-center">
                <AlertTriangle size={14} className="text-amber-300" />
              </div>
              <div className="leading-tight">
                <h2 className="text-sm font-semibold text-white tracking-tight italic">Cancel Subscription</h2>
                <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-oracle-muted opacity-70 mt-0.5">
                  Drop back to Free
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              disabled={stage === 'submitting'}
              className="w-8 h-8 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 flex items-center justify-center transition-colors disabled:opacity-60"
              aria-label="Close"
            >
              <X size={14} className="text-white/80" />
            </button>
          </div>

          {/* Body */}
          <div className="p-6 space-y-4">
            {stage === 'confirm' && cancellableTier && (
              <>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full" style={{ background: tierMeta[cancellableTier].color, boxShadow: `0 0 8px ${tierMeta[cancellableTier].color}` }} />
                  <span className="text-[12px] font-mono uppercase tracking-[0.18em] text-white">
                    Currently on {tierMeta[cancellableTier].label}
                  </span>
                </div>
                <p className="text-[13px] text-white/85 leading-relaxed">
                  Cancellation drops your account to Free immediately. You can re-upgrade at any time from the same menu.
                </p>
                <div className="bg-white/[0.03] border border-white/10 rounded-xl p-4">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-oracle-muted mb-3">
                    You will lose access to
                  </p>
                  <ul className="space-y-1.5">
                    {WHAT_YOU_LOSE[cancellableTier].map((line) => (
                      <li key={line} className="text-[12px] text-white/80 flex gap-2">
                        <span className="text-oracle-muted">•</span>
                        <span>{line}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <p className="text-[10px] text-oracle-muted opacity-70">
                  Your account, journal, and any saved API keys are unaffected.
                </p>
              </>
            )}

            {stage === 'confirm' && !cancellableTier && (
              <p className="text-[13px] text-oracle-muted">
                You're already on the Free tier — nothing to cancel.
              </p>
            )}

            {stage === 'submitting' && (
              <div className="flex items-center gap-3 p-4 rounded-xl bg-white/[0.03] border border-white/10">
                <Loader2 size={18} className="text-oracle-accent animate-spin" />
                <span className="text-[13px] text-white/85">Updating your subscription…</span>
              </div>
            )}

            {stage === 'done' && (
              <div className="flex items-center gap-3 p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                <Check size={18} className="text-emerald-300" />
                <div>
                  <p className="text-sm text-white font-medium">Subscription cancelled</p>
                  <p className="text-[11px] text-oracle-muted mt-0.5">You're now on the Free tier.</p>
                </div>
              </div>
            )}

            {stage === 'error' && (
              <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20">
                <p className="text-sm text-red-300 font-medium mb-1">Cancellation failed</p>
                <p className="text-[11px] text-oracle-muted leading-relaxed break-words font-mono">{errorMessage}</p>
              </div>
            )}
          </div>

          {/* Footer actions */}
          <div className="flex items-center justify-end gap-2 px-6 py-4 border-t border-white/5 bg-black/30">
            {stage === 'confirm' && (
              <>
                <button
                  onClick={onClose}
                  className="px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-white/[0.04] hover:bg-white/[0.08] text-white border border-white/10 transition-colors"
                >
                  Keep {cancellableTier ? tierMeta[cancellableTier].label : 'subscription'}
                </button>
                <button
                  onClick={confirm}
                  disabled={!cancellableTier}
                  className="px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-red-500/80 hover:bg-red-500 text-white disabled:bg-white/[0.05] disabled:text-oracle-muted disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                >
                  Confirm cancellation
                </button>
              </>
            )}
            {stage === 'submitting' && (
              <button disabled className="px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-red-500/60 text-white flex items-center gap-2">
                <Loader2 size={11} className="animate-spin" />
                Cancelling…
              </button>
            )}
            {(stage === 'done' || stage === 'error') && (
              <button
                onClick={onClose}
                className="px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest bg-white/[0.04] hover:bg-white/[0.08] text-white border border-white/10 transition-colors"
              >
                Close
              </button>
            )}
          </div>

          {/* Crown footer for visual symmetry with upgrade flow */}
          {stage === 'confirm' && cancellableTier && (
            <div className="px-6 pb-4 -mt-2">
              <p className="text-[10px] text-oracle-muted opacity-60 flex items-center gap-1.5">
                <Crown size={10} className="opacity-60" />
                Need to upgrade again later? Use the Account menu or ⌘K → Upgrade.
              </p>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body,
  );
};
