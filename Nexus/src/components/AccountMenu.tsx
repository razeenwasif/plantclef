import { useState, useRef, useEffect, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { LogIn, LogOut, ChevronDown, Crown, ShieldCheck, Eye, Sparkles, User as UserIcon, Check } from 'lucide-react';
import { useAuth, tierMeta } from '../lib/auth';
import type { Tier } from '../lib/auth';
import { AuthModal } from './AuthModal';

interface AccountMenuProps {
  onCancelSubscription?: () => void;
}

export const AccountMenu = ({ onCancelSubscription }: AccountMenuProps = {}) => {
  const { user, loading, realTier, effectiveTier, isAdmin, viewingAs, signOut, viewAsTier, upgradeTier } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  const [menuOpen, setMenuOpen] = useState(false);
  const anchorRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuPos, setMenuPos] = useState<{ top: number; right: number } | null>(null);

  // Recompute dropdown position when opened, and on window resize while open
  useLayoutEffect(() => {
    if (!menuOpen) return;
    const update = () => {
      const r = anchorRef.current?.getBoundingClientRect();
      if (r) setMenuPos({ top: r.bottom + 8, right: window.innerWidth - r.right });
    };
    update();
    window.addEventListener('resize', update);
    window.addEventListener('scroll', update, true);
    return () => {
      window.removeEventListener('resize', update);
      window.removeEventListener('scroll', update, true);
    };
  }, [menuOpen]);

  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (menuRef.current?.contains(target)) return;
      if (anchorRef.current?.contains(target)) return;
      setMenuOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [menuOpen]);

  if (loading) {
    return (
      <div className="w-9 h-9 rounded-xl bg-white/5 border border-white/10 animate-pulse" />
    );
  }

  if (!user) {
    return (
      <>
        <button
          onClick={() => { setAuthMode('signin'); setAuthOpen(true); }}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-oracle-accent/20 to-oracle-pink/20 hover:from-oracle-accent/30 hover:to-oracle-pink/30 border border-oracle-accent/30 text-[10px] font-black uppercase tracking-[0.2em] text-white transition-all shadow-[0_0_20px_rgb(var(--oracle-accent)/0.15)]"
        >
          <LogIn size={12} />
          Sign In
        </button>
        <AuthModal open={authOpen} initialMode={authMode} onClose={() => setAuthOpen(false)} />
      </>
    );
  }

  const meta = tierMeta[effectiveTier];
  const initials = (user.displayName || user.email || '?')
    .split(/[ @.]/).filter(Boolean).slice(0, 2)
    .map(s => s[0]?.toUpperCase()).join('');

  return (
    <div className="relative">
      <button
        ref={anchorRef}
        onClick={() => setMenuOpen((v) => !v)}
        className="flex items-center gap-2 pl-1.5 pr-2 sm:pl-2 sm:pr-3 py-1 sm:py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"
      >
        <div
          className="w-8 h-8 sm:w-7 sm:h-7 rounded-lg flex items-center justify-center text-[10px] font-black flex-shrink-0"
          style={{ background: meta.bg, color: meta.color, border: `1px solid ${meta.color}40` }}
        >
          {initials || <UserIcon size={12} />}
        </div>
        <div className="hidden md:flex flex-col items-start leading-none">
          <span className="text-[9px] font-mono text-oracle-muted opacity-60 truncate max-w-[120px]">
            {user.email}
          </span>
          <span className="text-[9px] font-black uppercase tracking-widest mt-0.5" style={{ color: meta.color }}>
            {meta.label}
            {isAdmin && viewingAs && viewingAs !== 'admin' && (
              <span className="text-oracle-muted opacity-60"> · viewing</span>
            )}
          </span>
        </div>
        <ChevronDown size={12} className={`hidden md:flex text-oracle-muted transition-transform ${menuOpen ? 'rotate-180' : ''}`} />
      </button>

      {createPortal(
        <AnimatePresence>
          {menuOpen && menuPos && (
            <motion.div
              ref={menuRef}
              initial={{ opacity: 0, y: -8, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -8, scale: 0.97 }}
              transition={{ duration: 0.15 }}
              style={{ position: 'fixed', top: menuPos.top, right: menuPos.right }}
              className="w-72 glass-panel rounded-2xl p-2 z-[300] shadow-[0_25px_60px_rgba(0,0,0,0.6)]"
            >
            {/* User info */}
            <div className="px-3 py-3 border-b border-white/5 mb-2">
              <p className="text-[12px] text-white font-bold tracking-tight truncate">
                {user.displayName || user.email}
              </p>
              {user.displayName && (
                <p className="text-[10px] text-oracle-muted font-mono mt-0.5 truncate">{user.email}</p>
              )}
              <div className="mt-3 flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ background: tierMeta[realTier].color, boxShadow: `0 0 8px ${tierMeta[realTier].color}` }}
                  />
                  <span className="text-[10px] font-black uppercase tracking-widest" style={{ color: tierMeta[realTier].color }}>
                    {tierMeta[realTier].label} Tier
                  </span>
                </div>
                {realTier !== 'admin' && realTier !== 'pro' && (
                  <button
                    onClick={() => { upgradeTier('pro'); }}
                    className="flex items-center gap-1 px-2 py-1 rounded-md bg-oracle-accent/20 hover:bg-oracle-accent/30 border border-oracle-accent/40 text-[9px] font-black uppercase tracking-widest text-white"
                  >
                    <Crown size={10} />
                    Upgrade
                  </button>
                )}
                {(realTier === 'pro' || realTier === 'field') && onCancelSubscription && (
                  <button
                    onClick={() => { setMenuOpen(false); onCancelSubscription(); }}
                    className="flex items-center gap-1 px-2 py-1 rounded-md bg-white/[0.04] hover:bg-red-500/20 border border-white/10 hover:border-red-500/30 text-[9px] font-black uppercase tracking-widest text-oracle-muted hover:text-red-300 transition-colors"
                    title="Cancel your subscription and drop back to Free"
                  >
                    Cancel
                  </button>
                )}
              </div>
              <p className="text-[10px] text-oracle-muted mt-2 opacity-70">{tierMeta[realTier].description}</p>
            </div>

            {/* Admin: View as ... */}
            {isAdmin && (
              <div className="px-3 py-2 border-b border-white/5 mb-2">
                <div className="flex items-center gap-2 mb-2">
                  <Eye size={12} className="text-oracle-pink" />
                  <span className="text-[9px] font-black text-oracle-muted uppercase tracking-[0.2em]">View As</span>
                  {viewingAs && viewingAs !== 'admin' && (
                    <span className="ml-auto text-[9px] font-mono text-oracle-pink">impersonating</span>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-1">
                  {(['free', 'pro', 'field', 'admin'] as Tier[]).map((t) => {
                    const tm = tierMeta[t];
                    const active = (viewingAs ?? 'admin') === t;
                    return (
                      <button
                        key={t}
                        onClick={() => viewAsTier(t === 'admin' ? null : t)}
                        className={`flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg border text-[10px] font-black uppercase tracking-widest transition-all ${
                          active ? 'border-white/20 bg-white/[0.06]' : 'border-white/5 hover:bg-white/[0.04]'
                        }`}
                        style={active ? { color: tm.color } : { color: '#94a3b8' }}
                      >
                        <span className="flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full" style={{ background: tm.color }} />
                          {tm.label}
                        </span>
                        {active && <Check size={10} />}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* QA: quick tier switcher (non-admin) — temporary until real billing exists */}
            {!isAdmin && (
              <div className="px-3 py-2 border-b border-white/5 mb-2">
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles size={12} className="text-oracle-cyan" />
                  <span className="text-[9px] font-black text-oracle-muted uppercase tracking-[0.2em]">Dev · Set Tier</span>
                </div>
                <div className="grid grid-cols-3 gap-1">
                  {(['free', 'pro', 'field'] as const).map((t) => {
                    const tm = tierMeta[t];
                    const active = realTier === t;
                    return (
                      <button
                        key={t}
                        onClick={() => upgradeTier(t)}
                        className={`px-2 py-1.5 rounded-lg border text-[9px] font-black uppercase tracking-widest transition-all ${
                          active ? 'border-white/20 bg-white/[0.06]' : 'border-white/5 hover:bg-white/[0.04]'
                        }`}
                        style={{ color: active ? tm.color : '#94a3b8' }}
                      >
                        {tm.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Actions */}
            <button
              className="w-full flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-white/5 text-left text-[11px] text-oracle-muted hover:text-white transition-colors"
            >
              <ShieldCheck size={14} />
              <span>Account & Billing</span>
              <span className="ml-auto text-[8px] font-mono opacity-40 uppercase">Soon</span>
            </button>

            <button
              onClick={() => { setMenuOpen(false); signOut(); }}
              className="w-full flex items-center gap-3 px-3 py-2 rounded-xl hover:bg-red-500/10 text-left text-[11px] text-red-300/80 hover:text-red-300 transition-colors"
            >
              <LogOut size={14} />
              Sign Out
            </button>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </div>
  );
};
