import { useState, useEffect, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search, ArrowUp, ArrowDown, CornerDownLeft, Command,
  LayoutGrid, Play, BarChart3, Cpu, Terminal,
  Network, BookOpen,
  LogIn, LogOut, Eye, Crown, ChevronRight,
} from 'lucide-react';
import { useAuth, tierMeta } from '../lib/auth';
import type { Tier } from '../lib/auth';

export type PaletteAction =
  | { kind: 'tab'; tab: string }
  | { kind: 'signin' }
  | { kind: 'signout' }
  | { kind: 'viewAs'; tier: Tier | null }
  | { kind: 'upgrade'; tier: Exclude<Tier, 'admin'> }
  | { kind: 'startRun'; podId: string | null }
  | { kind: 'cancelSubscription' };

interface CommandItem {
  id: string;
  group: string;
  label: string;
  description?: string;
  icon: any;
  keywords?: string[];
  rightHint?: string;
  action: PaletteAction;
  accent?: 'accent' | 'pink' | 'emerald' | 'cyan' | 'red';
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onAction: (action: PaletteAction) => void;
}

export const CommandPalette = ({ open, onClose, onAction }: CommandPaletteProps) => {
  const { user, isAdmin, realTier, viewingAs } = useAuth();
  const [query, setQuery] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Build the canonical action list (depends on auth state)
  const items: CommandItem[] = useMemo(() => {
    const tabs: CommandItem[] = [
      { id: 't:cluster',      group: 'Go to', label: 'Fleet',        icon: LayoutGrid, keywords: ['gpu', 'cluster', 'silicon', 'fleet'], action: { kind: 'tab', tab: 'cluster' }, rightHint: 'G F' },
      { id: 't:mission',      group: 'Go to', label: 'Mission',      icon: Play,       keywords: ['training', 'jobs', 'queue'], action: { kind: 'tab', tab: 'mission' }, rightHint: 'G M' },
      { id: 't:analytics',    group: 'Go to', label: 'Analytics',    icon: BarChart3,  keywords: ['charts', 'metrics', 'f1'], action: { kind: 'tab', tab: 'analytics' } },
      { id: 't:hardware',     group: 'Go to', label: 'Silicon',      icon: Cpu,        keywords: ['vitals', 'blackwell', '3d'], action: { kind: 'tab', tab: 'hardware' } },
      { id: 't:terminal',     group: 'Go to', label: 'Console',      icon: Terminal,   keywords: ['logs', 'terminal'], action: { kind: 'tab', tab: 'terminal' } },
      { id: 't:architecture', group: 'Go to', label: 'Architecture', icon: Network,    keywords: ['engineering', 'control plane', 'agent', 'firestore', 'schema', 'how built'], action: { kind: 'tab', tab: 'architecture' } },
      { id: 't:methodology',  group: 'Go to', label: 'Methodology',  icon: BookOpen,   keywords: ['design', 'rationale', 'why', 'rollout', 'slices', 'decisions'], action: { kind: 'tab', tab: 'methodology' } },
    ];

    const actions: CommandItem[] = [
      {
        id: 'a:startrun',
        group: 'Actions',
        label: 'Start a Training Run…',
        description: 'Submit a request to one of the online pods',
        icon: Play,
        keywords: ['train', 'run', 'launch', 'job', 'start', 'mission', 'spawn', 'request'],
        action: { kind: 'startRun', podId: null },
        accent: 'accent',
      },
    ];

    const account: CommandItem[] = user
      ? [
          { id: 'a:signout', group: 'Account', label: 'Sign Out', icon: LogOut, action: { kind: 'signout' }, accent: 'red' },
        ]
      : [
          { id: 'a:signin', group: 'Account', label: 'Sign In', description: 'Email / password or Google', icon: LogIn, action: { kind: 'signin' }, accent: 'accent' },
        ];

    if (user && realTier !== 'pro' && realTier !== 'admin') {
      account.push({ id: 'a:upgrade', group: 'Account', label: 'Upgrade to Pro', description: 'Unlimited identifications, rare-species alerts', icon: Crown, action: { kind: 'upgrade', tier: 'pro' }, accent: 'accent' });
    }
    if (user && (realTier === 'pro' || realTier === 'field')) {
      account.push({
        id: 'a:cancelSubscription',
        group: 'Account',
        label: 'Cancel Subscription…',
        description: `Drop from ${realTier === 'pro' ? 'Pro' : 'Field'} back to Free`,
        icon: LogOut,
        keywords: ['cancel', 'unsubscribe', 'downgrade', 'subscription', 'plan', 'billing'],
        action: { kind: 'cancelSubscription' },
        accent: 'red',
      });
    }

    const adminItems: CommandItem[] = isAdmin
      ? (['free', 'pro', 'field', 'admin'] as Tier[]).map((t) => {
          const tm = tierMeta[t];
          const current = (viewingAs ?? 'admin') === t;
          return {
            id: `va:${t}`,
            group: 'Admin · View as',
            label: `${tm.label}${current ? '  (current)' : ''}`,
            description: tm.description,
            icon: Eye,
            keywords: ['view', 'impersonate', 'tier', t],
            action: { kind: 'viewAs', tier: t === 'admin' ? null : t },
            accent: 'pink',
          };
        })
      : [];

    return [...tabs, ...actions, ...account, ...adminItems];
  }, [user, isAdmin, realTier, viewingAs]);

  // Filter by query
  const filtered = useMemo(() => {
    const q = query.toLowerCase().trim();
    if (!q) return items;
    return items.filter((i) => {
      if (i.label.toLowerCase().includes(q)) return true;
      if (i.group.toLowerCase().includes(q)) return true;
      if (i.description?.toLowerCase().includes(q)) return true;
      if (i.keywords?.some((k) => k.toLowerCase().includes(q))) return true;
      return false;
    });
  }, [items, query]);

  // Group filtered items preserving order
  const grouped = useMemo(() => {
    const map = new Map<string, CommandItem[]>();
    filtered.forEach((i) => {
      const arr = map.get(i.group) ?? [];
      arr.push(i);
      map.set(i.group, arr);
    });
    return [...map.entries()];
  }, [filtered]);

  // Reset state on open
  useEffect(() => {
    if (!open) return;
    setQuery('');
    setSelectedIdx(0);
    // focus input after the animation kicks in
    const t = setTimeout(() => inputRef.current?.focus(), 60);
    return () => clearTimeout(t);
  }, [open]);

  // Clamp selection if filtered shrinks
  useEffect(() => {
    if (selectedIdx >= filtered.length) setSelectedIdx(Math.max(0, filtered.length - 1));
  }, [filtered.length, selectedIdx]);

  // Keyboard navigation while open
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); return; }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIdx((i) => Math.min(i + 1, filtered.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIdx((i) => Math.max(i - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const item = filtered[selectedIdx];
        if (item) { onAction(item.action); onClose(); }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, filtered, selectedIdx, onAction, onClose]);

  // Auto-scroll selected item into view
  useEffect(() => {
    const el = document.querySelector<HTMLDivElement>(`[data-palette-idx="${selectedIdx}"]`);
    el?.scrollIntoView({ block: 'nearest' });
  }, [selectedIdx]);

  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
          className="fixed inset-0 z-[400] flex items-start justify-center p-4 pt-[15vh] bg-black/70 backdrop-blur-md"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 500, damping: 38 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-[600px] glass-panel rounded-2xl overflow-hidden shadow-[0_25px_80px_rgba(0,0,0,0.65)]"
          >
            {/* Search input */}
            <div className="flex items-center gap-3 px-5 pt-4 pb-4 border-b border-white/5">
              <Search size={16} className="text-oracle-muted" />
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => { setQuery(e.target.value); setSelectedIdx(0); }}
                placeholder="Search commands, pages, or actions…"
                className="flex-1 bg-transparent outline-none text-sm text-white placeholder:text-oracle-muted/50"
              />
              <kbd className="text-[10px] font-mono text-oracle-muted bg-white/5 border border-white/10 rounded px-2 py-0.5">
                esc
              </kbd>
            </div>

            {/* Results */}
            <div className="max-h-[50vh] overflow-y-auto custom-scrollbar py-2">
              {filtered.length === 0 && (
                <div className="px-5 py-10 text-center">
                  <p className="text-sm text-oracle-muted">No matches for <span className="text-white font-mono">"{query}"</span></p>
                </div>
              )}

              {grouped.map(([group, groupItems]) => (
                <div key={group} className="px-2 py-1">
                  <div className="px-3 py-2 text-[10px] font-semibold text-oracle-muted/60 tracking-[0.18em] uppercase">
                    {group}
                  </div>
                  {groupItems.map((item) => {
                    const idx = filtered.indexOf(item);
                    const selected = idx === selectedIdx;
                    const accentColor = accentToColor(item.accent);
                    return (
                      <div
                        key={item.id}
                        data-palette-idx={idx}
                        onMouseEnter={() => setSelectedIdx(idx)}
                        onClick={() => { onAction(item.action); onClose(); }}
                        className={`flex items-center gap-3 mx-1 px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                          selected ? 'bg-white/[0.06]' : 'hover:bg-white/[0.03]'
                        }`}
                      >
                        <div
                          className={`w-7 h-7 rounded-md flex items-center justify-center transition-colors ${
                            selected ? '' : 'bg-white/[0.03]'
                          }`}
                          style={selected ? { background: `${accentColor}1A`, color: accentColor } : { color: 'rgba(255,255,255,0.7)' }}
                        >
                          <item.icon size={14} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className={`text-sm truncate ${selected ? 'text-white' : 'text-white/85'}`}>
                            {item.label}
                          </p>
                          {item.description && (
                            <p className="text-[11px] text-oracle-muted truncate opacity-70">{item.description}</p>
                          )}
                        </div>
                        {item.rightHint && (
                          <kbd className="text-[10px] font-mono text-oracle-muted bg-white/5 border border-white/10 rounded px-1.5 py-0.5">
                            {item.rightHint}
                          </kbd>
                        )}
                        {selected && (
                          <ChevronRight size={14} className="text-oracle-muted" />
                        )}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            {/* Footer hints */}
            <div className="flex items-center justify-between px-5 py-2.5 border-t border-white/5 text-[10px] text-oracle-muted">
              <div className="flex items-center gap-4">
                <Hint label="navigate"><ArrowUp size={10} /><ArrowDown size={10} /></Hint>
                <Hint label="select"><CornerDownLeft size={10} /></Hint>
                <Hint label="close">esc</Hint>
              </div>
              <div className="flex items-center gap-1.5">
                <Command size={10} />
                <span className="font-mono">Nexus Palette</span>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
};

const Hint = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div className="flex items-center gap-1.5">
    <span className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-white/5 border border-white/10 text-white/70">
      {children}
    </span>
    <span>{label}</span>
  </div>
);

const accentToColor = (a?: CommandItem['accent']): string => {
  switch (a) {
    case 'pink':    return '#e2e8f0';
    case 'emerald': return '#10b981';
    case 'cyan':    return '#e2e8f0';
    case 'red':     return '#f87171';
    case 'accent':  return '#a4c4dc';
    default:        return '#a4c4dc';
  }
};
