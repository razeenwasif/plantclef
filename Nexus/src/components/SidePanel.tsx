import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X, ChevronRight,
  LayoutGrid, Play, BarChart3, Cpu, Terminal,
  Network, BookOpen,
} from 'lucide-react';
import { NexusMark } from './NexusMark';

export interface NavItem {
  id: string;
  icon: any;
  label: string;
  description: string;
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

// Two groups — Operations (live state) and Reference (how it's built /
// why those choices). Reference parallels Oracle's Research + Systems
// tabs; Operations is Nexus's bread-and-butter.
export const NAV_GROUPS: NavGroup[] = [
  {
    label: 'Operations',
    items: [
      { id: 'cluster',   icon: LayoutGrid, label: 'Fleet',     description: 'GPU cluster vitals' },
      { id: 'mission',   icon: Play,       label: 'Mission',   description: 'Training jobs' },
      { id: 'analytics', icon: BarChart3,  label: 'Analytics', description: 'Signal analysis' },
      { id: 'hardware',  icon: Cpu,        label: 'Silicon',   description: '3D neural-core vitals' },
      { id: 'terminal',  icon: Terminal,   label: 'Console',   description: 'Live cluster logs' },
    ],
  },
  {
    label: 'Reference',
    items: [
      { id: 'architecture', icon: Network,  label: 'Architecture', description: 'Control plane, agents, schema' },
      { id: 'methodology',  icon: BookOpen, label: 'Methodology',  description: 'Design rationale + rollout' },
    ],
  },
];

export const findNavItem = (id: string): NavItem | null => {
  for (const g of NAV_GROUPS) for (const i of g.items) if (i.id === id) return i;
  return null;
};

interface SidePanelProps {
  open: boolean;
  onClose: () => void;
  activeTab: string;
  onSelect: (tab: string) => void;
}

export const SidePanel = ({ open, onClose, activeTab, onSelect }: SidePanelProps) => {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  return createPortal(
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={onClose}
            className="fixed inset-0 z-[180] backdrop-blur-2xl bg-black/40"
            aria-hidden
          />

          <motion.aside
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', stiffness: 380, damping: 38 }}
            className="fixed left-0 top-0 bottom-0 w-full sm:w-80 z-[190] flex flex-col bg-oracle-bg/95 border-r border-white/10 shadow-[24px_0_80px_rgba(0,0,0,0.7)]"
            role="navigation"
            aria-label="Primary"
          >
            <div className="flex items-center justify-between p-5 border-b border-white/5">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-black border border-white/15 shadow-[0_0_14px_rgba(255,255,255,0.08)] flex items-center justify-center text-white">
                  <NexusMark size={14} />
                </div>
                <div className="leading-tight">
                  <p className="text-sm font-semibold text-white italic tracking-tight">NEXUS</p>
                  <p className="text-[9px] font-semibold uppercase tracking-[0.22em] text-oracle-muted opacity-70 mt-0.5">Navigate</p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] border border-white/10 flex items-center justify-center transition-colors"
                aria-label="Close navigation"
              >
                <X size={14} className="text-white/80" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-7">
              {NAV_GROUPS.map((group) => (
                <div key={group.label}>
                  <p className="text-[9px] font-semibold uppercase tracking-[0.25em] text-oracle-muted opacity-50 mb-2.5 px-3">
                    {group.label}
                  </p>
                  <div className="space-y-0.5">
                    {group.items.map((item) => {
                      const active = activeTab === item.id;
                      return (
                        <button
                          key={item.id}
                          onClick={() => { onSelect(item.id); onClose(); }}
                          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors group text-left ${
                            active
                              ? 'bg-oracle-accent/15'
                              : 'hover:bg-white/[0.04]'
                          }`}
                        >
                          <div
                            className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors ${
                              active ? 'bg-oracle-accent/25' : 'bg-white/[0.03] group-hover:bg-white/[0.06]'
                            }`}
                          >
                            <item.icon size={14} className={active ? 'text-oracle-accent' : 'text-oracle-muted group-hover:text-white'} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className={`text-sm font-medium ${active ? 'text-white' : 'text-white/85 group-hover:text-white'}`}>
                              {item.label}
                            </p>
                            <p className="text-[10px] text-oracle-muted truncate opacity-70">
                              {item.description}
                            </p>
                          </div>
                          {active && (
                            <ChevronRight size={12} className="text-oracle-accent flex-shrink-0" />
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>

            <div className="p-3 border-t border-white/5">
              <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/[0.02] border border-white/5 text-[10px] text-oracle-muted">
                <span>Quick jump</span>
                <kbd className="font-mono text-[9px] bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-white/80">⌘K</kbd>
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>,
    document.body
  );
};
