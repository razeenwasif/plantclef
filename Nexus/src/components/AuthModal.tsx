import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Mail, Lock, User as UserIcon, Loader2, Zap, AlertCircle } from 'lucide-react';
import { useAuth } from '../lib/auth';

type Mode = 'signin' | 'signup';

interface AuthModalProps {
  open: boolean;
  initialMode?: Mode;
  onClose: () => void;
}

export const AuthModal = ({ open, initialMode = 'signin', onClose }: AuthModalProps) => {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { signIn, signUp, signInWithGoogle } = useAuth();

  useEffect(() => {
    if (open) {
      setMode(initialMode);
      setError(null);
    }
  }, [open, initialMode]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      if (mode === 'signin') await signIn(email, password);
      else                   await signUp(email, password, displayName || undefined);
      onClose();
    } catch (err: any) {
      setError(prettyError(err?.code) || err?.message || 'Authentication failed.');
    } finally {
      setSubmitting(false);
    }
  };

  const onGoogle = async () => {
    setError(null);
    setSubmitting(true);
    try {
      await signInWithGoogle();
      onClose();
    } catch (err: any) {
      setError(prettyError(err?.code) || err?.message || 'Google sign-in failed.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          className="fixed inset-0 z-[200] flex items-center justify-center p-4 bg-black/80 backdrop-blur-xl"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}
            onClick={(e) => e.stopPropagation()}
            className="glass-panel w-full max-w-md rounded-3xl p-8 relative overflow-hidden"
          >
            {/* Glow accent */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-oracle-accent/10 rounded-full blur-[100px] -translate-y-32 translate-x-32 pointer-events-none" />

            {/* Close */}
            <button
              onClick={onClose}
              className="absolute top-4 right-4 w-9 h-9 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 flex items-center justify-center transition-colors"
            >
              <X size={16} className="text-white/80" />
            </button>

            {/* Brand mark */}
            <div className="flex items-center gap-3 mb-8 relative z-10">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-oracle-accent to-oracle-pink p-px shadow-[0_0_25px_rgba(124,58,237,0.5)]">
                <div className="w-full h-full bg-oracle-bg rounded-[11px] flex items-center justify-center">
                  <Zap size={16} className="text-white" fill="currentColor" />
                </div>
              </div>
              <div>
                <h2 className="text-lg font-bold text-white tracking-tight italic uppercase">
                  Oracle<span className="font-extralight text-oracle-muted not-italic"> Access</span>
                </h2>
                <p className="text-[9px] text-oracle-accent font-black tracking-[0.3em] uppercase opacity-60">
                  {mode === 'signin' ? 'Welcome Back' : 'Create Account'}
                </p>
              </div>
            </div>

            {/* Mode toggle */}
            <div className="flex p-1 rounded-xl bg-black/40 border border-white/5 mb-6 relative z-10">
              {(['signin', 'signup'] as Mode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => { setMode(m); setError(null); }}
                  className={`flex-1 px-4 py-2 rounded-lg text-[10px] font-black uppercase tracking-[0.2em] transition-all ${
                    mode === m ? 'bg-oracle-accent/30 text-white shadow-[0_0_15px_rgba(124,58,237,0.25)]' : 'text-oracle-muted hover:text-white'
                  }`}
                >
                  {m === 'signin' ? 'Sign In' : 'Sign Up'}
                </button>
              ))}
            </div>

            {/* Form */}
            <form onSubmit={onSubmit} className="space-y-4 relative z-10">
              {mode === 'signup' && (
                <Field icon={UserIcon} label="Display Name" type="text" value={displayName} onChange={setDisplayName} placeholder="optional" />
              )}
              <Field icon={Mail} label="Email" type="email" value={email} onChange={setEmail} placeholder="you@field.org" required />
              <Field icon={Lock} label="Password" type="password" value={password} onChange={setPassword} placeholder="••••••••" required minLength={6} />

              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="flex items-start gap-2 px-3 py-2.5 rounded-xl bg-red-500/10 border border-red-500/30 text-[11px] text-red-300"
                  >
                    <AlertCircle size={14} className="flex-shrink-0 mt-0.5" />
                    <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              <button
                type="submit"
                disabled={submitting}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-oracle-accent to-oracle-pink text-white text-[11px] font-black uppercase tracking-[0.2em] shadow-[0_0_25px_rgba(124,58,237,0.4)] hover:shadow-[0_0_35px_rgba(124,58,237,0.6)] disabled:opacity-50 disabled:cursor-not-allowed transition-shadow"
              >
                {submitting && <Loader2 size={14} className="animate-spin" />}
                {mode === 'signin' ? 'Authenticate' : 'Create Account'}
              </button>
            </form>

            {/* Divider */}
            <div className="flex items-center gap-4 my-6 relative z-10">
              <div className="flex-1 h-px bg-white/10" />
              <span className="text-[9px] font-black text-oracle-muted uppercase tracking-[0.3em] opacity-60">or</span>
              <div className="flex-1 h-px bg-white/10" />
            </div>

            {/* Google */}
            <button
              onClick={onGoogle}
              disabled={submitting}
              className="w-full flex items-center justify-center gap-3 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-[11px] font-black uppercase tracking-[0.2em] text-white transition-colors disabled:opacity-50 relative z-10"
            >
              <GoogleGlyph />
              Continue with Google
            </button>

            <p className="text-[10px] text-oracle-muted text-center mt-6 opacity-60 relative z-10">
              {mode === 'signin' ? "Don't have an account? " : 'Already have one? '}
              <button
                type="button"
                onClick={() => { setMode(mode === 'signin' ? 'signup' : 'signin'); setError(null); }}
                className="text-oracle-accent hover:text-oracle-pink font-bold uppercase tracking-widest"
              >
                {mode === 'signin' ? 'Sign Up' : 'Sign In'}
              </button>
            </p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

const Field = ({ icon: Icon, label, type, value, onChange, placeholder, required, minLength }: any) => (
  <div>
    <label className="text-[9px] font-black text-oracle-muted uppercase tracking-[0.2em] opacity-60 mb-2 block">
      {label}
    </label>
    <div className="relative">
      <Icon size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-oracle-muted" />
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        required={required}
        minLength={minLength}
        className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white/[0.03] border border-white/10 text-[12px] text-white placeholder:text-oracle-muted/50 focus:outline-none focus:border-oracle-accent/50 focus:bg-white/[0.06] focus:shadow-[0_0_20px_rgba(124,58,237,0.15)] transition-all"
      />
    </div>
  </div>
);

const GoogleGlyph = () => (
  <svg width="16" height="16" viewBox="0 0 48 48">
    <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3c-1.6 4.7-6 8-11.3 8-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.9 1.2 8 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5z"/>
    <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.6 16 18.9 13 24 13c3.1 0 5.9 1.2 8 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z"/>
    <path fill="#4CAF50" d="M24 44c5.2 0 9.8-2 13.3-5.2l-6.1-5.2C29.1 35.2 26.7 36 24 36c-5.2 0-9.7-3.3-11.3-8l-6.5 5C9.5 39.6 16.2 44 24 44z"/>
    <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.4-2.3 4.4-4.2 5.7l6.1 5.2C40.8 35.7 44 30.3 44 24c0-1.3-.1-2.4-.4-3.5z"/>
  </svg>
);

const prettyError = (code?: string): string | null => {
  if (!code) return null;
  const map: Record<string, string> = {
    'auth/email-already-in-use': 'That email is already registered. Try signing in.',
    'auth/invalid-email':        'That email address looks malformed.',
    'auth/weak-password':        'Password too weak — use at least 6 characters.',
    'auth/invalid-credential':   'Invalid email or password.',
    'auth/user-not-found':       'No account matches that email.',
    'auth/wrong-password':       'Incorrect password.',
    'auth/too-many-requests':    'Too many attempts — try again later.',
    'auth/popup-closed-by-user': 'Sign-in was cancelled.',
    'auth/operation-not-allowed':'This sign-in method is disabled in Firebase Console.',
  };
  return map[code] || null;
};
