import { createContext, useContext, useEffect, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import {
  onAuthStateChanged,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
  signOut as fbSignOut,
  updateProfile,
} from 'firebase/auth';
import type { User } from 'firebase/auth';
import { doc, onSnapshot, setDoc, updateDoc, serverTimestamp, getDoc } from 'firebase/firestore';
import { auth, googleProvider, db } from './firebase';

export type Tier = 'free' | 'pro' | 'field' | 'admin';

const ADMIN_EMAILS = ['razeen.wasif66@gmail.com'];

interface AuthState {
  user: User | null;
  loading: boolean;
  realTier: Tier;
  effectiveTier: Tier;
  isAdmin: boolean;
  viewingAs: Tier | null;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, displayName?: string) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
  viewAsTier: (tier: Tier | null) => void;
  upgradeTier: (tier: Exclude<Tier, 'admin'>) => Promise<void>;
  cancelSubscription: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

const viewAsStorageKey = 'oracle.viewAs';

const isAdminEmail = (email: string | null | undefined): boolean =>
  !!email && ADMIN_EMAILS.includes(email.toLowerCase());

/**
 * Ensure a users/{uid} doc exists. Self-create runs through the strict
 * `tier == 'free'` rule; the admin promotion happens as a separate write
 * with a custom code path that the rules allow because the email matches
 * `ADMIN_EMAILS` (rules check `request.auth.token.email`).
 */
const ensureUserDoc = async (user: User): Promise<void> => {
  const ref = doc(db, 'users', user.uid);
  const snap = await getDoc(ref);
  if (!snap.exists()) {
    await setDoc(ref, {
      email:       user.email ?? '',
      displayName: user.displayName ?? '',
      tier:        'free',
      createdAt:   serverTimestamp(),
      updatedAt:   serverTimestamp(),
    });
  }
  // Admin grant: if email is in the allow-list and current tier isn't admin,
  // bump to admin. Allowed by rules because the caller is an admin email.
  if (isAdminEmail(user.email)) {
    const current = (await getDoc(ref)).data();
    if (current?.tier !== 'admin') {
      await updateDoc(ref, { tier: 'admin', updatedAt: serverTimestamp() });
    }
  }
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [realTier, setRealTier] = useState<Tier>('free');
  const [viewingAs, setViewingAs] = useState<Tier | null>(() => {
    const stored = localStorage.getItem(viewAsStorageKey);
    return (stored === 'free' || stored === 'pro' || stored === 'field' || stored === 'admin') ? stored : null;
  });

  // Track auth state
  useEffect(() => {
    const unsub = onAuthStateChanged(auth, async (u) => {
      setUser(u);
      if (!u) {
        setRealTier('free');
        setLoading(false);
        return;
      }
      try {
        await ensureUserDoc(u);
      } catch (err) {
        console.warn('ensureUserDoc failed', err);
      }
      // Subscribe to the user doc; setLoading(false) after first snapshot
      const unsubDoc = onSnapshot(
        doc(db, 'users', u.uid),
        (snap) => {
          const data = snap.data();
          const t = (data?.tier ?? 'free') as Tier;
          setRealTier(t);
          setLoading(false);
        },
        (err) => {
          console.warn('users doc snapshot error', err);
          setLoading(false);
        }
      );
      // Stash cleanup on the user object so we tear down per session
      (u as any).__oracleUnsubDoc = unsubDoc;
    });
    return () => unsub();
  }, []);

  // Tear down per-user doc subscription when user changes
  useEffect(() => {
    return () => {
      const u = user as any;
      if (u && typeof u.__oracleUnsubDoc === 'function') {
        u.__oracleUnsubDoc();
      }
    };
  }, [user]);

  const isAdmin = realTier === 'admin';
  const effectiveTier = isAdmin && viewingAs ? viewingAs : realTier;

  const signIn = useCallback(async (email: string, password: string) => {
    await signInWithEmailAndPassword(auth, email, password);
  }, []);

  const signUp = useCallback(async (email: string, password: string, displayName?: string) => {
    const cred = await createUserWithEmailAndPassword(auth, email, password);
    if (displayName) await updateProfile(cred.user, { displayName });
  }, []);

  const signInWithGoogle = useCallback(async () => {
    await signInWithPopup(auth, googleProvider);
  }, []);

  const signOut = useCallback(async () => {
    localStorage.removeItem(viewAsStorageKey);
    setViewingAs(null);
    await fbSignOut(auth);
  }, []);

  const viewAsTier = useCallback((tier: Tier | null) => {
    if (!isAdmin) return;
    setViewingAs(tier);
    if (tier) localStorage.setItem(viewAsStorageKey, tier);
    else      localStorage.removeItem(viewAsStorageKey);
  }, [isAdmin]);

  const upgradeTier = useCallback(async (tier: Exclude<Tier, 'admin'>) => {
    if (!user || realTier === 'admin') return;
    try {
      await updateDoc(doc(db, 'users', user.uid), {
        tier,
        updatedAt: serverTimestamp(),
      });
      // realTier will refresh via onSnapshot
    } catch (err) {
      console.warn('upgradeTier failed', err);
    }
  }, [user, realTier]);

  /**
   * Cancel an active Pro / Field subscription. Drops the user back to
   * 'free' immediately (no end-of-period semantics — there is no billing
   * backend yet). Also records `previous_tier` and `cancelled_at` for the
   * audit trail; both are written self-attributed and accepted by the
   * existing /users/{uid} self-update rule because `tier='free'` satisfies
   * isAllowedTier() and `createdAt` is left untouched.
   *
   * No-ops if the user is already on free or is an admin.
   */
  const cancelSubscription = useCallback(async () => {
    if (!user) return;
    if (realTier === 'admin' || realTier === 'free') return;
    try {
      await updateDoc(doc(db, 'users', user.uid), {
        tier: 'free',
        previous_tier: realTier,
        cancelled_at: serverTimestamp(),
        updatedAt: serverTimestamp(),
      });
      // realTier refreshes via onSnapshot
    } catch (err) {
      console.warn('cancelSubscription failed', err);
      throw err;   // let the caller surface the rules error
    }
  }, [user, realTier]);

  const value: AuthState = {
    user, loading, realTier, effectiveTier, isAdmin, viewingAs,
    signIn, signUp, signInWithGoogle, signOut, viewAsTier, upgradeTier, cancelSubscription,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthState => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
};

// Tier capability map — single source of truth for what each tier can do
export const tierCapabilities = (tier: Tier) => ({
  identifyDailyLimit: tier === 'free' ? 5 : Infinity,
  canSaveToJournal:   tier !== 'free',
  canGenerateReports: tier === 'pro' || tier === 'field' || tier === 'admin',
  canIssueApiKeys:    tier === 'pro' || tier === 'admin',
  rareSpeciesAlerts:  tier !== 'free',
  offlineField:       tier === 'field' || tier === 'admin',
});

export const tierMeta: Record<Tier, { label: string; color: string; bg: string; description: string }> = {
  free:  { label: 'Free',  color: '#94a3b8', bg: 'rgba(148,163,184,0.15)', description: '5 identifications per day' },
  pro:   { label: 'Pro',   color: '#7c3aed', bg: 'rgba(124,58,237,0.15)',  description: 'Unlimited IDs + rare-species alerts' },
  field: { label: 'Field', color: '#10b981', bg: 'rgba(16,185,129,0.15)',  description: 'Pro + offline mobile bundle' },
  admin: { label: 'Admin', color: '#ec4899', bg: 'rgba(236,72,153,0.15)',  description: 'Full platform access · tier QA' },
};
