// Umbrella registry — parent products that can claim the Nexus shell
// as a downstream offering. The shell renders a breadcrumb ("Oracle /
// NEXUS") when the visitor has access to one of these umbrellas, and
// stands alone ("NEXUS") when they don't.
//
// Today: empty. The standalone Nexus backend doesn't know which
// visitors are also Oracle customers (separate auth tenants); when /
// if we add per-product entitlements via a custom-claims bridge or
// shared identity provider, register the umbrella here:
//
//   {
//     id: 'oracle',
//     name: 'Oracle',
//     marketingUrl: 'https://oracle-neuro-sym.web.app',
//     icon: Zap,
//     hasAccess: (user) => Boolean(user?.products?.includes('oracle')),
//   }

import type { LucideIcon } from 'lucide-react';
import type { User } from 'firebase/auth';
import { useAuth } from './auth';
import type { Tier } from './auth';

export interface Umbrella {
  id: string;
  name: string;
  marketingUrl: string;
  icon: LucideIcon;
  hasAccess: (user: User | null, realTier: Tier | null) => boolean;
}

export const UMBRELLA_REGISTRY: Umbrella[] = [];

export const resolveActiveUmbrella = (
  user: User | null,
  realTier: Tier | null,
): Umbrella | null => {
  for (const u of UMBRELLA_REGISTRY) {
    if (u.hasAccess(user, realTier)) return u;
  }
  return null;
};

export const useActiveUmbrella = (): Umbrella | null => {
  const { user, realTier } = useAuth();
  return resolveActiveUmbrella(user, realTier);
};
