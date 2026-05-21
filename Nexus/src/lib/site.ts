// Nexus — standalone deployment, single site, single Hosting target.
// The shared-codebase era's hostname-based SITE detection has been
// stripped; everything here is hardcoded to the Nexus shell.
//
// THEME_COLORS lives here so components that need bare hex strings
// (Recharts stroke props, three.js light + material colours, GlowBar
// inline-style color) can pull from a single source of truth without
// duplicating the palette across files.

export const SITE_HOSTNAME: string =
  typeof window !== 'undefined' ? window.location.hostname : 'nexus-cluster.web.app';

export const SITE_TAGLINE = 'Distributed Training Console';

// Theme colours for places Tailwind classes can't reach.
export interface ThemeColors {
  chartPrimary: string;
  chartSeries: string[];
  accent: string;
  vramBar: string;
  hbmBar: string;
  saturationBar: string;
  vitalsBall: { training: string; idle: string };
  vitalsSpot: string;
}

export const THEME_COLORS: ThemeColors = {
  chartPrimary: '#cfd8e6',
  chartSeries: ['#cfd8e6', '#a4c4dc', '#bee3eb', '#94a3b8', '#cbd5e1', '#64748b'],
  accent: '#a4c4dc',
  vramBar: '#a4c4dc',
  hbmBar: '#cfd8e6',
  saturationBar: '#94a3b8',
  vitalsBall: { training: '#a4c4dc', idle: '#cfd8e6' },
  vitalsSpot: '#cfd8e6',
};
