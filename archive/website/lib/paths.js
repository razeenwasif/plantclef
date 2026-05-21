// Build a path that respects Next's basePath when the site is hosted under a
// subpath (GitHub Pages: /PlantCLEF2026). `next/link` and `next/image` already
// handle this, but plain `<img src>` and external links do not.

export const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

export function asset(p) {
  if (!p) return p;
  if (/^(https?:|data:|mailto:)/.test(p)) return p;
  return basePath + (p.startsWith("/") ? p : "/" + p);
}
