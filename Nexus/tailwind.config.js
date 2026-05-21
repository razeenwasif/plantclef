/** @type {import('tailwindcss').Config} */
//
// Nexus palette. Single-tenant — no umbrella-product palette switch (the
// shared-codebase era had a body[data-site=...] toggle; standalone Nexus
// doesn't need it). Tokens are written as `rgb(var(--nexus-token) / <alpha-value>)`
// so Tailwind alpha modifiers (bg-nexus-accent/30) still work.
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,tsx,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        nexus: {
          bg:     "rgb(var(--nexus-bg)     / <alpha-value>)",
          panel:  "rgba(8, 8, 12, 0.55)",
          border: "rgba(255, 255, 255, 0.07)",
          accent: "rgb(var(--nexus-accent) / <alpha-value>)",
          cyan:   "rgb(var(--nexus-cyan)   / <alpha-value>)",
          pink:   "rgb(var(--nexus-pink)   / <alpha-value>)",
          text:   "rgb(var(--nexus-text)   / <alpha-value>)",
          muted:  "rgb(var(--nexus-muted)  / <alpha-value>)"
        },
        // Alias the old `oracle-*` namespace to the same tokens so files
        // copied from the shared codebase keep compiling without a name
        // rewrite. Both `text-nexus-accent` and `text-oracle-accent` work.
        oracle: {
          bg:     "rgb(var(--nexus-bg)     / <alpha-value>)",
          panel:  "rgba(8, 8, 12, 0.55)",
          border: "rgba(255, 255, 255, 0.07)",
          accent: "rgb(var(--nexus-accent) / <alpha-value>)",
          cyan:   "rgb(var(--nexus-cyan)   / <alpha-value>)",
          pink:   "rgb(var(--nexus-pink)   / <alpha-value>)",
          text:   "rgb(var(--nexus-text)   / <alpha-value>)",
          muted:  "rgb(var(--nexus-muted)  / <alpha-value>)"
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'glow-spin': 'spin 10s linear infinite',
        'float': 'floating 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        floating: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-15px)' },
        }
      }
    },
  },
  plugins: [],
}
