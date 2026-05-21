# Multi-Oracle Architecture

How the codebase scales when Oracle expands from Plant to sister
domains (Bird, Sea, Land). Written down so we don't re-derive this
when Oracle #2 lands.

## 1. Current state — single repo, hostname-routed

One repo, one dashboard. Sister Oracles are served from the same
React app, distinguished by hostname. The decision is already
encoded in the codebase:

- `dashboard/src/App.tsx` — hostname-based router selects which
  Oracle "site" renders (work from the cluster Hosting target
  scaffold).
- `firebase.json` + `.firebaserc` — multi-target Hosting scaffold
  (`oracle-neuro-sym` + `cluster` today; future sister Oracles
  add more targets).
- `dashboard/src/index.css` — CSS variables drive the palette
  (`--oracle-bg`, `--oracle-accent`, etc.). Each sister Oracle
  overrides just the palette; everything else (Newsreader + Inter
  type, Tailwind utility classes, component chrome) is shared.
- `lib/auth.ts`, `lib/journal.ts`, Firestore rules, the training
  plane (`src/training/`, `src/cluster/`), pod agent + telemetry
  — single source of truth, all Oracles use the same primitives.

## 2. Why monorepo now

Sister Oracles reuse **~80%** of the surface:

- Identical chrome (header, navigation, command palette,
  AccountMenu, AuthPage, SidePanel).
- Identical auth + tier system (Firebase Auth + tier metadata).
- Identical Firestore schemas (`/users/{uid}/journal`, `/pods`,
  `/runs`, `/run_requests`).
- Identical training plane (CUDA + TPU launcher, telemetry,
  liveness, cluster manifest, control plane).
- Identical neuro-symbolic reasoning chassis (AC-3 constraint
  propagator — only the priors are domain-specific).

Duplicating that across repos means inevitable drift: one Oracle
gets a Firestore schema migration, the others don't; one fixes a
training-loop bug, the others still hit it. Single-repo keeps the
platform coherent and lets a fix land everywhere at once.

What changes per Oracle today is mostly **content + priors**, not
infrastructure:

- Domain encoder (BioCLIP for plants → a bird-specific encoder for
  Oracle Bird).
- Taxonomy + phenology priors fed into AC-3.
- Identification copy + UI labels.
- Demo data (`SpeciesDiscovery.tsx` feed, `Atlas` reference pins).

These are configurable, not structural.

## 3. When to split into workspaces

Single repo is the right call **until one of these triggers fires**:

1. **Dependency conflict.** Two Oracles need incompatible major
   versions of a shared dep — e.g. Oracle Bird is on React 20,
   Oracle Plant is still on React 19, and the migration can't be
   coordinated. Workspaces resolve this cleanly.
2. **Training-stack divergence > 50%.** If Oracle Sea ends up
   needing a fundamentally different training loop (e.g. video +
   audio fusion instead of image identification), packaging a
   shared `core` becomes natural.
3. **Team ownership boundary.** When a different team owns Oracle
   Bird end-to-end and shared-repo PR review becomes friction.
4. **Build-time blowup.** When one dashboard's CI takes >10
   minutes because it builds every Oracle's assets even on a
   plant-only change. (Today: ~1.5s. Plenty of headroom.)

Do **not** split for aesthetic reasons ("it'd be cleaner") if none
of these triggers have fired. The cost of premature workspace
splits is real: cross-package version coordination, package
publishing logistics, harder global refactors.

## 4. How to split when the trigger fires

Bun workspaces are already in use (`dashboard/bun.lock`), so the
migration is mechanical:

```
Oracle/
├── apps/
│   ├── oracle-plant/      ← today's dashboard/
│   ├── oracle-bird/       ← new
│   └── oracle-sea/        ← new
├── packages/
│   ├── core/              ← chrome, auth, journal, palette,
│   │                        AccountMenu, command palette, etc.
│   ├── training/          ← what's in src/training today
│   ├── cluster/           ← what's in src/cluster today
│   └── theme/             ← CSS variable definitions per Oracle
├── package.json           ← root workspace manifest
└── firebase.json          ← still multi-target
```

Root `package.json` declares:

```json
{
  "workspaces": ["apps/*", "packages/*"]
}
```

Each `apps/oracle-{domain}/` is a thin shell that imports
`@oracle/core` for chrome, overrides the palette via
`@oracle/theme`, and supplies its own domain logic (encoder,
priors, demo data).

### What stays shared vs splits

| Concern                       | Shared (`packages/`) | Per-Oracle (`apps/`) |
|-------------------------------|----------------------|----------------------|
| Header, nav, command palette  | ✓                    |                      |
| AccountMenu, AuthPage         | ✓                    |                      |
| Firestore schemas + rules     | ✓                    |                      |
| Auth + tier system            | ✓                    |                      |
| Training plane (CUDA/TPU)     | ✓                    |                      |
| AC-3 propagator (chassis)     | ✓                    |                      |
| Telemetry + control plane     | ✓                    |                      |
| Domain encoder                |                      | ✓                    |
| Taxonomy + phenology priors   |                      | ✓                    |
| Identification copy / labels  |                      | ✓                    |
| Demo data (Discover / Atlas)  |                      | ✓                    |
| Palette tokens                |                      | ✓ (via `@oracle/theme`) |

### Migration order (when the day comes)

1. Carve `packages/theme/` out first — palette is the cleanest
   seam and validates the workspace plumbing.
2. Then `packages/core/` (the chrome + auth + journal layer).
3. `packages/training/` + `packages/cluster/` last — they're the
   biggest extraction and depend least on the workspace topology.
4. Rename `dashboard/` → `apps/oracle-plant/`. CI deploy target
   becomes `apps/oracle-plant/dist`.

Don't try to do all of this in one PR. Each step above is its own
landing.

## 5. Decision log

- **2026-05** — chose monorepo + hostname routing during the
  field-guide reimagine. Cluster Hosting target scaffolded as the
  proof point. Workspace split deferred until a trigger from §3
  fires.
