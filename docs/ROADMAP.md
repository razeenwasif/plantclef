# Roadmap

Working roadmap for the Oracle identification platform. Updated as
initiatives ship; reflects the state of `main`.

> **Nexus has moved.** The cluster-monitoring console that lived in this
> repo's `dashboard/` (under `SITE === 'nexus'` gating) was extracted to
> its own repo + Firebase project. See `~/Nexus` and
> https://nexus-cluster.web.app. This document only tracks Oracle now.

## Currently active

Nothing big. Background polish:

- Pod-agent service-account credentials need to be re-minted against the
  `nexus-cluster` Firebase project so existing training hosts re-register
  there instead of `oracle-neuro-sym`. Tracked in the Nexus repo's
  roadmap; mentioned here only as the cleanup tail of the Nexus
  extraction.

## Reimagine: from ops-dashboard skin to consumer product

The Nexus extraction left Oracle running on the same React shell as
Nexus — same header / left sidebar / glass-panel grammar / status-pill
language. The only differentiation is colour (Oracle violet / cyan /
pink vs. Nexus monochrome + ice-blue) and background (cyber-grid +
bg-orbs vs. LightTrails). At a *layout* and *interaction* level the two
products read as twins. That's wrong for Oracle: it's a consumer
identification product, not an ops tool, and the current shell
foregrounds its weakest dimension (parity with an ops console) instead
of its strongest (the identification action).

This entry is the design + IA plan for a fresh Oracle. **Nexus stays as
is** — ops engineers want familiar dashboard patterns, so the
Blade-Runner-flavoured monochrome aesthetic is the right home there.
Oracle is the one that needs to look like a *product*, not a console.

### Visual direction: modern field guide

Drop the Blade Runner / cyberpunk vocabulary entirely. Move toward
**premium, modern, sleek** — the kind of restraint Linear, Stripe,
Anthropic, and Vercel use. With Oracle's subject matter (species
identification) there's a specific framing that no ML competitor has
taken: a **modern field guide**. Linnaeus-meets-Stripe — the visual
heritage of a naturalist's notebook executed with current product-
design discipline.

Concretely:

| Lever | Today | Reimagine |
|---|---|---|
| **Surface** | Near-black `#05050f` + heavy glass panels with 64 px backdrop-blur | Warm paper `#FAF7F2` (or warm dark `#1A1816` if light feels too clinical), hairline 1 px borders at ~8 % opacity, no glass, no glow |
| **Palette** | Violet `#7c3aed` + cyan `#06b6d4` + pink `#ec4899` triad with neon shadows | One restrained botanical accent (deep green `#3A5A3F` or sepia `#8B6F47`); semantic colours stay (emerald = live, red = error) |
| **Typography** | `Inter` everywhere, italic uppercase tracking-widest for headings | Display serif for headings (Newsreader / GT Sectra), neutral sans for body (Inter / Geist). Sentence case, calm hierarchy. |
| **Motion** | Animated stroke-dashoffset edges, pulsing dots, `MeshDistortMaterial` blob | Slow, restrained micro-interactions only; no continuous animation in the chrome |
| **Imagery** | Abstract cyber-grid + bg-orb gradients | Real organism photography — high-res hero shots, used as a marquee element rather than wallpaper |
| **Spacing** | Dense, dashboard-tight (~16 px section padding) | Generous, editorial (~64 px section padding) |

The hero on the Identify tab should feel like turning the page of a
beautifully laid-out reference book, not opening a control panel.

### IA changes

Today Oracle has nine peer tabs (Identify, Atlas, Journal, Discover,
Research, Systems, Trust, Reports, API). Visually they're all equal —
which makes Identify, the *whole point of the product*, feel like one
admin surface among many.

Proposed restructure:

- **Identify becomes the entire landing surface.** Not a tab — the
  product. Drop-zone hero, big result card, recent identifications
  scrolling beneath. No left-sidebar dominating the layout.
- **Atlas, Journal, Discover become contextual lenses on user data,**
  not equal-weight tabs. Pattern: a single small top nav with
  "Identify · Library · Globe · Discoveries" links, or even a
  command-bar at the top that swaps view contexts in place.
- **Research, Systems, Trust, API move behind an "About this engine"
  link in a corner.** Keeps them accessible for recruiters and curious
  engineers — the "open Oracle, see the working note" instinct
  survives — without dominating the consumer flow.
- **The header strip simplifies.** Drop the ⌘K search bar, the AEST
  clock, the Account menu, the gradient brand chip. Replace with a
  thin top bar: serif wordmark on the left, a single "Sign in"
  affordance on the right.
- **The Reports tab is genuinely demo-quality today and should be
  honestly labelled.** Either elevate it to working ecological-report
  generation backed by a real LLM, or quietly retire it. Don't keep
  half-finished features in the consumer surface.

### Concrete component changes

These are what changes inside the React tree, ranked from "trivial,
huge perceptual win" to "structural":

1. **Palette + typography swap.** Replace the
   `tailwind.config.js` `oracle` tokens and load the display serif via
   `@font-face`. Half a day; the existing CSS-variable indirection
   means every component remaps automatically.
2. **Drop cyber-grid + bg-orb + glass-panel.** Move to flat warm
   surface with hairline borders. ~half a day across the shared
   component set.
3. **Rewrite the header / sidebar / footer chrome.** The Linear /
   Vercel pattern: minimal top bar, no permanent left sidebar (panel
   slides in on demand from a single hamburger / `K` action).
   ~1 day.
4. **Re-author the Identify hero.** Drop-zone with a generous frame,
   one large result card with high-res organism imagery,
   subtle slide-in animation. ~1 day.
5. **Editorialise Research + Systems.** Today they're long-form
   technical pages styled like a dashboard tab. Re-typeset as
   editorial: paper-style line-length, prominent figure captions,
   pull-quotes, table-of-contents on the side. Doesn't change content,
   only typography + layout. ~1-2 days each.
6. **Reduce or retire Reports.** Honest re-scope — either commit to
   real Gemma-class LLM integration (multi-day effort) or remove the
   tab. No middle ground.

### Tradeoffs

- **Visual continuity across the portfolio breaks.** Today Oracle +
  Nexus feel like sibling products (same shell); afterwards they read
  as different categories of thing (consumer product vs. ops tool).
  Probably the right outcome — they *are* different categories. But
  if you ever want a future product to share visual DNA with Oracle,
  it now has to inherit the field-guide language rather than the
  dashboard one.
- **The "open Oracle, see the science" instinct weakens.** Recruiters
  currently land on Oracle and within two clicks are looking at
  ablation tables. After the reimagine, the science lives behind one
  more click (the "About this engine" link). Mitigation: a single
  small "Read the working note · PlantCLEF 2026" CTA in the Identify
  hero's footer preserves the recruiter path explicitly.
- **The Research / Systems editorialisation is unglamorous work.**
  Re-typesetting 800 lines of existing prose isn't visually exciting,
  but it's where the perceived "premium" comes from. Skipping it
  leaves the marketing surface looking great and the dive-deep
  surfaces looking unchanged — incongruous.

### Phased rollout

Each phase ships independently; the visual change is most jarring
between phase 1 and 2, so consider doing them together in a single
release if you have the bandwidth.

1. **Phase 1 · Palette + typography + chrome.** Tailwind tokens swap,
   serif headings, warm paper surface, drop cyber-grid + glass.
   Visually unmistakable change. **~1-2 days.**
2. **Phase 2 · IA collapse.** Identify becomes the landing surface,
   Atlas/Journal/Discover demote to contextual lenses, About link
   hides the engineering pages. **~1-2 days.**
3. **Phase 3 · Hero + identification ceremony.** Drop-zone redesign,
   result card, real organism imagery, micro-animations.
   **~1-2 days.**
4. **Phase 4 · Editorial Research + Systems.** Re-typeset the deep
   pages without changing content. **~2-3 days.**
5. **Phase 5 · Reports decision.** Either build it for real or remove
   it. **Either ~0 days or several.**

**Total ~5-10 days** depending on Reports decision and how much
editorial polish goes into phase 4. Each phase leaves Oracle in a
shippable state, so you can stop after phase 2 if it's already
landing the way you want.

### What this initiative is NOT

- **A Nexus reimagine.** Nexus is genre-appropriate as-is.
- **An IA simplification of Nexus.** The Operations + Reference group
  structure is right for an ops console.
- **A logo / brand-mark redesign.** The Oracle wordmark and the
  identity story stay; what changes is the typography setting and the
  surrounding visual world.

## Recently shipped

### Paper (PlantCLEF 2026 working note)

- Restructured: `report/` consolidated into `archive/report/`
- 10 TikZ figures: anchor pipeline, phenology pipeline, train + quadrat
  image grids, test-set day-of-year calendar wheel, aggregate deltas,
  unfreeze sweep, species long-tail, val-vs-kaggle, phenology
  illustration
- Ablation table inserted into `main.tex`
- Headline `0.418265` Macro F1 normalised across the body, captions, and
  ablation context
- 4 unified Manim scenes (pipeline overview with zoom-levels + overlap
  beats + mixed-resolution ensemble, plus the phenology deep dive)
- Blender hero shot scaffold (`paper/animations/blender/hero_quadrat.py`)

### Dashboard

- Pipeline-snake hand-rolled SVG diagram (replaced Mermaid, which
  couldn't keep the layout snaked under dagre or ELK)
- PostProcessLab — three interactive demos (logit adjustment, sigmoid
  threshold, phenology) for the Research tab
- Research tab framing — PlantCLEF 2026 case study header, honest
  abstract, ablation table embedded
- Universal-identifier rebrand: de-plant-specified copy in side-nav
  descriptions, Research framed as a case study rather than the
  product's identity
- `no-cache` HTML + immutable hashed-asset cache headers so new deploys
  reach browsers immediately
- Console-tab scroll bug fixed (`min-h-0` on the log container + direct
  `scrollTop` instead of `scrollIntoView`)

### Cluster split (now reverted)

The cluster split (single React build, two Firebase Hosting targets:
`oracle-neuro-sym.web.app` + `oraclenexus.web.app`) shipped, the Nexus
visual decoupling shipped on top of it, and then the Nexus shell was
extracted into its own repo. The `oraclenexus.web.app` Hosting target,
the Nexus-only components, and the cluster-monitoring Python scripts are
no longer in this repo. See `~/Nexus` for the standalone version.

### Portfolio

- Oracle entry rewritten honestly: BioCLIP 2.5 ViT-H/14 partial fine-
  tune, tile-ensemble inference, `0.418` Macro F1
- Nexus added as a standalone entry pointing at `nexus-cluster.web.app`
- Prism gained the `· AutoML Workbench` suffix
- Tagline + About broadened for junior / grad reads — dropped CUDA-
  kernel and neuro-symbolic specifics

## Future / parking lot

- **Active site monitoring.** No alerting on Firebase Hosting / Firestore
  yet. If usage grows, consider a free-tier uptime check.
- **Per-tier model gating.** Pro tier UI exists but the actual model
  swap (different inference recipe per tier) is mocked.
- **Reports tab evolution.** Currently demo-quality. Real ecological-
  report generation would need a Gemma-class LLM wired up.
- **The two distributed-training stretch tasks** (checkpoint-resume
  idempotency, leader-elected seed-sweep coordinator) moved with Nexus.
