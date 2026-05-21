# ORACLE — Product Evolution Roadmap

> From training-monitor → marketable identification platform.
> The neon Command Center stays as the "Pro/Dev" face. New public-facing pages live alongside it.

## Strategic framing

ORACLE has two audiences hiding inside one codebase:

1. **Operators / researchers** — Fleet, Mission, Analytics, Silicon, Console, Research tabs. The backstage.
2. **End users / customers** — people who want to identify a plant, log a discovery, hit an API, read a generated field report, or trust the model. The frontstage.

The current dashboard speaks only to (1). We keep all existing tabs and refine later. Every new pillar below ships as a **new top-level tab/page**, so the operator surface is undisturbed.

---

## Pillars (build order)

### 1. Identify — the upload flow `[SHIPPED ✓]`
The single most monetizable feature. Live on `feature/product-evolution`.

- Drag-and-drop or file-picker for a photo.
- Mock inference pipeline animation walking through Encode → BioCLIP → DINOv2 → ConvNeXt-V2 → SWA Calibrate.
- Top-5 species with circular confidence rings, expandable to citation chips (GBIF / Wikipedia / iNaturalist).
- Grad-CAM-style attribution heatmap toggle on the uploaded image.
- Daily-quota chip ("3 / 5 free today") + "Upgrade to Pro" CTA + exhausted-quota banner.
- "Save to Journal" stub wired for Pillar #3.

**Tier model:** Free 5/day · Pro unlimited + rare-species alerts + GPS auto-tag · Field (offline mobile).
**Follow-up work:** swap `MOCK_RESULTS` for a real `POST /api/identify` call; wire heatmap to real Grad-CAM output from the backbone.

---

### 2. Atlas — interactive phytogeographic globe + map `[SHIPPED ✓]`
Two views, one tab. Designed to be the landing-page hero and an analyst's working surface in the same component.

**Globe view** (marketing hero):
- Rotating 3D earth (three.js, already in deps) with real-time pins for recent identifications.
- Colour-coded by domain (Flora / Avian / Marine / Terrestrial / Urban).
- Pulse animation on each new ID, click-through to species detail.

**Map view** (analyst):
- 2D phytogeographic heatmap using **MapLibre GL** (free; same renderer quality as Mapbox, no per-load billing).
- Custom SVG markers for rare-tier species; size-by-confidence, colour-by-domain.
- Density heatmap layer toggle.
- Time-range filter: last 24h / 7d / all-time.
- "Trending taxa" sidebar.

Toggle between Globe ↔ Map in the header strip. Shared state (filters, time range).

---

### 3. Auth & Tier Management — Firebase identity + role-based access `[SHIPPED ✓]`
Foundation layer that gates Journal, Reports, and the Devs console. Without this, the freemium / Pro model is theatre.

**Pages:**
- `/login` — email/password + Google OAuth (Firebase Auth providers).
- `/signup` — same, with tier defaulting to `free`.
- `/account` — profile (avatar, display name, tier badge), billing portal hook, API-key visibility (for Pro+), sign-out.

**Tiers:** `free` (default) · `pro` (paid) · `field` (offline-mobile bundle, paid) · `admin` (internal).

**Admin privileges (for testing all tiers):**
- Seed user **razeen.wasif66@gmail.com** with `admin` role on first sign-in (one-time Firestore bootstrap or Cloud Function trigger).
- Header chip visible only to admins: **"View as → Free / Pro / Field / Admin"** toggle. Instantly impersonates the chosen tier client-side so every quota, paywall, and Pro-only feature can be QA'd without logging out. Server still trusts the real claim, so it's strictly a UI preview.
- Admin-only `/admin` panel (later): grant/revoke tiers, view recent signups, override quota for support cases.

**Gating model:**
- Public (no login required): Fleet, Identify (5/day quota by IP fingerprint), Atlas, Discover, Mission, Analytics, Research, Trust, Silicon, Console.
- Login required: Journal (Pillar 4), Reports (Pillar 7), Devs API keys (Pillar 5).
- Free-tier quota tracking moves from client state to Firestore so 5/day actually means 5/day across devices.

**Engineering:**
- `firebase/auth` + `firebase/firestore` SDKs (web).
- React context: `AuthProvider` exposing `{ user, tier, effectiveTier, signIn, signOut, viewAsTier }` (admin-only `viewAsTier` override).
- Firestore custom claims for fast tier reads on protected operations.
- Firestore security rules enforcing tier at the data layer (defense in depth).
- Login state survives reloads via Firebase's built-in persistence.

---

### 4. Journal — the personal herbarium / field log `[SHIPPED ✓]`
Pokémon-Go-for-plants but for adults. The retention engine.

- Every authenticated ID becomes a stamp in the user's collection.
- Streaks ("identified 7 days running"), rarity score, completion progress per biome.
- Exportable PDF field reports *(simple per-user; B2B reports live in Pillar #6).*
- Shareable cards: every Discovery card gets a one-tap export → 1080×1080 image with photo + species + confidence + ORACLE watermark. Free distribution.
- Optional public profile / leaderboard.

---

### 5. Developer Console — the public API tier `[SHIPPED ✓]`
Make the "1.2M API requests/day" sidebar stat real.

- API key issuance and management.
- Live usage graph + Stripe-metered billing.
- `POST /identify` endpoint docs with cURL / Python / JS examples.
- Webhooks for async batches.
- Target customers: conservation orgs, agritech, camera-trap projects, citizen-science platforms.

---

### 6. Why Trust ORACLE — provenance & benchmarks page `[SHIPPED ✓]`
**Separate from the existing Research tab.** Research = the academic working note. Trust = the B2B sales asset.

- Live benchmark leaderboard (auto-updated when a new run completes).
- Model cards per backbone (BioCLIP, DINOv2, ConvNeXt-V2).
- Training data lineage with sources and licenses.
- Reproducibility receipts: seed, commit hash, dataset hash.
- PlantCLEF 2026 final result banner when it lands.
- **AC-3 constraint-pruning visualization** — interactive D3 node-link diagram showing how the neuro-symbolic layer prunes candidate species based on taxonomic / biogeographic rules. "We don't just pattern-match — we reason." This is the moat competitors can't fake.
- "Audited by" partner logos (placeholder until partners exist).

---

### 7. Reports — agentic ecological reporting `[SHIPPED ✓]`
The highest-ROI B2B feature. Conservation NGOs, state forestry departments, biodiversity consultancies, agritech all need this.

- Button: **"Generate Ecological Health Report for Land Managers"**.
- Pipeline: recent identifications + GBIF/IUCN metadata → structured prompt → local Gemma 4 (already in backend) → markdown → PDF via `@react-pdf/renderer` (or server-side wkhtmltopdf for richer layouts).
- Report sections: site overview map, species richness, rare/threatened taxa flagged, invasive species flagged, phenological notes, recommended actions, methods + provenance.
- Templates: Quick Brief (1 page) · Standard (5–10 pages) · Audit-Grade (30+ pages with appendix).
- Per-organisation branding (logo, signoff block) on Pro tier.
- Webhook for scheduled monthly reports (Pro tier).

---

## Quick-win flair (parallel, cheap)

- [ ] **Sound design** — subtle synth pulse on confidence-locked IDs. Off by default, toggle in header.
- [ ] **Onboarding tour** — 30-second motion-driven walkthrough for new visitors (Fleet → Identify → Atlas).
- [ ] **Share cards** — universal export button on any discovery card / identification result.
- [ ] **"Field" theme** — high-contrast outdoor-sun variant of the dark UI for mobile field use. Studio (current) and Field; one toggle, no light-mode rebrand.
- [ ] **Replace placeholder picsum images** in `SpeciesDiscovery.tsx` with real model outputs on real GBIF photos.
- [ ] **Live status footer** — replace "PLATFORM SOVEREIGN" static text with real-time inference QPS.
- [ ] **Ambient header pulse** — small reactive indicator in the header that beats with inference QPS. Scoped, brand-consistent — not a full "heartbeat canvas" page.
- [ ] **Keyboard shortcuts** — `/` to focus identify, `g f` for Fleet, `g i` for Identify, etc.

---

## Explicitly out of scope

- **Light "Botanical Green" mode** — dilutes the neon-cyber brand. Field theme covers the legitimate outdoor-readability need.
- **Generative "Ecological Heartbeat" canvas page** — risk of decorative cruft without clear semantic mapping. The header pulse covers the ambient-motion want at a fraction of the cost.

---

## Tab layout (after all pillars)

```
[Fleet] [Identify ✓] [Atlas ✓] [Journal★] [Discover] [Mission]
[Reports★] [Analytics] [Research] [Trust★] [Devs★] [Silicon] [Console]
```

★ = new in this roadmap. The header nav is horizontally scrollable (with masked fade edges) so we can keep adding tabs without an overflow menu.

**Auth-related UI** lives in the header (top-right): unauthenticated → "Sign In" chip; authenticated → avatar + tier badge → dropdown to Account / Admin (if applicable) / Sign Out. No tab needed.

---

## Engineering notes

- Package manager: **Bun** (not npm). `bun install`, `bun run build`, `bun run dev`. `dashboard/bun.lock` is authoritative.
- Deploy: `./scripts/redeploy_frontend.sh` from project root → Firebase Hosting at https://oracle-neuro-sym.web.app. Run after each frontend change ready for live preview.
- All new pages ship as components under `dashboard/src/components/`, lazy-loaded where heavy (Atlas globe + MapLibre, Reports PDF renderer).
- Mock data first; swap in real model / DB / auth later. Each pillar should be demoable end-to-end on its own branch before merging.
- Keep glassmorphic / neon vocabulary consistent: `glass-panel`, `glass-button`, `oracle-accent`, `oracle-cyan`, `oracle-pink`.
- Build target stays Vite + React 19 + Tailwind 4 + framer-motion + three.js + maplibre-gl. Adds for upcoming pillars: `firebase` (Auth + Firestore), `d3` (Trust AC-3 viz), `@react-pdf/renderer` (Reports).
- The horizontal nav uses CSS `mask-image` + scroll-into-view for the active tab; no plugin required.
