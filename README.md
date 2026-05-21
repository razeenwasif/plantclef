# 🌿 ORACLE: Neural Identifying & Taxonomic Reasoning Orchestrator

[![NVIDIA Blackwell](https://img.shields.io/badge/NVIDIA-Blackwell%20Optimized-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Rust](https://img.shields.io/badge/Rust-High--Performance%20I%2FO-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **PlantCLEF 2026 Official Implementation**  
> An elite, neuro-symbolic ensemble designed to identify 7,800+ plant species with high-fidelity botanical reasoning and absolute hardware saturation.

---

## 🚀 The Speed-of-Light Stack

ORACLE is engineered for **NVIDIA Blackwell (RTX 5090 / PRO 6000)** clusters, pushing the theoretical limits of distributed throughput and VRAM efficiency.

### ⚡ Blackwell "Native-Path" Compute
*   **FP8 Surge:** 2.0x throughput increase via native 8-bit transformer kernels (`TransformerEngine`).
*   **FlashAttention-4:** Full activation of **TCGEN05** hardware for extreme ViT-L sequence lengths.
*   **Zero-Copy Preprocessing:** Custom CUDA C++ kernels for Retinex illumination normalization, eliminating CPU-GPU bottlenecks.
*   **Unified Graph Fusion:** `torch.compile` with full **CUDA Graphs** capture for static, low-latency execution paths.

### 🧬 Zero-Latency Data Infrastructure
*   **nvJPEG Hardware Decoding:** 100% offload of JPEG decompression to dedicated GPU silicon via NVIDIA DALI.
*   **Rust-Powered Swarm:** SIMD-optimized metadata and resizing engines (550+ imgs/sec) that bypass the Python GIL.
*   **RAM-Disk Satiation:** Automated dataset synchronization to `/dev/shm` for zero-latency I/O.

---

## 🏗️ Multi-Language Architecture

The pipeline is architected as a polyglot system to maximize hardware occupancy across 8x NVIDIA GPUs.

| Component | Language | Responsibility |
| :--- | :--- | :--- |
| **Brain** | `Python` | Model logic, training loops, and research orchestration (PyTorch/DALI). |
| **Muscles** | `Rust` | High-throughput I/O, dataset sharding, and real-time resizing. |
| **Nervous System** | `Go` | Cluster health monitoring, telemetry API, and mission dispatching. |
| **Foundations** | `C++/CUDA` | Fused loss kernels, Retinex filters, and custom SAHI tiling engines. |

---

## 🛠️ Usage Protocol

The ORACLE pipeline is managed via the unified **`oracle.py`** CLI.

### ⚡ Dual-Accelerator Support

ORACLE trains on both NVIDIA CUDA clusters and Google Cloud TPU VMs through the same CLI. Switch with `--mode`:

```bash
./oracle.py train --phase p2a --role sprint --mode cuda     # NCCL + torchrun
./oracle.py train --phase p2a --role sprint --mode tpu      # PJRT + xmp.spawn
./oracle.py train --phase p2a --role sprint --mode auto     # detect: TPU_NAME → tpu, else cuda
```

The selector flows through `launch_oracle.sh`, which branches the NCCL/Blackwell vs XLA/PJRT environments, and through `src/training/accelerator.py`, which abstracts device, dtype, autocast, distributed init, and graph-step semantics. The data layer also branches: NVIDIA DALI on CUDA, `webdataset` on TPU (same `.tar` shards, equivalent transforms). See **[docs/TPU.md](docs/TPU.md)** for the full guide.

### 🧭 Cluster Manifest

For multi-host clusters, the topology lives in a single `cluster.yaml` instead of being scattered across env vars and file-lock dances:

```bash
./oracle.py train --phase p2a --cluster configs/cluster.yaml
```

The manifest derives `ORACLE_NNODES`, `ORACLE_MASTER_IP`, `ORACLE_NODE_RANK`, device counts, and accelerator mode deterministically. Hostname auto-detect picks the right entry; pass `--host-id` to override. Schema + worked examples in **[docs/CLUSTER.md](docs/CLUSTER.md)**; copyable starting point at `configs/cluster.example.yaml`.

### 1. Foundation Caching (Phase 1)
Build the foundation feature cache from 1.4M images.
```bash
./oracle.py train --phase p1 --role sprint
```

### 2. Head Warmup (Phase 2a)
Train three unique warmup baselines on extracted features.
```bash
./oracle.py train --phase p2a --role sprint
```

### 3. Student Distillation (Phase 2b-student)
Perform multi-seed fine-tuning and student distillation.
```bash
./oracle.py train --phase p2b-student --role sprint
```

### 4. Asymmetric Dual-Teacher Distillation (Phase 2.5 - ad-td)
Train the high-efficiency "Trinity" student.
```bash
./oracle.py train --phase ad-td --role sprint -- --mode extract
./oracle.py train --phase ad-td --role sprint -- --mode train
```

### 5. Ultimate Ensemble Inference
Generate the final submission using the unified mega-ensemble pipeline.
```bash
./oracle.py infer --ensemble
```

---

## 📊 Performance Benchmarks

| Optimization | Speedup | Impact |
| :--- | :--- | :--- |
| **WebDataset + NVMe** | 5.0× | Zero I/O wait, sequential shard reads. |
| **Rust Preprocessing** | 8.0× | SIMD-parallel resize at 550+ img/s. |
| **Teacher Logit Cache** | 3.0× | Eliminates teacher forward passes during KD. |
| **Blackwell FP8** | 2.0× | Maximum Tensor Core utilization. |

---

## 🎛️ Neon Command Center

The ORACLE dashboard is a dual-surface app: an **operator console** for the training fleet and a **public-facing identification platform** built on the same models.

**Live:** [oracle-neuro-sym.web.app](https://oracle-neuro-sym.web.app)
**Local dev:**
```bash
cd dashboard && bun install && bun run dev
```
**Deploy:** `./scripts/redeploy_frontend.sh` from the project root.

### 🛰 Operator Console
Real-time monitoring of the distributed cluster:
- **Fleet** — per-GPU utilisation, temp, VRAM, power, assigned job.
- **Mission** — running / queued / failed training jobs with throughput and cost.
- **Analytics** — Vector-F1 confidence delta, SWA calibration, long-tail recall.
- **Silicon** — 3D neural-core vitals, NVLink bandwidth, GDDR7 throughput.
- **Console** — live cluster log stream with anomaly highlighting.
- **Research** — the PlantCLEF 2026 working note rendered inline.

### 🌿 Product Features

#### Identify ✓
Drag-and-drop a photo of any plant; ORACLE returns the top-5 species candidates with citation-grade provenance.
- Live inference pipeline animation (DALI → BioCLIP → DINOv2 → ConvNeXt-V2 → SWA calibration).
- Circular confidence rings; expandable rows linking out to **GBIF**, **Wikipedia**, **iNaturalist**.
- Grad-CAM-style attribution heatmap toggle on the uploaded image.
- Rarity tagging (common / uncommon / rare) and family-level taxonomy.
- Daily free-tier quota (5/day) with "Upgrade to Pro" path for unlimited use, rare-species alerts, and EXIF GPS auto-tagging.

#### Atlas ✓
Interactive global biodiversity feed with two views in one tab:
- **Globe** — rotating wireframe earth (three.js + react-three-fiber) with colour-coded pins for recent identifications, pulsing rings on rare-tier finds, click-through to species detail. Drag to rotate, scroll to zoom.
- **Map** — MapLibre GL 2D phytogeographic view on a dark-cartographic basemap. Custom glowing markers sized by rarity, click to inspect.
- Shared filter strip: domain toggles (Flora / Terrestrial / Avian / Marine / Urban), time window (24h / 7 days / All).
- **Trending Taxa** sidebar ranks the most-spotted species in the active window.
- Live counters for rare pins and active domains.

#### Auth & Tier Management ✓
Firebase Authentication powers identity and tier-based feature gating.
- **Sign In / Sign Up modal** — email/password and Google OAuth, full-screen blurred overlay with toggle between modes.
- **Tier system** — `free` (5 IDs/day), `pro` (unlimited + rare-species alerts), `field` (Pro + offline bundle), `admin` (everything).
- **Account menu** in the header — avatar with tier-coloured initials, dropdown showing email, real tier badge, tier description, and an upgrade CTA.
- **Admin "View as…" toggle** — admins can impersonate any tier client-side to QA every paywall and quota without logging out. State persists in `localStorage`.
- **razeen.wasif66@gmail.com** is seeded as admin on first sign-in via the `ADMIN_EMAILS` list in `src/lib/auth.tsx`.
- **Tier-gated quota** — `Identify` now reads its daily limit from `tierCapabilities(effectiveTier)`. Free shows the `X / 5` chip and "Upgrade to Pro" CTA; Pro+ shows `∞`.

> **One-time Firebase Console setup required:** enable Email/Password and Google sign-in providers under *Authentication → Sign-in method*. The web SDK config is already wired via `dashboard/.env` (`VITE_FIREBASE_*`).

#### Journal ✓
Your personal herbarium. Every identification you make is auto-saved as a stamp in a private collection.
- **Stats strip**: total entries, unique species, rarity score (rare=10 · uncommon=3 · common=1), day streak.
- **Pokédex-style grid** — square thumbnails with rarity badges and confidence chips; filter by domain, search by species/family, sort by recency / rarity / confidence.
- **Detail modal** — full image, taxonomy, habitat, observation timestamp, citation links to GBIF / Wikipedia / iNaturalist.
- **Share Card export** — one-click 1080×1080 PNG with photo, species, confidence, rarity chip, ORACLE watermark. Built with the Canvas API; free distribution for socials.
- **Auto-save** from Identify — when a logged-in user completes an identification, the top-1 result is saved silently. The Identify button switches to "Saved · View in Journal" linking straight to the new entry.
- **Login gate** — unauthenticated users see a CTA explaining the value and a "Sign In" button.
- Storage: per-user `localStorage` (last 200 entries, images resized to 480 px JPEG). Firestore sync is a follow-up.

#### API · Developer Console ✓
The public API surface for ORACLE — keys, quotas, code, and plans in one tab.
- **API Keys management** — create, reveal, copy, revoke (Pro+ tier). Live vs Test environments, masked prefix by default, "Just created · copy now" highlight on freshly minted keys.
- **30-day usage chart** — recharts area graph with synthetic series until real metering ships.
- **Quick-start examples** — cURL, Python, and JavaScript snippets for `POST /v1/identify` with one-click copy.
- **Endpoints reference** — method-coloured table covering identify, batch, species lookup, webhooks, and usage.
- **Plans strip** — Free / Pro / Enterprise cards with feature lists and inline upgrade buttons; highlights the current plan.
- Tier gating: docs are public, keys require Pro (auto-granted to admins). Free users see an upgrade gate.

#### Trust ✓
The B2B sales asset — the *Why you can trust ORACLE* page. Distinct from the Research tab (which is the academic working note).
- **Hero strip** — macro-F1, species coverage, training corpus, audit lineage.
- **PlantCLEF 2026 banner** — placeholder for the official rank, lights up the moment the leaderboard publishes.
- **AC-3 constraint visualisation (the centrepiece)** — interactive SVG flow diagram showing how 12 visual candidates get pruned through Biogeographic → Phenology → Taxonomic constraint columns to reach the final top-5. Hover any species to see exactly which constraint kept or pruned it. Animated bezier edges, glow-on-hover, red strike-throughs on rejections with reason text.
- **Model cards** — three side-by-side cards for BioCLIP (304 M, MIT, Tree-of-Life), DINOv2 (1.1 B, CC-BY-NC, LVD-142M), and ConvNeXt-V2 (660 M, Apache-2.0, ImageNet-22K).
- **Training data lineage table** — GBIF (CC0), iNaturalist (CC-BY-NC), PlantCLEF (research-only), Herbarium scans. Per-source counts, share bars, license, source links.
- **Reproducibility receipt** — commit hash, run ID, dataset SHA-256, seeds, eval timestamp. Each row copyable.
- **Auditors strip** — "Pending" placeholders until peer review and partner integrations finalise.

#### Reports ✓
Agentic ecological reporting. Turn your journal observations into publish-ready PDFs.
- **Three templates** — Quick Brief (1 page), Standard (5–10 pages), Audit-Grade (30+ pages with provenance appendix).
- **Configurator** — organisation name, observation window (7 / 30 / 90 days), optional stakeholder-context prompt for the agent, demo-dataset toggle for empty journals.
- **Live agentic pipeline** — animated 5-stage progress (Collate → GBIF/IUCN cross-ref → Gemma 4 synthesis → SWA calibration → Layout) modelled on the Identify pipeline.
- **In-app document preview** — full report rendered on cream paper inside the app: gradient cover page, executive summary, community composition table, species inventory, rare/threatened taxa flags with amber alert chips, numbered recommendations, methods, and a provenance appendix (audit template only).
- **Download PDF** uses the browser's native print pipeline via an `@media print` stylesheet that hides all chrome and reveals only `.oracle-report` at A4. Yields a clean, paginated PDF with no extra libraries.
- **Tier gating** — sign-in required; Pro tier (free for admin) unlocks all templates and custom branding. Free sees an upgrade gate.
- **Real Gemma 4 integration** — currently deterministic synthesis from Journal + tone variants per template. Backend LLM service drop-in lands when the inference path is wired (the data shape and prompt scaffolding are already in `src/lib/reports.ts`).

---

## 📑 Documentation
- [🧬 Technical Pipeline Guide](docs/PIPELINE.md)
- [🏗️ System Architecture UML](docs/ARCHITECTURE_UML.md)
- [📚 Codebase Deep-Dive](docs/CODEBASE_GUIDE.md)

---

<p align="center">
  <b>Developed for PlantCLEF 2026</b><br>
  <i>Pushing the boundaries of botanical AI.</i>
</p>
