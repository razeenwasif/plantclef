# PlantCLEF 2026 Website (Next.js 14)

Next.js 14 App Router app for the team's PlantCLEF 2026 submission recap.
Static export, no backend. Lives at **arm-wision.github.io** (org root
site, served from the domain root with no subpath).

## Quick start

```bash
cd website
npm install
npm run dev          # localhost:3000
npm run build        # static export to ./out/
```

The build produces a fully static `out/` directory.

## Layout

```
website/
├── app/
│   ├── layout.jsx          # html shell, Google Fonts
│   ├── page.jsx            # composes the section components
│   └── globals.css         # design tokens, layout, charts
├── components/
│   ├── Topbar.jsx          ("use client") scroll-aware nav
│   ├── Hero.jsx            real LUCAS quadrat thumbnails
│   ├── Stats.jsx
│   ├── Abstract.jsx
│   ├── Tasks.jsx           ("use client") tabbed task description
│   ├── Dataset.jsx
│   ├── ChartsSection.jsx   wrapper for the five Recharts figures
│   ├── Timeline.jsx
│   ├── Leaderboard.jsx     ("use client") sortable per-experiment table
│   ├── Baselines.jsx
│   ├── TeamBlock.jsx
│   ├── BibTeX.jsx          ("use client") copy-to-clipboard
│   ├── Reveal.jsx          ("use client") scroll-reveal animations
│   ├── Footer.jsx
│   └── charts/
│       ├── theme.js                shared Recharts color tokens
│       ├── ScoreProgression.jsx    Figure 1 (line)
│       ├── UnfreezeSweep.jsx       Figure 2 (val vs Kaggle dual axis)
│       ├── ValVsKaggle.jsx         Figure 3 (scatter + Pearson r)
│       ├── E015PerEpoch.jsx        Figure 4 (per-epoch lines)
│       └── SpeciesLongTail.jsx     Figure 5 (cap histogram)
├── lib/
│   ├── team.js             single source of truth for the team list
│   ├── paths.js            asset() helper that respects optional BASE_PATH
│   └── chart_data.js       five chart datasets, regenerated from CSVs
└── public/
    ├── data/               JSON copies of the submission + experiment data
    └── quadrats/           six real LUCAS test images (480x480 jpgs)
```

## Deploy

GitHub Actions workflow at `.github/workflows/deploy.yml` builds the
static export on every push to main that touches `website/**`, then
uploads `website/out/` to GitHub Pages.

The site is served at `https://arm-wision.github.io/` because the repo
inside the `arm-wision` org is named `arm-wision.github.io` (root site,
no subpath).

## Refreshing the Kaggle data

```bash
kaggle competitions submissions plantclef-2026 -v --page-size 200 \
    > public/data/kaggle_submissions.csv
# then re-run the aggregator script (originally in
# website/plantclef/data/build_data.py before the move) and paste the
# updated literals back into lib/chart_data.js
```

## Optional subpath deploy

If we ever need to host under a subpath again (e.g. legacy
`*.github.io/PlantCLEF2026`), set `BASE_PATH=/<repo>` in the build env
and `next.config.mjs` will flip basePath + assetPrefix automatically.
The `asset()` helper in `lib/paths.js` keeps raw `<img src>` values
working under any subpath.
