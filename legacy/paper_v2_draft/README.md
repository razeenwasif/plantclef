# Long-form research draft (archived)

This is the CVPR-style technical-report draft that pre-dated the
official PlantCLEF 2026 working note (the citable artifact lives in
`~/Research/PlantCLEF2026/report/main.tex`). It was retired once its
content was decomposed into the four living docs:

* [`../../BACKLOG.md`](../../BACKLOG.md) - future-work directions
* [`../../docs/SYSTEMS_DESIGN.md`](../../docs/SYSTEMS_DESIGN.md) - hardware / kernel / distributed-training techniques
* [`../../docs/ALGORITHMS.md`](../../docs/ALGORITHMS.md) - ML / mathematical techniques
* [`../../docs/RELATED_WORK.md`](../../docs/RELATED_WORK.md) - citation bank

The `.tex` source and rendered PDF are preserved here for posterity
(easier to skim the connective prose than to read four reference docs
in series). LaTeX build artefacts (`.aux`, `.fdb_latexmk`, `.fls`,
`.log`, `.out`), the 459 MB manim virtualenv, and 41 MB of rendered
animation outputs were intentionally dropped on import; rebuild from
`animations/` source if you need them.

## Contents

| Path                                       | What it is                                                                 |
|--------------------------------------------|----------------------------------------------------------------------------|
| `plantclef2026_research_paper.tex`         | Long-form draft, 510 lines, 54 subsections.                                |
| `plantclef2026_research_paper.pdf`         | Rendered PDF (380 KB).                                                     |
| `cvpr.sty`, `ieeenat_fullname.bst`         | CVPR template + bib style for rebuild.                                     |
| `figures/inference_pipeline.tex`           | TikZ pipeline diagram.                                                     |
| `animations/scene_*.py`                    | Manim scene sources (pipeline overview, Bayesian aggregation, phenology, logit adjustment). |
| `animations/{Makefile,style.py,requirements.txt}` | Manim build glue.                                                          |
