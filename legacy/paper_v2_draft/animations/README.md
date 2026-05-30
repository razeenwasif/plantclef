# PlantCLEF 2026 inference pipeline — Manim animations

Four 3Blue1Brown-style animated scenes that walk through the post-processing
math of the winning submission, narration-ready for a course video
presentation.

## Scenes

| File                              | Length | What it shows |
|-----------------------------------|--------|---------------|
| `scene_pipeline_overview.py`      | ~60 s  | End-to-end pipeline walkthrough: quadrat → tiles → BioCLIP → aggregation → post-processing → species set |
| `scene_logit_adjustment.py`       | ~45 s  | The class-prior logit adjustment `z' = z − τ·log π̃` with τ slider 0 → 0.25 boosting the long tail |
| `scene_bayesian_aggregation.py`   | ~45 s  | 16-tile entropy weighting `w = exp(−H)` collapsing per-tile posteriors into one image-level distribution |
| `scene_phenology.py`              | ~30 s  | Boltzmann phenology mask rotating around the calendar, modulating a species probability |

`style.py` holds the shared palette + helpers so every scene matches the
dashboard's glass-panel aesthetic (oracle-accent violet, cyan, pink on
near-black).

## Prerequisites

Manim Community Edition (the actively-maintained fork of 3Blue1Brown's
original). Tested with manim 0.18+.

```bash
# System deps (apt — Manim needs Cairo, Pango, ffmpeg, and a LaTeX install)
sudo apt-get install -y \
    build-essential \
    libcairo2-dev libpango1.0-dev \
    ffmpeg \
    texlive texlive-latex-extra texlive-fonts-extra dvisvgm

# Python deps
cd paper/animations
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rendering

```bash
# Single scene, medium quality (faster):
make logit             # → media/videos/scene_logit_adjustment/720p30/*.mp4

# Single scene, broadcast quality:
make logit Q=h         # 1080p60, much slower

# All four scenes:
make all

# Drop into an editor:
make all && open media/videos/   # macOS
make all && xdg-open media/videos/   # Linux
```

Quality flags map to manim's `-q` argument: `l` (480p15), `m` (720p30),
`h` (1080p60), `k` (4K60).

## Narration anchors

Each scene has `# NARRATION:` comments at every major beat indicating
suggested voiceover text and timing. The `wait()` calls between beats
leave gaps long enough for natural speech; trim them in post if your
read is faster.

## Output

Renders land under `media/videos/<scene_name>/<quality>/*.mp4`. The
`media/` directory is gitignored.
