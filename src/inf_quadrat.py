"""
inf_quadrat.py - thin entry point for PlantCLEF-style quadrat inference.

Forwards command-line arguments to `quadrat/inf_script.py`, the paper's
anchor recipe (4x4 grid, 224 + 336 dual-resolution, softmax-mean,
LA tau=0.25, T=0.03, adaptive k in [2, 10]). For the full argument
reference run with --help.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

INF_PATH = (Path(__file__).resolve().parent.parent
            / "quadrat" / "inf_script.py")

if not INF_PATH.is_file():
    raise SystemExit(f"Could not find quadrat inference script at {INF_PATH}.")

runpy.run_path(str(INF_PATH), run_name="__main__")
