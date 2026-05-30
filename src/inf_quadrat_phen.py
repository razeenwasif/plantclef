"""
inf_quadrat_phen.py - thin entry point for the phenology-aware quadrat
inference pipeline (paper Pivot 3).

Forwards command-line arguments to `quadrat/inf_script_phen.py`:
anchor + multi-scale tiling at {1.0, 0.8} + ExG vegetation filter +
entropy-weighted Bayesian aggregation + circular Gaussian DOY prior.
For the full argument reference run with --help.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

INF_PATH = (Path(__file__).resolve().parent.parent
            / "quadrat" / "inf_script_phen.py")

if not INF_PATH.is_file():
    raise SystemExit(f"Could not find phenology inference script at {INF_PATH}.")

runpy.run_path(str(INF_PATH), run_name="__main__")
