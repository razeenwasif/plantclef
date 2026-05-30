"""
inf_single_plant.py - thin entry point for PlantNet-style whole-image
single-plant inference.

PlantNet-300K images already contain one centred subject per image, so
no tile aggregation is required. Forwards to
`single_plant/inf_script_whole.py`. For the full argument reference run
with --help.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

INF_PATH = (Path(__file__).resolve().parent.parent
            / "single_plant" / "inf_script_whole.py")

if not INF_PATH.is_file():
    raise SystemExit(f"Could not find single-plant inference script at {INF_PATH}.")

runpy.run_path(str(INF_PATH), run_name="__main__")
