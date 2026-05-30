"""
train.py - thin entry point for the BioCLIP-2.5 multitask training pipeline.

The actual model definition, dataset, transforms, and training loop
live in `../shared/bioclip25_multitask/`. This file forwards command-
line arguments to that script so callers can launch training from the
repository root with:

    python src/train.py [args]
    torchrun --nproc_per_node=2 src/train.py [args]

For the full argument reference see
`shared/bioclip25_multitask/train.py` or run with --help.

The recipe reported in the paper (BioCLIP 2.5 ViT-H/14, per-head MLPs
for species/genus/family, last-4-blocks fine-tune on the i001 manifest)
is documented in `shared/bioclip25_multitask/README.md` and reproduced
by the staged invocation:

    bash shared/bioclip25_multitask/scripts/train_head_only.sh
    bash shared/bioclip25_multitask/scripts/train_last_blocks.sh \\
         outputs/train_head_only/checkpoints/best.pt
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

EXP_DIR = (Path(__file__).resolve().parent.parent
           / "shared" / "bioclip25_multitask")
TRAIN_PATH = EXP_DIR / "train.py"

if not TRAIN_PATH.is_file():
    raise SystemExit(
        f"Could not find the BioCLIP 2.5 multitask training script at {TRAIN_PATH}.\n"
        f"This file is a thin shim; the real training code lives in shared/."
    )

sys.path.insert(0, str(EXP_DIR))

runpy.run_path(str(TRAIN_PATH), run_name="__main__")
