"""Inference phase entry point.

Usage:
    python phases/inference/run.py --config configs/inference.yaml
    torchrun --nproc_per_node=1 phases/inference/run.py --config configs/inference.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from .config import load
from .pipeline import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="PlantCLEF 2026 inference pipeline")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to inference.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--limit",      type=int, default=None, help="Process only first N images")
    parser.add_argument("--checkpoint", action="append", help="Path to checkpoint (can specify multiple)")
    parser.add_argument("--resolution", action="append", type=int, help="Native resolution for checkpoint (can specify multiple)")
    parser.add_argument("--batch_size", type=int, default=None)
    args, _ = parser.parse_known_args()

    cfg = load(args.config)

    if args.checkpoint:
        resolutions = args.resolution or [cfg.resolution] * len(args.checkpoint)
        # Zip them together
        cfg.checkpoints = [{"path": p, "resolution": r} for p, r in zip(args.checkpoint, resolutions)]
    elif args.resolution:
        # No --checkpoint override, but --resolution given. Override existing
        # checkpoints' native resolution + cfg.resolution + tile_size in lockstep.
        new_res = args.resolution[0]  # `action='append'` always returns a list
        print(f"[Inference] Overriding resolution {cfg.resolution} → {new_res} (from --resolution).")
        cfg.resolution = new_res
        if hasattr(cfg, "tile_size"):
            cfg.tile_size = new_res
        if getattr(cfg, "checkpoints", None):
            for ck in cfg.checkpoints:
                if isinstance(ck, dict):
                    ck["resolution"] = new_res

    if args.limit is not None:
        cfg.limit = args.limit

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    run_inference(cfg)


if __name__ == "__main__":
    main()
