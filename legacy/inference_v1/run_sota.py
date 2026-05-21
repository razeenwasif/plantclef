"""Inference phase entry point.

Usage:
    python phases/inference/run.py --config configs/inference_v2.yaml
    torchrun --nproc_per_node=1 phases/inference/run.py --config configs/inference_v2.yaml
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from .config import load
from .pipeline_sota import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="PlantCLEF 2026 inference pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to inference_v2.yaml")
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

    if args.limit is not None:
        cfg.limit = args.limit
    
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    run_inference(cfg)


if __name__ == "__main__":
    main()
