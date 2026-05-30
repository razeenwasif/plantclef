"""
Tiled multi-label inference for the DINOv3 classifier on PlantCLEF quadrats.

Pipeline (per image):
  1. Generate overlapping tiles with src_experiments/004_bioclip_few_shot/tiling.py.
  2. Encode each tile with the LoRA-fine-tuned DINOv3.
  3. Forward through the classifier head -> per-tile sigmoid probabilities.
  4. Aggregate tile probabilities to an image-level score (max / mean_top_m / noisy_or).
  5. Apply threshold + top-N to produce the multi-label species list.
  6. Write a PlantCLEF-format submission CSV.

Outputs:
  {output-dir}/{run_slug}/
      run_config.json
      submission.csv               (quadrat_id, species_ids)
      predictions_scored.csv       (quadrat_id, rank, species_id, score)
      summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# Sibling
from model import DINOv3MultiLabel, build_default_transform, DEFAULT_HUB_NAME

# 004 utilities
_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
if str(_FOUR) not in sys.path:
    sys.path.insert(0, str(_FOUR))
from tiling import get_tiles, get_multiscale_tiles  # type: ignore  # noqa: E402
from aggregation import aggregate_scores, apply_threshold, AGG_MODES  # type: ignore  # noqa: E402
from dataset import DEFAULT_SPECIES_CSV  # type: ignore  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "./outputs"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DINOv3 tiled multi-label inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Phase-2 checkpoint (.pth) produced by train.py --phase 2.")
    p.add_argument("--images-root", required=True,
                   help="Directory containing test quadrat images.")
    p.add_argument("--species-csv", default=DEFAULT_SPECIES_CSV)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    # Tiling
    p.add_argument("--tile-size", type=int, default=384,
                   help="Square tile side (must be a multiple of 16 for DINOv3).")
    p.add_argument("--tile-overlap", type=int, default=128)
    p.add_argument("--multiscale", type=str, default="",
                   help="Comma-separated tile sizes for multi-scale tiling, e.g. '256,384'.")
    p.add_argument("--stride-ratio", type=float, default=0.5,
                   help="Used only with --multiscale.")

    # Aggregation / thresholding
    p.add_argument("--agg-mode", choices=AGG_MODES, default="noisy_or")
    p.add_argument("--agg-top-m", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--per-species-threshold-file", default=None,
                   help="Optional JSON {species_id: threshold} for per-class cutoffs.")

    # Inference perf
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true",
                   help="Use bf16 autocast for backbone forward.")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: str, device: str) -> tuple[DINOv3MultiLabel, list[str]]:
    """Reconstruct the LoRA-wrapped DINOv3 from a Phase-2 checkpoint."""
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    backbone_name = ckpt.get("backbone_name", DEFAULT_HUB_NAME)
    backbone_source = ckpt.get("backbone_source", "torch_hub")
    species_ids = ckpt["species_ids"]
    n_classes = ckpt["n_classes"]
    lora_r = ckpt.get("lora_r", 32)
    lora_alpha = ckpt.get("lora_alpha", 64)

    model = DINOv3MultiLabel(
        n_classes=n_classes,
        backbone_source=backbone_source,
        backbone_name=backbone_name,
        pretrained=False,  # weights come from the checkpoint
    )
    model.apply_lora(r=lora_r, alpha=lora_alpha, dropout=0.0)
    model.backbone.load_state_dict(ckpt["backbone_state"])
    model.head.load_state_dict(ckpt["head_state"])
    if "gem_state" in ckpt:
        model.gem.load_state_dict(ckpt["gem_state"])
    model = model.to(device).eval()
    logger.info(f"  loaded model with {n_classes} classes (epoch {ckpt.get('epoch', '?')})")
    return model, species_ids


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_tiles_to_probs(
    model: DINOv3MultiLabel,
    transform,
    tiles: list[Image.Image],
    device: str,
    batch_size: int,
    use_bf16: bool,
) -> torch.Tensor:
    """Returns (n_tiles, n_classes) sigmoid probabilities on CPU."""
    if not tiles:
        return torch.empty(0, model.n_classes)

    out_chunks: list[torch.Tensor] = []
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i: i + batch_size]
        batch = torch.stack([transform(t) for t in batch_tiles]).to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=use_bf16 and device.startswith("cuda"),
        ):
            logits = model(batch)
        probs = torch.sigmoid(logits.float()).cpu()
        out_chunks.append(probs)
    return torch.cat(out_chunks, dim=0)


def process_image(
    image_path: Path,
    model: DINOv3MultiLabel,
    transform,
    species_ids: list[str],
    args: argparse.Namespace,
    per_species_thresholds: dict[str, float] | None,
) -> tuple[list[str], list[float]]:
    image = Image.open(image_path).convert("RGB")

    # Single- or multi-scale tiling
    if args.multiscale:
        tile_sizes = [int(s) for s in args.multiscale.split(",") if s.strip()]
        tiles, _ = get_multiscale_tiles(image, tile_sizes, stride_ratio=args.stride_ratio)
    else:
        stride = max(1, args.tile_size - args.tile_overlap)
        tiles, _ = get_tiles(image, args.tile_size, stride)

    if not tiles:
        return [], []

    tile_probs = encode_tiles_to_probs(
        model, transform, tiles, args.device, args.batch_size, args.bf16
    )

    # tile_probs are already sigmoid'd; aggregation modes that re-sigmoid (noisy_or)
    # would be wrong if applied to probs. Convert back to logits for noisy_or.
    if args.agg_mode == "noisy_or":
        eps = 1e-6
        clamped = tile_probs.clamp(eps, 1 - eps)
        tile_scores = torch.log(clamped / (1 - clamped))  # logit
    else:
        tile_scores = tile_probs

    image_scores = aggregate_scores(
        tile_scores, mode=args.agg_mode, top_m=args.agg_top_m
    )
    # noisy_or returns probabilities; max/mean over probs is also probability-scale
    # ('mean' on probs gives a smoothed prob; 'max' picks the strongest tile).

    return apply_threshold(
        image_scores,
        species_ids,
        threshold=args.threshold,
        top_n=args.top_n,
        per_species_thresholds=per_species_thresholds,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def make_run_slug(args: argparse.Namespace) -> str:
    ckpt_stem = Path(args.checkpoint).stem
    parts = [ckpt_stem, args.agg_mode, f"thr{args.threshold:.2f}"]
    if args.multiscale:
        parts.append(f"ms{args.multiscale.replace(',', 'x')}")
    else:
        parts.append(f"ts{args.tile_size}")
    return "_".join(parts)


def main() -> None:
    args = parse_args()

    if args.tile_size % 16 != 0 and not args.multiscale:
        sys.exit(f"--tile-size must be a multiple of 16 for DINOv3, got {args.tile_size}")
    if args.multiscale:
        for s in (int(x) for x in args.multiscale.split(",")):
            if s % 16 != 0:
                sys.exit(f"--multiscale entry {s} is not a multiple of 16")

    images_root = Path(args.images_root)
    if not images_root.exists():
        sys.exit(f"ERROR: --images-root not found: {images_root}")

    out_root = Path(args.output_dir)
    run_slug = make_run_slug(args)
    out_dir = out_root / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run slug: {run_slug}")
    logger.info(f"Output dir: {out_dir}")

    # Load model
    model, species_ids = load_model_from_checkpoint(args.checkpoint, args.device)
    transform = build_default_transform(args.tile_size if not args.multiscale else 384)

    # Per-species thresholds (optional)
    per_species_thresholds: dict[str, float] | None = None
    if args.per_species_threshold_file:
        thr_path = Path(args.per_species_threshold_file)
        if thr_path.exists():
            per_species_thresholds = json.load(thr_path.open())
            logger.info(f"Loaded per-species thresholds: {len(per_species_thresholds)} entries")
        else:
            logger.warning(f"Threshold file not found: {thr_path}")

    # Find images
    image_paths = find_images(images_root)
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        sys.exit(f"ERROR: no images found in {images_root}")
    logger.info(f"Found {len(image_paths)} images")

    # Save run config
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(
            {**vars(args), "n_classes": len(species_ids), "n_images": len(image_paths)},
            f, indent=2,
        )

    # Inference loop
    submission_rows: list[dict] = []
    scored_rows: list[dict] = []
    errors = 0
    t0 = time.perf_counter()

    for i, image_path in enumerate(image_paths):
        quadrat_id = image_path.stem
        try:
            pred_ids, pred_scores = process_image(
                image_path, model, transform, species_ids, args, per_species_thresholds
            )
        except Exception as exc:
            logger.error(f"Error on {image_path.name}: {exc}")
            errors += 1
            pred_ids, pred_scores = [], []

        ids_str = "[" + ", ".join(pred_ids) + "]"
        submission_rows.append({"quadrat_id": quadrat_id, "species_ids": ids_str})
        for rank, (sid, score) in enumerate(zip(pred_ids, pred_scores), start=1):
            scored_rows.append({
                "quadrat_id": quadrat_id, "rank": rank,
                "species_id": sid, "score": f"{score:.6f}",
            })

        if (i + 1) % 25 == 0 or i + 1 == len(image_paths):
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            logger.info(
                f"  {i+1:>5}/{len(image_paths)} processed | "
                f"{rate:.2f} img/s | last: {quadrat_id} -> {len(pred_ids)} species"
            )

    total_secs = time.perf_counter() - t0

    # Submission
    sub_path = out_dir / "submission.csv"
    with open(sub_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quadrat_id", "species_ids"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(submission_rows)
    logger.info(f"Submission: {sub_path} ({len(submission_rows)} rows)")

    scored_path = out_dir / "predictions_scored.csv"
    with open(scored_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quadrat_id", "rank", "species_id", "score"])
        w.writeheader()
        w.writerows(scored_rows)
    logger.info(f"Scored: {scored_path} ({len(scored_rows)} rows)")

    summary = {
        "run_slug": run_slug,
        "checkpoint": args.checkpoint,
        "n_images": len(image_paths),
        "n_processed": len(image_paths) - errors,
        "n_errors": errors,
        "total_secs": round(total_secs, 2),
        "per_image_secs": round(total_secs / max(len(image_paths), 1), 3),
        "agg_mode": args.agg_mode,
        "threshold": args.threshold,
        "tile_size": args.tile_size,
        "multiscale": args.multiscale,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {len(image_paths)} images in {total_secs:.1f}s "
          f"({total_secs / max(len(image_paths), 1):.3f}s/image)\n"
          f"Submission: {sub_path}")


if __name__ == "__main__":
    main()
