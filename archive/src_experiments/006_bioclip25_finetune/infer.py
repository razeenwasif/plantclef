"""
Test-time inference: tiled prediction on quadrat images → submission.csv

Pipeline (per image)
--------------------
  1. Apply optional preprocessing (margin crop, JPEG recompression)
  2. Tile the image using the selected tiling mode
  3. Apply BioCLIP preprocessing to each tile
  4. Batch-encode tiles through the frozen backbone → embeddings
  5. Apply linear head → per-tile logits
  6. Aggregate → image-level scores
  7. Output top-N species predictions

Outputs
-------
  {output_dir}/
    submission.csv          PlantCLEF format: quadrat_id, species_ids
    predictions_scored.csv  quadrat_id, rank, species_id, score
    run_config.json
    summary.json
    tile_preview.png        (if --save-tile-preview is set, first image only)

Submission format
-----------------
  "quadrat_id","species_ids"
  "CBN-Pla-B1-20130724","[1395806]"
  "CBN-PdlC-A1-20130807","[1351284, 1494911, 1381367, 1396535, 1412857]"

Usage examples
--------------
  # Basic whole-image inference
  python infer.py --checkpoint ./outputs/train/checkpoints/best.pt

  # 3x3 grid tiling
  python infer.py --checkpoint ... --tile-mode grid_3x3

  # Sliding window (tile 224, stride 112)
  python infer.py --checkpoint ... --tile-mode sliding --tile-size 224 --stride 112

  # Dense multiscale: whole + 2x2 + 3x3 + 4x4, 25% overlap
  python infer.py --checkpoint ... --tile-mode multiscale_dense \\
                  --scales 1,2,3,4 --overlap 0.25

  # Five-crop preset
  python infer.py --checkpoint ... --tile-mode five_crop

  # Generic NxN grid via --grid-size
  python infer.py --checkpoint ... --tile-mode grid --grid-size 6

  # Multi-scale + topk_mean(5) + lanczos
  python infer.py --checkpoint ... --tile-mode multiscale \\
                  --agg-mode topk_mean --topk-agg 5 --interp lanczos

  # Smoke test (first 5 images) with tile preview
  python infer.py --checkpoint ... --limit 5 --save-tile-preview
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
from PIL import Image

from dataset import DEFAULT_TEST_IMAGES_DIR
from model import load_bioclip25_probe
from transforms import build_inference_preprocessor
from tiling import (
    TILING_MODES, AGG_MODES,
    tile_image, encode_tiles, classify_tiles, aggregate_logits,
    save_tile_preview, _tile_image_with_info,
)
from utils import (
    setup_logging,
    resolve_device,
    topk_predictions,
    save_json,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "./outputs/infer"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_scales(s: str) -> list[int]:
    """Parse a comma-separated list of ints, e.g. '1,2,3,4' → [1, 2, 3, 4]."""
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--scales must be comma-separated integers, got: {s!r}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BioCLIP 2.5 tiled inference → PlantCLEF submission.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--checkpoint", required=True,
                   help="Path to best.pt from train.py")

    # Data
    p.add_argument("--test-dir", default=DEFAULT_TEST_IMAGES_DIR,
                   help="Directory containing test quadrat images.")

    # Tiling - mode
    p.add_argument(
        "--tile-mode", default="whole", choices=list(TILING_MODES),
        help=(
            "Tiling strategy for test-time inference.\n"
            "  whole               : full image as a single tile\n"
            "  grid_NxN (N=2..8)   : fixed NxN grid (grid_2x2 through grid_8x8)\n"
            "  grid                : generic NxN grid; set N with --grid-size\n"
            "  hstrips_2/3         : 2 or 3 horizontal strips\n"
            "  vstrips_2/3         : 2 or 3 vertical strips\n"
            "  center_crop         : single centred square crop (--tile-size)\n"
            "  five_crop           : center + 4 corners (5 tiles, --tile-size)\n"
            "  sliding             : sliding window (--tile-size, --stride)\n"
            "  multiscale          : whole + grid_2x2 + grid_4x4  [21 tiles]\n"
            "  multiscale_dense    : whole + grids from --scales + opt. sliding"
        ),
    )

    # Tiling - geometry
    p.add_argument("--overlap", type=float, default=0.0,
                   help="Tile overlap ratio for grid/strip modes (e.g. 0.25 = 25%% overlap on each side).")
    p.add_argument("--tile-size", type=int, default=224,
                   help="Tile pixel size for sliding / center_crop / five_crop modes.")
    p.add_argument("--stride", type=int, default=0,
                   help="Sliding-window stride in pixels (0 = tile_size // 2).")
    p.add_argument("--grid-size", type=int, default=None,
                   help="Grid N for the generic 'grid' mode (NxN tiles).")
    p.add_argument("--scales", type=_parse_scales, default=None,
                   metavar="N,N,...",
                   help="Comma-separated grid sizes for multiscale_dense "
                        "(e.g. '1,2,3,4').  Scale 1 is the whole image.")
    p.add_argument("--max-tiles-per-image", type=int, default=None,
                   metavar="N",
                   help="Hard cap on tiles per image; excess tiles are "
                        "dropped deterministically (first N kept). "
                        "Useful for dense/sliding modes to limit memory.")

    # Aggregation
    p.add_argument("--agg-mode", default="max", choices=list(AGG_MODES),
                   help="max = element-wise max | topk_mean = mean of top-k tiles")
    p.add_argument("--topk-agg", type=int, default=5,
                   help="k for topk_mean aggregation.")

    # Inference preprocessing
    p.add_argument("--img-size",     type=int,   default=224)
    p.add_argument("--interp",       default="bicubic",
                   choices=["bicubic", "lanczos", "bilinear"])
    p.add_argument("--margin-crop",  type=float, default=0.0,
                   help="Fraction of shorter side to crop on each border (0 = none).")
    p.add_argument("--jpeg-quality", type=int,   default=0,
                   help="JPEG recompression quality before tiling (0 = none).")
    p.add_argument("--jpeg-subsampling", type=int, default=0,
                   choices=[0, 1, 2],
                   help="JPEG chroma subsampling: 0=4:4:4  1=4:2:2  2=4:2:0")

    # Output
    p.add_argument("--top-n",           type=int, default=20,
                   help="Number of species to include in submission per image.")
    p.add_argument("--tile-batch-size", type=int, default=32,
                   help="Tiles per backbone forward pass.")
    p.add_argument("--output-dir",      default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device",          default="auto")
    p.add_argument("--limit",           type=int, default=0,
                   help="Process only first N images (0 = all; for debugging).")

    # Debug
    p.add_argument("--save-tile-preview", action="store_true",
                   help="Save a PNG showing tile bounding boxes for the first "
                        "test image to {output_dir}/tile_preview.png.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir=str(out_dir))

    logger.info("=" * 60)
    logger.info("BioCLIP 2.5 Linear Probe Inference")
    logger.info(f"  checkpoint       : {args.checkpoint}")
    logger.info(f"  test_dir         : {args.test_dir}")
    logger.info(f"  tile_mode        : {args.tile_mode}")
    logger.info(f"  overlap          : {args.overlap}")
    logger.info(f"  tile_size        : {args.tile_size}")
    logger.info(f"  stride           : {args.stride or 'tile_size//2'}")
    logger.info(f"  grid_size        : {args.grid_size or 'n/a'}")
    logger.info(f"  scales           : {args.scales or 'n/a'}")
    logger.info(f"  max_tiles        : {args.max_tiles_per_image or 'unlimited'}")
    logger.info(f"  agg_mode         : {args.agg_mode}  (topk={args.topk_agg})")
    logger.info(f"  interp           : {args.interp}")
    logger.info(f"  margin_crop      : {args.margin_crop}")
    logger.info(f"  jpeg_quality     : {args.jpeg_quality or 'none'}")
    logger.info(f"  top_n            : {args.top_n}")
    logger.info(f"  device           : {device}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model, idx_to_species = load_bioclip25_probe(args.checkpoint, device=device)
    model.eval()
    num_classes = len(idx_to_species)
    logger.info(f"  {num_classes:,} classes")

    # ------------------------------------------------------------------
    # Inference preprocessing
    # ------------------------------------------------------------------
    preproc = build_inference_preprocessor(args)
    logger.info(f"  preproc: {preproc}")

    # ------------------------------------------------------------------
    # Test images
    # ------------------------------------------------------------------
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        sys.exit(f"ERROR: test directory not found: {test_dir}")

    test_images = sorted([p for p in test_dir.rglob("*") if p.suffix in _IMAGE_EXTS])
    if not test_images:
        sys.exit(f"ERROR: no images found under {test_dir}")

    if args.limit > 0:
        test_images = test_images[: args.limit]
        logger.info(f"Limiting to first {args.limit} images")

    logger.info(f"Found {len(test_images):,} test images")

    # ------------------------------------------------------------------
    # Shared tiling kwargs (passed through to tile_image every iteration)
    # ------------------------------------------------------------------
    tiling_kwargs = dict(
        mode=args.tile_mode,
        overlap_ratio=args.overlap,
        tile_size=args.tile_size,
        stride=args.stride,
        grid_size=args.grid_size,
        scales=args.scales,
        max_tiles=args.max_tiles_per_image,
    )

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    submission_rows: list[dict] = []
    scored_rows:     list[dict] = []
    errors   = 0
    t_start  = time.perf_counter()
    first_logged = False

    for i, img_path in enumerate(test_images):
        quadrat_id = img_path.stem
        try:
            image = Image.open(img_path).convert("RGB")

            tiles = tile_image(image, **tiling_kwargs)

            if not first_logged:
                # Detailed first-image diagnostics
                pairs = _tile_image_with_info(image, **tiling_kwargs)
                logger.info(
                    f"First image {img_path.name}: "
                    f"size={image.size}  tile_mode={args.tile_mode}  "
                    f"n_tiles={len(tiles)}"
                )
                preview_boxes = pairs[:5]
                for info, _ in preview_boxes:
                    logger.info(
                        f"  tile[{info.tile_index}] "
                        f"({info.left},{info.top})-({info.right},{info.bottom})  "
                        f"[{info.right - info.left}x{info.bottom - info.top}px]"
                        f"  mode={info.mode_name}"
                    )
                if len(pairs) > 5:
                    logger.info(f"  ... ({len(pairs) - 5} more tiles)")

                if args.save_tile_preview:
                    preview_path = str(out_dir / "tile_preview.png")
                    save_tile_preview(
                        image,
                        output_path=preview_path,
                        **tiling_kwargs,
                    )

                first_logged = True

            feats = encode_tiles(
                backbone_encode_fn=model.encode,
                preprocess=preproc,
                tiles=tiles,
                device=device,
                batch_size=args.tile_batch_size,
            )

            tile_logits = classify_tiles(
                head=model.head, tile_features=feats, device=device
            )

            img_logits = aggregate_logits(
                tile_logits, mode=args.agg_mode, topk=args.topk_agg
            )

            pred_species, pred_scores = topk_predictions(
                img_logits, idx_to_species, top_n=args.top_n
            )

        except Exception as exc:
            logger.warning(f"Error on {img_path.name}: {exc}")
            errors += 1
            pred_species = []
            pred_scores  = []

        species_ids_str = "[" + ", ".join(
            str(int(s)) for s in pred_species if s.isdigit()
        ) + "]"
        submission_rows.append({
            "quadrat_id": quadrat_id,
            "species_ids": species_ids_str,
        })

        for rank, (sid, score) in enumerate(
            zip(pred_species, pred_scores), start=1
        ):
            scored_rows.append({
                "quadrat_id": quadrat_id,
                "rank":       rank,
                "species_id": sid,
                "score":      score,
            })

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            rate    = (i + 1) / elapsed
            logger.info(
                f"  [{i+1:>5}/{len(test_images)}]  "
                f"{elapsed:.0f}s elapsed  {rate:.1f} img/s  errors={errors}"
            )

    total_secs = time.perf_counter() - t_start
    logger.info(
        f"Inference complete: {len(test_images):,} images  "
        f"{total_secs:.0f}s  "
        f"({len(test_images)/max(total_secs,1):.1f} img/s)  "
        f"errors={errors}"
    )

    # ------------------------------------------------------------------
    # submission.csv
    # ------------------------------------------------------------------
    sub_path = out_dir / "submission.csv"
    with open(sub_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["quadrat_id", "species_ids"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(submission_rows)
    logger.info(f"Submission saved: {sub_path}  ({len(submission_rows):,} rows)")

    # ------------------------------------------------------------------
    # predictions_scored.csv
    # ------------------------------------------------------------------
    scored_path = out_dir / "predictions_scored.csv"
    if scored_rows:
        with open(scored_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["quadrat_id", "rank", "species_id", "score"],
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            writer.writerows(scored_rows)
        logger.info(f"Scored predictions saved: {scored_path}")

    # ------------------------------------------------------------------
    # run_config.json + summary.json
    # ------------------------------------------------------------------
    run_config = {
        **vars(args),
        "device":       device,
        "n_classes":    num_classes,
        "n_images":     len(test_images),
        "total_secs":   round(total_secs, 1),
        "errors":       errors,
        "preprocessor": repr(preproc),
    }
    save_json(run_config, str(out_dir / "run_config.json"))
    save_json(
        {
            "n_images":       len(test_images),
            "n_predictions":  len(submission_rows),
            "errors":         errors,
            "total_secs":     round(total_secs, 1),
            "images_per_sec": round(len(test_images) / max(total_secs, 1), 2),
            "tile_mode":      args.tile_mode,
            "agg_mode":       args.agg_mode,
        },
        str(out_dir / "summary.json"),
    )


if __name__ == "__main__":
    main()
