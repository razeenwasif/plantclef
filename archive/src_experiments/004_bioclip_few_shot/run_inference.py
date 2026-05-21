"""
Few-shot tile-based inference on PlantCLEF quadrat images.

Pipeline
--------
For each test image:
  1. Generate overlapping tiles.
  2. Encode each tile with the BioCLIP model.
  3. Score tiles against the species support bank (prototype or KNN mode).
  4. Optionally fuse with text-prompt similarity (controlled by --text-alpha).
  5. Aggregate tile scores to an image-level multi-label prediction.
  6. Apply threshold / top-N to produce the final species list.

Usage examples
--------------
# Minimal (uses default support bank, bioclip, prototype mode)
python run_inference.py \\
    --bank-dir ./cache/bioclip_k5_random_seed42 \\
    --images-root /workspace/plantclef/kaggle_uploads/test/images

# KNN mode, top-20 species, BioCLIP 2
python run_inference.py \\
    --bank-dir ./cache/bioclip-2_k10_random_seed42 \\
    --images-root /path/to/test/images \\
    --scoring-mode knn \\
    --top-n 20

# With text fusion (alpha=0.7 image + 0.3 text)
python run_inference.py \\
    --bank-dir ./cache/bioclip_k5_random_seed42 \\
    --images-root /path/to/test/images \\
    --text-alpha 0.7 \\
    --species-csv /path/to/species_lookup_with_gbif_cleaned_names.csv

# Smoke test: process first 5 images only
python run_inference.py \\
    --bank-dir ./cache/bioclip_k5_random_seed42 \\
    --images-root /path/to/test/images \\
    --limit 5

Output directory structure
--------------------------
{output-dir}/{run_slug}/
    run_config.json        CLI args + runtime metadata
    submission.csv         PlantCLEF format: quadrat_id, species_ids
    predictions_scored.csv image_name, rank, species_id, score
    summary.json           timing, counts, parameters
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

# Local modules
from models import load_model, resolve_device, default_batch_size, resolve_model_name
from tiling import get_tiles, encode_image_tiles
from few_shot import SupportBank, score_prototype, score_knn, fuse_scores
from aggregation import aggregate_scores, apply_threshold, AGG_MODES
from dataset import DEFAULT_SPECIES_CSV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default test image root - override with --images-root
DEFAULT_TEST_IMAGES = "/workspace/plantclef/kaggle_uploads/test/images"
DEFAULT_OUTPUT_DIR = "./outputs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BioCLIP few-shot inference on quadrat images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Support bank
    p.add_argument(
        "--bank-dir", required=True,
        help="Directory containing a built support bank (output of build_support_bank.py).",
    )

    # Model (must match the bank)
    p.add_argument("--model-name", default=None,
                   help="Model name. Defaults to the model recorded in the bank's metadata.")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Tile encoding batch size. Defaults to model-specific value.")

    # Data
    p.add_argument("--images-root", default=DEFAULT_TEST_IMAGES,
                   help="Directory of test/quadrat images.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Root output directory.")

    # Scoring
    p.add_argument(
        "--scoring-mode", default="prototype", choices=["prototype", "knn"],
        help="prototype: compare tile to mean support embedding. "
             "knn: compare tile to all support embeddings (aggregated per species).",
    )

    # Tiling
    p.add_argument("--tile-size", type=int, default=224)
    p.add_argument("--tile-overlap", type=int, default=112,
                   help="Pixel overlap between adjacent tiles (stride = tile_size - overlap).")

    # Aggregation
    p.add_argument(
        "--agg-mode", default="max", choices=list(AGG_MODES),
        help="Tile-to-image aggregation mode.",
    )
    p.add_argument("--agg-top-m", type=int, default=3,
                   help="Number of tiles for mean_top_m mode.")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Global score threshold below which species are excluded.")
    p.add_argument("--top-n", type=int, default=5,
                   help="Maximum number of species to predict per image (0 = no cap).")
    p.add_argument(
        "--per-species-threshold-file", default=None,
        help="Optional JSON file mapping species_id -> threshold override.",
    )

    # Optional text fusion
    p.add_argument(
        "--text-alpha", type=float, default=1.0,
        help=(
            "Weight for image few-shot scores in fusion: "
            "final = alpha * image + (1-alpha) * text. "
            "1.0 = image only (no text). Requires --species-csv."
        ),
    )
    p.add_argument("--species-csv", default=DEFAULT_SPECIES_CSV,
                   help="Enriched species CSV for text prompt building (used when --text-alpha < 1.0).")
    p.add_argument("--text-batch-size", type=int, default=256,
                   help="Batch size for text encoding (only used when text fusion is active).")

    # Limit / debug
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N images.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_images(images_root: str) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths: list[Path] = []
    for ext in exts:
        paths.extend(Path(images_root).glob(ext))
    return sorted(set(paths))


def load_bank_metadata(bank_dir: str) -> dict:
    meta_path = Path(bank_dir) / "bank_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def build_text_features(
    model,
    tokenizer,
    species_ids: list[str],
    species_csv: str,
    text_batch_size: int,
    device: str,
) -> torch.Tensor:
    """Build text prototype embeddings for text-fusion mode."""
    from pathlib import Path as _Path
    import sys as _sys

    # Reuse the prompt builder from the zero-shot v2 experiment
    v2_dir = (
        _Path(__file__).parent.parent / "002_bioclip_tile_zero_shot_v2"
    )
    if str(v2_dir) not in _sys.path:
        _sys.path.insert(0, str(v2_dir))

    try:
        from prompt_builder import load_species_labels, build_all_prompts
        from utils import encode_text_features_from_prompts
    except ImportError as exc:
        logger.error(
            f"Text fusion requires the prompt builder from 002_bioclip_tile_zero_shot_v2. "
            f"Could not import: {exc}"
        )
        raise

    labels = load_species_labels(species_csv)
    # Keep only species that are in the bank
    id_set = set(species_ids)
    labels = [lbl for lbl in labels if lbl.species_id in id_set]

    prompt_lists = build_all_prompts(labels, mode="scientific")
    label_ids = [lbl.species_id for lbl in labels]

    text_feats = encode_text_features_from_prompts(
        model, tokenizer, prompt_lists, device, batch_size=text_batch_size
    )
    # Map back to the bank's species ordering
    id_to_feat = {sid: text_feats[i] for i, sid in enumerate(label_ids)}
    ordered = torch.stack(
        [id_to_feat.get(sid, torch.zeros(text_feats.shape[1])) for sid in species_ids]
    )
    return F.normalize(ordered, dim=-1)


def make_run_slug(bank_dir: str, scoring_mode: str, agg_mode: str, text_alpha: float) -> str:
    bank_slug = Path(bank_dir).name
    slug = f"{bank_slug}_{scoring_mode}_{agg_mode}"
    if text_alpha < 1.0:
        slug += f"_textalpha{text_alpha:.2f}"
    return slug


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

def process_image(
    image_path: Path,
    model,
    transform,
    bank: SupportBank,
    scoring_mode: str,
    agg_mode: str,
    agg_top_m: int,
    tile_size: int,
    stride: int,
    batch_size: int,
    top_n: int,
    threshold: float,
    per_species_thresholds: dict[str, float] | None,
    device: str,
    text_feats: torch.Tensor | None = None,
    text_alpha: float = 1.0,
) -> tuple[list[str], list[float]]:
    """Run few-shot inference on a single image. Returns (pred_ids, pred_scores)."""
    from PIL import Image as _PIL
    image = _PIL.open(image_path).convert("RGB")
    tiles, _ = get_tiles(image, tile_size, stride)

    tile_emb = encode_image_tiles(model, transform, tiles, device, batch_size=batch_size)
    # tile_emb: (n_tiles, dim), on CPU, L2-normalised

    # Score tiles against support bank
    if scoring_mode == "prototype":
        tile_scores = score_prototype(tile_emb, bank, device=device)
    elif scoring_mode == "knn":
        tile_scores = score_knn(tile_emb, bank, device=device)
    else:
        raise ValueError(f"Unknown scoring mode: {scoring_mode}")

    # Optional text fusion
    if text_feats is not None and text_alpha < 1.0:
        # Build text-side tile scores (cosine sim of each tile to text prototypes)
        protos_text = text_feats.to(device)
        q = tile_emb.to(device)
        text_tile_scores = (q @ protos_text.T).cpu()
        tile_scores = fuse_scores(tile_scores, text_tile_scores, alpha=text_alpha)

    # Aggregate tiles → image-level score
    image_scores = aggregate_scores(tile_scores, mode=agg_mode, top_m=agg_top_m)

    return apply_threshold(
        image_scores,
        bank.species_ids,
        threshold=threshold,
        top_n=top_n,
        per_species_thresholds=per_species_thresholds,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    stride = args.tile_size - args.tile_overlap
    if stride <= 0:
        sys.exit(
            f"ERROR: tile-overlap ({args.tile_overlap}) must be < tile-size ({args.tile_size})"
        )

    # ------------------------------------------------------------------
    # Load support bank
    # ------------------------------------------------------------------
    bank_dir = Path(args.bank_dir)
    bank_file = bank_dir / "bank.pt"
    if not bank_file.exists():
        sys.exit(
            f"ERROR: no bank.pt found in {bank_dir}. "
            "Run build_support_bank.py first."
        )
    logger.info(f"Loading support bank from {bank_file} ...")
    bank = SupportBank.load(bank_file)
    logger.info(f"  {len(bank.species_ids)} species loaded")

    # ------------------------------------------------------------------
    # Determine model name
    # ------------------------------------------------------------------
    bank_meta = load_bank_metadata(args.bank_dir)
    if args.model_name:
        model_name = args.model_name
    elif "model_name" in bank_meta:
        model_name = bank_meta["model_name"]
        logger.info(f"Using model from bank metadata: {model_name}")
    else:
        model_name = "bioclip"
        logger.warning(f"No model name found; defaulting to '{model_name}'")

    batch_size = args.batch_size or default_batch_size(model_name)

    run_slug = make_run_slug(args.bank_dir, args.scoring_mode, args.agg_mode, args.text_alpha)
    out_dir = Path(args.output_dir) / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BioCLIP Few-Shot Inference")
    print(f"  run_slug    : {run_slug}")
    print(f"  model       : {model_name}")
    print(f"  scoring     : {args.scoring_mode}")
    print(f"  agg_mode    : {args.agg_mode}")
    print(f"  threshold   : {args.threshold}")
    print(f"  top_n       : {args.top_n}")
    print(f"  tile_size   : {args.tile_size}  stride={stride}")
    print(f"  device      : {device}")
    print(f"  output dir  : {out_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info("Loading model ...")
    model, transform, tokenizer = load_model(model_name, device)

    # ------------------------------------------------------------------
    # Optional text features for fusion
    # ------------------------------------------------------------------
    text_feats = None
    if args.text_alpha < 1.0:
        logger.info(
            f"Building text features for fusion (alpha={args.text_alpha}) ..."
        )
        try:
            text_feats = build_text_features(
                model, tokenizer, bank.species_ids,
                args.species_csv, args.text_batch_size, device,
            )
            logger.info(f"  text_feats: {text_feats.shape}")
        except Exception as exc:
            logger.error(f"Text fusion disabled due to error: {exc}")
            text_feats = None

    # ------------------------------------------------------------------
    # Per-species threshold file
    # ------------------------------------------------------------------
    per_species_thresholds = None
    if args.per_species_threshold_file:
        thr_path = Path(args.per_species_threshold_file)
        if thr_path.exists():
            with open(thr_path) as f:
                per_species_thresholds = json.load(f)
            logger.info(
                f"Loaded per-species thresholds: {len(per_species_thresholds)} entries"
            )
        else:
            logger.warning(
                f"Per-species threshold file not found: {thr_path} - using global threshold"
            )

    # ------------------------------------------------------------------
    # Find test images
    # ------------------------------------------------------------------
    images_root = Path(args.images_root)
    if not images_root.exists():
        sys.exit(f"ERROR: images-root not found: {images_root}")
    image_paths = find_images(str(images_root))
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        sys.exit(f"ERROR: no images found in {images_root}")
    logger.info(f"Found {len(image_paths)} test images")

    # ------------------------------------------------------------------
    # Save run config
    # ------------------------------------------------------------------
    config = {
        **vars(args),
        "model_name_resolved": resolve_model_name(model_name),
        "device_resolved": device,
        "stride": stride,
        "batch_size_used": batch_size,
        "run_slug": run_slug,
        "n_species_in_bank": len(bank.species_ids),
        "n_images": len(image_paths),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    logger.info(f"Running inference (mode={args.scoring_mode}) ...")
    submission_rows: list[dict] = []
    scored_rows: list[dict] = []
    t0 = time.perf_counter()
    errors = 0

    for i, image_path in enumerate(image_paths):
        quadrat_id = image_path.stem
        print(
            f"  [{i+1:>5}/{len(image_paths)}] {quadrat_id} ...",
            end=" ", flush=True,
        )
        try:
            pred_ids, pred_scores = process_image(
                image_path=image_path,
                model=model,
                transform=transform,
                bank=bank,
                scoring_mode=args.scoring_mode,
                agg_mode=args.agg_mode,
                agg_top_m=args.agg_top_m,
                tile_size=args.tile_size,
                stride=stride,
                batch_size=batch_size,
                top_n=args.top_n,
                threshold=args.threshold,
                per_species_thresholds=per_species_thresholds,
                device=device,
                text_feats=text_feats,
                text_alpha=args.text_alpha,
            )
        except Exception as exc:
            logger.error(f"Error processing {image_path.name}: {exc}")
            errors += 1
            # Emit empty prediction row so quadrat is still in submission
            pred_ids, pred_scores = [], []

        ids_str = "[" + ", ".join(pred_ids) + "]"
        submission_rows.append({"quadrat_id": quadrat_id, "species_ids": ids_str})
        for rank, (sid, score) in enumerate(zip(pred_ids, pred_scores), start=1):
            scored_rows.append({
                "image_name": quadrat_id,
                "rank": rank,
                "species_id": sid,
                "score": f"{score:.6f}",
            })
        print(f"{len(pred_ids)} species  [{ids_str[:50]}...]")

    total_secs = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    sub_path = out_dir / "submission.csv"
    with open(sub_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["quadrat_id", "species_ids"], quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(submission_rows)
    logger.info(f"Submission CSV saved: {sub_path}  ({len(submission_rows)} rows)")

    scored_path = out_dir / "predictions_scored.csv"
    with open(scored_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_name", "rank", "species_id", "score"]
        )
        writer.writeheader()
        writer.writerows(scored_rows)
    logger.info(f"Scored predictions saved: {scored_path}  ({len(scored_rows)} rows)")

    summary = {
        "run_slug": run_slug,
        "model_name": model_name,
        "scoring_mode": args.scoring_mode,
        "agg_mode": args.agg_mode,
        "threshold": args.threshold,
        "top_n": args.top_n,
        "tile_size": args.tile_size,
        "stride": stride,
        "n_species_in_bank": len(bank.species_ids),
        "n_images": len(image_paths),
        "n_processed": len(submission_rows) - errors,
        "n_errors": errors,
        "total_secs": round(total_secs, 2),
        "per_image_secs": round(total_secs / max(len(submission_rows), 1), 3),
        "device": device,
        "text_alpha": args.text_alpha,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Inference complete: {run_slug}")
    print(f"  Images processed : {len(submission_rows)} / {len(image_paths)}")
    if errors:
        print(f"  Errors           : {errors}")
    print(f"  Inference time   : {total_secs:.1f}s  "
          f"({total_secs / max(len(submission_rows), 1):.3f}s/image)")
    print(f"  Submission CSV   : {sub_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
