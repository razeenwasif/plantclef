"""
Multi-model BioCLIP zero-shot inference with enriched species prompts.

Supports BioCLIP 1, 2, and 2.5 with SAHI-style tile max-pool aggregation
and prompt ensembling over multiple text templates per species.

Usage examples
--------------
# BioCLIP 1, scientific prompts only
python run_inference.py \\
    --model-name hf-hub:imageomics/bioclip \\
    --species-csv path/to/species_lookup_with_gbif_cleaned_names.csv \\
    --images-root path/to/images \\
    --prompt-mode scientific

# BioCLIP 2, all prompts
python run_inference.py \\
    --model-name hf-hub:imageomics/bioclip-2 \\
    --prompt-mode all

# BioCLIP 2.5 (larger model, use lower batch size)
python run_inference.py \\
    --model-name hf-hub:imageomics/bioclip-2.5-vith14 \\
    --batch-size 16 \\
    --prompt-mode all

# Smoke test: only process 5 images
python run_inference.py --limit 5

Output directory structure
--------------------------
{output_dir}/{run_slug}/
    run_config.json         CLI args + runtime metadata
    prompt_table.csv        species_id, n_prompts, prompts (JSON list)
    prompt_summary.json     aggregate stats about prompts
    submission.csv          PlantCLEF format: quadrat_id, species_ids
    predictions_topk.csv    image_name, rank 1..k, species_id, score
    summary.json            timing, counts, any eval metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import open_clip
from PIL import Image

# Local modules
from utils import (
    get_tiles,
    encode_text_features_from_prompts,
    encode_image_tiles,
    compute_tile_logits,
    aggregate_tile_logits,
    image_top_k,
)
from prompt_builder import (
    load_species_labels,
    build_all_prompts,
    prompt_stats,
    PROMPT_MODES,
    SpeciesLabel,
)

# ---------------------------------------------------------------------------
# Known model defaults
# ---------------------------------------------------------------------------

# Batch sizes that work on ~16 GB VRAM. Can be overridden with --batch-size.
_MODEL_DEFAULT_BATCH = {
    "hf-hub:imageomics/bioclip":          64,
    "hf-hub:imageomics/bioclip-2":        64,
    "hf-hub:imageomics/bioclip-2.5-vith14": 16,
}

DEFAULT_SPECIES_CSV = (
    "/root/workspace/PlantCLEF2026/src_experiments/"
    "002_bioclip_tile_zero_shot_v2/data/"
    "species_lookup_with_gbif_cleaned_names.csv"
)
DEFAULT_IMAGES_ROOT = "/workspace/plantclef/kaggle_uploads/test/images"
DEFAULT_OUTPUT_DIR = "./outputs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-model BioCLIP zero-shot inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model-name",
        default="hf-hub:imageomics/bioclip",
        help="OpenCLIP model identifier. One of: "
             "hf-hub:imageomics/bioclip, "
             "hf-hub:imageomics/bioclip-2, "
             "hf-hub:imageomics/bioclip-2.5-vith14",
    )

    # Data paths
    parser.add_argument("--species-csv", default=DEFAULT_SPECIES_CSV,
                        help="Path to enriched species CSV (species_lookup_with_gbif_cleaned_names.csv)")
    parser.add_argument("--images-root", default=DEFAULT_IMAGES_ROOT,
                        help="Directory containing test images (flat, jpg/jpeg/png)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Root output directory. Outputs go under {output_dir}/{run_slug}/")

    # Prompt configuration
    parser.add_argument(
        "--prompt-mode",
        default="scientific",
        choices=list(PROMPT_MODES),
        help=(
            "scientific: family A only | "
            "scientific_common: A+B+D | "
            "scientific_family: A+C | "
            "all: A+B+C+D"
        ),
    )
    parser.add_argument("--max-common-names", type=int, default=3,
                        help="Max extra English common names per species (beyond primary)")
    parser.add_argument("--max-synonyms", type=int, default=2,
                        help="Max GBIF synonyms to include as additional scientific names")

    # Tiling
    parser.add_argument("--tile-size", type=int, default=224,
                        help="Square tile side length in pixels")
    parser.add_argument("--tile-overlap", type=int, default=112,
                        help="Overlap between adjacent tiles in pixels (stride = tile_size - tile_overlap)")

    # Inference
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Image tile encoding batch size. Defaults to model-specific value.")
    parser.add_argument("--text-batch-size", type=int, default=256,
                        help="Text prompt encoding batch size")
    parser.add_argument("--device", default="auto",
                        help="Device: 'auto', 'cuda', 'cpu', 'cuda:0', etc.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top species to predict per image")

    # Limiting / debug
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N images (for smoke tests)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def make_run_slug(model_name: str, prompt_mode: str) -> str:
    """Derive a filesystem-safe run identifier from model + prompt mode."""
    model_slug = model_name.split("/")[-1].replace(".", "-")
    return f"{model_slug}_{prompt_mode}"


def find_images(images_root: str) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths: list[Path] = []
    for ext in exts:
        paths.extend(Path(images_root).glob(ext))
    return sorted(set(paths))


def load_model(model_name: str, device: str):
    """Load OpenCLIP model, preprocessing transform, and tokenizer."""
    print(f"  Loading model '{model_name}' ...")
    model, _, transform = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    return model, transform, tokenizer


def save_prompt_table(
    out_dir: Path,
    labels: list[SpeciesLabel],
    prompt_lists: list[list[str]],
) -> None:
    """Save per-species prompt table to CSV."""
    path = out_dir / "prompt_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species_id", "canonical_scientific", "primary_common", "n_prompts", "prompts"])
        for label, prompts in zip(labels, prompt_lists):
            writer.writerow([
                label.species_id,
                label.canonical_scientific,
                label.primary_common,
                len(prompts),
                json.dumps(prompts),
            ])
    print(f"  Prompt table saved: {path}")


def save_submission_csv(out_dir: Path, rows: list[dict]) -> None:
    path = out_dir / "submission.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quadrat_id", "species_ids"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Submission CSV saved: {path}  ({len(rows)} rows)")


def save_topk_csv(out_dir: Path, topk_rows: list[dict]) -> None:
    path = out_dir / "predictions_topk.csv"
    fieldnames = ["image_name", "rank", "species_id", "species_name", "logit_score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topk_rows)
    print(f"  Top-k predictions saved: {path}  ({len(topk_rows)} rows)")


# ---------------------------------------------------------------------------
# Per-image inference
# ---------------------------------------------------------------------------

def process_image(
    image_path: Path,
    model,
    transform,
    text_feats: torch.Tensor,
    species_ids: list[str],
    logit_scale: float,
    tile_size: int,
    stride: int,
    batch_size: int,
    top_k: int,
    device: str,
) -> tuple[list[str], list[float]]:
    """Run tiled zero-shot inference on one image. Returns (top_ids, top_scores)."""
    image = Image.open(image_path).convert("RGB")
    tiles, _ = get_tiles(image, tile_size, stride)
    image_feats = encode_image_tiles(model, transform, tiles, device, batch_size=batch_size)
    image_feats = image_feats.to(device)
    tile_logits = compute_tile_logits(image_feats, text_feats, logit_scale)
    image_logits = aggregate_tile_logits(tile_logits)
    return image_top_k(image_logits, species_ids, k=top_k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    stride = args.tile_size - args.tile_overlap
    if stride <= 0:
        raise ValueError(
            f"tile-overlap ({args.tile_overlap}) must be less than tile-size ({args.tile_size})"
        )

    batch_size = args.batch_size or _MODEL_DEFAULT_BATCH.get(args.model_name, 64)
    run_slug = make_run_slug(args.model_name, args.prompt_mode)
    out_dir = Path(args.output_dir) / run_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Run: {run_slug}")
    print(f"Device: {device}  |  tile_size={args.tile_size}  stride={stride}  top_k={args.top_k}")
    print(f"Output dir: {out_dir}")
    print(f"{'='*60}\n")

    # ----- Save run config -----
    config = vars(args).copy()
    config.update({
        "device_resolved": device,
        "stride": stride,
        "batch_size_used": batch_size,
        "run_slug": run_slug,
        "output_path": str(out_dir),
    })
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ----- Load species labels -----
    print(f"Loading species from: {args.species_csv}")
    if not Path(args.species_csv).exists():
        sys.exit(f"ERROR: species CSV not found: {args.species_csv}")
    labels = load_species_labels(
        args.species_csv,
        max_common_names=args.max_common_names,
        max_synonyms=args.max_synonyms,
    )
    species_ids = [lbl.species_id for lbl in labels]
    id_to_name = {lbl.species_id: lbl.canonical_scientific for lbl in labels}
    print(f"  {len(labels)} species loaded")

    # ----- Build prompts -----
    print(f"\nBuilding prompts (mode='{args.prompt_mode}') ...")
    prompt_lists = build_all_prompts(labels, args.prompt_mode)
    stats = prompt_stats(prompt_lists)
    print(f"  {stats['total_prompts']} total prompts  "
          f"({stats['min_per_species']}-{stats['max_per_species']} per species, "
          f"avg {stats['avg_per_species']})")
    save_prompt_table(out_dir, labels, prompt_lists)
    with open(out_dir / "prompt_summary.json", "w") as f:
        json.dump({**stats, "prompt_mode": args.prompt_mode}, f, indent=2)

    # ----- Load model -----
    print(f"\nLoading model ...")
    model, transform, tokenizer = load_model(args.model_name, device)
    logit_scale = model.logit_scale.exp().item()
    print(f"  logit_scale: {logit_scale:.4f}")

    # ----- Encode text features -----
    print(f"\nEncoding species text features (batch_size={args.text_batch_size}) ...")
    t0_text = time.perf_counter()
    text_feats = encode_text_features_from_prompts(
        model, tokenizer, prompt_lists, device, batch_size=args.text_batch_size
    )
    text_feats = text_feats.to(device)
    text_encoding_secs = time.perf_counter() - t0_text
    print(f"  text_feats: {text_feats.shape}  ({text_encoding_secs:.1f}s)")

    # ----- Find images -----
    if not Path(args.images_root).exists():
        sys.exit(f"ERROR: images-root not found: {args.images_root}")
    image_paths = find_images(args.images_root)
    if args.limit:
        image_paths = image_paths[: args.limit]
    print(f"\nFound {len(image_paths)} images in: {args.images_root}")

    if not image_paths:
        sys.exit("ERROR: no images found - check --images-root")

    # ----- Run inference -----
    print(f"Running inference (batch_size={batch_size}) ...\n")
    submission_rows: list[dict] = []
    topk_rows: list[dict] = []
    t0_infer = time.perf_counter()
    errors = 0

    for i, image_path in enumerate(image_paths):
        quadrat_id = image_path.stem
        print(f"  [{i+1:>5}/{len(image_paths)}] {quadrat_id} ...", end=" ", flush=True)

        try:
            top_ids, top_scores = process_image(
                image_path, model, transform, text_feats,
                species_ids, logit_scale,
                args.tile_size, stride, batch_size, args.top_k, device,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            errors += 1
            continue

        # Submission row
        ids_str = "[" + ", ".join(top_ids) + "]"
        submission_rows.append({"quadrat_id": quadrat_id, "species_ids": ids_str})

        # Top-k detail rows
        for rank, (sid, score) in enumerate(zip(top_ids, top_scores), start=1):
            topk_rows.append({
                "image_name": quadrat_id,
                "rank": rank,
                "species_id": sid,
                "species_name": id_to_name.get(sid, ""),
                "logit_score": f"{score:.6f}",
            })

        print(f"{ids_str[:60]}")

    total_infer_secs = time.perf_counter() - t0_infer
    n_processed = len(submission_rows)

    # ----- Save outputs -----
    print()
    save_submission_csv(out_dir, submission_rows)
    save_topk_csv(out_dir, topk_rows)

    # ----- Save summary -----
    summary = {
        "run_slug": run_slug,
        "model_name": args.model_name,
        "prompt_mode": args.prompt_mode,
        "n_species": len(labels),
        **stats,
        "n_images_found": len(image_paths),
        "n_images_processed": n_processed,
        "n_errors": errors,
        "top_k": args.top_k,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "stride": stride,
        "batch_size": batch_size,
        "text_encoding_secs": round(text_encoding_secs, 2),
        "inference_total_secs": round(total_infer_secs, 2),
        "inference_per_image_secs": round(total_infer_secs / max(n_processed, 1), 3),
        "device": device,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {out_dir / 'summary.json'}")

    # ----- Final report -----
    print(f"\n{'='*60}")
    print(f"Run complete: {run_slug}")
    print(f"  Images processed : {n_processed} / {len(image_paths)}")
    if errors:
        print(f"  Errors           : {errors}")
    print(f"  Text encoding    : {text_encoding_secs:.1f}s")
    print(f"  Inference        : {total_infer_secs:.1f}s  "
          f"({total_infer_secs / max(n_processed, 1):.3f}s/image)")
    print(f"  Outputs          : {out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
