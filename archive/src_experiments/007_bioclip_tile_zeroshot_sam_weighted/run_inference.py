"""
BioCLIP zero-shot inference with SAM-based vegetation-aware tile weighting.

Experiment 007: extends the 002 baseline by scoring each image tile for
vegetation content (via RGB ExG or SAM masks) and using those scores to
softly reweight tile logits before image-level aggregation.

Key additions over 002
----------------------
  - Vegetation scoring per tile (rgb or sam method)
  - Configurable soft tile weighting: w_i = clip(alpha + beta*veg_i, min, max)
  - Multiple aggregation methods: max, mean, topk_mean, weighted_mean, weighted_topk_mean
  - Rich per-image and dataset-level visualizations
  - Per-tile metadata CSV (tile coords, veg_ratio, weight, top-1 species/score)

Usage examples
--------------
# Baseline (identical to 002 behaviour)
python run_inference.py --aggregation max --scoring rgb --limit 5

# SAM-weighted mean (needs SAM checkpoint)
python run_inference.py \\
    --scoring sam \\
    --sam-checkpoint ./checkpoints/sam_vit_b_01ec64.pth \\
    --aggregation weighted_mean \\
    --n-visualize 10 \\
    --limit 20

# Quick comparison: run both modes via --aggregation
python run_inference.py --aggregation weighted_mean --scoring rgb --n-visualize 5

Output structure
----------------
{output_dir}/{run_slug}/
    run_config.json
    prompt_table.csv
    prompt_summary.json
    submission.csv
    predictions_topk.csv
    tile_metadata.csv          <- NEW: per-tile veg/weight/species data
    comparison_{method}.csv    <- NEW: baseline vs weighted comparison
    summary.json
    visualizations/
        {image_id}/
            A_tile_overview.png
            B_veg_heatmap.png
            C_sam_tile??.png   (only when SAM masks available)
            D_tile_diagnostics.png
            E_prediction_comparison.png
        summary_veg_stats.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image

# Local modules
from utils import (
    get_tiles,
    encode_text_features_from_prompts,
    encode_image_tiles,
    compute_tile_logits,
    image_top_k,
)
from prompt_builder import (
    load_species_labels,
    build_all_prompts,
    prompt_stats,
    PROMPT_MODES,
    SpeciesLabel,
)
from aggregation import aggregate, AGGREGATION_METHODS
from sam_weighting import (
    load_sam_generator,
    score_tiles,
    compute_tile_weights,
)
from visualization import (
    save_image_visualizations,
    save_summary_statistics,
)
from config import (
    MODEL_DEFAULT_BATCH,
    DEFAULT_MODEL_NAME,
    DEFAULT_TILE_SIZE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_PROMPT_MODE,
    DEFAULT_TEXT_BATCH,
    DEFAULT_SPECIES_CSV,
    DEFAULT_IMAGES_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAM_CHECKPOINT,
    DEFAULT_SAM_MODEL_TYPE,
    DEFAULT_EXG_THRESHOLD,
    DEFAULT_SAM_MIN_GREENNESS,
    DEFAULT_WEIGHT_ALPHA,
    DEFAULT_WEIGHT_BETA,
    DEFAULT_WEIGHT_MIN,
    DEFAULT_WEIGHT_MAX,
    DEFAULT_AGGREGATION,
    DEFAULT_TOPK_TILES,
    DEFAULT_SCORING,
    DEFAULT_N_VISUALIZE,
    SCORING_METHODS,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BioCLIP zero-shot inference with SAM vegetation weighting (exp 007)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- BioCLIP model ----
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                   help="OpenCLIP model: bioclip | bioclip-2 | bioclip-2.5-vith14")
    p.add_argument("--device", default="auto",
                   help="Device: auto | cuda | cpu | cuda:0")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Tile encoding batch size (model-specific default if omitted)")
    p.add_argument("--text-batch-size", type=int, default=DEFAULT_TEXT_BATCH)

    # ---- Data paths ----
    p.add_argument("--species-csv", default=DEFAULT_SPECIES_CSV)
    p.add_argument("--images-root", default=DEFAULT_IMAGES_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    # ---- Prompts ----
    p.add_argument("--prompt-mode", default=DEFAULT_PROMPT_MODE, choices=list(PROMPT_MODES))
    p.add_argument("--max-common-names", type=int, default=3)
    p.add_argument("--max-synonyms", type=int, default=2)

    # ---- Tiling ----
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--tile-overlap", type=int, default=DEFAULT_TILE_OVERLAP)

    # ---- Aggregation ----
    p.add_argument("--aggregation", default=DEFAULT_AGGREGATION,
                   choices=list(AGGREGATION_METHODS),
                   help=(
                       "max=baseline max-pool | mean=uniform mean | "
                       "topk_mean=mean of top-k tiles | "
                       "weighted_mean=SAM-weighted mean | "
                       "weighted_topk_mean=weighted mean of top-k tiles"
                   ))
    p.add_argument("--topk-tiles", type=int, default=DEFAULT_TOPK_TILES,
                   help="k for topk_mean / weighted_topk_mean")
    p.add_argument("--also-run-baseline", action="store_true",
                   help="Also run max-pool baseline alongside chosen aggregation "
                        "(saves comparison CSV + prediction_comparison.png)")

    # ---- Vegetation scoring ----
    p.add_argument("--scoring", default=DEFAULT_SCORING, choices=list(SCORING_METHODS),
                   help="rgb=ExG pixel scoring (no model) | sam=SAM masks + ExG")
    p.add_argument("--sam-checkpoint", default=DEFAULT_SAM_CHECKPOINT,
                   help="Path to SAM checkpoint (.pth). Download with download_sam.py")
    p.add_argument("--sam-model-type", default=DEFAULT_SAM_MODEL_TYPE,
                   choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--exg-threshold", type=float, default=DEFAULT_EXG_THRESHOLD,
                   help="ExG value above which a pixel is counted as vegetation")
    p.add_argument("--sam-min-greenness", type=float, default=DEFAULT_SAM_MIN_GREENNESS,
                   help="Minimum green fraction of a SAM mask for it to be labelled vegetation")

    # ---- Tile weighting ----
    p.add_argument("--weight-alpha", type=float, default=DEFAULT_WEIGHT_ALPHA,
                   help="Base weight for a zero-vegetation tile (w = alpha + beta*veg_ratio)")
    p.add_argument("--weight-beta", type=float, default=DEFAULT_WEIGHT_BETA,
                   help="Slope: increase in weight per unit vegetation ratio")
    p.add_argument("--weight-min", type=float, default=DEFAULT_WEIGHT_MIN,
                   help="Minimum tile weight (prevents full suppression)")
    p.add_argument("--weight-max", type=float, default=DEFAULT_WEIGHT_MAX,
                   help="Maximum tile weight cap")

    # ---- Predictions ----
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help="Top-k species per image in outputs")

    # ---- Limits / debug ----
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N images (smoke test)")
    p.add_argument("--n-visualize", type=int, default=DEFAULT_N_VISUALIZE,
                   help="Number of images for which to save detailed visualizations (0=none)")
    p.add_argument("--no-save-tiles", action="store_true",
                   help="Skip saving per-tile metadata CSV (faster for large runs)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def make_run_slug(model_name: str, prompt_mode: str, aggregation: str, scoring: str) -> str:
    model_slug = model_name.split("/")[-1].replace(".", "-")
    return f"{model_slug}_{prompt_mode}_{aggregation}_{scoring}"


def find_images(images_root: str) -> list[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths: list[Path] = []
    for ext in exts:
        paths.extend(Path(images_root).glob(ext))
    return sorted(set(paths))


def load_model(model_name: str, device: str):
    print(f"  Loading BioCLIP model '{model_name}' ...")
    model, _, transform = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    return model, transform, tokenizer


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    aggregation: str,
    topk_tiles: int,
    scoring: str,
    sam_generator,
    weight_alpha: float,
    weight_beta: float,
    weight_min: float,
    weight_max: float,
    exg_threshold: float,
    sam_min_greenness: float,
    also_run_baseline: bool,
) -> dict:
    """
    Run tiled zero-shot inference on one image with optional SAM weighting.

    Returns a dict with:
      top_ids, top_scores       — chosen aggregation method
      baseline_ids, baseline_scores — max-pool baseline (if also_run_baseline)
      tiles, coords             — tile objects and coordinates
      veg_scores, weights       — vegetation scores and computed weights
      sam_masks_per_tile        — SAM masks (empty lists if scoring='rgb')
      tile_logits               — (n_tiles, n_species) tensor on CPU
    """
    image = Image.open(image_path).convert("RGB")
    tiles, coords = get_tiles(image, tile_size, stride)

    # ----- SAM / RGB vegetation scoring -----
    veg_scores, sam_masks_per_tile = score_tiles(
        tiles,
        method=scoring,
        sam_generator=sam_generator,
        exg_threshold=exg_threshold,
        min_greenness=sam_min_greenness,
    )
    weights = compute_tile_weights(
        veg_scores,
        alpha=weight_alpha,
        beta=weight_beta,
        w_min=weight_min,
        w_max=weight_max,
    )

    # ----- BioCLIP tile encoding -----
    image_feats = encode_image_tiles(model, transform, tiles, device, batch_size=batch_size)
    image_feats = image_feats.to(device)
    tile_logits = compute_tile_logits(image_feats, text_feats, logit_scale)

    # ----- Aggregation -----
    needs_weights = aggregation in ("weighted_mean", "weighted_topk_mean")
    image_logits = aggregate(
        tile_logits,
        method=aggregation,
        weights=weights if needs_weights else None,
        topk_tiles=topk_tiles,
    )
    top_ids, top_scores = image_top_k(image_logits, species_ids, k=top_k)

    # ----- Baseline (max-pool) for comparison -----
    baseline_ids, baseline_scores = [], []
    if also_run_baseline and aggregation != "max":
        baseline_logits = aggregate(tile_logits, method="max")
        baseline_ids, baseline_scores = image_top_k(baseline_logits, species_ids, k=top_k)

    return {
        "image":              image,
        "tiles":              tiles,
        "coords":             coords,
        "veg_scores":         veg_scores,
        "weights":            weights.tolist(),
        "sam_masks_per_tile": sam_masks_per_tile,
        "tile_logits":        tile_logits.cpu(),
        "top_ids":            top_ids,
        "top_scores":         top_scores,
        "baseline_ids":       baseline_ids,
        "baseline_scores":    baseline_scores,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    stride = args.tile_size - args.tile_overlap
    if stride <= 0:
        sys.exit(f"ERROR: tile-overlap ({args.tile_overlap}) must be < tile-size ({args.tile_size})")

    batch_size  = args.batch_size or MODEL_DEFAULT_BATCH.get(args.model_name, 64)
    run_slug    = make_run_slug(args.model_name, args.prompt_mode, args.aggregation, args.scoring)
    out_dir     = Path(args.output_dir) / run_slug
    viz_dir     = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ---- Print run header ----
    print(f"\n{'='*65}")
    print(f"Experiment 007 — BioCLIP + SAM Vegetation Weighting")
    print(f"  Run         : {run_slug}")
    print(f"  Device      : {device}  |  tile={args.tile_size}  stride={stride}")
    print(f"  Scoring     : {args.scoring}  |  Aggregation: {args.aggregation}")
    print(f"  Weights     : alpha={args.weight_alpha}  beta={args.weight_beta}  "
          f"[{args.weight_min}, {args.weight_max}]")
    print(f"  Output dir  : {out_dir}")
    print(f"{'='*65}\n")

    # ---- Save run config ----
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

    # ---- Load species ----
    print(f"Loading species from: {args.species_csv}")
    if not Path(args.species_csv).exists():
        sys.exit(f"ERROR: species CSV not found: {args.species_csv}")
    labels = load_species_labels(
        args.species_csv,
        max_common_names=args.max_common_names,
        max_synonyms=args.max_synonyms,
    )
    species_ids = [lbl.species_id for lbl in labels]
    id_to_name  = {lbl.species_id: lbl.canonical_scientific for lbl in labels}
    print(f"  {len(labels)} species loaded")

    # ---- Build prompts ----
    print(f"\nBuilding prompts (mode='{args.prompt_mode}') ...")
    prompt_lists = build_all_prompts(labels, args.prompt_mode)
    stats = prompt_stats(prompt_lists)
    print(f"  {stats['total_prompts']} prompts  "
          f"({stats['min_per_species']}–{stats['max_per_species']} / species)")

    prompt_path = out_dir / "prompt_table.csv"
    with open(prompt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["species_id", "canonical_scientific", "primary_common", "n_prompts", "prompts"])
        for label, prompts in zip(labels, prompt_lists):
            writer.writerow([label.species_id, label.canonical_scientific,
                             label.primary_common, len(prompts), json.dumps(prompts)])

    with open(out_dir / "prompt_summary.json", "w") as f:
        json.dump({**stats, "prompt_mode": args.prompt_mode}, f, indent=2)

    # ---- Load BioCLIP ----
    print(f"\nLoading BioCLIP model ...")
    model, transform, tokenizer = load_model(args.model_name, device)
    logit_scale = model.logit_scale.exp().item()
    print(f"  logit_scale: {logit_scale:.4f}")

    # ---- Encode text features ----
    print(f"\nEncoding text features (batch={args.text_batch_size}) ...")
    t0_text = time.perf_counter()
    text_feats = encode_text_features_from_prompts(
        model, tokenizer, prompt_lists, device, batch_size=args.text_batch_size
    )
    text_feats = text_feats.to(device)
    text_encoding_secs = time.perf_counter() - t0_text
    print(f"  text_feats: {text_feats.shape}  ({text_encoding_secs:.1f}s)")

    # ---- Load SAM (if requested) ----
    # SAM and BioCLIP 2.5 (ViT-H) compete for VRAM.  We try CUDA first; if OOM,
    # load_sam_generator automatically retries on CPU and emits a warning.
    sam_generator = None
    if args.scoring == "sam":
        print(f"\nLoading SAM model ...")
        sam_device = device  # load_sam_generator handles OOM fallback to cpu
        sam_generator = load_sam_generator(
            args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=sam_device,
        )
        if sam_generator is None:
            print("  SAM unavailable — falling back to RGB scoring.")
            args.scoring = "rgb"

    # ---- Find images ----
    if not Path(args.images_root).exists():
        sys.exit(f"ERROR: images-root not found: {args.images_root}")
    image_paths = find_images(args.images_root)
    if args.limit:
        image_paths = image_paths[: args.limit]
    print(f"\nFound {len(image_paths)} images in: {args.images_root}")
    if not image_paths:
        sys.exit("ERROR: no images found — check --images-root")

    # ---- Inference loop ----
    print(f"Running inference (aggregation={args.aggregation}, scoring={args.scoring}) ...\n")

    submission_rows:   list[dict] = []
    topk_rows:         list[dict] = []
    tile_meta_rows:    list[dict] = []
    comparison_rows:   list[dict] = []

    # Accumulators for dataset-level visualizations
    all_veg_scores:      list[float] = []
    all_weights:         list[float] = []
    all_top1_conf:       list[float] = []

    t0_infer = time.perf_counter()
    errors = 0
    n_visualized = 0

    for i, image_path in enumerate(image_paths):
        quadrat_id = image_path.stem
        print(f"  [{i+1:>5}/{len(image_paths)}] {quadrat_id} ...", end=" ", flush=True)

        try:
            result = process_image(
                image_path=image_path,
                model=model,
                transform=transform,
                text_feats=text_feats,
                species_ids=species_ids,
                logit_scale=logit_scale,
                tile_size=args.tile_size,
                stride=stride,
                batch_size=batch_size,
                top_k=args.top_k,
                device=device,
                aggregation=args.aggregation,
                topk_tiles=args.topk_tiles,
                scoring=args.scoring,
                sam_generator=sam_generator,
                weight_alpha=args.weight_alpha,
                weight_beta=args.weight_beta,
                weight_min=args.weight_min,
                weight_max=args.weight_max,
                exg_threshold=args.exg_threshold,
                sam_min_greenness=args.sam_min_greenness,
                also_run_baseline=args.also_run_baseline,
            )
        except Exception as exc:
            import traceback
            print(f"ERROR: {exc}")
            traceback.print_exc()
            errors += 1
            continue

        top_ids    = result["top_ids"]
        top_scores = result["top_scores"]
        tiles      = result["tiles"]
        coords     = result["coords"]
        veg_scores = result["veg_scores"]
        weights    = result["weights"]
        tile_logits = result["tile_logits"]

        # Accumulate dataset-level stats
        all_veg_scores.extend(veg_scores)
        all_weights.extend(weights)

        # Per-tile top-1 predictions (for diagnostics + dataset scatter)
        tile_probs = tile_logits.softmax(dim=-1)
        tile_top1_idx    = tile_probs.argmax(dim=1).tolist()
        tile_top1_logits = tile_logits.max(dim=1).values.tolist()
        tile_top1_species = [id_to_name.get(species_ids[idx], species_ids[idx])
                             for idx in tile_top1_idx]
        all_top1_conf.extend(tile_top1_logits)

        # ---- Submission row ----
        ids_str = "[" + ", ".join(top_ids) + "]"
        submission_rows.append({"quadrat_id": quadrat_id, "species_ids": ids_str})

        # ---- Top-k prediction rows ----
        for rank, (sid, score) in enumerate(zip(top_ids, top_scores), start=1):
            topk_rows.append({
                "image_name":  quadrat_id,
                "rank":        rank,
                "species_id":  sid,
                "species_name": id_to_name.get(sid, ""),
                "logit_score": f"{score:.6f}",
                "aggregation": args.aggregation,
            })

        # ---- Tile metadata rows ----
        if not args.no_save_tiles:
            for ti, (x1, y1, x2, y2) in enumerate(coords):
                tile_meta_rows.append({
                    "image_id":      quadrat_id,
                    "tile_idx":      ti,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "veg_ratio":     f"{veg_scores[ti]:.6f}",
                    "weight":        f"{weights[ti]:.6f}",
                    "top1_species_id":   species_ids[tile_top1_idx[ti]],
                    "top1_species_name": tile_top1_species[ti],
                    "top1_logit":        f"{tile_top1_logits[ti]:.6f}",
                })

        # ---- Baseline comparison rows ----
        if args.also_run_baseline and result["baseline_ids"]:
            for rank, (sid, score) in enumerate(zip(result["baseline_ids"],
                                                     result["baseline_scores"]), start=1):
                comparison_rows.append({
                    "image_name":  quadrat_id,
                    "rank":        rank,
                    "method":      "baseline_max",
                    "species_id":  sid,
                    "species_name": id_to_name.get(sid, ""),
                    "logit_score": f"{score:.6f}",
                })

        # ---- Visualizations ----
        do_viz = (args.n_visualize > 0) and (n_visualized < args.n_visualize)
        if do_viz:
            try:
                baseline_s = result["baseline_ids"]   if result["baseline_ids"] else top_ids
                baseline_c = result["baseline_scores"] if result["baseline_scores"] else top_scores
                save_image_visualizations(
                    image=result["image"],
                    image_id=quadrat_id,
                    tiles=tiles,
                    coords=coords,
                    veg_scores=veg_scores,
                    weights=weights,
                    sam_masks_per_tile=result["sam_masks_per_tile"],
                    tile_top1_species=tile_top1_species,
                    tile_top1_logits=tile_top1_logits,
                    baseline_top_species=[id_to_name.get(s, s) for s in baseline_s],
                    baseline_top_scores=baseline_c,
                    weighted_top_species=[id_to_name.get(s, s) for s in top_ids],
                    weighted_top_scores=top_scores,
                    viz_dir=viz_dir,
                )
                n_visualized += 1
            except Exception as exc:
                print(f"  [viz warning] {exc}")

        mean_veg = float(np.mean(veg_scores)) if veg_scores else 0.0
        print(f"{ids_str[:55]}  veg={mean_veg:.2f}")

    total_infer_secs = time.perf_counter() - t0_infer
    n_processed = len(submission_rows)

    # ---- Save outputs ----
    print()
    save_csv(out_dir / "submission.csv",
             submission_rows, ["quadrat_id", "species_ids"])
    print(f"  Submission CSV saved  ({len(submission_rows)} rows)")

    save_csv(out_dir / "predictions_topk.csv",
             topk_rows, ["image_name", "rank", "species_id", "species_name", "logit_score", "aggregation"])
    print(f"  Top-k predictions saved  ({len(topk_rows)} rows)")

    if not args.no_save_tiles and tile_meta_rows:
        save_csv(out_dir / "tile_metadata.csv",
                 tile_meta_rows,
                 ["image_id", "tile_idx", "x1", "y1", "x2", "y2",
                  "veg_ratio", "weight", "top1_species_id", "top1_species_name", "top1_logit"])
        print(f"  Tile metadata saved  ({len(tile_meta_rows)} rows)")

    if comparison_rows:
        save_csv(out_dir / "comparison_baseline.csv",
                 comparison_rows,
                 ["image_name", "rank", "method", "species_id", "species_name", "logit_score"])
        print(f"  Comparison CSV saved  ({len(comparison_rows)} rows)")

    # ---- Dataset-level visualizations ----
    if args.n_visualize > 0 and all_veg_scores:
        try:
            save_summary_statistics(all_veg_scores, all_weights, all_top1_conf, viz_dir)
            print(f"  Summary statistics plot saved to {viz_dir}/")
        except Exception as exc:
            print(f"  [viz warning] summary stats failed: {exc}")

    # ---- Summary JSON ----
    summary = {
        "experiment":             "007_bioclip_tile_zeroshot_sam_weighted",
        "run_slug":               run_slug,
        "model_name":             args.model_name,
        "prompt_mode":            args.prompt_mode,
        "aggregation":            args.aggregation,
        "scoring_method":         args.scoring,
        "n_species":              len(labels),
        **stats,
        "n_images_found":         len(image_paths),
        "n_images_processed":     n_processed,
        "n_errors":               errors,
        "top_k":                  args.top_k,
        "tile_size":              args.tile_size,
        "tile_overlap":           args.tile_overlap,
        "stride":                 stride,
        "weight_alpha":           args.weight_alpha,
        "weight_beta":            args.weight_beta,
        "weight_min":             args.weight_min,
        "weight_max":             args.weight_max,
        "exg_threshold":          args.exg_threshold,
        "mean_veg_ratio":         float(np.mean(all_veg_scores)) if all_veg_scores else None,
        "mean_tile_weight":       float(np.mean(all_weights)) if all_weights else None,
        "n_visualized":           n_visualized,
        "text_encoding_secs":     round(text_encoding_secs, 2),
        "inference_total_secs":   round(total_infer_secs, 2),
        "inference_per_image_secs": round(total_infer_secs / max(n_processed, 1), 3),
        "device":                 device,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Final report ----
    print(f"\n{'='*65}")
    print(f"Experiment 007 complete: {run_slug}")
    print(f"  Images processed : {n_processed} / {len(image_paths)}")
    if errors:
        print(f"  Errors           : {errors}")
    print(f"  Mean veg ratio   : {np.mean(all_veg_scores):.3f}" if all_veg_scores else "")
    print(f"  Mean tile weight : {np.mean(all_weights):.3f}"    if all_weights    else "")
    print(f"  Text encoding    : {text_encoding_secs:.1f}s")
    print(f"  Inference        : {total_infer_secs:.1f}s  "
          f"({total_infer_secs / max(n_processed, 1):.3f}s/image)")
    print(f"  Visualizations   : {n_visualized} images  →  {viz_dir}/")
    print(f"  Outputs          : {out_dir}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
