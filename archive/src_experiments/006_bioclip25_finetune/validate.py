"""
Standalone validation script with configurable tiling and preprocessing ablations.

Loads a trained checkpoint and evaluates on the validation split (the same
stratified 10% held out during training).

Supports:
  - whole-image mode
  - grid_2x2, grid_4x4, multiscale tiling
  - configurable interpolation (bicubic, lanczos)
  - optional margin crop
  - optional JPEG recompression (q=85, q=94)
  - configurable aggregation: max | topk_mean (k configurable)

Use this to compare tiling/preprocessing ablations without retraining.

Usage examples
--------------
  # Whole-image baseline
  python validate.py --checkpoint ./outputs/train/checkpoints/best.pt

  # 2x2 grid tiling, max aggregation
  python validate.py --checkpoint ... --tile-mode grid_2x2

  # Multi-scale tiling + topk_mean aggregation, top-5
  python validate.py --checkpoint ... --tile-mode multiscale --agg-mode topk_mean --topk-agg 5

  # Lanczos interpolation ablation
  python validate.py --checkpoint ... --interp lanczos

  # JPEG q=85 compression ablation
  python validate.py --checkpoint ... --jpeg-quality 85

  # Combined: lanczos + JPEG q=94 + 5% margin crop
  python validate.py --checkpoint ... --interp lanczos --jpeg-quality 94 --margin-crop 0.05
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import (
    DEFAULT_TRAIN_META_CSV,
    DEFAULT_TRAIN_IMAGE_ROOT,
    load_train_metadata,
    resolve_image_paths,
    build_class_mapping,
    build_val_split,
)
from model import load_bioclip25_probe
from transforms import build_inference_preprocessor, InferencePreprocessor
from tiling import (
    TILING_MODES, AGG_MODES,
    tile_image, encode_tiles, classify_tiles, aggregate_logits,
)
from utils import (
    setup_logging,
    resolve_device,
    compute_recall_at_k,
    topk_predictions,
    save_json,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "./outputs/validate"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validation with configurable tiling and preprocessing ablations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--checkpoint", required=True, help="Path to best.pt from train.py")

    # Data
    p.add_argument("--train-meta-csv",   default=DEFAULT_TRAIN_META_CSV)
    p.add_argument("--train-image-root", default=DEFAULT_TRAIN_IMAGE_ROOT)
    p.add_argument("--val-fraction",     type=float, default=0.1)
    p.add_argument("--val-seed",         type=int,   default=42)
    p.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N val images (0 = all, for quick smoke test).",
    )

    # Tiling
    p.add_argument("--tile-mode",    default="whole",  choices=list(TILING_MODES))
    p.add_argument("--overlap",      type=float, default=0.0,
                   help="Tile overlap ratio for grid modes (0 = no overlap).")

    # Aggregation
    p.add_argument("--agg-mode",  default="max", choices=list(AGG_MODES))
    p.add_argument("--topk-agg",  type=int, default=5,
                   help="k for topk_mean aggregation.")

    # Inference preprocessing
    p.add_argument("--img-size",    type=int,   default=224)
    p.add_argument("--interp",      default="bicubic",
                   choices=["bicubic", "lanczos", "bilinear"])
    p.add_argument("--margin-crop", type=float, default=0.0,
                   help="Fraction of shorter side to crop on each border (0 = none).")
    p.add_argument("--jpeg-quality", type=int, default=0,
                   help="JPEG recompression quality (0 = no recompression).")
    p.add_argument("--jpeg-subsampling", type=int, default=0,
                   choices=[0, 1, 2],
                   help="JPEG chroma subsampling: 0=4:4:4  1=4:2:2  2=4:2:0")

    # Inference
    p.add_argument("--val-top-n",       type=int, default=20)
    p.add_argument("--tile-batch-size", type=int, default=32)
    p.add_argument("--device",          default="auto")
    p.add_argument("--output-dir",      default=DEFAULT_OUTPUT_DIR)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Val-time dataset: returns raw PIL images + labels
# ---------------------------------------------------------------------------

class ValRawDataset(Dataset):
    """Returns (PIL Image, class_index, species_id_str)."""

    def __init__(
        self,
        df,
        species_to_idx: dict[str, int],
    ) -> None:
        self.df = df[df["resolved_path"].notna()].reset_index(drop=True)
        self.species_to_idx = species_to_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["resolved_path"]).convert("RGB")
        label = self.species_to_idx[row["species_id"]]
        return image, label, row["species_id"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir=str(out_dir))

    logger.info("=" * 60)
    logger.info("BioCLIP 2.5 Linear Probe Validation")
    logger.info(f"  checkpoint     : {args.checkpoint}")
    logger.info(f"  tile_mode      : {args.tile_mode}")
    logger.info(f"  overlap        : {args.overlap}")
    logger.info(f"  agg_mode       : {args.agg_mode}  (topk={args.topk_agg})")
    logger.info(f"  interp         : {args.interp}")
    logger.info(f"  margin_crop    : {args.margin_crop}")
    logger.info(f"  jpeg_quality   : {args.jpeg_quality or 'none'}")
    logger.info(f"  jpeg_subsampling: {args.jpeg_subsampling}")
    logger.info(f"  device         : {device}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model, idx_to_species = load_bioclip25_probe(args.checkpoint, device=device)
    model.eval()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    preproc = build_inference_preprocessor(args)
    logger.info(f"Preprocessor: {preproc}")

    # ------------------------------------------------------------------
    # Validation data
    # ------------------------------------------------------------------
    logger.info("Loading validation split ...")
    df = load_train_metadata(args.train_meta_csv)
    df = resolve_image_paths(df, args.train_image_root, verify=False)
    df = df[df["resolved_path"].notna()].reset_index(drop=True)

    species_to_idx, _ = build_class_mapping(df)
    _, val_df = build_val_split(df, val_fraction=args.val_fraction, seed=args.val_seed)

    if args.limit > 0:
        val_df = val_df.head(args.limit).reset_index(drop=True)
        logger.info(f"Limiting to first {args.limit} val images")

    val_dataset = ValRawDataset(val_df, species_to_idx)
    logger.info(f"Val images: {len(val_dataset):,}")

    # ------------------------------------------------------------------
    # Inference loop (one image at a time for tiling support)
    # ------------------------------------------------------------------
    results:    list[dict] = []
    total_loss  = 0.0
    n_processed = 0
    t_start = time.perf_counter()

    for i in range(len(val_dataset)):
        image, label, gt_species_id = val_dataset[i]

        tiles = tile_image(image, mode=args.tile_mode, overlap_ratio=args.overlap)

        # Apply inference preprocessing to each tile
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

        # Cross-entropy against ground truth
        label_t = torch.tensor([label])
        loss = torch.nn.functional.cross_entropy(
            img_logits.unsqueeze(0), label_t
        ).item()
        total_loss += loss
        n_processed += 1

        pred_species, _ = topk_predictions(
            img_logits, idx_to_species, top_n=args.val_top_n
        )
        results.append({"gt_species_id": gt_species_id, "pred_ids": pred_species})

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t_start
            recall1_so_far = sum(
                1 for r in results if r["gt_species_id"] in r["pred_ids"][:1]
            ) / len(results)
            logger.info(
                f"  [{i+1:>6}/{len(val_dataset)}]  "
                f"loss={total_loss/n_processed:.4f}  "
                f"recall@1_so_far={recall1_so_far:.4f}  "
                f"{elapsed:.0f}s elapsed"
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = compute_recall_at_k(results, k_values=(1, 5, 10, 20))
    metrics["val_loss"]    = round(total_loss / max(n_processed, 1), 4)
    metrics["n_evaluated"] = n_processed
    total_secs = time.perf_counter() - t_start
    metrics["total_secs"]  = round(total_secs, 1)
    metrics["images_per_sec"] = round(n_processed / max(total_secs, 1), 2)

    logger.info("-" * 60)
    logger.info("VALIDATION RESULTS")
    for k, v in sorted(metrics.items()):
        logger.info(f"  {k:<20} {v}")
    logger.info("-" * 60)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    run_config = {
        **vars(args),
        "device": device,
        "n_classes": len(idx_to_species),
    }
    run_config["preprocessor"] = repr(preproc)

    save_json(run_config, str(out_dir / "validate_config.json"))
    save_json({"metrics": metrics, "run_config": run_config},
              str(out_dir / "validate_results.json"))
    logger.info(f"Results saved: {out_dir}/validate_results.json")


if __name__ == "__main__":
    main()
