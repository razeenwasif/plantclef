"""
Baseline submission using ONLY the Phase-1 linear-probe head.

Why this exists: the Phase-2 LoRA fine-tune collapsed to all-positive sigmoid
output (see diagnose_predictions.py). The Phase-1 softmax head reached
top-1 = 11.97% on held-out single-plant images and is the strongest known-good
artefact in this experiment. This script produces a Kaggle-format submission
from it so we have a real baseline score while Phase 2 retrains.

Pipeline (per quadrat image):
  1. Tile via 004_bioclip_few_shot/tiling.get_tiles.
  2. DINOv3 backbone (SSL-init, no LoRA) -> 2048-d (CLS + GeM-patches).
  3. Phase-1 head -> 7806-d logits -> softmax probabilities.
  4. Per tile, take top-K species (default K=3).
  5. Aggregate across tiles by max probability per species.
  6. Output the union of top-N species (default N=10) as the prediction set.
  7. Write submission_phase1_*.csv in PlantCLEF format.
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

from model import DINOv3MultiLabel, build_default_transform, DEFAULT_HUB_NAME

_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
sys.path.insert(0, str(_FOUR))
from tiling import get_tiles  # type: ignore  # noqa: E402
from dataset import load_species_ids, DEFAULT_SPECIES_CSV  # type: ignore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--phase1-head", required=True,
                   help="Path to phase1_head.pth (contains head_state, gem_state)")
    p.add_argument("--ssl-checkpoint", required=True,
                   help="SSL-pretrained DINOv3 backbone checkpoint")
    p.add_argument("--images-root", required=True,
                   help="Directory containing test quadrat images (recursive)")
    p.add_argument("--species-csv", default=DEFAULT_SPECIES_CSV)
    p.add_argument("--output-dir", default="./outputs")
    p.add_argument("--backbone-source", choices=("torch_hub", "timm"), default="torch_hub")
    p.add_argument("--backbone", default=DEFAULT_HUB_NAME)
    p.add_argument("--tile-size", type=int, default=384)
    p.add_argument("--tile-overlap", type=int, default=128)
    p.add_argument("--per-tile-top-k", type=int, default=3,
                   help="Top-K species kept per tile before cross-tile aggregation")
    p.add_argument("--final-top-n", type=int, default=10,
                   help="Final number of species per quadrat in the submission")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load_phase1_model(args: argparse.Namespace) -> tuple[DINOv3MultiLabel, list[str]]:
    species_ids = load_species_ids(args.species_csv)
    n_classes = len(species_ids)

    model = DINOv3MultiLabel(
        n_classes=n_classes,
        backbone_source=args.backbone_source,
        backbone_name=args.backbone,
        pretrained=True,
        ssl_checkpoint=args.ssl_checkpoint,
    )

    ckpt = torch.load(args.phase1_head, map_location="cpu", weights_only=False)
    model.head.load_state_dict(ckpt["head_state"])
    if "gem_state" in ckpt:
        model.gem.load_state_dict(ckpt["gem_state"])
    elif "gem_p" in ckpt:
        model.gem.p.data.copy_(ckpt["gem_p"])
    logger.info(f"Loaded Phase-1 head from {args.phase1_head}")
    model = model.to(args.device).eval()
    return model, species_ids


@torch.no_grad()
def encode_tiles(
    model: DINOv3MultiLabel,
    transform,
    tiles: list[Image.Image],
    device: str,
    batch_size: int,
    use_bf16: bool,
) -> torch.Tensor:
    """Returns (n_tiles, n_classes) softmax probabilities on CPU."""
    if not tiles:
        return torch.empty(0, model.n_classes)
    chunks = []
    for i in range(0, len(tiles), batch_size):
        batch = torch.stack([transform(t) for t in tiles[i: i + batch_size]]).to(
            device, non_blocking=True
        )
        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=use_bf16 and device.startswith("cuda"),
        ):
            logits = model(batch)
        probs = torch.softmax(logits.float(), dim=1).cpu()
        chunks.append(probs)
    return torch.cat(chunks, dim=0)


def aggregate_phase1(
    tile_probs: torch.Tensor,
    species_ids: list[str],
    per_tile_top_k: int,
    final_top_n: int,
) -> tuple[list[str], list[float]]:
    """
    Per tile: take top-K. Across tiles: max probability per species.
    Then return the final_top_n species by aggregated score.
    """
    if tile_probs.numel() == 0:
        return [], []
    n_tiles, n_classes = tile_probs.shape
    K = min(per_tile_top_k, n_classes)
    topk_vals, topk_idx = tile_probs.topk(K, dim=1)
    agg = torch.zeros(n_classes, dtype=torch.float32)
    for t in range(n_tiles):
        for k in range(K):
            idx = int(topk_idx[t, k])
            v = float(topk_vals[t, k])
            if v > agg[idx]:
                agg[idx] = v
    nonzero = (agg > 0).sum().item()
    N = min(final_top_n, int(nonzero))
    if N == 0:
        return [], []
    final_vals, final_idx = agg.topk(N)
    return (
        [species_ids[int(i)] for i in final_idx.tolist()],
        [float(v) for v in final_vals.tolist()],
    )


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    args = parse_args()
    if args.tile_size % 16 != 0:
        sys.exit("tile_size must be a multiple of 16 for DINOv3.")

    out_dir = Path(args.output_dir) / "phase1_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, species_ids = load_phase1_model(args)
    transform = build_default_transform(args.tile_size)
    stride = max(1, args.tile_size - args.tile_overlap)

    image_paths = find_images(Path(args.images_root))
    if args.limit:
        image_paths = image_paths[: args.limit]
    logger.info(f"Found {len(image_paths)} test images")

    sub_csv = out_dir / "submission_phase1_baseline.csv"
    scored_csv = out_dir / "predictions_phase1_baseline_scored.csv"

    t0 = time.time()
    with open(sub_csv, "w", newline="") as fsub, \
         open(scored_csv, "w", newline="") as fsco:
        sub_writer = csv.writer(fsub, quoting=csv.QUOTE_ALL)
        sub_writer.writerow(["quadrat_id", "species_ids"])
        sco_writer = csv.writer(fsco, quoting=csv.QUOTE_ALL)
        sco_writer.writerow(["quadrat_id", "rank", "species_id", "score"])

        for i, img_path in enumerate(image_paths):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as exc:
                logger.warning(f"Skipping {img_path.name}: {exc}")
                sub_writer.writerow([img_path.stem, "[]"])
                continue

            tiles, _ = get_tiles(image, args.tile_size, stride)
            if not tiles:
                sub_writer.writerow([img_path.stem, "[]"])
                continue

            tile_probs = encode_tiles(
                model, transform, tiles, args.device, args.batch_size, args.bf16
            )
            sp_ids, sp_scores = aggregate_phase1(
                tile_probs, species_ids, args.per_tile_top_k, args.final_top_n
            )
            sub_writer.writerow([img_path.stem, repr([int(s) for s in sp_ids])])
            for rank, (sid, score) in enumerate(zip(sp_ids, sp_scores), start=1):
                sco_writer.writerow([img_path.stem, rank, sid, f"{score:.6f}"])

            if (i + 1) % 25 == 0 or i + 1 == len(image_paths):
                rate = (i + 1) / (time.time() - t0)
                eta = (len(image_paths) - (i + 1)) / max(rate, 1e-9)
                logger.info(
                    f"  {i+1}/{len(image_paths)}  ({rate:.2f} img/s, ETA {eta:.0f}s)"
                )

    summary = {
        "n_images": len(image_paths),
        "elapsed_s": time.time() - t0,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "per_tile_top_k": args.per_tile_top_k,
        "final_top_n": args.final_top_n,
        "phase1_head": args.phase1_head,
        "ssl_checkpoint": args.ssl_checkpoint,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {sub_csv}")
    logger.info(f"Wrote {scored_csv}")
    logger.info(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
