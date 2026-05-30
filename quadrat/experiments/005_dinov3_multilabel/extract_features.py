"""
Phase 1 feature cache: run DINOv3 once over all single-plant training images,
save the (CLS + GeM-pooled patches) features and class labels to a single
torch tensor file. The cached file is consumed by `train.py --phase 1` to
fit the classifier head as a fast linear probe.

Output schema (.pt file, fp16 features):
    {
        "features":   (N, 2048) float16
        "labels":     (N,) int64
        "paths":      list[str], length N
        "species_ids": list[str], length n_classes
        "backbone":   str (timm model name)
        "embed_dim":  int (feature dim, = 2 * backbone embed dim)
    }

Usage:
    cd src_experiments/005_dinov3_multilabel
    python extract_features.py \
        --metadata-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
        --image-root   /workspace/plantclef/raw/train/images_max_side_800 \
        --species-csv  .../species_lookup_with_gbif_cleaned_names.csv \
        --output       ../../models/dinov3_v1/phase1_feature_cache.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Sibling imports from this experiment dir
from model import DINOv3MultiLabel, build_default_transform, DEFAULT_HUB_NAME
from mosaic_dataset import SinglePlantDataset

# Reuse 004's metadata utilities by adding its directory to sys.path
_FOUR = Path(__file__).resolve().parent.parent / "004_bioclip_few_shot"
if str(_FOUR) not in sys.path:
    sys.path.insert(0, str(_FOUR))

from dataset import (  # type: ignore  # noqa: E402
    load_train_metadata,
    resolve_image_paths,
    load_species_ids,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache DINOv3 features for Phase 1 linear probe.")
    p.add_argument("--metadata-csv", required=True,
                   help="Cleaned + verified single-plant training metadata CSV.")
    p.add_argument("--image-root", required=True,
                   help="Root dir of training images: {root}/{species_id}/{image_name}.")
    p.add_argument("--species-csv", required=True,
                   help="Species lookup CSV (defines class index ordering).")
    p.add_argument("--output", required=True,
                   help="Path for the cached .pt file.")
    p.add_argument("--backbone-source", choices=["torch_hub", "timm"], default="torch_hub",
                   help="DINOv3 loader. torch_hub matches the SSL pretrain in 005_dinov3_vegetation_adapt.")
    p.add_argument("--backbone", default=DEFAULT_HUB_NAME,
                   help="Model name within the chosen source.")
    p.add_argument("--ssl-checkpoint", default=None,
                   help="Path to SSL backbone .pt (e.g. .../outputs/ssl/backbone_final.pt).")
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N images (debug).")
    p.add_argument("--verify-paths", action="store_true",
                   help="Stat each image before extraction (slower, safer).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract(
    model: DINOv3MultiLabel,
    loader: DataLoader,
    device: str,
    feat_dim: int,
    n_total: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Run the backbone in eval mode and stream features into a fp16 buffer."""
    features = torch.empty(n_total, feat_dim, dtype=torch.float16)
    labels = torch.empty(n_total, dtype=torch.int64)
    paths: list[str] = [""] * n_total

    model.eval()
    cursor = 0
    t0 = time.time()
    log_every = max(1, len(loader) // 50)

    for step, (imgs, lbls, batch_paths) in enumerate(loader):
        # Drop rows with missing/corrupt images (label == -1 from the dataset)
        valid_mask = lbls != -1
        if not valid_mask.any():
            continue
        imgs = imgs[valid_mask].to(device, non_blocking=True)
        lbls = lbls[valid_mask]
        kept_paths = [bp for bp, m in zip(batch_paths, valid_mask.tolist()) if m]

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            feats = model.forward_features(imgs)
        feats = feats.to(torch.float16).cpu()

        n = feats.shape[0]
        if cursor + n > n_total:
            n = n_total - cursor
            feats = feats[:n]
            lbls = lbls[:n]
            kept_paths = kept_paths[:n]

        features[cursor:cursor + n] = feats
        labels[cursor:cursor + n] = lbls
        for i, p in enumerate(kept_paths):
            paths[cursor + i] = p
        cursor += n

        if step % log_every == 0 or step == len(loader) - 1:
            elapsed = time.time() - t0
            rate = cursor / max(elapsed, 1e-6)
            logger.info(
                f"  step {step:>5d}/{len(loader)} | cached {cursor:>9,d}/{n_total:,} "
                f"({100.0 * cursor / max(n_total, 1):.1f}%) | "
                f"{rate:.1f} img/s | elapsed {elapsed/60:.1f}m"
            )

    # Trim to actual count (in case some rows were dropped)
    return features[:cursor], labels[:cursor], paths[:cursor]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    logger.info("DINOv3 Phase-1 feature extraction")
    logger.info(f"  backbone   : {args.backbone}")
    logger.info(f"  resolution : {args.resolution}")
    logger.info(f"  device     : {args.device}")
    logger.info(f"  output     : {args.output}")

    # 1. Load species ordering (defines the classifier output dim)
    species_ids = load_species_ids(args.species_csv)
    n_classes = len(species_ids)
    logger.info(f"  classes    : {n_classes}")

    # 2. Load + resolve metadata
    df = load_train_metadata(args.metadata_csv)
    df = resolve_image_paths(df, args.image_root, verify=args.verify_paths)
    if args.limit > 0:
        df = df.head(args.limit).copy()
        logger.info(f"  --limit applied: using first {args.limit:,} rows")

    # 3. Build dataset / loader
    transform = build_default_transform(args.resolution)
    dataset = SinglePlantDataset(df, species_ids, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=args.num_workers > 0,
    )

    # 4. Build model (backbone only — head is unused here)
    model = DINOv3MultiLabel(
        n_classes=n_classes,
        backbone_source=args.backbone_source,
        backbone_name=args.backbone,
        pretrained=True,
        ssl_checkpoint=args.ssl_checkpoint,
    ).to(args.device)
    feat_dim = 2 * model.embed_dim

    # 5. Extract
    features, labels, paths = extract(
        model=model,
        loader=loader,
        device=args.device,
        feat_dim=feat_dim,
        n_total=len(dataset),
    )

    # 6. Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "paths": paths,
            "species_ids": species_ids,
            "backbone": args.backbone,
            "backbone_source": args.backbone_source,
            "ssl_checkpoint": args.ssl_checkpoint,
            "embed_dim": feat_dim,
            "resolution": args.resolution,
        },
        out,
    )
    logger.info(
        f"Saved {features.shape[0]:,} features ({features.shape[1]}-d, fp16) -> {out}"
    )


if __name__ == "__main__":
    main()
