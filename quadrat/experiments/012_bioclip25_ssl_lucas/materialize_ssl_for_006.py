"""
Bridge from SSL pretraining (this experiment) to supervised fine-tuning
(`src_experiments/006_bioclip25_finetune/train.py`).

The SSL trainer (`ssl_train.py`) saves a backbone-only state_dict. The 006
training script's `--resume` path expects a full `BioCLIP25LinearProbe`
state_dict (backbone + freshly-initialised head). This script materialises
that bridge:

    1. Load the SSL backbone state_dict from `--ssl-ckpt`.
    2. Build a fresh BioCLIP25LinearProbe with a random-init head.
    3. Replace the backbone weights with the SSL-adapted ones.
    4. Save the result as a 006-format checkpoint with epoch=-1 (so 006's
       `--resume` resumes at epoch 0 = fresh supervised training).

After running this, the user kicks off the team-best 010 last_blocks recipe
on top of the SSL-warm-started backbone:

    python src_experiments/006_bioclip25_finetune/train.py \\
        --resume <output of this script> \\
        --unfreeze-blocks 4 \\
        --epochs 5 \\
        --batch-size 128 \\
        --lr 1e-3 \\
        --backbone-lr-scale 0.1 \\
        --output-dir src_experiments/012_bioclip25_ssl_lucas/outputs/finetune

Run
---
    python materialize_ssl_for_006.py \\
        --ssl-ckpt src_experiments/012_bioclip25_ssl_lucas/outputs/ssl_bioclip25_backbone_ep5.pt \\
        --train-meta-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \\
        --out src_experiments/012_bioclip25_ssl_lucas/outputs/ssl_init_for_006.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

LOG = logging.getLogger("materialize_ssl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ssl-ckpt", required=True,
                   help="Output of ssl_train.py — contains backbone_state_dict.")
    p.add_argument("--train-meta-csv", required=True,
                   help="PlantCLEF train metadata CSV used to fix idx_to_species ordering.")
    p.add_argument("--train-image-root",
                   default="/workspace/plantclef/raw/train/images_max_side_800/",
                   help="Image root used by 006 for resolve_image_paths.")
    p.add_argument("--out", required=True, help="Output 006-format ckpt.")
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]),
                   help="Project root (so we can import from 006).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    sys.path.insert(0, str(Path(args.repo_root) / "src_experiments" / "006_bioclip25_finetune"))
    from model import BioCLIP25LinearProbe, BIOCLIP25_MODEL_NAME  # noqa: E402
    from dataset import (  # noqa: E402
        load_train_metadata, resolve_image_paths, build_class_mapping,
    )

    LOG.info(f"Loading SSL ckpt: {args.ssl_ckpt}")
    ssl = torch.load(args.ssl_ckpt, map_location="cpu", weights_only=False)
    backbone_sd = ssl["backbone_state_dict"]
    model_name = ssl.get("model_name", BIOCLIP25_MODEL_NAME)
    LOG.info(f"  model_name={model_name}, ssl_epoch={ssl.get('epoch', '?')}")
    LOG.info(f"  backbone keys: {len(backbone_sd):,}")

    LOG.info(f"Resolving species ordering from {args.train_meta_csv}")
    df = load_train_metadata(args.train_meta_csv)
    df = resolve_image_paths(df, args.train_image_root, verify=False)
    df = df[df["resolved_path"].notna()].reset_index(drop=True)
    _, idx_to_species = build_class_mapping(df)
    num_classes = len(idx_to_species)
    LOG.info(f"  num_classes={num_classes:,}")

    LOG.info("Building BioCLIP25LinearProbe (backbone fully frozen, fresh head)")
    model = BioCLIP25LinearProbe(num_classes=num_classes, model_name=model_name,
                                 unfreeze_blocks=0)

    missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
    LOG.info(f"  backbone load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        LOG.warning(f"  first 5 missing: {missing[:5]}")
    if unexpected:
        LOG.warning(f"  first 5 unexpected: {unexpected[:5]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": -1,  # 006's load_checkpoint sets start_epoch = epoch + 1 = 0
        "model_state_dict": model.state_dict(),
        "idx_to_species": idx_to_species,
        "config": {
            "model_name": model_name,
            "ssl_init": True,
            "ssl_source_ckpt": str(args.ssl_ckpt),
            "ssl_epoch": ssl.get("epoch"),
        },
        "metrics": {},
    }
    torch.save(ckpt, out_path)
    LOG.info(f"Saved 006-format ckpt → {out_path}")
    LOG.info(
        "Next step:\n"
        f"  python src_experiments/006_bioclip25_finetune/train.py \\\n"
        f"      --resume {out_path} \\\n"
        f"      --train-meta-csv {args.train_meta_csv} \\\n"
        f"      --train-image-root {args.train_image_root} \\\n"
        f"      --unfreeze-blocks 4 --epochs 5 \\\n"
        f"      --batch-size 128 --lr 1e-3 --backbone-lr-scale 0.1 \\\n"
        f"      --output-dir src_experiments/012_bioclip25_ssl_lucas/outputs/finetune"
    )


if __name__ == "__main__":
    main()
