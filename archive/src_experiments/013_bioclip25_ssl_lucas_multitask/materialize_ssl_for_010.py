"""
Bridge from SSL pretraining (012's `ssl_train.py`) to 010's multi-task
fine-tuning (`010_bioclip25_end_to_end_finetune_multitask/train.py`).

Why a new bridge: 012's `materialize_ssl_for_006.py` produced a
`BioCLIP25LinearProbe`-shaped ckpt (plain Linear head). 013 wants 010's
`BioCLIP25MultiTask` shape (LayerNorm → Linear(1024→1024) → GELU → Dropout
→ species_head, plus optional genus/family/order/class aux heads).

Steps:
  1. Load SSL ckpt → backbone state_dict.
  2. Read 010 metadata + taxonomy CSVs → build idx_to_{species,genus,family,...}.
  3. Build a fresh BioCLIP25MultiTask with use_taxonomy_heads=True.
  4. Replace `model.backbone` weights with the SSL state_dict.
  5. Save a 010-format ckpt: model_state_dict matches 010 exactly, epoch=-1,
     encoders embedded so 010/train.py with --resume + --resume-weights-only
     loads cleanly.

Run:
    python src_experiments/013_bioclip25_ssl_lucas_multitask/materialize_ssl_for_010.py \
        --ssl-ckpt   src_experiments/012_bioclip25_ssl_lucas/outputs/ssl_bioclip25_backbone_ep5.pt \
        --train-meta-csv /workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv \
        --taxonomy-csv   /workspace/working/workspace/PlantCLEF2026/src_experiments/002_bioclip_tile_zero_shot_v2/data/species_lookup_with_gbif_cleaned_names.csv \
        --out src_experiments/013_bioclip25_ssl_lucas_multitask/outputs/ssl_init_for_010.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

LOG = logging.getLogger("materialize_ssl_010")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ssl-ckpt", required=True,
                   help="012 ssl_train.py output — has backbone_state_dict.")
    p.add_argument("--train-meta-csv", required=True)
    p.add_argument("--taxonomy-csv",   required=True)
    p.add_argument("--out", required=True, help="Output 010-format ckpt.")
    p.add_argument("--use-taxonomy-heads", action="store_true", default=True,
                   help="Build aux heads (genus, family, order, class) in the ckpt.")
    p.add_argument("--no-taxonomy-heads", dest="use_taxonomy_heads",
                   action="store_false")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]),
                   help="Project root (so we can import 010 modules).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    sys.path.insert(0, str(Path(args.repo_root) / "src_experiments"
                             / "010_bioclip25_end_to_end_finetune_multitask"))
    from model import BioCLIP25MultiTask, BIOCLIP25_MODEL_NAME  # noqa: E402
    from dataset import load_metadata, build_label_encoders  # noqa: E402

    LOG.info(f"Loading SSL ckpt: {args.ssl_ckpt}")
    ssl = torch.load(args.ssl_ckpt, map_location="cpu", weights_only=False)
    backbone_sd = ssl["backbone_state_dict"]
    model_name = ssl.get("model_name", BIOCLIP25_MODEL_NAME)
    LOG.info(f"  model_name={model_name}, ssl_epoch={ssl.get('epoch', '?')}")
    LOG.info(f"  backbone keys: {len(backbone_sd):,}")

    LOG.info(f"Loading metadata: {args.train_meta_csv}")
    df = load_metadata(args.train_meta_csv, args.taxonomy_csv)
    encoders = build_label_encoders(df)

    num_species = len(encoders["idx_to_species"])
    num_genus   = len(encoders.get("idx_to_genus",  []))
    num_family  = len(encoders.get("idx_to_family", []))
    num_order   = len(encoders.get("idx_to_order",  []))
    num_class   = len(encoders.get("idx_to_class",  []))

    LOG.info(
        f"  encoders → species={num_species:,} genus={num_genus:,} "
        f"family={num_family:,} order={num_order:,} class={num_class:,}"
    )

    LOG.info(
        f"Building BioCLIP25MultiTask (use_taxonomy_heads={args.use_taxonomy_heads}, "
        f"hidden_dim={args.hidden_dim}, dropout={args.dropout})"
    )
    model = BioCLIP25MultiTask(
        num_species  = num_species,
        num_genus    = num_genus    if args.use_taxonomy_heads else 0,
        num_family   = num_family   if args.use_taxonomy_heads else 0,
        num_order    = num_order    if args.use_taxonomy_heads else 0,
        num_class    = num_class    if args.use_taxonomy_heads else 0,
        model_name   = model_name,
        hidden_dim   = args.hidden_dim,
        dropout      = args.dropout,
        use_taxonomy_heads = args.use_taxonomy_heads,
    )

    missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
    LOG.info(f"  backbone load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        LOG.warning(f"  first 5 missing:    {missing[:5]}")
    if unexpected:
        LOG.warning(f"  first 5 unexpected: {unexpected[:5]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        # epoch=-1 so 010's load_checkpoint(... +1) = 0; ALSO --resume-weights-only
        # forces start_epoch=0 anyway. Safe either way.
        "epoch": -1,
        "model_state_dict": model.state_dict(),
        "encoders": encoders,
        "config": {
            "model_name":          model_name,
            "hidden_dim":          args.hidden_dim,
            "dropout":             args.dropout,
            "use_taxonomy_heads":  args.use_taxonomy_heads,
            "num_species":         num_species,
            "num_genus":           num_genus  if args.use_taxonomy_heads else 0,
            "num_family":          num_family if args.use_taxonomy_heads else 0,
            "num_order":           num_order  if args.use_taxonomy_heads else 0,
            "num_class":           num_class  if args.use_taxonomy_heads else 0,
            "ssl_init":            True,
            "ssl_source_ckpt":     str(args.ssl_ckpt),
            "ssl_epoch":           ssl.get("epoch"),
        },
        "metrics": {},
    }
    torch.save(ckpt, out_path)
    LOG.info(f"Saved 010-format ckpt → {out_path}")
    LOG.info(
        "Next:\n"
        "  cd <repo> && python src_experiments/010_bioclip25_end_to_end_finetune_multitask/train.py \\\n"
        f"      --resume {out_path} --resume-weights-only \\\n"
        f"      --train-meta-csv {args.train_meta_csv} \\\n"
        f"      --taxonomy-csv   {args.taxonomy_csv} \\\n"
        "      --train-image-root /workspace/plantclef/raw/train/images_max_side_800 \\\n"
        "      --unfreeze-last-n-blocks 4 \\\n"
        "      --use-taxonomy-heads \\\n"
        "      --hidden-dim 1024 --dropout 0.2 \\\n"
        "      --epochs 5 --batch-size 128 --precision bf16 \\\n"
        "      --backbone-lr 1e-6 --head-lr 1e-4 \\\n"
        "      --output-dir src_experiments/013_bioclip25_ssl_lucas_multitask/outputs/finetune"
    )


if __name__ == "__main__":
    main()
