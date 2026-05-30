"""
single_plant/inf_script_whole.py
================================

Whole-image inference for PlantNet-300K-style data, where every image
already contains a single centred subject (no quadrat aggregation).

Pipeline
--------
* load checkpoint  : `shared/bioclip25_multitask/model.py::load_checkpoint_model`
* preprocess       : `shared/bioclip25_multitask/transforms.py::val_transform`
                     applied at one or two resolutions (default 224 + 336)
* forward          : centre-cropped image  ->  species_logits
* aggregate        : softmax-mean across resolutions
* output           : a submission CSV with one row per image

This script is deliberately the simplest-possible inference path: no
tile sampling, no anchor / k selection, no logit adjustment. Those
strategies are quadrat-specific and live in `quadrat/`.

Usage
-----

    python single_plant/inf_script_whole.py \\
        --checkpoint /path/to/best.pt \\
        --image-root /mnt/d/PlantNet-300k/images/images/test \\
        --output     outputs/single_plant_whole/submission.csv \\
        --resolutions 224 336 \\
        --top-k 5

The image-root may be either
  * a directory containing JPEGs / a flat tree (recursively scanned), or
  * a manifest CSV with an ``image_path`` column.

The script writes a CSV with columns
  ``image_id, top1_species_id, top1_prob, topk_species_ids``
where ``topk_species_ids`` is a semicolon-separated list of the K
highest-probability species, sorted descending.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

EXP_DIR = (Path(__file__).resolve().parent.parent
           / "shared" / "bioclip25_multitask")
if not EXP_DIR.is_dir():
    raise SystemExit(
        f"Expected shared dir not found: {EXP_DIR}\n"
        f"Edit EXP_DIR in this file if the BioCLIP 2.5 multitask sources live elsewhere."
    )
sys.path.insert(0, str(EXP_DIR))

from model import load_checkpoint_model       # noqa: E402
from transforms import val_transform          # noqa: E402
from utils import resolve_device, amp_autocast  # noqa: E402


# ─── Positional-embedding resampling ──────────────────────────────────────
# Identical to the helper in quadrat/inf_script.py: open_clip ViT-H/14
# ships with pos_embed sized for 224 px (16x16 patch grid + CLS = 257
# entries) and won't auto-interpolate on forward. Resample bicubically
# for any other input size. Idempotent at the source grid.

def _resample_pos_embed(model, res: int) -> None:
    visual     = model.backbone.visual
    patch_size = visual.conv1.kernel_size[0]    # 14 for ViT-H/14
    new_grid   = res // patch_size

    pe       = visual.positional_embedding      # (N+1, D)
    cls_pe   = pe[:1]
    grid_pe  = pe[1:]
    old_grid = int(math.sqrt(grid_pe.shape[0]))
    if old_grid == new_grid:
        return

    grid_pe = (
        grid_pe.reshape(old_grid, old_grid, -1).permute(2, 0, 1).unsqueeze(0)
    )
    grid_pe = F.interpolate(
        grid_pe.float(),
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=False,
    ).to(pe.dtype)
    grid_pe = grid_pe.squeeze(0).permute(1, 2, 0).reshape(new_grid * new_grid, -1)

    visual.positional_embedding = torch.nn.Parameter(
        torch.cat([cls_pe, grid_pe], dim=0)
    )


# ─── Image source: directory walk or manifest CSV ─────────────────────────

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def gather_images(image_root: Path) -> list[tuple[str, Path]]:
    """Return list of (image_id, absolute_path) tuples.

    image_root may be a directory (recursively walked) or a CSV with an
    `image_path` column (optionally also `image_id`).
    """
    if image_root.is_dir():
        items: list[tuple[str, Path]] = []
        for p in sorted(image_root.rglob("*")):
            if p.suffix.lower() in _VALID_EXTS and p.is_file():
                items.append((p.stem, p.resolve()))
        return items

    if image_root.is_file() and image_root.suffix.lower() == ".csv":
        df = pd.read_csv(image_root)
        if "image_path" not in df.columns:
            raise SystemExit(f"manifest {image_root} missing image_path column")
        if "image_id" not in df.columns:
            df["image_id"] = df["image_path"].map(lambda p: Path(p).stem)
        return [
            (str(row.image_id), Path(row.image_path).resolve())
            for row in df.itertuples(index=False)
        ]

    raise SystemExit(f"image-root must be a directory or CSV: {image_root}")


# ─── Inference core ───────────────────────────────────────────────────────

@torch.inference_mode()
def infer_one_image(
    model: torch.nn.Module,
    img: Image.Image,
    resolutions: tuple[int, ...],
    device: str,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> torch.Tensor:
    """Run softmax-mean inference across resolutions. Returns shape [S]."""
    probs_acc: list[torch.Tensor] = []
    for res in resolutions:
        _resample_pos_embed(model, res)
        x = val_transform(res)(img).unsqueeze(0).to(device, non_blocking=True)
        with amp_autocast(device, enabled=amp_enabled, dtype=amp_dtype):
            sp_logits, *_ = model(x)
        probs_acc.append(F.softmax(sp_logits.float(), dim=-1)[0])
    return torch.stack(probs_acc, dim=0).mean(dim=0)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Whole-image (single-plant) inference for BioCLIP 2.5 multitask checkpoints.",
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to a shared/bioclip25_multitask trained .pt")
    p.add_argument("--image-root", required=True, type=Path,
                   help="Directory of images (recursively scanned) OR a CSV with image_path")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination submission CSV")
    p.add_argument("--resolutions", type=int, nargs="+", default=[224, 336],
                   help="One or more resolutions to ensemble via softmax-mean")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default=None,
                   help="cuda / cpu / mps; auto-detected if omitted")
    p.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only process this many images (smoke test)")
    args = p.parse_args()

    device = args.device if args.device else resolve_device()
    amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    amp_enabled = args.precision != "fp32"

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[inf_single_plant] device={device}  precision={args.precision}  "
          f"resolutions={tuple(args.resolutions)}")
    print(f"[inf_single_plant] loading checkpoint {args.checkpoint}")
    model, encoders, _config = load_checkpoint_model(str(args.checkpoint), device=device)
    model.eval()

    idx_to_species = encoders.get("idx_to_species")
    if idx_to_species is None:
        raise SystemExit(
            "Checkpoint missing encoders['idx_to_species']. Train with the "
            "multitask code in shared/bioclip25_multitask/ to embed the encoder."
        )

    items = gather_images(args.image_root)
    if args.limit:
        items = items[: args.limit]
    print(f"[inf_single_plant] {len(items):,} images to process")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows: list[tuple[str, str, float, str]] = []
    errors = 0

    for i, (image_id, path) in enumerate(items, 1):
        try:
            with Image.open(path) as raw:
                img = raw.convert("RGB")
            probs = infer_one_image(
                model, img, tuple(args.resolutions), device, amp_dtype, amp_enabled
            )
            topk = torch.topk(probs, k=min(args.top_k, probs.numel()))
            top_idx = topk.indices.tolist()
            top_p   = topk.values.tolist()
            top_species = [str(idx_to_species[i]) for i in top_idx]
            rows.append(
                (
                    image_id,
                    top_species[0],
                    float(top_p[0]),
                    ";".join(top_species),
                )
            )
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [warn] {path}: {e}", file=sys.stderr)
            continue

        if i % args.log_every == 0 or i == len(items):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta = (len(items) - i) / max(rate, 1e-6)
            print(
                f"  {i:>7,}/{len(items):,}  ·  {rate:6.2f} img/s  ·  "
                f"ETA {eta:6.0f}s  ·  errors={errors}"
            )

    with args.output.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "top1_species_id", "top1_prob", "topk_species_ids"])
        w.writerows(rows)

    dt = time.time() - t0
    print(
        f"[inf_single_plant] wrote {len(rows):,} rows to {args.output}\n"
        f"[inf_single_plant] {dt:.1f}s total  ·  {len(rows)/max(dt,1e-6):.2f} img/s  ·  errors={errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
