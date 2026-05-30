"""
Geometry-correct LUCAS pseudo-quadrats so they match the PlantCLEF 2026 test
distribution before SSL.

EDA finding (`eda/plantclef2026_eda.ipynb`, commit f87e0a7):

    Test (N=2105):  800 px max-side, aspect ≈ 1.00 (std 0.10),  0.60 Mpx.
    LUCAS (N=139):  1740 × 1299,     aspect 1.34,                2.33 Mpx.

LUCAS is NOT a drop-in proxy. The 008 Phase B v3 pseudo-distillation got
0.227 partly because LUCAS shape/scale never matched test. We attack that
here with a deterministic preprocessing pass:

    1. Center-crop to square (aspect 1.34 → 1.00).
    2. Resize so max-side = 800 px (matches test pixel budget).
    3. Save as JPEG quality 95 to `--out-dir`.

Run once on RunPod, output is then the SSL training corpus.

    python prepare_lucas.py \\
        --in-dir /workspace/plantclef/raw/pseudo_quadrats \\
        --out-dir /workspace/plantclef/processed/lucas_aspect_corrected \\
        --max-side 800 \\
        --workers 16
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

LOG = logging.getLogger("prepare_lucas")


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True, help="Root of LUCAS pseudo_quadrats.")
    p.add_argument("--out-dir", required=True, help="Where to write corrected JPEGs.")
    p.add_argument("--max-side", type=int, default=800,
                   help="Max-side in pixels of output. PlantCLEF test is ~800.")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0,
                   help="Optional cap for smoke tests (0 = all).")
    return p.parse_args()


def list_inputs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)


def correct_one(args: tuple[Path, Path, int, int]) -> tuple[Path, str]:
    src, dst, max_side, quality = args
    if dst.exists():
        return src, "skip"
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            short = min(w, h)
            x0 = (w - short) // 2
            y0 = (h - short) // 2
            im = im.crop((x0, y0, x0 + short, y0 + short))

            if short > max_side:
                im = im.resize((max_side, max_side), Image.BICUBIC)

            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=quality)
        return src, "ok"
    except Exception as e:
        return src, f"err: {e}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = list_inputs(in_root)
    if args.limit:
        paths = paths[: args.limit]
    LOG.info(f"Found {len(paths):,} LUCAS images under {in_root}")

    jobs: list[tuple[Path, Path, int, int]] = []
    for src in paths:
        rel = src.relative_to(in_root).with_suffix(".jpg")
        dst = out_root / rel
        jobs.append((src, dst, args.max_side, args.quality))

    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(correct_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            src, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                LOG.warning(f"{src} → {status}")
            if (i + 1) % 1000 == 0:
                LOG.info(f"  progress: {i+1:,}/{len(jobs):,}  ok={n_ok} skip={n_skip} err={n_err}")

    LOG.info(f"Done. ok={n_ok:,} skip={n_skip:,} err={n_err:,}  out={out_root}")


if __name__ == "__main__":
    main()
