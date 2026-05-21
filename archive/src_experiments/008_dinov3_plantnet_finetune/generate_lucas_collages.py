"""
Generate LUCAS-background synthetic collages for Phase B v2 (LUCAS variant).

Why
---
The original 50 k collages used single-plant training photos as backgrounds.
That meant backgrounds were close-up plant shots, not top-down quadrats — which
is why v1's Phase B val F1 of 0.46 translated to 0.0002 on real Kaggle test
quadrats. LUCAS pseudo-quadrats are real top-down 50×50 cm vegetation-plot
photos and close the domain gap.

But the existing 11,734 SAM stickers cover only 1,586 rare species (< 20
training images each). If we used LUCAS backgrounds + stickers only, the
classifier would get positive labels for only 20 % of species — the rare
long-tail that's *least* likely to appear in test quadrats. So we add
rectangle-pasted single-plant crops (soft-edged) drawn from all 7,806 species
to keep label coverage broad.

Pipeline per collage
--------------------
1. Pick random LUCAS pseudo-quadrat → resize shortest-side to CANVAS then
   random-crop to CANVAS × CANVAS.
2. Paste K ∈ [K_MIN, K_MAX] foreground objects:
     - 40 % of objects: existing SAM stickers (alpha-preserved rotation)
     - 60 % of objects: single-plant image → center-square-crop → scale →
       paste with feathered rectangular alpha (no rotation — avoids black
       corners without the precision of SAM alpha).
3. Labels = union of species IDs of all pasted objects (LUCAS bg species
   unknown — we accept the missing-positive noise; ASL tolerates it).
4. Save JPEG q=85 to match test-time preprocessing byte distribution.

Parallelised with multiprocessing.Pool (1 task = 1 collage).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lucas-paths", default="/tmp/lucas_paths.txt",
                   help="Text file with one LUCAS image path per line.")
    p.add_argument("--sticker-dir", default="/workspace/plantclef/processed/stickers")
    p.add_argument("--singleplant-csv", default="/workspace/plantclef/processed/train_metadata_cleaned_verified_stratified.csv")
    p.add_argument("--singleplant-root", default="/workspace/plantclef/raw/train/images_max_side_800")
    p.add_argument("--output-dir", default="/workspace/plantclef/processed/collages_lucas")
    p.add_argument("--output-csv", default="/workspace/plantclef/processed/synthetic_collages_lucas.csv")
    p.add_argument("--n-collages", type=int, default=80000)
    p.add_argument("--canvas-size", type=int, default=448)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=6)
    p.add_argument("--scale-min", type=float, default=0.15)
    p.add_argument("--scale-max", type=float, default=0.40)
    p.add_argument("--sticker-prob", type=float, default=0.40,
                   help="Per-object probability of using a SAM sticker (when one is available for the sampled species).")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--singleplant-per-species", type=int, default=15,
                   help="Cap single-plant images indexed per species (keeps RAM bounded).")
    return p.parse_args()


# ---------- global indices (populated in main, shared via fork) ----------
_LUCAS_PATHS: list[str] = []
_STICKERS: dict[str, list[str]] = {}
_SINGLEPLANT: dict[str, list[str]] = {}
_ALL_SPECIES: list[str] = []
_ARGS: argparse.Namespace | None = None


def load_lucas(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_sticker_index(root: str) -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    rp = Path(root)
    for sd in rp.iterdir():
        if sd.is_dir():
            pngs = [str(p) for p in sd.glob("*.png")]
            if pngs:
                idx[sd.name] = pngs
    return idx


def load_singleplant_index(csv_path: str, root: str, cap: int) -> dict[str, list[str]]:
    """Parse training CSV → map species_id → list of image paths."""
    idx: dict[str, list[str]] = {}
    with open(csv_path) as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            sid = row.get("species_id") or row.get("species_ids")
            img = row.get("image_name")
            if not sid or not img:
                continue
            p = f"{root}/{sid}/{img}"
            idx.setdefault(sid, []).append(p)
    for sid in idx:
        if len(idx[sid]) > cap:
            random.Random(1337).shuffle(idx[sid])
            idx[sid] = idx[sid][:cap]
    return idx


# --------------------- compositing helpers --------------------------------

def _prep_canvas(lucas_path: str, canvas: int) -> np.ndarray | None:
    img = cv2.imread(lucas_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = canvas / min(h, w)
    nh, nw = int(h * s + 0.5), int(w * s + 0.5)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # random crop to canvas × canvas
    top = random.randint(0, nh - canvas)
    left = random.randint(0, nw - canvas)
    return img[top:top + canvas, left:left + canvas].copy()


def _paste_sticker(bg: np.ndarray, sticker_path: str, scale: float,
                   angle: float, cx: int, cy: int) -> bool:
    """Paste a SAM-extracted RGBA sticker onto `bg` in-place. Return True on success."""
    stk = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
    if stk is None or stk.shape[-1] != 4:
        return False
    H, W = bg.shape[:2]
    sh, sw = stk.shape[:2]
    target = int(max(H, W) * scale)
    if target < 8:
        return False
    s = target / max(sh, sw)
    nh, nw = max(8, int(sh * s)), max(8, int(sw * s))
    stk = cv2.resize(stk, (nw, nh), interpolation=cv2.INTER_AREA)
    # rotate with transparent background
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
    stk = cv2.warpAffine(stk, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # place so (cx, cy) is centre
    x0 = cx - nw // 2
    y0 = cy - nh // 2
    # clip to bg
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + nw), min(H, y0 + nh)
    if x1c <= x0c or y1c <= y0c:
        return False
    sx0, sy0 = x0c - x0, y0c - y0
    sx1, sy1 = sx0 + (x1c - x0c), sy0 + (y1c - y0c)
    patch = stk[sy0:sy1, sx0:sx1]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    rgb = patch[:, :, :3].astype(np.float32)
    region = bg[y0c:y1c, x0c:x1c].astype(np.float32)
    blended = rgb * alpha + region * (1.0 - alpha)
    bg[y0c:y1c, x0c:x1c] = blended.astype(np.uint8)
    return True


def _paste_rectangle(bg: np.ndarray, plant_path: str, scale: float,
                     cx: int, cy: int, feather_frac: float = 0.12) -> bool:
    """Paste a feathered center-square crop of a single-plant JPG onto `bg`."""
    img = cv2.imread(plant_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    img = img[top:top + side, left:left + side]
    H, W = bg.shape[:2]
    target = int(max(H, W) * scale)
    if target < 8:
        return False
    img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    feather = max(3, int(target * feather_frac))
    # build soft rectangular alpha (1 in centre, Gaussian-feathered to 0 at edge)
    alpha = np.zeros((target, target), dtype=np.float32)
    alpha[feather:target - feather, feather:target - feather] = 1.0
    k = 2 * feather + 1
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    # paste
    x0 = cx - target // 2
    y0 = cy - target // 2
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + target), min(H, y0 + target)
    if x1c <= x0c or y1c <= y0c:
        return False
    sx0, sy0 = x0c - x0, y0c - y0
    sx1, sy1 = sx0 + (x1c - x0c), sy0 + (y1c - y0c)
    rgb = img[sy0:sy1, sx0:sx1].astype(np.float32)
    a = alpha[sy0:sy1, sx0:sx1, None]
    region = bg[y0c:y1c, x0c:x1c].astype(np.float32)
    blended = rgb * a + region * (1.0 - a)
    bg[y0c:y1c, x0c:x1c] = blended.astype(np.uint8)
    return True


# ---------- per-collage worker ----------

def make_one(i: int) -> dict | None:
    """Generate one collage. Returns {'image_name': ..., 'species_ids': ...} or None on failure."""
    rng = random.Random(_ARGS.seed + i)
    # background
    for _ in range(5):
        bg_path = rng.choice(_LUCAS_PATHS)
        bg = _prep_canvas(bg_path, _ARGS.canvas_size)
        if bg is not None:
            break
    else:
        return None

    labels: set[str] = set()
    K = rng.randint(_ARGS.k_min, _ARGS.k_max)
    for _ in range(K):
        sid = rng.choice(_ALL_SPECIES)
        use_sticker = (sid in _STICKERS) and (rng.random() < _ARGS.sticker_prob)
        scale = rng.uniform(_ARGS.scale_min, _ARGS.scale_max)
        cx = rng.randint(0, _ARGS.canvas_size - 1)
        cy = rng.randint(0, _ARGS.canvas_size - 1)
        ok = False
        if use_sticker:
            stk_path = rng.choice(_STICKERS[sid])
            angle = rng.uniform(0, 360)
            ok = _paste_sticker(bg, stk_path, scale, angle, cx, cy)
        else:
            if sid in _SINGLEPLANT:
                plant_path = rng.choice(_SINGLEPLANT[sid])
                ok = _paste_rectangle(bg, plant_path, scale, cx, cy)
            elif sid in _STICKERS:  # fallback if no single-plant
                stk_path = rng.choice(_STICKERS[sid])
                angle = rng.uniform(0, 360)
                ok = _paste_sticker(bg, stk_path, scale, angle, cx, cy)
        if ok:
            labels.add(sid)

    if not labels:
        return None

    name = f"collage_lucas_{i:07d}.jpg"
    out_path = os.path.join(_ARGS.output_dir, name)
    cv2.imwrite(out_path, bg, [cv2.IMWRITE_JPEG_QUALITY, _ARGS.jpeg_quality])
    return {"image_name": name, "species_ids": ",".join(sorted(labels))}


def _init_worker(args, lucas, stickers, singleplant, all_species):
    global _ARGS, _LUCAS_PATHS, _STICKERS, _SINGLEPLANT, _ALL_SPECIES
    _ARGS = args
    _LUCAS_PATHS = lucas
    _STICKERS = stickers
    _SINGLEPLANT = singleplant
    _ALL_SPECIES = all_species


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    global _ARGS
    _ARGS = parse_args()
    random.seed(_ARGS.seed)
    os.makedirs(_ARGS.output_dir, exist_ok=True)

    LOG.info("Loading LUCAS paths …")
    lucas = load_lucas(_ARGS.lucas_paths)
    LOG.info(f"  LUCAS images: {len(lucas):,}")

    LOG.info("Indexing stickers …")
    stickers = load_sticker_index(_ARGS.sticker_dir)
    LOG.info(f"  sticker species: {len(stickers):,}, files: {sum(len(v) for v in stickers.values()):,}")

    LOG.info("Indexing single-plant training images …")
    singleplant = load_singleplant_index(_ARGS.singleplant_csv, _ARGS.singleplant_root,
                                         _ARGS.singleplant_per_species)
    LOG.info(f"  single-plant species: {len(singleplant):,}, indexed files: {sum(len(v) for v in singleplant.values()):,}")

    all_species = sorted(set(list(singleplant.keys()) + list(stickers.keys())))
    LOG.info(f"  union species (sampler pool): {len(all_species):,}")

    # CSV handle (line-buffered so progress is durable)
    LOG.info(f"Generating {_ARGS.n_collages:,} collages with {_ARGS.workers} workers …")
    t0 = time.time()
    n_ok = 0
    n_err = 0
    with open(_ARGS.output_csv, "w") as fout:
        fout.write("image_name;species_ids\n")
        pool = Pool(_ARGS.workers, initializer=_init_worker,
                    initargs=(_ARGS, lucas, stickers, singleplant, all_species))
        try:
            for i, rec in enumerate(pool.imap_unordered(make_one, range(_ARGS.n_collages), chunksize=64)):
                if rec is None:
                    n_err += 1
                    continue
                fout.write(f"{rec['image_name']};{rec['species_ids']}\n")
                n_ok += 1
                if n_ok % 2000 == 0:
                    fout.flush()
                    dt = time.time() - t0
                    rate = n_ok / max(dt, 1e-6)
                    eta = (_ARGS.n_collages - n_ok) / max(rate, 1e-6)
                    LOG.info(f"  {n_ok:,}/{_ARGS.n_collages:,} ok ({n_err} err), "
                             f"rate={rate:.1f}/s, eta={eta/60:.1f}m")
        finally:
            pool.close()
            pool.join()
    dt = time.time() - t0
    LOG.info(f"Done in {dt/60:.1f} min. {n_ok:,} ok, {n_err:,} err.")
    LOG.info(f"  CSV:   {_ARGS.output_csv}")
    LOG.info(f"  Root:  {_ARGS.output_dir}")


if __name__ == "__main__":
    main()
