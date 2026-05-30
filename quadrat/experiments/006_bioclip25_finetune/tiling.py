"""
Tiled inference for high-resolution quadrat images.

Tiling modes
------------
  whole            : single tile = the full image (unresized)
  grid_NxN         : NxN grid  (N=2..8 named; or use generic 'grid' + --grid-size)
  grid             : generic NxN grid driven by --grid-size
  hstrips_N        : N horizontal strips  (N=2,3)
  vstrips_N        : N vertical strips    (N=2,3)
  center_crop      : single centred square crop
  five_crop        : center + 4 corners  (5 tiles)
  sliding          : sliding-window tiles driven by --tile-size / --stride
  multiscale       : whole + grid_2x2 + grid_4x4  (backward-compat alias)
  multiscale_dense : whole + grids from --scales + opt. sliding window

Tile metadata
-------------
  Every tile carries a TileInfo namedtuple:
    (left, top, right, bottom, mode_name, tile_index)
  The public tile_image() function returns List[PIL.Image] for the existing
  encode_tiles() pipeline.  Internally, _tile_image_with_info() returns
  List[Tuple[TileInfo, PIL.Image]] which is used for debug previews.

Overlap
-------
  overlap_ratio : fraction of cell size added on each side during grid/strip
                  tiling.  0.0 = exact grid, 0.25 = 25% margin each side.

Aggregation modes
-----------------
  max         : element-wise max over all tile logits
  topk_mean   : average logits from the k tiles with the highest max-logit
"""

from __future__ import annotations

import logging
import re
from typing import Callable, NamedTuple

import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Named grid modes up to 8×8. The dispatcher handles any grid_NxN pattern,
# so adding larger grids here just makes them available as --tile-mode choices.
TILING_MODES = (
    "whole",
    "grid_2x2", "grid_3x3", "grid_4x4", "grid_5x5",
    "grid_6x6", "grid_7x7", "grid_8x8",
    "grid",
    "hstrips_2", "hstrips_3",
    "vstrips_2", "vstrips_3",
    "center_crop",
    "five_crop",
    "sliding",
    "multiscale",
    "multiscale_dense",
)

AGG_MODES = ("max", "topk_mean")

# Regex for any grid_NxN mode string (N must match on both sides).
_GRID_RE = re.compile(r"^grid_(\d+)x(\d+)$")

# Warn if a single image produces more tiles than this (before max-tiles cap)
_TILE_COUNT_WARN = 256


# ---------------------------------------------------------------------------
# Tile metadata
# ---------------------------------------------------------------------------

class TileInfo(NamedTuple):
    left:       int
    top:        int
    right:      int
    bottom:     int
    mode_name:  str
    tile_index: int


# ---------------------------------------------------------------------------
# Low-level crop helpers  (all return List[Tuple[TileInfo, Image]])
# ---------------------------------------------------------------------------

def _clamp_box(left: float, top: float, right: float, bottom: float,
               w: int, h: int) -> tuple[int, int, int, int]:
    """Round and clamp a crop box to image bounds."""
    l = max(0, int(round(left)))
    t = max(0, int(round(top)))
    r = min(w, int(round(right)))
    b = min(h, int(round(bottom)))
    # Guarantee non-zero area
    r = max(l + 1, r)
    b = max(t + 1, b)
    return l, t, r, b


def _make_tiles(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    mode_name: str,
) -> list[tuple[TileInfo, Image.Image]]:
    """Crop each box from *image* and attach TileInfo."""
    w, h = image.size
    result: list[tuple[TileInfo, Image.Image]] = []
    for idx, (l, t, r, b) in enumerate(boxes):
        cl, ct, cr, cb = _clamp_box(l, t, r, b, w, h)
        info = TileInfo(cl, ct, cr, cb, mode_name, idx)
        result.append((info, image.crop((cl, ct, cr, cb))))
    return result


# ---- whole -----------------------------------------------------------------

def _tiles_whole(image: Image.Image) -> list[tuple[TileInfo, Image.Image]]:
    w, h = image.size
    return _make_tiles(image, [(0, 0, w, h)], "whole")


# ---- NxN grid --------------------------------------------------------------

def _tiles_grid(
    image: Image.Image,
    grid_size: int,
    overlap_ratio: float = 0.0,
    mode_name: str | None = None,
) -> list[tuple[TileInfo, Image.Image]]:
    w, h = image.size
    cell_w = w / grid_size
    cell_h = h / grid_size
    mode_name = mode_name or f"grid_{grid_size}x{grid_size}"

    boxes: list[tuple[float, float, float, float]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = (col + 1) * cell_w
            y2 = (row + 1) * cell_h
            if overlap_ratio > 0.0:
                dx = cell_w * overlap_ratio
                dy = cell_h * overlap_ratio
                x1 -= dx
                y1 -= dy
                x2 += dx
                y2 += dy
            boxes.append((x1, y1, x2, y2))

    return _make_tiles(image, boxes, mode_name)


# ---- horizontal / vertical strips ------------------------------------------

def _tiles_hstrips(
    image: Image.Image,
    n_strips: int,
    overlap_ratio: float = 0.0,
) -> list[tuple[TileInfo, Image.Image]]:
    w, h = image.size
    strip_h = h / n_strips
    boxes: list[tuple[float, float, float, float]] = []
    for i in range(n_strips):
        y1 = i * strip_h
        y2 = (i + 1) * strip_h
        if overlap_ratio > 0.0:
            dy = strip_h * overlap_ratio
            y1 -= dy
            y2 += dy
        boxes.append((0, y1, w, y2))
    return _make_tiles(image, boxes, f"hstrips_{n_strips}")


def _tiles_vstrips(
    image: Image.Image,
    n_strips: int,
    overlap_ratio: float = 0.0,
) -> list[tuple[TileInfo, Image.Image]]:
    w, h = image.size
    strip_w = w / n_strips
    boxes: list[tuple[float, float, float, float]] = []
    for i in range(n_strips):
        x1 = i * strip_w
        x2 = (i + 1) * strip_w
        if overlap_ratio > 0.0:
            dx = strip_w * overlap_ratio
            x1 -= dx
            x2 += dx
        boxes.append((x1, 0, x2, h))
    return _make_tiles(image, boxes, f"vstrips_{n_strips}")


# ---- center_crop / five_crop -----------------------------------------------

def _tiles_center_crop(
    image: Image.Image,
    tile_size: int,
) -> list[tuple[TileInfo, Image.Image]]:
    w, h = image.size
    s = min(w, h, tile_size)
    cx, cy = w // 2, h // 2
    boxes = [(cx - s // 2, cy - s // 2, cx - s // 2 + s, cy - s // 2 + s)]
    return _make_tiles(image, boxes, "center_crop")


def _tiles_five_crop(
    image: Image.Image,
    tile_size: int,
) -> list[tuple[TileInfo, Image.Image]]:
    w, h = image.size
    s = min(w, h, tile_size)
    cx, cy = w // 2, h // 2
    half = s // 2
    boxes = [
        # center
        (cx - half, cy - half, cx - half + s, cy - half + s),
        # top-left
        (0, 0, s, s),
        # top-right
        (w - s, 0, w, s),
        # bottom-left
        (0, h - s, s, h),
        # bottom-right
        (w - s, h - s, w, h),
    ]
    return _make_tiles(image, boxes, "five_crop")


# ---- sliding window --------------------------------------------------------

def _tiles_sliding(
    image: Image.Image,
    tile_size: int,
    stride: int,
) -> list[tuple[TileInfo, Image.Image]]:
    """
    Sliding window over the image.  The last column/row always includes the
    right/bottom edge of the image to avoid missing content.
    """
    w, h = image.size
    # If image is smaller than tile_size, fall back to whole
    if w <= tile_size and h <= tile_size:
        return _tiles_whole(image)

    effective_w = max(tile_size, w)
    effective_h = max(tile_size, h)

    xs = list(range(0, effective_w - tile_size + 1, stride))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))

    ys = list(range(0, effective_h - tile_size + 1, stride))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))

    boxes: list[tuple[float, float, float, float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for y in ys:
        for x in xs:
            box = (x, y, min(x + tile_size, w), min(y + tile_size, h))
            if box not in seen:
                seen.add(box)
                boxes.append(box)

    return _make_tiles(image, boxes, "sliding")


# ---- multiscale (backward-compat) ------------------------------------------

def _tiles_multiscale(
    image: Image.Image,
    overlap_ratio: float = 0.0,
) -> list[tuple[TileInfo, Image.Image]]:
    """whole + grid_2x2 + grid_4x4  (original 21-tile multiscale)."""
    tiles = _tiles_whole(image)
    tiles += _tiles_grid(image, 2, overlap_ratio, mode_name="grid_2x2")
    tiles += _tiles_grid(image, 4, overlap_ratio, mode_name="grid_4x4")
    return tiles


# ---- multiscale_dense -------------------------------------------------------

def _tiles_multiscale_dense(
    image: Image.Image,
    scales: list[int],
    overlap_ratio: float = 0.0,
    tile_size: int = 224,
    stride: int = 0,
    include_sliding: bool = False,
) -> list[tuple[TileInfo, Image.Image]]:
    """
    whole image + one NxN grid per scale N + optional sliding window.

    scales         : list of grid sizes, e.g. [1, 2, 3, 4]
                     scale=1 is equivalent to whole (skipped, whole always added)
    include_sliding: if True, also append sliding-window tiles
    """
    tiles: list[tuple[TileInfo, Image.Image]] = _tiles_whole(image)
    for n in scales:
        if n <= 1:
            continue  # whole already included
        tiles += _tiles_grid(image, n, overlap_ratio, mode_name=f"grid_{n}x{n}")
    if include_sliding:
        eff_stride = stride if stride > 0 else max(1, tile_size // 2)
        tiles += _tiles_sliding(image, tile_size, eff_stride)
    return tiles


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _tile_image_with_info(
    image: Image.Image,
    mode: str,
    overlap_ratio: float = 0.0,
    tile_size: int = 224,
    stride: int = 0,
    grid_size: int | None = None,
    scales: list[int] | None = None,
    max_tiles: int | None = None,
) -> list[tuple[TileInfo, Image.Image]]:
    """
    Internal function returning (TileInfo, PIL.Image) pairs.
    Applies max_tiles cap with deterministic truncation if set.
    """
    w, h = image.size

    if mode == "whole":
        tiles = _tiles_whole(image)

    elif m := _GRID_RE.match(mode):
        # Handles grid_2x2, grid_3x3, ..., grid_8x8 and any other grid_NxN
        rows, cols = int(m.group(1)), int(m.group(2))
        if rows != cols:
            raise ValueError(
                f"Only square grids supported; got {mode!r} ({rows}x{cols})"
            )
        tiles = _tiles_grid(image, rows, overlap_ratio)

    elif mode == "grid":
        n = grid_size if grid_size is not None else 3
        tiles = _tiles_grid(image, n, overlap_ratio, mode_name=f"grid_{n}x{n}")

    elif mode == "hstrips_2":
        tiles = _tiles_hstrips(image, 2, overlap_ratio)
    elif mode == "hstrips_3":
        tiles = _tiles_hstrips(image, 3, overlap_ratio)
    elif mode == "vstrips_2":
        tiles = _tiles_vstrips(image, 2, overlap_ratio)
    elif mode == "vstrips_3":
        tiles = _tiles_vstrips(image, 3, overlap_ratio)

    elif mode == "center_crop":
        tiles = _tiles_center_crop(image, tile_size)

    elif mode == "five_crop":
        tiles = _tiles_five_crop(image, tile_size)

    elif mode == "sliding":
        eff_stride = stride if stride > 0 else max(1, tile_size // 2)
        tiles = _tiles_sliding(image, tile_size, eff_stride)

    elif mode == "multiscale":
        tiles = _tiles_multiscale(image, overlap_ratio)

    elif mode == "multiscale_dense":
        eff_scales = scales if scales else [1, 2, 3, 4]
        eff_stride = stride if stride > 0 else max(1, tile_size // 2)
        tiles = _tiles_multiscale_dense(
            image,
            scales=eff_scales,
            overlap_ratio=overlap_ratio,
            tile_size=tile_size,
            stride=eff_stride,
        )

    else:
        raise ValueError(
            f"Unknown tiling mode {mode!r}. Choose from: {TILING_MODES}"
        )

    if len(tiles) > _TILE_COUNT_WARN:
        logger.warning(
            f"tile_image: mode={mode!r} produced {len(tiles)} tiles for "
            f"image size {w}x{h} - this may be slow. "
            f"Consider --max-tiles-per-image to cap."
        )

    if max_tiles is not None and len(tiles) > max_tiles:
        logger.info(
            f"tile_image: truncating {len(tiles)} → {max_tiles} tiles "
            f"(--max-tiles-per-image={max_tiles})"
        )
        tiles = tiles[:max_tiles]

    return tiles


def tile_image(
    image: Image.Image,
    mode: str,
    overlap_ratio: float = 0.0,
    tile_size: int = 224,
    stride: int = 0,
    grid_size: int | None = None,
    scales: list[int] | None = None,
    max_tiles: int | None = None,
) -> list[Image.Image]:
    """
    Tile a PIL image according to *mode*.

    Parameters
    ----------
    image         : PIL Image (RGB)
    mode          : one of TILING_MODES  (or any valid grid_NxN string)
    overlap_ratio : overlap fraction for grid / strip modes (0 = no overlap)
    tile_size     : pixel size for sliding / center_crop / five_crop modes
    stride        : sliding-window stride (0 = tile_size // 2)
    grid_size     : grid N for the generic 'grid' mode
    scales        : list of grid sizes for 'multiscale_dense'
    max_tiles     : hard cap on number of tiles (deterministic truncation)

    Returns
    -------
    list of PIL Images
    """
    pairs = _tile_image_with_info(
        image, mode,
        overlap_ratio=overlap_ratio,
        tile_size=tile_size,
        stride=stride,
        grid_size=grid_size,
        scales=scales,
        max_tiles=max_tiles,
    )
    return [img for _, img in pairs]


# ---------------------------------------------------------------------------
# Debug tile preview
# ---------------------------------------------------------------------------

def save_tile_preview(
    image: Image.Image,
    mode: str,
    output_path: str,
    overlap_ratio: float = 0.0,
    tile_size: int = 224,
    stride: int = 0,
    grid_size: int | None = None,
    scales: list[int] | None = None,
    max_tiles: int | None = None,
    max_preview_tiles: int = 64,
) -> None:
    """
    Draw tile bounding boxes on a downscaled copy of *image* and save as PNG.
    """
    pairs = _tile_image_with_info(
        image, mode,
        overlap_ratio=overlap_ratio,
        tile_size=tile_size,
        stride=stride,
        grid_size=grid_size,
        scales=scales,
        max_tiles=max_tiles,
    )

    W, H = image.size
    MAX_SIDE = 1024
    scale = min(1.0, MAX_SIDE / max(W, H))
    preview_w = max(1, int(W * scale))
    preview_h = max(1, int(H * scale))
    preview = image.resize((preview_w, preview_h), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(preview, "RGBA")

    colors = [
        (255, 80,  80,  140),
        (80,  200, 80,  140),
        (80,  120, 255, 140),
        (255, 200, 0,   140),
        (200, 80,  255, 140),
    ]

    capped = pairs[:max_preview_tiles]
    for i, (info, _) in enumerate(capped):
        color = colors[i % len(colors)]
        l = int(info.left  * scale)
        t = int(info.top   * scale)
        r = int(info.right * scale)
        b = int(info.bottom * scale)
        draw.rectangle([l, t, r - 1, b - 1], outline=color[:3], width=2)
        draw.text((l + 3, t + 2), str(info.tile_index), fill=(255, 255, 255, 230))

    preview.save(output_path, format="PNG")
    logger.info(
        f"Tile preview saved: {output_path}  "
        f"({len(pairs)} tiles shown, capped at {max_preview_tiles})"
    )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_tiles(
    backbone_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    preprocess: Callable[[Image.Image], torch.Tensor],
    tiles: list[Image.Image],
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Encode a list of tiles through the backbone.

    Returns (n_tiles, embed_dim) float32 CPU tensor; NOT L2-normalised.
    """
    if not tiles:
        raise ValueError("encode_tiles: received an empty tile list")

    all_features: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_imgs = torch.stack(
                [preprocess(t) for t in tiles[i : i + batch_size]]
            ).to(device)
            feats = backbone_encode_fn(batch_imgs)
            all_features.append(feats.cpu())

    return torch.cat(all_features, dim=0)


# ---------------------------------------------------------------------------
# Classification (head forward)
# ---------------------------------------------------------------------------

@torch.no_grad()
def classify_tiles(
    head: torch.nn.Module,
    tile_features: torch.Tensor,
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Apply the linear head to tile features. Returns (n_tiles, num_classes) on CPU."""
    head.eval()
    all_logits: list[torch.Tensor] = []
    for i in range(0, len(tile_features), batch_size):
        batch = tile_features[i : i + batch_size].to(device)
        all_logits.append(head(batch).cpu())
    return torch.cat(all_logits, dim=0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_logits(
    tile_logits: torch.Tensor,
    mode: str = "max",
    topk: int = 5,
) -> torch.Tensor:
    """
    Aggregate per-tile logits to a single image-level score.

    Parameters
    ----------
    tile_logits : (n_tiles, num_classes) tensor
    mode        : 'max' | 'topk_mean'
    topk        : k for topk_mean

    Returns
    -------
    (num_classes,) image-level logit tensor
    """
    if tile_logits.shape[0] == 0:
        raise ValueError("aggregate_logits: empty tile_logits tensor")

    if mode == "max":
        return tile_logits.max(dim=0).values

    elif mode == "topk_mean":
        n_tiles = tile_logits.shape[0]
        k = min(topk, n_tiles)
        tile_scores = tile_logits.max(dim=1).values
        topk_idx = tile_scores.topk(k).indices
        return tile_logits[topk_idx].mean(dim=0)

    else:
        raise ValueError(
            f"Unknown aggregation mode {mode!r}. Choose from: {AGG_MODES}"
        )


# ---------------------------------------------------------------------------
# All-in-one convenience function
# ---------------------------------------------------------------------------

def infer_image(
    image: Image.Image,
    backbone_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    head: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: str,
    tile_mode: str = "whole",
    overlap_ratio: float = 0.0,
    tile_size: int = 224,
    stride: int = 0,
    grid_size: int | None = None,
    scales: list[int] | None = None,
    max_tiles: int | None = None,
    agg_mode: str = "max",
    topk_agg: int = 5,
    tile_batch_size: int = 32,
) -> torch.Tensor:
    """Full tiled inference pipeline for a single image. Returns (num_classes,) logits."""
    tiles = tile_image(
        image,
        mode=tile_mode,
        overlap_ratio=overlap_ratio,
        tile_size=tile_size,
        stride=stride,
        grid_size=grid_size,
        scales=scales,
        max_tiles=max_tiles,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"infer_image: mode={tile_mode!r}, n_tiles={len(tiles)}, "
            f"img_size={image.size}"
        )

    feats = encode_tiles(
        backbone_encode_fn=backbone_encode_fn,
        preprocess=preprocess,
        tiles=tiles,
        device=device,
        batch_size=tile_batch_size,
    )
    tile_logits = classify_tiles(head=head, tile_features=feats, device=device)
    return aggregate_logits(tile_logits, mode=agg_mode, topk=topk_agg)
