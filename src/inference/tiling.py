"""Sliding-window tile generation for high-resolution images.

Supports:
- Configurable tile size and stride (fractional overlap).
- Full-image coverage including right/bottom border tiles.
- Optional multi-scale tiling (image is downscaled before tiling, coordinates
  are reported in the *original* image's pixel space).
"""

from __future__ import annotations

import logging
from typing import Iterator, TYPE_CHECKING

from PIL import Image

import torch
import numpy as np
try:
    import plantclef_ext
except ImportError:
    plantclef_ext = None

if TYPE_CHECKING:
    from .config import TilingConfig
from .types import TileSpec

logger = logging.getLogger(__name__)


def generate_tiles(
    image: Image.Image,
    image_id: str,
    cfg: TilingConfig,
) -> list[tuple[TileSpec, Image.Image]]:
    """Generate all tiles for one image across all configured scales."""
    
    if cfg.method == "grid":
        return _generate_grid_tiles(image, image_id, cfg)

    # Attempt CUDA acceleration for 1.0 scale (most common)
    if plantclef_ext is not None and torch.cuda.is_available() and len(cfg.scales) == 1 and cfg.scales[0] == 1.0:
        tiles = _generate_tiles_cuda(image, image_id, cfg)
        if tiles: # Only return if CUDA actually produced tiles
            return tiles

    # Standard Fallback (CPU Sliding Window)
    results: list[tuple[TileSpec, Image.Image]] = []
    tile_id = 0

    for scale in cfg.scales:
        scaled = _scale_image(image, scale)
        scale_tiles = _slide_window(scaled, image_id, cfg.tile_size, cfg.stride, scale)

        for spec, tile_img in scale_tiles:
            spec.tile_id = tile_id
            results.append((spec, tile_img))
            tile_id += 1

    return results


def _generate_grid_tiles(image: Image.Image, image_id: str, cfg: TilingConfig) -> list[tuple[TileSpec, Image.Image]]:
    """Divide image into a fixed GxG grid."""
    results = []
    W, H = image.width, image.height
    G = cfg.grid_size
    
    tile_w = W // G
    tile_h = H // G
    
    tile_id = 0
    for row in range(G):
        for col in range(G):
            x0 = col * tile_w
            y0 = row * tile_h
            # Last tile in row/col takes remainder
            x1 = W if col == (G - 1) else (x0 + tile_w)
            y1 = H if row == (G - 1) else (y0 + tile_h)
            
            tile = image.crop((x0, y0, x1, y1))
            if tile.mode != "RGB":
                tile = tile.convert("RGB")
                
            spec = TileSpec(
                tile_id=tile_id,
                image_id=image_id,
                x0=x0, y0=y0,
                x1=x1, y1=y1,
                scale=1.0
            )
            results.append((spec, tile))
            tile_id += 1
    return results


def _generate_tiles_cuda(image: Image.Image, image_id: str, cfg: TilingConfig) -> list[tuple[TileSpec, Image.Image]]:
    """CUDA-accelerated tiling via custom plantclef_ext kernel."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # C++ Kernel expects (C, H, W)
    img_tensor = torch.from_numpy(np.array(image)).cuda().permute(2, 0, 1)
    
    overlap = 1.0 - (cfg.stride / cfg.tile_size)
    
    y_starts = _anchor_points(image.height, cfg.tile_size, cfg.stride)
    x_starts = _anchor_points(image.width, cfg.tile_size, cfg.stride)
    expected_count = len(y_starts) * len(x_starts)

    tiles_tensor = plantclef_ext.extract_tiles(
        img_tensor.float(), 
        cfg.tile_size, 
        cfg.tile_size, 
        float(overlap)
    )
    
    if tiles_tensor.size(0) != expected_count:
        return [] 

    results = []
    tile_id = 0
    for y0 in y_starts:
        for x0 in x_starts:
            tile_cpu = tiles_tensor[tile_id].permute(1, 2, 0).byte().cpu().numpy()
            tile_img = Image.fromarray(tile_cpu)
            spec = TileSpec(
                tile_id=tile_id,
                image_id=image_id,
                x0=x0, y0=y0,
                x1=x0 + cfg.tile_size,
                y1=y0 + cfg.tile_size,
                scale=1.0
            )
            results.append((spec, tile_img))
            tile_id += 1
    return results


def _scale_image(image: Image.Image, scale: float) -> Image.Image:
    """Return a resized copy of *image* at the given scale factor."""
    if abs(scale - 1.0) < 1e-6:
        return image
    w = max(1, int(round(image.width * scale)))
    h = max(1, int(round(image.height * scale)))
    return image.resize((w, h), Image.LANCZOS)


def _slide_window(
    image: Image.Image,
    image_id: str,
    tile_size: int,
    stride: int,
    scale: float,
) -> Iterator[tuple[TileSpec, Image.Image]]:
    """Yield ``(TileSpec, tile_image)`` pairs for a single scale level."""
    W, H = image.width, image.height

    if W <= tile_size and H <= tile_size:
        tile = image.crop((0, 0, W, H))
        if tile.mode != "RGB":
            tile = tile.convert("RGB")
        spec = TileSpec(
            tile_id=0,
            image_id=image_id,
            x0=0,
            y0=0,
            x1=int(round(W / scale)),
            y1=int(round(H / scale)),
            scale=scale,
        )
        yield spec, tile
        return

    y_starts = _anchor_points(H, tile_size, stride)
    x_starts = _anchor_points(W, tile_size, stride)

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)

            tile = image.crop((x0, y0, x1, y1))
            if tile.mode != "RGB":
                tile = tile.convert("RGB")

            spec = TileSpec(
                tile_id=0,           
                image_id=image_id,
                x0=int(round(x0 / scale)),
                y0=int(round(y0 / scale)),
                x1=int(round(x1 / scale)),
                y1=int(round(y1 / scale)),
                scale=scale,
            )
            yield spec, tile


def _anchor_points(dim: int, tile_size: int, stride: int) -> list[int]:
    """Compute tile start positions along one axis with a snapped final tile."""
    if dim <= tile_size:
        return [0]

    points: list[int] = list(range(0, dim - tile_size, stride))
    last = dim - tile_size
    if not points or points[-1] < last:
        points.append(last)

    return points


def count_tiles(
    image_width: int,
    image_height: int,
    cfg: TilingConfig,
) -> int:
    """Estimate the number of tiles that would be generated for an image."""
    total = 0
    for scale in cfg.scales:
        W = max(1, int(round(image_width * scale)))
        H = max(1, int(round(image_height * scale)))
        nx = len(_anchor_points(W, cfg.tile_size, cfg.stride))
        ny = len(_anchor_points(H, cfg.tile_size, cfg.stride))
        total += nx * ny
    return total
