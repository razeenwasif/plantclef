"""
Tiling and image-embedding utilities.

Adapted from 002_bioclip_tile_zero_shot_v2/utils.py with minor extensions:
  - multiscale tiling support
  - explicit type annotations
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def get_tiles(
    image: Image.Image,
    tile_size: int,
    stride: int,
) -> tuple[list[Image.Image], list[tuple[int, int, int, int]]]:
    """
    Slice an image into overlapping square tiles.

    Tiles are aligned to the right/bottom edges so the full image is covered.
    Boundary tiles smaller than tile_size are resized up to tile_size.

    Parameters
    ----------
    image     : PIL Image (RGB)
    tile_size : side length of each square tile in pixels
    stride    : step size between tile origins (stride = tile_size - overlap)

    Returns
    -------
    tiles  : list of PIL Images, each tile_size x tile_size
    coords : list of (x1, y1, x2, y2) pixel coordinates
    """
    w, h = image.size
    tiles: list[Image.Image] = []
    coords: list[tuple[int, int, int, int]] = []

    y = 0
    while y + tile_size <= h or y == 0:
        x = 0
        while x + tile_size <= w or x == 0:
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tile = image.crop((x1, y1, x2, y2))
            if tile.size[0] < tile_size or tile.size[1] < tile_size:
                tile = tile.resize((tile_size, tile_size), Image.BILINEAR)
            tiles.append(tile)
            coords.append((x1, y1, x2, y2))
            if x + stride >= w:
                break
            x += stride
        if y + stride >= h:
            break
        y += stride

    return tiles, coords


def get_multiscale_tiles(
    image: Image.Image,
    tile_sizes: list[int],
    stride_ratio: float = 0.5,
) -> tuple[list[Image.Image], list[tuple[int, int, int, int, int]]]:
    """
    Generate tiles at multiple scales and return them with their scale tag.

    Parameters
    ----------
    image        : PIL Image (RGB)
    tile_sizes   : list of tile side lengths to use
    stride_ratio : stride = int(tile_size * stride_ratio)

    Returns
    -------
    all_tiles  : flat list of PIL Images
    all_coords : list of (x1, y1, x2, y2, tile_size) tuples
    """
    all_tiles: list[Image.Image] = []
    all_coords: list[tuple[int, int, int, int, int]] = []
    for ts in tile_sizes:
        stride = max(1, int(ts * stride_ratio))
        tiles, coords = get_tiles(image, ts, stride)
        all_tiles.extend(tiles)
        all_coords.extend((x1, y1, x2, y2, ts) for x1, y1, x2, y2 in coords)
    return all_tiles, all_coords


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_tiles(
    model: Any,
    transform: Any,
    tiles: list[Image.Image],
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Encode a list of image tiles and return L2-normalised embeddings.

    Parameters
    ----------
    model      : OpenCLIP model (on `device`, in eval mode)
    transform  : image preprocessing callable
    tiles      : list of PIL Images
    device     : torch device string
    batch_size : number of tiles to encode per forward pass

    Returns
    -------
    Tensor of shape (n_tiles, embed_dim), float32, on CPU, L2-normalised
    """
    if not tiles:
        raise ValueError("encode_image_tiles: received an empty tile list")

    all_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = torch.stack(
                [transform(t) for t in tiles[i : i + batch_size]]
            ).to(device)
            emb = model.encode_image(batch)
            emb = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)  # (n_tiles, dim)
