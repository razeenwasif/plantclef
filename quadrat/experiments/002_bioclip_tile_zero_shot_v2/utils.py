"""
Shared utilities for zero-shot BioCLIP inference.

Reuses the core tiling and scoring logic from bioclip_tile_zero_shot,
and adds prompt-ensemble text encoding for multi-model support.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def get_tiles(image: Image.Image, tile_size: int, stride: int) -> tuple[list[Image.Image], list[tuple[int, int, int, int]]]:
    """
    Slice an image into overlapping square tiles.

    Tiles are aligned to the right/bottom edges so the full image is covered.
    Any tile smaller than tile_size (at the image boundary) is resized to tile_size.

    Returns
    -------
    tiles  : list of PIL Images, each tile_size × tile_size
    coords : list of (x1, y1, x2, y2) in pixel coordinates
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


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_text_features_from_prompts(
    model: Any,
    tokenizer: Any,
    prompt_lists: list[list[str]],
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Encode pre-built per-species prompt lists and return one embedding per species.

    Process
    -------
    1. Flatten all prompts across all species.
    2. Encode in batches (normalise each embedding).
    3. For each species: average its prompt embeddings, then renormalise.

    This is the prompt-ensembling approach from the official BioCLIP zero-shot example.

    Parameters
    ----------
    model        : OpenCLIP model (already on `device` and in eval mode)
    tokenizer    : model-matched tokenizer
    prompt_lists : one list of prompts per species
    device       : torch device string
    batch_size   : tokeniser / encode batch size for text

    Returns
    -------
    Tensor of shape (n_species, embed_dim), float32, L2-normalised
    """
    flat_prompts: list[str] = []
    species_sizes: list[int] = []
    for prompts in prompt_lists:
        flat_prompts.extend(prompts)
        species_sizes.append(len(prompts))

    all_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(flat_prompts), batch_size):
            batch = flat_prompts[i : i + batch_size]
            tokens = tokenizer(batch).to(device)
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu())

    all_emb = torch.cat(all_embeddings, dim=0)  # (total_prompts, dim)

    species_embeddings: list[torch.Tensor] = []
    offset = 0
    for size in species_sizes:
        chunk = all_emb[offset : offset + size]   # (n_prompts_i, dim)
        avg = chunk.mean(dim=0)                    # (dim,)
        avg = F.normalize(avg, dim=0)
        species_embeddings.append(avg)
        offset += size

    return torch.stack(species_embeddings, dim=0)  # (n_species, dim)


def encode_image_tiles(
    model: Any,
    transform: Any,
    tiles: list[Image.Image],
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Encode a list of image tiles and return L2-normalised embeddings.

    Returns
    -------
    Tensor of shape (n_tiles, embed_dim), float32, on CPU
    """
    all_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = torch.stack([transform(t) for t in tiles[i : i + batch_size]]).to(device)
            emb = model.encode_image(batch)
            emb = F.normalize(emb, dim=-1)
            all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)  # (n_tiles, dim)


# ---------------------------------------------------------------------------
# Scoring and aggregation
# ---------------------------------------------------------------------------

def compute_tile_logits(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    """
    Compute scaled cosine-similarity logits between tiles and species embeddings.

    Parameters
    ----------
    image_feats : (n_tiles, dim) - must be on same device as text_feats
    text_feats  : (n_species, dim)
    logit_scale : scalar (model.logit_scale.exp())

    Returns
    -------
    (n_tiles, n_species) float tensor
    """
    return logit_scale * image_feats @ text_feats.T


def aggregate_tile_logits(tile_logits: torch.Tensor) -> torch.Tensor:
    """
    SAHI-style max-pooling over tiles.

    Takes the per-species maximum logit across all tiles, which preserves
    the strongest local signal without diluting it with background tiles.

    Returns
    -------
    (n_species,) tensor
    """
    return tile_logits.max(dim=0).values


def tile_top_k(tile_logits: torch.Tensor, k: int = 3) -> tuple[Any, Any]:
    """
    Per-tile top-k using softmax probabilities.

    Returns
    -------
    indices : (n_tiles, k) numpy int array
    scores  : (n_tiles, k) numpy float array
    """
    probs = tile_logits.softmax(dim=-1)
    top = probs.topk(k, dim=-1)
    return top.indices.numpy(), top.values.numpy()


def image_top_k(
    image_logits: torch.Tensor,
    species_ids: list[str],
    k: int = 5,
) -> tuple[list[str], list[float]]:
    """
    Top-k species IDs and raw logit scores from aggregated image logits.

    Returns
    -------
    top_species_ids : list of str, length k
    top_scores      : list of float, length k
    """
    top = image_logits.topk(k)
    top_species_ids = [species_ids[i] for i in top.indices.tolist()]
    top_scores = top.values.tolist()
    return top_species_ids, top_scores
