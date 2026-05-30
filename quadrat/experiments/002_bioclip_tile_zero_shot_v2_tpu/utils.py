"""
Shared utilities for zero-shot BioCLIP inference — TPU/XLA-compatible version.

Changes from the CUDA version
------------------------------
1. encode_text_features_from_prompts and encode_image_tiles accept an optional
   mark_step_fn callback (called after each encode batch) so the caller can
   flush the XLA graph on TPU without this module needing a direct torch_xla
   import.
2. tile_top_k uses .cpu().numpy() instead of .numpy() so it is safe on any
   device (XLA tensors do not support direct .numpy() conversion).
3. All other logic is identical to the CUDA version.

Cross-image batching helpers (for high-throughput TPU inference)
-----------------------------------------------------------------
preprocess_tiles_to_tensor  Apply the OpenCLIP transform to a list of PIL
                             tiles and return a stacked float tensor.
encode_tiles_tensor          Encode a pre-stacked tile tensor on `device`
                             and return L2-normalised features **on device**
                             (no CPU copy).  Use this when accumulating tiles
                             across images so that one large forward pass
                             services many images at once.
"""

from __future__ import annotations

from typing import Any, Callable

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
    Any tile smaller than tile_size (at the image boundary) is resized to
    tile_size.

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
    device,
    batch_size: int = 256,
    mark_step_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """
    Encode pre-built per-species prompt lists and return one embedding per
    species.

    Process
    -------
    1. Flatten all prompts across all species.
    2. Encode in batches (normalise each embedding).
    3. For each species: average its prompt embeddings, then renormalise.

    This is the prompt-ensembling approach from the official BioCLIP zero-shot
    example.

    Parameters
    ----------
    model        : OpenCLIP model (already on `device` and in eval mode)
    tokenizer    : model-matched tokenizer
    prompt_lists : one list of prompts per species
    device       : torch device (str or torch.device)
    batch_size   : tokeniser / encode batch size for text
    mark_step_fn : optional callable invoked after each batch to flush the XLA
                   graph on TPU (pass device_utils.mark_step bound to backend).

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
            # Move to CPU before accumulating to avoid holding live XLA tensors.
            all_embeddings.append(emb.cpu())
            # Flush the XLA execution graph after each batch so it does not
            # grow unboundedly (no-op on non-XLA backends).
            if mark_step_fn is not None:
                mark_step_fn()

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
    device,
    batch_size: int = 64,
    mark_step_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """
    Encode a list of image tiles and return L2-normalised embeddings.

    Parameters
    ----------
    model        : OpenCLIP model
    transform    : preprocessing transform
    tiles        : list of PIL Images
    device       : torch device (str or torch.device)
    batch_size   : image encoding batch size
    mark_step_fn : optional callable to flush the XLA graph after each batch.

    Returns
    -------
    Tensor of shape (n_tiles, embed_dim), float32, on CPU
    """
    all_embeddings: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = torch.stack(
                [transform(t) for t in tiles[i : i + batch_size]]
            ).to(device)
            emb = model.encode_image(batch)
            emb = F.normalize(emb, dim=-1)
            # Move to CPU immediately to avoid accumulating live device tensors.
            all_embeddings.append(emb.cpu())
            if mark_step_fn is not None:
                mark_step_fn()
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
    image_feats : (n_tiles, dim) — must be on the same device as text_feats
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
    # Use .cpu() before .numpy() — required for XLA/TPU tensors and harmless
    # on CPU/CUDA.
    return top.indices.cpu().numpy(), top.values.cpu().numpy()


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


# ---------------------------------------------------------------------------
# Cross-image batching helpers
# ---------------------------------------------------------------------------

def preprocess_tiles_to_tensor(
    tiles: list[Image.Image],
    transform,
) -> torch.Tensor:
    """
    Apply an OpenCLIP preprocessing transform to a list of PIL tiles and
    return a stacked float tensor of shape (n_tiles, C, H, W).

    Runs entirely on the CPU.  Call this in a worker thread so image prep
    overlaps with TPU computation.
    """
    return torch.stack([transform(t) for t in tiles])


def encode_tiles_tensor(
    model: Any,
    tiles_tensor: torch.Tensor,
    device,
    mark_step_fn: Callable[[], None] | None = None,
) -> torch.Tensor:
    """
    Encode a pre-stacked tile tensor and return L2-normalised embeddings
    **on `device`** (no CPU copy).

    Unlike encode_image_tiles, this function does NOT move the result to CPU
    after encoding.  The caller keeps the tensor alive on-device so that
    multiple images can be encoded together before any CPU materialisation.

    Parameters
    ----------
    model        : OpenCLIP model (already on `device`, eval mode)
    tiles_tensor : (n_tiles, C, H, W) float tensor — may be on CPU on entry,
                   will be transferred to `device` inside this call.
    device       : target device for inference
    mark_step_fn : optional XLA mark_step callback

    Returns
    -------
    Tensor of shape (n_tiles, embed_dim), float32, L2-normalised, on `device`
    """
    with torch.no_grad():
        batch = tiles_tensor.to(device)
        feats = model.encode_image(batch)
        feats = F.normalize(feats, dim=-1)
        if mark_step_fn is not None:
            mark_step_fn()
    return feats
