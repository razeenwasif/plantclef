"""
SAM-based and RGB-based vegetation scoring for image tiles.

Two scoring methods are supported:

  rgb  — Excess Green Index (ExG = 2G - R - B) applied pixel-wise.
         Fast, no model required.  Works well for outdoor plant imagery.

  sam  — SAM SamAutomaticMaskGenerator segments the tile into regions;
         each region is scored for greenness with ExG.  Regions above
         a greenness threshold count as vegetation.  This is slower but
         produces spatially aware masks useful for visualisation.

If SAM scoring is requested but the checkpoint is absent the code falls
back to RGB scoring and prints a warning.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from config import (
    DEFAULT_SAM_MODEL_TYPE,
    SAM_POINTS_PER_SIDE,
    SAM_PRED_IOU_THRESH,
    SAM_STABILITY_SCORE_THRESH,
    SAM_MIN_MASK_REGION_AREA,
    DEFAULT_EXG_THRESHOLD,
    DEFAULT_SAM_MIN_GREENNESS,
    DEFAULT_WEIGHT_ALPHA,
    DEFAULT_WEIGHT_BETA,
    DEFAULT_WEIGHT_MIN,
    DEFAULT_WEIGHT_MAX,
    SCORING_METHODS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_uint8_rgb(tile: Image.Image) -> np.ndarray:
    """PIL Image → (H, W, 3) uint8 array."""
    return np.array(tile.convert("RGB"), dtype=np.uint8)


def _exg_greenness(
    img_rgb: np.ndarray,
    exg_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return per-pixel ExG values and a boolean vegetation mask.

    ExG = 2G - R - B (range: -510 to 510 for uint8 inputs cast to float32)
    """
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    exg = 2.0 * g - r - b
    return exg, (exg > exg_threshold)


# ---------------------------------------------------------------------------
# RGB scoring
# ---------------------------------------------------------------------------

def score_tile_rgb(
    tile: Image.Image,
    exg_threshold: float = DEFAULT_EXG_THRESHOLD,
) -> float:
    """
    Return the fraction of tile pixels classified as vegetation via ExG.

    An ExG threshold of 20 requires a pixel to be meaningfully greener
    than the mean of red and blue channels before it is counted.
    """
    img = _to_uint8_rgb(tile)
    _, mask = _exg_greenness(img, exg_threshold)
    return float(mask.mean())


# ---------------------------------------------------------------------------
# SAM model loading
# ---------------------------------------------------------------------------

# Known SAM checkpoint filename fragments → model type
_SAM_FILENAME_MAP = {
    "vit_h": "vit_h",
    "vit_l": "vit_l",
    "vit_b": "vit_b",
}


def detect_sam_model_type(checkpoint_path: str) -> str | None:
    """
    Infer SAM model type from the checkpoint filename.

    Returns 'vit_b', 'vit_l', 'vit_h', or None if unrecognisable.
    Example: 'sam_vit_h_4b8939.pth' → 'vit_h'
    """
    name = Path(checkpoint_path).name.lower()
    for fragment, model_type in _SAM_FILENAME_MAP.items():
        if fragment in name:
            return model_type
    return None


def load_sam_generator(
    checkpoint_path: str,
    model_type: str = DEFAULT_SAM_MODEL_TYPE,
    device: str = "cpu",
    points_per_side: int = SAM_POINTS_PER_SIDE,
    pred_iou_thresh: float = SAM_PRED_IOU_THRESH,
    stability_score_thresh: float = SAM_STABILITY_SCORE_THRESH,
    min_mask_region_area: int = SAM_MIN_MASK_REGION_AREA,
):
    """
    Load SAM and return a SamAutomaticMaskGenerator.

    Auto-detects the model type from the checkpoint filename so you don't
    need to pass --sam-model-type explicitly when the filename is standard.
    Falls back to rgb scoring (returns None) if the checkpoint is absent.

    If CUDA is requested but OOM occurs, automatically retries on CPU.
    """
    if not Path(checkpoint_path).exists():
        warnings.warn(
            f"SAM checkpoint not found at '{checkpoint_path}'. "
            "Falling back to RGB vegetation scoring.  "
            "Run:  python download_sam.py --model-type vit_h  (or vit_b/vit_l)",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        warnings.warn(
            "segment-anything is not installed. Falling back to RGB scoring.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # Auto-detect model type from filename; warn if it differs from the argument
    detected = detect_sam_model_type(checkpoint_path)
    if detected is not None and detected != model_type:
        print(f"  [SAM] Auto-detected model type '{detected}' from filename "
              f"(overrides --sam-model-type '{model_type}')")
        model_type = detected
    elif detected is None:
        print(f"  [SAM] Could not detect model type from filename; "
              f"using --sam-model-type='{model_type}'")

    def _build(target_device: str):
        print(f"  Loading SAM ({model_type}) from {checkpoint_path} on {target_device} ...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam = sam.to(target_device)
        sam.eval()
        return SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=0,
            min_mask_region_area=min_mask_region_area,
        )

    # Try requested device; fall back to CPU on OOM
    try:
        generator = _build(device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device != "cpu":
            warnings.warn(
                f"CUDA OOM loading SAM on '{device}' — retrying on CPU. "
                "SAM scoring will be slower but correct.  "
                "Consider --sam-model-type vit_b for a smaller SAM model.",
                RuntimeWarning,
                stacklevel=2,
            )
            import torch
            torch.cuda.empty_cache()
            generator = _build("cpu")
        else:
            raise

    print(f"  SAM ready — model={model_type}  device={generator.predictor.model.device}  "
          f"points_per_side={points_per_side}")
    return generator


# ---------------------------------------------------------------------------
# SAM scoring
# ---------------------------------------------------------------------------

def score_tile_sam(
    sam_generator,
    tile: Image.Image,
    exg_threshold: float = DEFAULT_EXG_THRESHOLD,
    min_greenness: float = DEFAULT_SAM_MIN_GREENNESS,
) -> tuple[float, list[dict]]:
    """
    Score a tile for vegetation using SAM-generated masks.

    Algorithm
    ---------
    1. Run SamAutomaticMaskGenerator on the tile.
    2. For each mask, compute the fraction of its pixels that pass ExG.
    3. Masks where greenness >= min_greenness are labelled 'vegetation'.
    4. vegetation_ratio = union-area of veg masks / total tile area.

    Returns
    -------
    vegetation_ratio : float
    masks            : list of SAM mask dicts, each augmented with:
                       'greenness' (float) and 'is_vegetation' (bool)
    """
    img = _to_uint8_rgb(tile)
    h, w = img.shape[:2]

    try:
        masks = sam_generator.generate(img)
    except Exception as exc:
        warnings.warn(f"SAM generation failed ({exc}); using RGB fallback.", RuntimeWarning)
        return score_tile_rgb(tile, exg_threshold=exg_threshold), []

    if not masks:
        return score_tile_rgb(tile, exg_threshold=exg_threshold), []

    exg_map, pixel_veg_mask = _exg_greenness(img, exg_threshold)
    veg_union = np.zeros((h, w), dtype=bool)

    for m in masks:
        seg = m["segmentation"]  # (H, W) bool
        n_pixels = seg.sum()
        if n_pixels == 0:
            m["greenness"] = 0.0
            m["is_vegetation"] = False
            continue

        greenness = float(pixel_veg_mask[seg].mean())
        m["greenness"] = greenness
        m["is_vegetation"] = greenness >= min_greenness

        if m["is_vegetation"]:
            veg_union |= seg

    vegetation_ratio = float(veg_union.mean())
    return vegetation_ratio, masks


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def score_tiles(
    tiles: list[Image.Image],
    method: str = "rgb",
    sam_generator=None,
    exg_threshold: float = DEFAULT_EXG_THRESHOLD,
    min_greenness: float = DEFAULT_SAM_MIN_GREENNESS,
) -> tuple[list[float], list[list[dict]]]:
    """
    Score all tiles and return vegetation ratios and (optional) SAM masks.

    Parameters
    ----------
    tiles         : list of PIL Images (square, tile_size × tile_size)
    method        : 'rgb' or 'sam'
    sam_generator : SamAutomaticMaskGenerator or None (triggers RGB fallback)
    exg_threshold : ExG pixel threshold
    min_greenness : fraction of mask pixels that must be green for it to count

    Returns
    -------
    veg_scores : list[float]       one per tile, range [0, 1]
    all_masks  : list[list[dict]]  SAM masks per tile (empty lists for rgb mode)
    """
    if method not in SCORING_METHODS:
        raise ValueError(f"Unknown scoring method '{method}'. Choose from {SCORING_METHODS}.")

    use_sam = (method == "sam") and (sam_generator is not None)

    veg_scores: list[float] = []
    all_masks: list[list[dict]] = []

    for tile in tiles:
        if use_sam:
            score, masks = score_tile_sam(
                sam_generator, tile,
                exg_threshold=exg_threshold,
                min_greenness=min_greenness,
            )
        else:
            score = score_tile_rgb(tile, exg_threshold=exg_threshold)
            masks = []

        veg_scores.append(score)
        all_masks.append(masks)

    return veg_scores, all_masks


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_tile_weights(
    veg_scores: list[float],
    alpha: float = DEFAULT_WEIGHT_ALPHA,
    beta: float = DEFAULT_WEIGHT_BETA,
    w_min: float = DEFAULT_WEIGHT_MIN,
    w_max: float = DEFAULT_WEIGHT_MAX,
) -> np.ndarray:
    """
    Convert vegetation ratios to soft tile weights.

    Formula: w_i = clip(alpha + beta * veg_ratio_i, w_min, w_max)

    With defaults (alpha=0.5, beta=1.0):
      veg_ratio=0.0  →  weight=0.5   (background tile still contributes)
      veg_ratio=0.5  →  weight=1.0   (mixed tile gets unit weight)
      veg_ratio=1.0  →  weight=1.5   (pure-veg tile boosted by 50%)
    """
    scores = np.array(veg_scores, dtype=np.float32)
    return np.clip(alpha + beta * scores, w_min, w_max)
