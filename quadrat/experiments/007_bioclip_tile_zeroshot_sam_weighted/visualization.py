"""
Visualizations for the SAM-weighted BioCLIP experiment.

Each public function saves one figure to disk and returns the output path.

Panels produced
---------------
A. tile_grid_overview      - full image with tile bounding boxes coloured by veg_ratio
B. tile_veg_heatmap        - grid of tile thumbnails annotated with veg/weight/rank
C. sam_mask_overlay        - side-by-side: tile | SAM masks | binary veg mask
D. tile_diagnostics        - per-tile table: veg_ratio, weight, top species, score
E. prediction_comparison   - baseline vs weighted top-k side-by-side
F  summary_statistics      - dataset-level histograms and scatter plots
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless rendering — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from config import (
    MAX_TILE_VIZ_PER_IMAGE,
    MAX_SAM_VIZ_TILES,
    VIZ_THUMBNAIL_SIZE,
    VIZ_DPI,
)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _veg_colour(veg_ratio: float) -> tuple[float, float, float]:
    """Map veg_ratio [0,1] to an RGB colour: red (0) → yellow (0.5) → green (1)."""
    r = max(0.0, 1.0 - 2.0 * veg_ratio)
    g = min(1.0, 2.0 * veg_ratio)
    return r, g, 0.15


def _weight_colour(w: float, w_min: float = 0.1, w_max: float = 2.0) -> tuple[float, float, float]:
    """Map weight to blue gradient."""
    t = (w - w_min) / max(w_max - w_min, 1e-6)
    t = float(np.clip(t, 0, 1))
    return (0.2 + 0.6 * (1 - t), 0.4 + 0.3 * (1 - t), 0.6 + 0.4 * t)


# ---------------------------------------------------------------------------
# A. Tile grid overview
# ---------------------------------------------------------------------------

def save_tile_grid_overview(
    image: Image.Image,
    coords: list[tuple[int, int, int, int]],
    veg_scores: list[float],
    weights: list[float],
    output_path: Path,
) -> Path:
    """
    Draw the original quadrat image with tile bounding boxes coloured by vegetation ratio.

    Green border = high vegetation, red border = low vegetation.
    Each tile is labelled with its index and veg_ratio.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: coloured by veg_ratio
    for ax, colour_fn, label in [
        (axes[0], lambda v, w: _veg_colour(v), "Vegetation ratio"),
        (axes[1], lambda v, w: _weight_colour(w), "Tile weight"),
    ]:
        ax.imshow(np.array(image))
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")

        for idx, (x1, y1, x2, y2) in enumerate(coords):
            colour = colour_fn(veg_scores[idx], weights[idx])
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colour, facecolor=(*colour, 0.15),
            )
            ax.add_patch(rect)
            val = veg_scores[idx] if ax is axes[0] else weights[idx]
            ax.text(
                x1 + 3, y1 + 14, f"{idx}:{val:.2f}",
                fontsize=6, color="white",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5),
            )

    # Colourbar proxy
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn,
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes[0], fraction=0.03, pad=0.02, label="Veg ratio [0–1]")

    fig.suptitle(f"Tile overview  ({len(coords)} tiles)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# B. Tile vegetation heatmap
# ---------------------------------------------------------------------------

def save_tile_veg_heatmap(
    tiles: list[Image.Image],
    coords: list[tuple[int, int, int, int]],
    veg_scores: list[float],
    weights: list[float],
    output_path: Path,
    max_tiles: int = MAX_TILE_VIZ_PER_IMAGE,
    thumb_size: int = VIZ_THUMBNAIL_SIZE,
) -> Path:
    """
    Grid of tile thumbnails, each annotated with index / veg_ratio / weight.
    Bordered by a colour reflecting vegetation level.
    """
    n = min(len(tiles), max_tiles)
    ncols = min(n, 8)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.9))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i in range(n):
        ax = axes_flat[i]
        thumb = tiles[i].resize((thumb_size, thumb_size), Image.BILINEAR)
        ax.imshow(np.array(thumb))
        colour = _veg_colour(veg_scores[i])
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
        x1, y1, x2, y2 = coords[i]
        ax.set_title(
            f"t{i}  v={veg_scores[i]:.2f}  w={weights[i]:.2f}\n({x1},{y1})–({x2},{y2})",
            fontsize=6, pad=2,
        )

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("Tile vegetation heatmap", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# C. SAM mask overlay
# ---------------------------------------------------------------------------

def save_sam_mask_overlay(
    tile: Image.Image,
    masks: list[dict],
    veg_score: float,
    tile_idx: int,
    output_path: Path,
) -> Path:
    """
    Three-panel figure: original tile | SAM masks coloured | binary vegetation mask.

    Each SAM mask is drawn with a random colour.  Vegetation masks are drawn
    with green tones, non-vegetation with grey.  The panel title shows veg_ratio.
    """
    img_np = np.array(tile.convert("RGB"))
    h, w = img_np.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    # Panel 1: original tile
    axes[0].imshow(img_np)
    axes[0].set_title("Original tile", fontsize=10)
    axes[0].axis("off")

    # Panel 2: SAM mask overlay
    overlay = img_np.copy().astype(np.float32)
    if masks:
        # Sort by area (largest first) so small masks stay visible
        sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        rng = np.random.default_rng(seed=42)
        for m in sorted_masks:
            seg = m["segmentation"]
            is_veg = m.get("is_vegetation", False)
            if is_veg:
                colour = rng.uniform([0.0, 0.5, 0.0], [0.5, 1.0, 0.5], 3)
            else:
                colour = rng.uniform([0.3, 0.3, 0.3], [0.7, 0.7, 0.7], 3)
            overlay[seg] = overlay[seg] * 0.4 + colour * 0.6 * 255

        axes[1].imshow(overlay.astype(np.uint8))
        axes[1].set_title(
            f"SAM masks ({len(masks)} total)\ngreen = vegetation",
            fontsize=10,
        )
    else:
        axes[1].imshow(img_np)
        axes[1].set_title("SAM masks\n(none generated)", fontsize=10)
    axes[1].axis("off")

    # Panel 3: binary vegetation mask
    if masks:
        veg_union = np.zeros((h, w), dtype=bool)
        for m in masks:
            if m.get("is_vegetation", False):
                veg_union |= m["segmentation"]
        binary = np.zeros((h, w, 3), dtype=np.uint8)
        binary[veg_union]  = [80, 200, 80]
        binary[~veg_union] = [30, 30, 30]
        axes[2].imshow(binary)
    else:
        axes[2].imshow(np.zeros((h, w, 3), dtype=np.uint8))
    axes[2].set_title(f"Veg mask  ratio={veg_score:.3f}", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(f"Tile {tile_idx}  —  vegetation ratio = {veg_score:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# D. Tile diagnostics panel
# ---------------------------------------------------------------------------

def save_tile_diagnostics(
    tiles: list[Image.Image],
    veg_scores: list[float],
    weights: list[float],
    tile_top1_species: list[str],   # top-1 predicted species per tile
    tile_top1_scores: list[float],  # top-1 logit per tile
    output_path: Path,
    max_tiles: int = MAX_TILE_VIZ_PER_IMAGE,
    thumb_size: int = VIZ_THUMBNAIL_SIZE,
) -> Path:
    """
    Compact table-style panel: one row per tile.

    Columns: thumbnail | veg bar | weight | top-1 species | logit
    """
    n = min(len(tiles), max_tiles)
    row_h = 1.1
    fig_h = n * row_h + 0.8
    fig, axes = plt.subplots(n, 5, figsize=(14, fig_h),
                              gridspec_kw={"width_ratios": [1, 2, 1, 3, 1]})
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Tile", "Veg ratio", "Weight", "Top-1 species", "Logit"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8, fontweight="bold", pad=3)

    for i in range(n):
        # Thumbnail
        thumb = tiles[i].resize((thumb_size, thumb_size), Image.BILINEAR)
        axes[i, 0].imshow(np.array(thumb))
        axes[i, 0].set_ylabel(f"t{i}", fontsize=7, rotation=0, labelpad=20, va="center")
        axes[i, 0].axis("off")

        # Veg ratio bar
        axes[i, 1].barh([0], [veg_scores[i]], color=_veg_colour(veg_scores[i]), height=0.6)
        axes[i, 1].barh([0], [1.0], color="lightgrey", height=0.6, zorder=0)
        axes[i, 1].set_xlim(0, 1)
        axes[i, 1].set_ylim(-0.5, 0.5)
        axes[i, 1].text(min(veg_scores[i] + 0.03, 0.97), 0, f"{veg_scores[i]:.3f}",
                        va="center", fontsize=7)
        axes[i, 1].axis("off")

        # Weight
        axes[i, 2].text(0.5, 0.5, f"{weights[i]:.3f}",
                        ha="center", va="center", fontsize=8,
                        color=_weight_colour(weights[i]))
        axes[i, 2].axis("off")

        # Top-1 species (truncated)
        species_short = textwrap.shorten(tile_top1_species[i], width=35, placeholder="…")
        axes[i, 3].text(0.02, 0.5, species_short,
                        ha="left", va="center", fontsize=7,
                        transform=axes[i, 3].transAxes)
        axes[i, 3].axis("off")

        # Logit score
        axes[i, 4].text(0.5, 0.5, f"{tile_top1_scores[i]:.2f}",
                        ha="center", va="center", fontsize=8)
        axes[i, 4].axis("off")

    fig.suptitle("Per-tile diagnostics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# E. Image-level prediction comparison
# ---------------------------------------------------------------------------

def save_prediction_comparison(
    baseline_species: list[str],
    baseline_scores: list[float],
    weighted_species: list[str],
    weighted_scores: list[float],
    image_id: str,
    output_path: Path,
    extra_methods: Optional[dict[str, tuple[list[str], list[float]]]] = None,
) -> Path:
    """
    Side-by-side top-k comparison between baseline and weighted predictions.

    extra_methods: {label: (species_list, scores_list)} for additional methods.
    """
    methods = {"Baseline (max)": (baseline_species, baseline_scores),
               "Weighted mean": (weighted_species, weighted_scores)}
    if extra_methods:
        methods.update(extra_methods)

    n_methods = len(methods)
    k = max(len(baseline_species), len(weighted_species))

    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, max(3, k * 0.7 + 1.5)))

    for ax, (method_name, (species, scores)) in zip(axes if n_methods > 1 else [axes], methods.items()):
        y_pos = list(range(len(species) - 1, -1, -1))
        colours = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(species))]
        bars = ax.barh(y_pos, scores, color=colours, edgecolor="white", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [textwrap.shorten(s, width=30, placeholder="…") for s in species],
            fontsize=8,
        )
        ax.set_xlabel("Logit score", fontsize=9)
        ax.set_title(method_name, fontsize=10, fontweight="bold")
        for bar, score in zip(bars, scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{score:.2f}", va="center", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Prediction comparison  —  {image_id}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# F. Dataset-level summary statistics
# ---------------------------------------------------------------------------

def save_summary_statistics(
    all_veg_scores: list[float],
    all_weights: list[float],
    all_top1_confidences: list[float],
    output_dir: Path,
) -> list[Path]:
    """
    Four-panel summary figure saved to output_dir/summary_veg_stats.png.

    Panels:
      1. Histogram of vegetation ratios across all tiles
      2. Histogram of tile weights
      3. Scatter: veg_ratio vs top-1 BioCLIP confidence (per tile)
      4. Scatter: veg_ratio vs tile weight (sanity check)
    """
    veg  = np.array(all_veg_scores, dtype=np.float32)
    wts  = np.array(all_weights,    dtype=np.float32)
    conf = np.array(all_top1_confidences, dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. Veg ratio histogram
    axes[0, 0].hist(veg, bins=40, color="#2ecc71", edgecolor="white", linewidth=0.5)
    axes[0, 0].set_xlabel("Vegetation ratio", fontsize=10)
    axes[0, 0].set_ylabel("# tiles", fontsize=10)
    axes[0, 0].set_title("Distribution of vegetation ratios", fontsize=11)
    axes[0, 0].axvline(veg.mean(), color="red", linestyle="--", label=f"mean={veg.mean():.3f}")
    axes[0, 0].legend(fontsize=9)

    # 2. Tile weight histogram
    axes[0, 1].hist(wts, bins=40, color="#3498db", edgecolor="white", linewidth=0.5)
    axes[0, 1].set_xlabel("Tile weight", fontsize=10)
    axes[0, 1].set_ylabel("# tiles", fontsize=10)
    axes[0, 1].set_title("Distribution of tile weights", fontsize=11)
    axes[0, 1].axvline(wts.mean(), color="red", linestyle="--", label=f"mean={wts.mean():.3f}")
    axes[0, 1].legend(fontsize=9)

    # 3. Veg ratio vs BioCLIP confidence
    if len(conf) == len(veg):
        axes[1, 0].scatter(veg, conf, alpha=0.3, s=8, color="#e74c3c")
        # Trend line
        if len(veg) > 10:
            z = np.polyfit(veg, conf, 1)
            p = np.poly1d(z)
            xline = np.linspace(veg.min(), veg.max(), 100)
            axes[1, 0].plot(xline, p(xline), "k--", linewidth=1.2, alpha=0.7)
        axes[1, 0].set_xlabel("Vegetation ratio", fontsize=10)
        axes[1, 0].set_ylabel("Top-1 BioCLIP logit", fontsize=10)
        axes[1, 0].set_title("Veg ratio vs BioCLIP confidence", fontsize=11)
    else:
        axes[1, 0].text(0.5, 0.5, "Confidence data\nnot available",
                        ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].axis("off")

    # 4. Veg ratio vs weight (sanity check)
    axes[1, 1].scatter(veg, wts, alpha=0.3, s=8, color="#9b59b6")
    axes[1, 1].set_xlabel("Vegetation ratio", fontsize=10)
    axes[1, 1].set_ylabel("Tile weight", fontsize=10)
    axes[1, 1].set_title("Veg ratio vs tile weight (sanity)", fontsize=11)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Dataset vegetation statistics  ({len(veg):,} tiles)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()

    out_path = output_dir / "summary_veg_stats.png"
    fig.savefig(out_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close(fig)
    return [out_path]


# ---------------------------------------------------------------------------
# Per-image convenience wrapper
# ---------------------------------------------------------------------------

def save_image_visualizations(
    image: Image.Image,
    image_id: str,
    tiles: list[Image.Image],
    coords: list[tuple[int, int, int, int]],
    veg_scores: list[float],
    weights: list[float],
    sam_masks_per_tile: list[list[dict]],
    tile_top1_species: list[str],
    tile_top1_logits: list[float],
    baseline_top_species: list[str],
    baseline_top_scores: list[float],
    weighted_top_species: list[str],
    weighted_top_scores: list[float],
    viz_dir: Path,
    extra_predictions: Optional[dict[str, tuple[list[str], list[float]]]] = None,
    max_tiles: int = MAX_TILE_VIZ_PER_IMAGE,
    max_sam_tiles: int = MAX_SAM_VIZ_TILES,
) -> list[Path]:
    """
    Save all per-image visualizations into viz_dir/{image_id}/.

    Returns list of saved paths.
    """
    img_dir = viz_dir / image_id
    img_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # A. Tile grid overview
    saved.append(save_tile_grid_overview(
        image, coords, veg_scores, weights,
        img_dir / "A_tile_overview.png",
    ))

    # B. Tile vegetation heatmap
    saved.append(save_tile_veg_heatmap(
        tiles, coords, veg_scores, weights,
        img_dir / "B_veg_heatmap.png",
        max_tiles=max_tiles,
    ))

    # C. SAM mask overlays (for the tiles with highest veg scores)
    if any(len(m) > 0 for m in sam_masks_per_tile):
        top_tile_idx = sorted(
            range(len(veg_scores)), key=lambda i: veg_scores[i], reverse=True
        )[:max_sam_tiles]
        for ti in top_tile_idx:
            if ti < len(sam_masks_per_tile):
                saved.append(save_sam_mask_overlay(
                    tiles[ti], sam_masks_per_tile[ti], veg_scores[ti],
                    tile_idx=ti,
                    output_path=img_dir / f"C_sam_tile{ti:02d}.png",
                ))

    # D. Per-tile diagnostics
    saved.append(save_tile_diagnostics(
        tiles, veg_scores, weights,
        tile_top1_species, tile_top1_logits,
        img_dir / "D_tile_diagnostics.png",
        max_tiles=max_tiles,
    ))

    # E. Prediction comparison
    saved.append(save_prediction_comparison(
        baseline_top_species, baseline_top_scores,
        weighted_top_species, weighted_top_scores,
        image_id=image_id,
        output_path=img_dir / "E_prediction_comparison.png",
        extra_methods=extra_predictions,
    ))

    return saved
