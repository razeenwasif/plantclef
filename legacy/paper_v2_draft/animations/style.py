"""
Shared palette + helpers so every scene matches the dashboard's
glass-panel aesthetic. Imported by each scene_*.py.
"""

from manim import (
    BLACK, BackgroundRectangle, Mobject, Rectangle, Text, VGroup,
    config,
)

# ── Palette (mirrors dashboard tailwind tokens) ─────────────────────────
PLANTCLEF_BG       = "#0B0B12"            # near-black background
PLANTCLEF_FG       = "#E2E8F0"            # default text on dark
PLANTCLEF_MUTED    = "#94A3B8"            # secondary text
PLANTCLEF_ACCENT   = "#7C3AED"            # violet — emphasis / model nodes
PLANTCLEF_CYAN     = "#06B6D4"            # cyan — stage / data
PLANTCLEF_PINK     = "#EC4899"            # pink — post-processing / focal
PLANTCLEF_EMERALD  = "#10B981"            # emerald — success / emit
PLANTCLEF_AMBER    = "#FBBF24"            # amber — warn / filtered
PLANTCLEF_DARK_GLASS = "#15151F"          # subtle panel fill

# Aliases that read better in scene code
COLOR_DATA      = PLANTCLEF_CYAN
COLOR_MODEL     = PLANTCLEF_ACCENT
COLOR_POST      = PLANTCLEF_PINK
COLOR_OK        = PLANTCLEF_EMERALD
COLOR_WARN      = PLANTCLEF_AMBER
COLOR_TEXT      = PLANTCLEF_FG
COLOR_DIM       = PLANTCLEF_MUTED

# ── Default config (call from each scene's __init__ if needed) ──────────
def apply_global_config() -> None:
    """Set Manim's global config so the canvas matches the rest of the deck."""
    config.background_color = PLANTCLEF_BG


# ── Tiny composable mobjects ────────────────────────────────────────────
def title(text: str, size: int = 38, color: str = COLOR_TEXT) -> Text:
    """Section title. Use as the opening shot of each scene."""
    return Text(text, font="Sans", weight="SEMIBOLD",
                font_size=size, color=color)


def subtitle(text: str, size: int = 22, color: str = COLOR_DIM) -> Text:
    return Text(text, font="Sans", font_size=size, color=color)


def caption(text: str, size: int = 18, color: str = COLOR_DIM) -> Text:
    return Text(text, font="Sans", font_size=size, color=color)


def code_span(text: str, size: int = 24, color: str = COLOR_DATA) -> Text:
    """A monospaced code chip (e.g. for variable names inline)."""
    return Text(text, font="Mono", font_size=size, color=color)


def panel(
    width: float, height: float,
    stroke: str = "#FFFFFF22",
    fill: str = PLANTCLEF_DARK_GLASS,
    fill_opacity: float = 0.6,
    corner_radius: float = 0.12,
) -> Rectangle:
    """A glass-panel rectangle for grouping content."""
    rect = Rectangle(
        width=width, height=height,
        stroke_color=stroke, stroke_width=1.0,
        fill_color=fill, fill_opacity=fill_opacity,
    )
    rect.round_corners(corner_radius)
    return rect


def shadowed(mob: Mobject, opacity: float = 0.0) -> VGroup:
    """Wrap a mobject in a transparent background rect so it can be safely
    placed over a busier scene without competing with strokes underneath."""
    bg = BackgroundRectangle(mob, color=BLACK, fill_opacity=opacity, buff=0.1)
    return VGroup(bg, mob)
