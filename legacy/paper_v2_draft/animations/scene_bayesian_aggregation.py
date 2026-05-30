"""
Scene 3 — Bayesian Aggregation across Tiles.

Story arc (≈45 s):
  0:00  Title card.
  0:04  16-tile (4×4) grid materialises; each tile shows a tiny posterior bar
        chart over a few species.
  0:12  Zoom into one tile; its 5-species distribution swells; compute
        Shannon entropy H_t = −Σ p log p on screen.
  0:22  Pull back; entropy numbers appear on every tile; tiles glow by
        weight w_t = exp(−H_t).
  0:30  Weighted-sum animation: every tile's distribution streams toward
        a single aggregate distribution at the bottom; weights scale the
        contribution.
  0:38  Final aggregate posterior settles, top-3 species labelled.
  0:42  Callout: "focused tiles dominate; ambiguous ones get muted".
  0:45  Out.

Render with:
    cd paper/animations && make aggregation
"""

from __future__ import annotations

import math
import random
from typing import List

from manim import (
    AnimationGroup, DOWN, FadeIn, FadeOut, GrowFromCenter,
    LEFT, MathTex, Rectangle, RIGHT, Scene,
    Text, Transform, UP, VGroup, Write,
    rate_functions, ReplacementTransform,
)

from style import (
    COLOR_DATA, COLOR_DIM, COLOR_MODEL, COLOR_OK,
    COLOR_POST, COLOR_TEXT, COLOR_WARN,
    apply_global_config, title, subtitle, caption, panel,
)


# ── A reproducible mock per-tile posterior set ──────────────────────────
# 5 species columns (drawn from the same 10-species toy set used elsewhere).
SPECIES_SHORT = [
    "T. pratense", "P. lanceolata", "G. verum", "K. arvensis", "S. pratensis",
]

# 16 tiles, each a 5-vector over species. Some tiles are confident
# (low entropy, peaked), others are flat (high entropy, ambiguous). The
# weighted aggregate ends up dominated by the confident, on-target tiles.
random.seed(7)
def _synth_posteriors() -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(16):
        # Tile "type" controls flatness.
        kind = i % 4
        if kind == 0:        # confident on T. pratense
            base = [3.5, 0.3, 0.1, 0.0, 0.0]
        elif kind == 1:      # confident on P. lanceolata
            base = [0.4, 3.2, 0.4, 0.1, 0.0]
        elif kind == 2:      # somewhat split between K. arvensis / S. pratensis (rare)
            base = [0.2, 0.4, 0.4, 1.4, 1.2]
        else:                # flat / ambiguous (background tile)
            base = [0.4, 0.4, 0.3, 0.5, 0.4]
        # Small noise
        z = [b + random.gauss(0, 0.05) for b in base]
        m = max(z); ez = [math.exp(zz - m) for zz in z]; s = sum(ez)
        out.append([e / s for e in ez])
    return out


TILE_POSTERIORS = _synth_posteriors()


def entropy(p: List[float]) -> float:
    return -sum(pi * math.log(pi + 1e-9) for pi in p)


# Geometry. 4×4 grid centred near top of frame.
GRID_TOP_Y    = 1.8
TILE_SIZE     = 0.95
TILE_GAP      = 0.18
GRID_ORIGIN_X = -(2 * (TILE_SIZE + TILE_GAP)) + TILE_SIZE / 2  # left edge of leftmost tile
BAR_W_INSIDE  = TILE_SIZE - 0.18


def _tile_xy(idx: int) -> tuple[float, float]:
    r, c = divmod(idx, 4)
    x = GRID_ORIGIN_X + c * (TILE_SIZE + TILE_GAP)
    y = GRID_TOP_Y - r * (TILE_SIZE + TILE_GAP)
    return x, y


def make_tile_with_bars(idx: int, posterior: List[float]) -> VGroup:
    """One 4×4-grid tile with a stacked mini bar chart inside."""
    x, y = _tile_xy(idx)
    box = Rectangle(
        width=TILE_SIZE, height=TILE_SIZE,
        stroke_color="#FFFFFF22", stroke_width=0.7,
        fill_color="#FFFFFF06", fill_opacity=1.0,
    ).move_to([x, y, 0])

    # Mini bars: vertical strips, each at ~width = BAR_W_INSIDE / N.
    n = len(posterior)
    cell_w = BAR_W_INSIDE / n
    base_y = y - TILE_SIZE / 2 + 0.08
    max_h = TILE_SIZE - 0.18
    bars = VGroup()
    for j, p in enumerate(posterior):
        h = max(0.02, p * max_h)
        bx = x - BAR_W_INSIDE / 2 + cell_w * (j + 0.5)
        col = COLOR_DATA if j < 2 else (COLOR_POST if j >= 3 else COLOR_MODEL)
        bar = Rectangle(
            width=cell_w * 0.7, height=h,
            stroke_width=0, fill_color=col, fill_opacity=0.85,
        ).move_to([bx, base_y + h / 2, 0])
        bars.add(bar)
    return VGroup(box, bars)


class BayesianAggregation(Scene):

    def construct(self) -> None:
        apply_global_config()

        # ── 0:00 - 0:04 · Title ────────────────────────────────────────
        # NARRATION: "Each tile of the quadrat lands a posterior over
        # the species set. The question is how to combine them."
        t = title("Bayesian Aggregation Across Tiles", size=42)
        s = subtitle("Entropy weighting collapses 16 tile posteriors into one", size=20)
        s.next_to(t, DOWN, buff=0.25)
        VGroup(t, s).move_to([0, 2.6, 0])
        self.play(Write(t), run_time=1.1)
        self.play(FadeIn(s, shift=UP * 0.2), run_time=0.7)
        self.wait(1.4)
        self.play(FadeOut(VGroup(t, s)), run_time=0.5)

        # ── 0:04 - 0:12 · 4×4 tile grid materialises ───────────────────
        # NARRATION: "Sixteen tiles. Each gives a five-species
        # distribution. Some are confident on the bull's-eye; others
        # are ambiguous, looking at background."
        tiles = VGroup(*[
            make_tile_with_bars(i, TILE_POSTERIORS[i]) for i in range(16)
        ])
        self.play(
            AnimationGroup(*[GrowFromCenter(t) for t in tiles],
                           lag_ratio=0.05),
            run_time=2.4,
        )
        # Small inline legend.
        legend_lbl = Text("each tile · 5-species posterior", font="Sans",
                          font_size=16, color=COLOR_DIM)
        legend_lbl.move_to([0, -1.45, 0])
        self.play(FadeIn(legend_lbl, shift=UP * 0.15), run_time=0.5)
        self.wait(2.2)
        self.play(FadeOut(legend_lbl), run_time=0.3)

        # ── 0:12 - 0:22 · Zoom into tile 0, compute entropy ────────────
        # NARRATION: "Pick one tile. Its entropy is the negative sum of
        # p log p — low when the tile is sure of a species, high when
        # it's not."
        # Reuse tile 0 (highly confident on T. pratense) for visual punch.
        focus_idx = 0
        focus_tile = tiles[focus_idx].copy()
        # Move it to centre-left, scale it up.
        focus_tile.generate_target()
        focus_tile.target.scale(2.6).move_to([-3.7, -1.0, 0])
        # Fade the original grid while we examine.
        self.play(
            tiles.animate.set_opacity(0.2),
            FadeIn(focus_tile),
            run_time=0.6,
        )
        self.play(
            focus_tile.animate.scale(2.6).move_to([-3.6, -0.4, 0]),
            run_time=0.9,
        )

        # Show the posterior beside it.
        post = TILE_POSTERIORS[focus_idx]
        H = entropy(post)
        post_label = MathTex(
            r"p_t \,=\, "
            + r"[\, " + r",\, ".join(f"{p:.2f}" for p in post) + r" \,]",
            font_size=28,
        )
        post_label.set_color(COLOR_TEXT)
        post_label.move_to([1.8, 0.7, 0])
        ent_expr = MathTex(
            r"H_t", r"\;=\;", r"-\!\sum_s p_s \log p_s",
            r"\;=\;", f"{H:.3f}",
            font_size=32,
        )
        ent_expr[0].set_color(COLOR_OK)
        ent_expr[2].set_color(COLOR_TEXT)
        ent_expr[4].set_color(COLOR_OK)
        ent_expr.move_to([1.8, -0.4, 0])

        self.play(FadeIn(post_label), run_time=0.7)
        self.wait(1.0)
        self.play(Write(ent_expr), run_time=1.4)
        self.wait(2.0)
        self.play(
            FadeOut(post_label),
            FadeOut(ent_expr),
            FadeOut(focus_tile),
            tiles.animate.set_opacity(1.0),
            run_time=0.7,
        )

        # ── 0:22 - 0:30 · Entropy + weight overlay on every tile ───────
        # NARRATION: "Now do it for every tile. Tiles glow brighter when
        # they're confident; we use w = exp minus H."
        H_per_tile = [entropy(p) for p in TILE_POSTERIORS]
        w_per_tile = [math.exp(-h) for h in H_per_tile]
        max_w = max(w_per_tile)
        # Re-skin tile borders by w (more confident → brighter cyan border).
        new_glows = VGroup()
        h_labels = VGroup()
        for i in range(16):
            w_rel = w_per_tile[i] / max_w
            glow = Rectangle(
                width=TILE_SIZE + 0.05, height=TILE_SIZE + 0.05,
                stroke_width=1.6,
                stroke_color=COLOR_OK,
                fill_opacity=0,
            )
            x, y = _tile_xy(i)
            glow.move_to([x, y, 0])
            glow.set_stroke(opacity=0.25 + 0.75 * w_rel)
            new_glows.add(glow)
            # Tiny "H = ..." label inside-top of each tile.
            lbl = Text(f"H={H_per_tile[i]:.2f}", font="Monospace", font_size=11,
                       color=COLOR_DIM)
            lbl.move_to([x, y + TILE_SIZE / 2 - 0.12, 0])
            h_labels.add(lbl)
        weight_def = MathTex(r"w_t", r"=", r"\exp(-H_t)", font_size=34)
        weight_def[0].set_color(COLOR_POST)
        weight_def[2].set_color(COLOR_TEXT)
        weight_def.move_to([0, -1.6, 0])
        self.play(
            AnimationGroup(*[FadeIn(g) for g in new_glows], lag_ratio=0.04),
            AnimationGroup(*[FadeIn(l) for l in h_labels], lag_ratio=0.03),
            FadeIn(weight_def, shift=UP * 0.2),
            run_time=2.5,
        )
        self.wait(2.0)

        # ── 0:30 - 0:38 · Streams into aggregate posterior ─────────────
        # NARRATION: "The aggregate is the weighted sum of per-tile
        # distributions. Sharp tiles dominate; blurry ones contribute
        # almost nothing."
        # Build the aggregate posterior at the bottom of the frame.
        agg = [0.0] * 5
        for w, post in zip(w_per_tile, TILE_POSTERIORS):
            for k in range(5):
                agg[k] += w * post[k]
        sm = sum(agg)
        agg = [a / sm for a in agg]

        # Aggregate bar chart container.
        agg_panel = panel(width=10.0, height=1.5)
        agg_panel.move_to([0, -3.0, 0])
        agg_title = Text("aggregate posterior  ·  Σ wₜ · pₜ",
                         font="Sans", font_size=15, color=COLOR_DIM)
        agg_title.move_to([0, -3.0 + 0.55, 0])

        # Initially empty bars; we'll grow them as tiles "stream" in.
        bar_w_each = 1.6
        bar_gap    = 0.32
        agg_y      = -3.0 - 0.15
        agg_bars = VGroup()
        agg_labels = VGroup()
        for k in range(5):
            bx = (k - 2) * (bar_w_each + bar_gap)
            container = Rectangle(
                width=bar_w_each, height=0.5,
                stroke_color="#FFFFFF22", stroke_width=0.5,
                fill_color="#FFFFFF08", fill_opacity=1.0,
            ).move_to([bx, agg_y, 0])
            agg_bars.add(container)
            lbl = Text(SPECIES_SHORT[k], font="Sans", slant="ITALIC",
                       font_size=12, color=COLOR_DIM)
            lbl.move_to([bx, agg_y - 0.45, 0])
            agg_labels.add(lbl)

        self.play(
            FadeIn(agg_panel),
            FadeIn(agg_title),
            FadeIn(agg_bars),
            FadeIn(agg_labels),
            run_time=0.8,
        )

        # Stream effect: for each tile, send a brief comet to each
        # aggregate bar weighted by w * p.
        stream_anims = []
        for i in range(16):
            x_t, y_t = _tile_xy(i)
            w_rel = w_per_tile[i] / max_w
            # Single visual "comet" per tile, sized by weight; even though
            # each tile distributes over 5 species, one comet keeps the
            # animation legible at this density.
            top_k = max(range(5), key=lambda k: TILE_POSTERIORS[i][k])
            dot = Rectangle(width=0.08, height=0.08, stroke_width=0,
                            fill_color=COLOR_OK, fill_opacity=0.85)
            dot.move_to([x_t, y_t, 0])
            target_x = (top_k - 2) * (bar_w_each + bar_gap)
            stream_anims.append(
                AnimationGroup(
                    FadeIn(dot, run_time=0.12),
                    dot.animate(rate_func=rate_functions.ease_in_out_sine)
                       .move_to([target_x, agg_y, 0])
                       .set_opacity(0.0),
                    lag_ratio=0,
                )
            )
        self.play(
            AnimationGroup(*stream_anims, lag_ratio=0.08),
            run_time=3.6,
        )

        # Now reveal the final aggregate bars sized by the actual weighted
        # values.
        max_agg = max(agg)
        final_bars = VGroup()
        final_labels = VGroup()
        for k in range(5):
            bx = (k - 2) * (bar_w_each + bar_gap)
            h = (agg[k] / max_agg) * 0.4 + 0.04
            bar = Rectangle(
                width=bar_w_each - 0.16, height=h,
                stroke_width=0, fill_color=COLOR_OK, fill_opacity=0.95,
            ).move_to([bx, agg_y - 0.25 + h / 2, 0])
            final_bars.add(bar)
            pct = Text(f"{agg[k]*100:0.1f}%", font="Monospace", font_size=13,
                       color=COLOR_TEXT)
            pct.move_to([bx, agg_y - 0.45 - 0.18, 0])
            final_labels.add(pct)
        # Move species labels up to make room for the % labels.
        new_label_pos = [[(k - 2) * (bar_w_each + bar_gap), agg_y + 0.43, 0]
                         for k in range(5)]
        self.play(
            *[agg_labels[k].animate.move_to(new_label_pos[k]) for k in range(5)],
            run_time=0.4,
        )
        self.play(
            AnimationGroup(*[GrowFromCenter(b) for b in final_bars], lag_ratio=0.05),
            FadeIn(final_labels),
            run_time=1.4,
        )
        self.wait(1.2)

        # ── 0:42 - 0:45 · Callout ──────────────────────────────────────
        # NARRATION: "Focused tiles win. Ambiguous tiles get muted. The
        # aggregate is the model's best image-level guess."
        cal = caption("focused tiles win · ambiguous tiles get muted",
                      size=18, color=COLOR_POST)
        cal.move_to([0, -1.6, 0])
        # Replace the weight definition with this callout.
        self.play(ReplacementTransform(weight_def, cal), run_time=0.7)
        self.wait(2.0)
