"""
Scene 1 — End-to-end inference pipeline walkthrough.

Story arc (≈82 s):
  0:00  Title.
  0:04  A quadrat appears (centre-left); brief subtitle.
  0:10  4×4 grid tiling: 16 tile squares emerge.
  0:16  Zoom levels: grid scales 1.0 → 0.8 → 1.0 to convey the
        multi-scale forward pass (scales {1.0, 0.8}).
  0:23  Overlap: a plant lands on a tile seam (bisected, red);
        25 % stride overlay shows a 5×5 effective grid in which the
        plant lies fully inside a shifted tile (emerald glow).
  0:30  Vegetation filter: ExG = 2G−R−B; sky/dirt tiles dim out.
  0:38  Surviving tiles fan into a BioCLIP encoder block; logit
        vector pulses out the right side.
  0:46  Mixed-resolution ensemble: one tile is downsampled to 224
        and 336 px in parallel; each forward produces a posterior;
        the two are fused as a weighted mean in probability space
        (336 logits are peakier — softmax then mean, not sum).
  0:56  16 logit vectors → entropy weighting → one aggregate
        (compressed bayesian-aggregation summary).
  1:04  Aggregate gets modulated by ×m̃ (phenology), −τ·log π̃
        (LA), σ(·) > T (threshold).
  1:14  Species set emerges; Macro-F1 = 0.418 reveal.
  1:22  Hand-off: phenology deep dive (folded in from scene_phenology.py).
  1:22  Phenology title.
  1:26  Calendar dial draws with GBIF radial histogram.
  1:38  Boltzmann formula reveal: m̃_d(s) = ((c+1)/max)^β.
  1:44  Pointer sweeps Jan → Dec; multiplier × probability bar.
  1:52  Out.

Render with:
    cd paper/animations && make pipeline
"""

from __future__ import annotations

import math
import random
from typing import List

import numpy as np
from manim import (
    AnimationGroup, AnnularSector, Annulus, Create, DOWN, FadeIn, FadeOut,
    GrowFromCenter, Indicate, LEFT, MathTex, Rectangle, RIGHT, Scene, Square,
    Text, Transform, UP, VGroup, ValueTracker, Write, always_redraw,
    rate_functions, ReplacementTransform, Line, Arrow, Circle,
)

from style import (
    COLOR_DATA, COLOR_DIM, COLOR_MODEL, COLOR_OK,
    COLOR_POST, COLOR_TEXT, COLOR_WARN,
    apply_global_config, title, subtitle, caption, panel,
)


random.seed(42)


# ─── Phenology beat data (folded in from scene_phenology.py) ────────────
DOY_COUNTS   = [12, 18, 56, 142, 380, 720, 950, 880, 540, 220, 84, 24]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DIAL_CX, DIAL_CY = -2.6, 0.0
DIAL_R_INNER     = 1.05
DIAL_R_OUTER     = 1.85


def boltzmann_mask(counts, beta):
    cmax = max(counts) + 1
    return [((c + 1) / cmax) ** beta for c in counts]


def make_tile(x: float, y: float, size: float, hue: str) -> Rectangle:
    return Rectangle(
        width=size, height=size,
        stroke_color="#FFFFFF40", stroke_width=0.7,
        fill_color=hue, fill_opacity=0.55,
    ).move_to([x, y, 0])


# Simulated tile "vegetation amounts" — used by the ExG filter beat.
# Indices 0..15 in row-major order. The 4-7 (bottom-right corner) tiles
# are "sky / dirt" and get dimmed.
TILE_GREEN = [
    0.85, 0.92, 0.40, 0.10,
    0.78, 0.81, 0.55, 0.05,
    0.30, 0.74, 0.69, 0.12,
    0.20, 0.55, 0.72, 0.88,
]


def green_for_tile(g: float) -> str:
    # Map a green-fraction to a hex green-ish hue.
    if g < 0.18:
        return "#5b5340"   # sandy / dirty (low)
    if g < 0.4:
        return "#3a5b48"   # mossy
    return "#6dab66" if g < 0.7 else "#94c87a"


class PipelineOverview(Scene):

    def construct(self) -> None:
        apply_global_config()

        # ── 0:00 - 0:03 · Hand-off from the Blender hero shot ─────────
        # The preceding Blender clip ends on a near-top-down view of a
        # quadrat with the 4×4 grid faintly drawn. We open the Manim
        # scene with a matching 2D quadrat near-centred and full-size,
        # then slide + scale it to the pipeline-left position. The
        # cross-cut reads as "the camera lands, we're now in 2D" rather
        # than a hard fade-to-title.
        quadrat = Square(side_length=4.0,
                         stroke_color=COLOR_DATA, stroke_width=1.5,
                         fill_color="#3a5b48", fill_opacity=0.7)
        quadrat.move_to([0, 0, 0])
        self.play(FadeIn(quadrat), run_time=0.5)
        self.wait(0.4)

        # Slide + scale into the pipeline-left position the rest of the
        # scene uses, with the input-label fading in below.
        quadrat_target = Square(side_length=2.8,
                                stroke_color=COLOR_DATA, stroke_width=1.5,
                                fill_color="#3a5b48", fill_opacity=0.7)
        quadrat_target.move_to([-4.0, 0.2, 0])
        quadrat_label = caption("input · vegetation quadrat",
                                size=16, color=COLOR_DIM)
        quadrat_label.move_to([-4.0, -2.2, 0])
        self.play(
            Transform(quadrat, quadrat_target),
            FadeIn(quadrat_label, shift=UP * 0.15),
            run_time=1.2,
        )
        self.wait(0.7)

        # ── 0:10 - 0:16 · 4×4 grid tiling ──────────────────────────────
        # NARRATION: "Sixteen tiles, four-forty-eight pixels each."
        tile_size = 2.8 / 4 - 0.06
        tile_origin_x = -4.0
        tile_origin_y = 0.2
        tiles = VGroup()
        for r in range(4):
            for c in range(4):
                idx = r * 4 + c
                x = tile_origin_x + (c - 1.5) * (tile_size + 0.06)
                y = tile_origin_y - (r - 1.5) * (tile_size + 0.06)
                tiles.add(make_tile(x, y, tile_size, green_for_tile(TILE_GREEN[idx])))
        grid_center = [tile_origin_x, tile_origin_y, 0]
        self.play(
            FadeOut(quadrat),
            AnimationGroup(*[GrowFromCenter(t) for t in tiles], lag_ratio=0.04),
            run_time=1.6,
        )
        new_label = caption("4 × 4 grid · 448-px tiles",
                            size=16, color=COLOR_DIM)
        new_label.move_to([-4.0, -2.2, 0])
        self.play(Transform(quadrat_label, new_label), run_time=0.4)
        self.wait(0.6)

        # ── 0:16 - 0:23 · Zoom levels (scales 1.0, 0.8) ────────────────
        # NARRATION: "We do this at two zoom levels. Scale one-point-zero
        # gives BioCLIP global context. Scale point-eight zooms in,
        # giving small plants a tighter crop and more pixel budget."
        scale_label = MathTex(r"\text{scale} = ", "1.0", font_size=30)
        scale_label[0].set_color(COLOR_DIM)
        scale_label[1].set_color(COLOR_DATA)
        scale_label.move_to([0, 2.4, 0])
        self.play(FadeIn(scale_label, shift=DOWN * 0.15), run_time=0.5)
        self.wait(0.8)

        # Scale the whole 4×4 grid down to 0.8 around its centre.
        # Visually the tiles "breathe in" — same tile count, smaller
        # spatial extent in the image, which is exactly what scale=0.8
        # means: every tile covers less of the canvas at this zoom.
        new_scale_label = MathTex(r"\text{scale} = ", "0.8", font_size=30)
        new_scale_label[0].set_color(COLOR_DIM)
        new_scale_label[1].set_color(COLOR_POST)
        new_scale_label.move_to(scale_label.get_center())
        scales_caption = caption(
            "scale 0.8 · smaller receptive field · more pixels per plant",
            size=14, color=COLOR_POST,
        )
        scales_caption.move_to([-4.0, -2.55, 0])
        self.play(
            tiles.animate.scale(0.8, about_point=grid_center),
            Transform(scale_label, new_scale_label),
            FadeIn(scales_caption, shift=UP * 0.15),
            run_time=1.2,
        )
        self.wait(1.6)

        # Restore scale 1.0 for the rest of the pipeline.
        back_label = MathTex(r"\text{scale} = ", "1.0", font_size=30)
        back_label[0].set_color(COLOR_DIM)
        back_label[1].set_color(COLOR_DATA)
        back_label.move_to(scale_label.get_center())
        self.play(
            tiles.animate.scale(1 / 0.8, about_point=grid_center),
            Transform(scale_label, back_label),
            FadeOut(scales_caption),
            run_time=1.0,
        )
        self.play(FadeOut(scale_label), run_time=0.3)

        # ── 0:23 - 0:30 · Overlap · 25 % stride ────────────────────────
        # NARRATION: "And we tile with overlap. A twenty-five-percent
        # stride means tiles overshoot their neighbours, so a plant on
        # a tile edge is never bisected — it lands fully inside at
        # least one of the shifted tiles."
        # Pick an edge between tile (1,1) and (1,2) and put a "plant"
        # marker right on the seam.
        plant_x = tile_origin_x + (1.5 - 1.5) * (tile_size + 0.06) + tile_size / 2 + 0.03
        plant_y = tile_origin_y - (1 - 1.5) * (tile_size + 0.06)
        plant = Circle(radius=0.13, stroke_width=0,
                       fill_color="#a3e29a", fill_opacity=0.95)
        plant.move_to([plant_x, plant_y, 0])
        # A faint red line through it to visualise the bisection.
        bisect_x = plant_x
        bisect = Line(
            [bisect_x, plant_y + 0.16, 0],
            [bisect_x, plant_y - 0.16, 0],
            stroke_color="#fb7185", stroke_width=2.0,
        )
        plant_label = caption("plant on a tile edge → bisected",
                              size=13, color="#fb7185")
        plant_label.move_to([-4.0, -2.55, 0])
        self.play(FadeIn(plant, scale=0.6),
                  Create(bisect),
                  FadeIn(plant_label, shift=UP * 0.15),
                  run_time=0.9)
        self.wait(1.3)

        # Now overlay a second 5×5 grid shifted by 25% stride
        # (i.e. tiles offset by 75% of tile width). With this overlap
        # the plant ends up entirely inside the shifted tile centred
        # on the seam.
        stride = 0.75 * (tile_size + 0.06)
        overlap_tiles = VGroup()
        # Position overlap tiles to clearly cover the seam: 5 columns × 5 rows.
        for r in range(5):
            for c in range(5):
                x = tile_origin_x + (c - 2) * stride
                y = tile_origin_y - (r - 2) * stride
                t = Rectangle(
                    width=tile_size, height=tile_size,
                    stroke_color=COLOR_POST, stroke_width=0.9,
                    fill_color=COLOR_POST, fill_opacity=0.0,
                )
                t.move_to([x, y, 0])
                overlap_tiles.add(t)

        new_label2 = caption(
            "25 % stride · 5 × 5 effective views per scale",
            size=14, color=COLOR_POST,
        )
        new_label2.move_to([-4.0, -2.55, 0])
        self.play(
            FadeOut(bisect),
            AnimationGroup(*[FadeIn(t) for t in overlap_tiles], lag_ratio=0.02),
            Transform(plant_label, new_label2),
            run_time=1.1,
        )
        # Highlight one specific overlap tile that fully contains the plant.
        # The shifted tile centred at column-2, row-2 of the overlap grid
        # straddles the original seam.
        cover_tile = overlap_tiles[2 * 5 + 2]  # row=2, col=2
        cover_glow = cover_tile.copy()
        cover_glow.set_stroke(color=COLOR_OK, width=2.2, opacity=1.0)
        self.play(
            Indicate(plant, color=COLOR_OK, scale_factor=1.3),
            Transform(cover_tile, cover_glow),
            run_time=0.9,
        )
        self.wait(1.2)
        # Clean up the overlap layer + plant marker before continuing.
        self.play(
            FadeOut(overlap_tiles),
            FadeOut(plant),
            FadeOut(plant_label),
            run_time=0.5,
        )

        # ── 0:30 - 0:38 · Vegetation filter ────────────────────────────
        # NARRATION: "Tiles that aren't mostly vegetation are dropped.
        # The Excess-Green index is just two-G minus R minus B."
        exg_eq = MathTex(r"\mathrm{ExG} \;=\; 2G - R - B",
                         font_size=30, color=COLOR_DATA)
        exg_eq.move_to([0, 2.3, 0])
        exg_caption = caption("drop tiles with < 15% green pixels",
                              size=14, color=COLOR_DIM)
        exg_caption.next_to(exg_eq, DOWN, buff=0.15)
        self.play(Write(exg_eq), FadeIn(exg_caption), run_time=1.0)

        # Dim out tiles whose simulated green-fraction is low.
        dim_anims = []
        for idx, g in enumerate(TILE_GREEN):
            if g < 0.18:
                dim_anims.append(tiles[idx].animate.set_opacity(0.1).set_stroke(opacity=0.2))
        self.play(*dim_anims, run_time=1.4)
        self.wait(1.6)
        self.play(FadeOut(exg_eq), FadeOut(exg_caption), run_time=0.4)

        # ── 0:26 - 0:34 · BioCLIP encoder block ────────────────────────
        # NARRATION: "Each surviving tile is forwarded through BioCLIP
        # 2.5 ViT-H/14 at two resolutions, 224 and 336."
        biolip_box = panel(width=3.6, height=1.6)
        biolip_box.set_stroke(COLOR_MODEL, width=1.6)
        biolip_box.move_to([1.2, 0.2, 0])
        biolip_label = Text("BioCLIP 2.5 ViT-H/14",
                            font="Sans", weight="SEMIBOLD",
                            font_size=22, color=COLOR_MODEL)
        biolip_label.move_to([1.2, 0.5, 0])
        biolip_sub = caption("last 4 blocks unfrozen · @ 224, 336 px",
                             size=13, color=COLOR_DIM)
        biolip_sub.move_to([1.2, -0.05, 0])
        self.play(FadeIn(biolip_box, shift=LEFT * 0.3),
                  Write(biolip_label),
                  FadeIn(biolip_sub), run_time=1.0)

        # Animate "active" tiles streaming into the BioCLIP box.
        stream_anims = []
        for idx, g in enumerate(TILE_GREEN):
            if g >= 0.18:
                src = tiles[idx]
                comet = Square(side_length=0.18,
                               stroke_width=0,
                               fill_color=COLOR_DATA, fill_opacity=0.9)
                comet.move_to(src.get_center())
                stream_anims.append(AnimationGroup(
                    FadeIn(comet, run_time=0.12),
                    comet.animate(rate_func=rate_functions.ease_in_out_sine)
                         .move_to(biolip_box.get_center())
                         .set_opacity(0.0),
                    lag_ratio=0,
                ))
        self.play(AnimationGroup(*stream_anims, lag_ratio=0.06), run_time=2.5)

        # Output: a logit-vector emerging from the right edge of BioCLIP.
        logit_bars = VGroup()
        for k in range(8):
            h = random.uniform(0.1, 0.85)
            bar = Rectangle(
                width=0.16, height=h,
                stroke_width=0,
                fill_color=COLOR_OK, fill_opacity=0.85,
            )
            bar.move_to([3.6 + k * 0.22, 0.2 - 0.5 + h / 2, 0])
            logit_bars.add(bar)
        logit_caption = caption("per-tile logit vector",
                                size=13, color=COLOR_DIM)
        logit_caption.move_to([4.4, -1.0, 0])
        self.play(
            AnimationGroup(*[GrowFromCenter(b) for b in logit_bars], lag_ratio=0.04),
            FadeIn(logit_caption),
            run_time=1.0,
        )
        self.wait(1.2)
        self.play(FadeOut(VGroup(biolip_box, biolip_label, biolip_sub,
                                 logit_bars, logit_caption,
                                 quadrat_label, tiles)),
                  run_time=0.6)

        # ── 0:46 - 0:54 · Mixed-resolution ensemble ────────────────────
        # NARRATION: "And we do every forward at two resolutions.
        # Two-twenty-four is BioCLIP's native receptive field — broad
        # context. Three-thirty-six gives the encoder a tighter crop
        # so small or partly-occluded plants get more pixel budget.
        # The two posteriors are weighted-meaned in probability space —
        # softmaxed first, because three-thirty-six logits are peakier
        # and would otherwise out-vote two-twenty-four on raw magnitude."
        # One representative tile in the centre.
        seed_tile = Square(side_length=1.0,
                           stroke_color=COLOR_DATA, stroke_width=1.2,
                           fill_color="#3a5b48", fill_opacity=0.75)
        seed_tile.move_to([-5.2, 0.5, 0])
        seed_lbl = caption("one tile · 448 px", size=12, color=COLOR_DIM)
        seed_lbl.move_to([-5.2, -0.25, 0])
        self.play(FadeIn(seed_tile, scale=0.85), FadeIn(seed_lbl), run_time=0.7)
        self.wait(0.7)

        # Two resized copies — 224 px (cyan flow) and 336 px (pink flow).
        tile_224 = Square(side_length=0.72,
                          stroke_color=COLOR_DATA, stroke_width=1.1,
                          fill_color="#3a5b48", fill_opacity=0.75)
        tile_224.move_to([-2.8, 1.5, 0])
        lbl_224 = MathTex(r"224 \times 224", font_size=18, color=COLOR_DATA)
        lbl_224.next_to(tile_224, UP, buff=0.15)

        tile_336 = Square(side_length=1.05,
                          stroke_color=COLOR_POST, stroke_width=1.1,
                          fill_color="#3a5b48", fill_opacity=0.75)
        tile_336.move_to([-2.8, -0.5, 0])
        lbl_336 = MathTex(r"336 \times 336", font_size=18, color=COLOR_POST)
        lbl_336.next_to(tile_336, UP, buff=0.15)

        self.play(
            ReplacementTransform(seed_tile.copy(), tile_224),
            ReplacementTransform(seed_tile.copy(), tile_336),
            FadeIn(lbl_224),
            FadeIn(lbl_336),
            run_time=1.0,
        )
        self.wait(0.4)

        # Mini-BioCLIP boxes for each resolution.
        bio_224 = panel(width=1.6, height=0.7, fill="#15151F", fill_opacity=0.9)
        bio_224.set_stroke(COLOR_DATA, width=1.0)
        bio_224.move_to([-0.6, 1.5, 0])
        bio_224_lbl = caption("BioCLIP", size=11, color=COLOR_DATA)
        bio_224_lbl.move_to(bio_224.get_center())

        bio_336 = panel(width=1.6, height=0.7, fill="#15151F", fill_opacity=0.9)
        bio_336.set_stroke(COLOR_POST, width=1.0)
        bio_336.move_to([-0.6, -0.5, 0])
        bio_336_lbl = caption("BioCLIP", size=11, color=COLOR_POST)
        bio_336_lbl.move_to(bio_336.get_center())

        self.play(
            FadeIn(bio_224, shift=LEFT * 0.15), FadeIn(bio_224_lbl),
            FadeIn(bio_336, shift=LEFT * 0.15), FadeIn(bio_336_lbl),
            run_time=0.6,
        )

        # Per-resolution probability bars. 336 px is slightly peakier
        # to convey "three-thirty-six logits are peakier" — the reason
        # the fusion is done in probability space, not logit space.
        def make_bars(center_x: float, center_y: float, heights: list[float],
                       color: str) -> VGroup:
            g = VGroup()
            for k, h in enumerate(heights):
                bar = Rectangle(
                    width=0.18, height=h,
                    stroke_width=0, fill_color=color, fill_opacity=0.9,
                )
                bar.move_to([center_x + (k - 2.5) * 0.22, center_y - 0.4 + h / 2, 0])
                g.add(bar)
            return g

        # Heights chosen to total ≈ 1 once normalised; 336 has a single
        # spike, 224 is broader.
        bars_224 = make_bars(2.0,  1.5, [0.18, 0.30, 0.62, 0.36, 0.16, 0.10], COLOR_DATA)
        bars_336 = make_bars(2.0, -0.5, [0.06, 0.10, 0.95, 0.18, 0.06, 0.04], COLOR_POST)

        # softmax-→-fusion ribbon: a small operator pill.
        fusion_pill = panel(width=2.4, height=0.85, fill="#15151F", fill_opacity=0.95)
        fusion_pill.set_stroke(COLOR_OK, width=1.2)
        fusion_pill.move_to([4.6, 0.5, 0])
        fusion_eq = MathTex(
            r"\bar p \;=\; \tfrac{1}{2}\!\left(p_{224} + p_{336}\right)",
            font_size=20,
        )
        fusion_eq.set_color(COLOR_OK)
        fusion_eq.move_to(fusion_pill.get_center())

        # Final fused bars.
        bars_fused = make_bars(4.6, -1.0, [0.12, 0.20, 0.78, 0.27, 0.11, 0.07], COLOR_OK)

        # Draw the two posteriors emerging, then fuse.
        self.play(
            AnimationGroup(*[GrowFromCenter(b) for b in bars_224], lag_ratio=0.04),
            AnimationGroup(*[GrowFromCenter(b) for b in bars_336], lag_ratio=0.04),
            run_time=0.9,
        )
        self.wait(0.6)
        self.play(
            FadeIn(fusion_pill),
            Write(fusion_eq),
            run_time=0.8,
        )
        # The two bar charts converge into the fused distribution at the
        # bottom-right via short comet streams, then the final fused bars
        # grow in place.
        comets = VGroup()
        for src in (bars_224, bars_336):
            for b in src:
                c = b.copy()
                comets.add(c)
        self.play(
            AnimationGroup(*[
                c.animate(rate_func=rate_functions.ease_in_out_sine)
                 .move_to(bars_fused.get_center())
                 .set_opacity(0.0)
                for c in comets
            ], lag_ratio=0.015),
            run_time=1.4,
        )
        self.play(
            AnimationGroup(*[GrowFromCenter(b) for b in bars_fused], lag_ratio=0.04),
            run_time=0.6,
        )
        peaky_note = caption(
            "fused in probability space · 336 is peakier; raw-logit sum would overweight it",
            size=12, color=COLOR_DIM,
        )
        peaky_note.move_to([0, -2.4, 0])
        self.play(FadeIn(peaky_note, shift=UP * 0.15), run_time=0.5)
        self.wait(1.6)
        self.play(
            FadeOut(VGroup(
                seed_tile, seed_lbl, tile_224, tile_336, lbl_224, lbl_336,
                bio_224, bio_224_lbl, bio_336, bio_336_lbl,
                bars_224, bars_336, fusion_pill, fusion_eq, bars_fused,
                peaky_note,
            )),
            run_time=0.6,
        )

        # ── 0:54 - 1:02 · Bayesian aggregation summary ─────────────────
        # NARRATION: "We Bayesian-aggregate across tiles. Confident
        # tiles dominate the image-level posterior."
        # Show 16 stacked mini-distributions on the left, collapsing
        # into one big distribution on the right.
        left_dists = VGroup()
        for i in range(16):
            row, col = divmod(i, 4)
            x = -5.0 + col * 0.45
            y = 1.6 - row * 0.75
            sub = VGroup()
            for k in range(5):
                h = random.uniform(0.05, 0.45)
                bar = Rectangle(
                    width=0.07, height=h,
                    stroke_width=0,
                    fill_color=COLOR_DATA, fill_opacity=0.85,
                )
                bar.move_to([x + (k - 2) * 0.09, y - 0.3 + h / 2, 0])
                sub.add(bar)
            left_dists.add(sub)
        agg_title = caption("16 per-tile posteriors  →  weighted sum  →  1",
                            size=16, color=COLOR_DIM)
        agg_title.move_to([0, -2.6, 0])
        self.play(
            AnimationGroup(*[GrowFromCenter(d) for d in left_dists], lag_ratio=0.03),
            FadeIn(agg_title),
            run_time=1.6,
        )

        # Streams converging to centre/right.
        agg_pos_x, agg_pos_y = 3.0, 0.2
        stream_anims2 = []
        for dist in left_dists:
            comet = Circle(radius=0.06,
                           stroke_width=0,
                           fill_color=COLOR_OK, fill_opacity=0.9)
            comet.move_to(dist.get_center())
            stream_anims2.append(AnimationGroup(
                FadeIn(comet, run_time=0.1),
                comet.animate(rate_func=rate_functions.ease_in_out_sine)
                     .move_to([agg_pos_x, agg_pos_y, 0])
                     .set_opacity(0.0),
                lag_ratio=0,
            ))
        # Build the aggregate distribution.
        agg_bars = VGroup()
        agg_heights = [0.85, 0.55, 0.32, 0.21, 0.13]
        for k, h in enumerate(agg_heights):
            bar = Rectangle(
                width=0.42, height=h,
                stroke_width=0,
                fill_color=COLOR_OK, fill_opacity=0.95,
            )
            bar.move_to([agg_pos_x + (k - 2) * 0.55, agg_pos_y - 0.4 + h / 2, 0])
            agg_bars.add(bar)
        self.play(
            AnimationGroup(*stream_anims2, lag_ratio=0.06),
            AnimationGroup(*[GrowFromCenter(b) for b in agg_bars], lag_ratio=0.08),
            run_time=2.6,
        )
        self.wait(0.6)
        self.play(FadeOut(VGroup(left_dists, agg_title)), run_time=0.4)

        # ── 0:42 - 0:52 · Post-processing chain ────────────────────────
        # NARRATION: "Now the post-processing stack. We apply a
        # phenology mask, the class-prior logit adjustment, and an
        # adaptive sigmoid threshold."
        # Three small "operator" badges that pulse over the aggregate.
        ops = [
            (r"\times \tilde m_d", COLOR_POST,  "phenology mask"),
            (r"-\tau \log \tilde\pi_s", COLOR_POST,  "logit adjustment · τ=0.25"),
            (r"\sigma(\cdot) > T", COLOR_POST,  "threshold · T=0.03"),
        ]
        ops_x = -4.6
        ops_y = 0.2
        for i, (sym, col, lbl) in enumerate(ops):
            badge = panel(width=2.6, height=0.95, fill="#15151F", fill_opacity=0.9)
            badge.set_stroke(col, width=1.2)
            badge.move_to([ops_x + i * 0.0, ops_y, 0])
            sym_tex = MathTex(sym, font_size=26, color=col)
            sym_tex.move_to(badge.get_center() + 0.18 * UP)
            lbl_tex = caption(lbl, size=11, color=COLOR_DIM)
            lbl_tex.move_to(badge.get_center() + 0.30 * DOWN)
            self.play(FadeIn(badge, shift=UP * 0.2),
                      Write(sym_tex),
                      FadeIn(lbl_tex), run_time=0.7)
            self.play(Indicate(agg_bars, color=col, scale_factor=1.06), run_time=0.6)
            self.play(FadeOut(VGroup(badge, sym_tex, lbl_tex)), run_time=0.3)
        self.wait(0.4)

        # ── 0:52 - 0:60 · Species emission + F1 reveal ─────────────────
        # NARRATION: "Every species above threshold is emitted. On the
        # leaderboard this reaches zero point four-one-eight."
        species_pill = VGroup(
            *[panel(width=2.6, height=0.5, fill="#15151F", fill_opacity=0.85)
              for _ in range(3)]
        )
        names = ["Trifolium pratense", "Plantago lanceolata", "Galium verum"]
        for i, p in enumerate(species_pill):
            p.move_to([-4.6, -1.0 - i * 0.7, 0])
            p.set_stroke(COLOR_OK, width=1.1)
        labels = VGroup()
        for i, n in enumerate(names):
            lbl = Text(n, font="Sans", slant="ITALIC",
                       font_size=18, color=COLOR_OK)
            lbl.move_to([-4.6, -1.0 - i * 0.7, 0])
            labels.add(lbl)
        self.play(
            AnimationGroup(*[FadeIn(species_pill[i], shift=LEFT * 0.2) for i in range(3)],
                           lag_ratio=0.12),
            AnimationGroup(*[FadeIn(labels[i], shift=LEFT * 0.2) for i in range(3)],
                           lag_ratio=0.12),
            run_time=1.5,
        )

        # F1 banner
        f1_pill = panel(width=5.2, height=1.1)
        f1_pill.set_stroke(COLOR_OK, width=1.6)
        f1_pill.move_to([2.6, -2.0, 0])
        f1_text = MathTex(
            r"\mathbf{Macro\text{-}F1}", r"=", r"\mathbf{0.418}",
            r"\ \text{public}",
            font_size=32,
        )
        f1_text[0].set_color(COLOR_TEXT)
        f1_text[2].set_color(COLOR_OK)
        f1_text[3].set_color(COLOR_DIM)
        f1_text.move_to(f1_pill.get_center())
        self.play(FadeIn(f1_pill), Write(f1_text), run_time=1.4)
        self.wait(2.5)
        # Clear the F1 card before pivoting into the phenology deep dive.
        self.play(
            FadeOut(VGroup(f1_pill, f1_text, species_pill, labels)),
            run_time=0.6,
        )

        # ── 1:22 - 1:52 · Phenology deep dive (merged from scene_phenology.py) ─
        # NARRATION: "Let's take a closer look at one of those post-
        # processing operators — the phenology penalty."
        beta = 1.0
        mask = boltzmann_mask(DOY_COUNTS, beta)

        ph_title = Text("Phenology Penalty · Deep Dive", font="Sans",
                        weight="SEMIBOLD", font_size=38, color=COLOR_TEXT)
        ph_sub = Text("A Boltzmann-softened GBIF prior over day-of-year",
                      font="Sans", font_size=20, color=COLOR_DIM)
        ph_sub.next_to(ph_title, DOWN, buff=0.25)
        VGroup(ph_title, ph_sub).move_to([0, 2.8, 0])
        self.play(Write(ph_title), run_time=1.0)
        self.play(FadeIn(ph_sub, shift=UP * 0.2), run_time=0.7)
        self.wait(1.1)
        self.play(FadeOut(VGroup(ph_title, ph_sub)), run_time=0.5)

        # Calendar dial.
        ring = Annulus(
            inner_radius=DIAL_R_INNER - 0.05,
            outer_radius=DIAL_R_INNER,
            color="#FFFFFF22", fill_opacity=0.6, stroke_width=0,
        ).move_to([DIAL_CX, DIAL_CY, 0])
        species_label = Text("Trifolium pratense",
                             font="Sans", slant="ITALIC",
                             font_size=18, color=COLOR_POST)
        species_label.move_to([DIAL_CX, DIAL_CY, 0])
        self.play(FadeIn(ring), Write(species_label), run_time=1.2)

        month_ticks = VGroup()
        month_lbls = VGroup()
        for i in range(12):
            theta = math.pi / 2 - 2 * math.pi * i / 12
            inner = np.array([DIAL_CX + DIAL_R_INNER * math.cos(theta),
                              DIAL_CY + DIAL_R_INNER * math.sin(theta), 0])
            outer = np.array([DIAL_CX + (DIAL_R_INNER + 0.06) * math.cos(theta),
                              DIAL_CY + (DIAL_R_INNER + 0.06) * math.sin(theta), 0])
            tick = Line(inner, outer, stroke_color="#FFFFFF55", stroke_width=1.5)
            month_ticks.add(tick)
            r_lbl = DIAL_R_OUTER + 0.30
            lbl_pos = np.array([DIAL_CX + r_lbl * math.cos(theta),
                                DIAL_CY + r_lbl * math.sin(theta), 0])
            lbl = Text(MONTH_LABELS[i][0], font="Monospace", font_size=14,
                       color=COLOR_DIM)
            lbl.move_to(lbl_pos)
            month_lbls.add(lbl)
        self.play(
            AnimationGroup(*[FadeIn(t) for t in month_ticks], lag_ratio=0.04),
            AnimationGroup(*[FadeIn(t) for t in month_lbls],  lag_ratio=0.04),
            run_time=1.4,
        )
        self.wait(0.7)

        # GBIF histogram radial bars.
        cmax = max(DOY_COUNTS) + 1
        radial_bars = VGroup()
        for i, c in enumerate(DOY_COUNTS):
            theta_mid = math.pi / 2 - 2 * math.pi * i / 12 - math.pi / 12
            r_outer_i = DIAL_R_INNER + 0.05 + ((c + 1) / cmax) * (DIAL_R_OUTER - DIAL_R_INNER - 0.05)
            sec = AnnularSector(
                inner_radius=DIAL_R_INNER + 0.02,
                outer_radius=r_outer_i,
                angle=2 * math.pi / 12 - 0.01,
                start_angle=theta_mid - math.pi / 12,
                arc_center=np.array([DIAL_CX, DIAL_CY, 0]),
                fill_color=COLOR_DATA,
                fill_opacity=0.32,
                stroke_color=COLOR_DATA,
                stroke_opacity=0.55,
                stroke_width=0.6,
            )
            radial_bars.add(sec)
        self.play(
            AnimationGroup(*[FadeIn(b) for b in radial_bars], lag_ratio=0.04),
            run_time=1.7,
        )
        self.wait(1.0)

        # Boltzmann formula.
        ph_formula = MathTex(
            r"\tilde m_d(s)", r"=",
            r"\left( \frac{n_{d, s} + 1}{\max_{d'} n_{d', s} + 1} \right)^{\!\beta}",
            font_size=32,
        )
        ph_formula[0].set_color(COLOR_POST)
        ph_formula[2].set_color(COLOR_TEXT)
        ph_formula.move_to([3.0, 1.4, 0])
        ph_beta_note = MathTex(r"\beta = 1", font_size=28, color=COLOR_POST)
        ph_beta_note.next_to(ph_formula, DOWN, buff=0.4)
        self.play(Write(ph_formula), run_time=1.3)
        self.play(FadeIn(ph_beta_note, shift=UP * 0.15), run_time=0.5)
        self.wait(2.4)

        # Pointer sweep + multiplier readout.
        raw_prob = 0.62
        prob_panel_x, prob_panel_y = 3.4, -1.2
        ph_prob_panel = panel(width=4.5, height=1.6)
        ph_prob_panel.move_to([prob_panel_x, prob_panel_y, 0])
        ph_prob_title = Text("post-LA visual probability  ×  m̃ₘ(s)",
                             font="Sans", font_size=14, color=COLOR_DIM)
        ph_prob_title.move_to([prob_panel_x, prob_panel_y + 0.55, 0])
        track_w = 3.5
        ph_track = Rectangle(
            width=track_w, height=0.45,
            stroke_color="#FFFFFF22", stroke_width=0.5,
            fill_color="#FFFFFF08", fill_opacity=1.0,
        ).move_to([prob_panel_x, prob_panel_y - 0.05, 0])
        self.play(FadeIn(ph_prob_panel), FadeIn(ph_prob_title), FadeIn(ph_track),
                  run_time=0.7)

        month_v = ValueTracker(0.0)

        def _ph_current_bar():
            m = int(month_v.get_value()) % 12
            eff = raw_prob * mask[m]
            w = max(0.02, eff * track_w)
            bar = Rectangle(
                width=w, height=0.45,
                stroke_width=0,
                fill_color=COLOR_OK, fill_opacity=0.95,
            )
            bar.move_to([prob_panel_x - track_w / 2 + w / 2, prob_panel_y - 0.05, 0])
            return bar

        def _ph_current_readout():
            m = int(month_v.get_value()) % 12
            mult = mask[m]
            eff = raw_prob * mult
            return Text(
                f"{MONTH_LABELS[m]} · m̃ = {mult:0.2f}  →  p = {eff:0.3f}",
                font="Monospace", font_size=14, color=COLOR_TEXT,
            ).move_to([prob_panel_x, prob_panel_y - 0.55, 0])

        def _ph_current_pointer():
            m = month_v.get_value()
            theta = math.pi / 2 - 2 * math.pi * m / 12
            inner = np.array([DIAL_CX, DIAL_CY, 0])
            outer = np.array([DIAL_CX + (DIAL_R_OUTER + 0.05) * math.cos(theta),
                              DIAL_CY + (DIAL_R_OUTER + 0.05) * math.sin(theta), 0])
            return Line(inner, outer, stroke_color=COLOR_POST, stroke_width=2.6)

        ph_bar      = always_redraw(_ph_current_bar)
        ph_readout  = always_redraw(_ph_current_readout)
        ph_pointer  = always_redraw(_ph_current_pointer)
        self.add(ph_bar, ph_readout, ph_pointer)
        self.play(month_v.animate.set_value(11.99),
                  rate_func=rate_functions.linear, run_time=6.0)
        self.play(month_v.animate.set_value(6.0),
                  rate_func=rate_functions.ease_in_out_sine, run_time=0.9)
        self.wait(1.5)
