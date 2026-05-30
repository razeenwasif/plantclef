"""
Scene 4 — Phenology Penalty.

Story arc (≈30 s):
  0:00  Title card.
  0:04  Calendar wheel (12-month dial) draws itself.
  0:08  GBIF day-of-year histogram for Trifolium pratense materialises
        around the dial as radial bars.
  0:14  Boltzmann softening reveal:  m̃_d(s) = ((c_d + 1) / max)^β.
  0:20  Pointer sweeps Jan → Dec; an external probability bar gets
        multiplicatively modulated by the current month's m̃.
  0:28  Out: pointer stops at peak summer, multiplier ≈ 1.0.

Render with:
    cd paper/animations && make phenology
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from manim import (
    AnimationGroup, AnnularSector, Annulus, Arc, Circle, DOWN, FadeIn, FadeOut, LEFT,
    Line, MathTex, RIGHT, Rectangle, Scene, Text, Transform,
    UP, VGroup, Write, rate_functions, smooth, ValueTracker,
    always_redraw,
)

from style import (
    COLOR_DATA, COLOR_DIM, COLOR_MODEL, COLOR_OK,
    COLOR_POST, COLOR_TEXT,
    apply_global_config, title, subtitle, caption, panel,
)


# Real-ish DOY histogram (sum across DOYs binned to 12 months) for
# Trifolium pratense — near-zero Dec-Feb, peak Jun-Jul.
DOY_COUNTS = [12, 18, 56, 142, 380, 720, 950, 880, 540, 220, 84, 24]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def boltzmann_mask(counts: List[int], beta: float) -> List[float]:
    """m̃_i = ((c_i + 1) / max(c + 1))^β"""
    cmax = max(counts) + 1
    return [((c + 1) / cmax) ** beta for c in counts]


# Geometry of the dial.
DIAL_CX, DIAL_CY = -2.6, 0.0
DIAL_R_INNER = 1.05
DIAL_R_OUTER = 1.85  # bars grow outward from the inner radius


class PhenologyPenalty(Scene):

    def construct(self) -> None:
        apply_global_config()
        beta = 1.0
        mask = boltzmann_mask(DOY_COUNTS, beta)

        # ── 0:00 - 0:04 · Title ────────────────────────────────────────
        # NARRATION: "Each species has a season. The phenology penalty
        # tells the model when a species should be visible."
        t = title("Phenology Penalty", size=42)
        s = subtitle("A Boltzmann-softened GBIF prior over day-of-year",
                     size=20)
        s.next_to(t, DOWN, buff=0.25)
        VGroup(t, s).move_to([0, 2.8, 0])
        self.play(Write(t), run_time=1.0)
        self.play(FadeIn(s, shift=UP * 0.2), run_time=0.7)
        self.wait(1.1)
        self.play(FadeOut(VGroup(t, s)), run_time=0.5)

        # ── 0:04 - 0:08 · Calendar dial draws itself ───────────────────
        # NARRATION: "Twelve months around the wheel."
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
            # Position month i so Jan is at top (12 o'clock), going clockwise.
            theta = math.pi / 2 - 2 * math.pi * i / 12
            inner = np.array([DIAL_CX + DIAL_R_INNER * math.cos(theta),
                              DIAL_CY + DIAL_R_INNER * math.sin(theta), 0])
            outer = np.array([DIAL_CX + (DIAL_R_INNER + 0.06) * math.cos(theta),
                              DIAL_CY + (DIAL_R_INNER + 0.06) * math.sin(theta), 0])
            tick = Line(inner, outer, stroke_color="#FFFFFF55", stroke_width=1.5)
            month_ticks.add(tick)
            # Label slightly outside the bars.
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

        # ── 0:08 - 0:14 · Histogram radial bars ────────────────────────
        # NARRATION: "Here are the observations of this species in GBIF,
        # binned by month."
        cmax = max(DOY_COUNTS) + 1
        radial_bars = VGroup()
        for i, c in enumerate(DOY_COUNTS):
            theta_mid = math.pi / 2 - 2 * math.pi * i / 12 - math.pi / 12
            # Each month occupies a 30° sector. Render as a thin Arc (annulus
            # slice) approximated by a Sector cut between two radii.
            # Manim Sector is filled; we want it to grow with the count.
            r_outer = DIAL_R_INNER + 0.05 + ((c + 1) / cmax) * (DIAL_R_OUTER - DIAL_R_INNER - 0.05)
            sec = AnnularSector(
                inner_radius=DIAL_R_INNER + 0.02,
                outer_radius=r_outer,
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

        # ── 0:14 - 0:20 · Boltzmann formula reveal beside ──────────────
        # NARRATION: "We don't use the raw histogram. We soften it with
        # a Boltzmann exponent — beta of one is the natural Bayesian
        # operating point reported in the paper."
        formula = MathTex(
            r"\tilde m_d(s)", r"=",
            r"\left( \frac{n_{d, s} + 1}{\max_{d'} n_{d', s} + 1} \right)^{\!\beta}",
            font_size=32,
        )
        formula[0].set_color(COLOR_POST)
        formula[2].set_color(COLOR_TEXT)
        formula.move_to([3.0, 1.4, 0])
        beta_note = MathTex(r"\beta = 1", font_size=28, color=COLOR_POST)
        beta_note.next_to(formula, DOWN, buff=0.4)
        self.play(Write(formula), run_time=1.3)
        self.play(FadeIn(beta_note, shift=UP * 0.15), run_time=0.5)
        self.wait(2.4)

        # ── 0:20 - 0:28 · Pointer sweep + probability multiplier ───────
        # NARRATION: "Watch how the visual posterior is multiplied by
        # the per-month mask. In winter the probability collapses; in
        # summer the model is amplified."
        # Build a static "raw probability" bar to the right that gets
        # multiplied by the current month's m̃.
        raw_prob = 0.62  # arbitrary visual posterior on the species
        prob_panel_x = 3.4
        prob_panel_y = -1.2
        prob_panel = panel(width=4.5, height=1.6)
        prob_panel.move_to([prob_panel_x, prob_panel_y, 0])
        prob_title = Text("post-LA visual probability  ×  m̃ₘ(s)",
                          font="Sans", font_size=14, color=COLOR_DIM)
        prob_title.move_to([prob_panel_x, prob_panel_y + 0.55, 0])

        # The "track" the bar fills.
        track_w = 3.5
        track = Rectangle(
            width=track_w, height=0.45,
            stroke_color="#FFFFFF22", stroke_width=0.5,
            fill_color="#FFFFFF08", fill_opacity=1.0,
        ).move_to([prob_panel_x, prob_panel_y - 0.05, 0])

        self.play(FadeIn(prob_panel), FadeIn(prob_title), FadeIn(track),
                  run_time=0.7)

        # Dynamic bar driven by ValueTracker for the month index.
        month_v = ValueTracker(0.0)

        def current_bar() -> Rectangle:
            m = int(month_v.get_value()) % 12
            mult = mask[m]
            eff = raw_prob * mult
            w = max(0.02, eff * track_w)
            bar = Rectangle(
                width=w, height=0.45,
                stroke_width=0,
                fill_color=COLOR_OK, fill_opacity=0.95,
            )
            bar.move_to([prob_panel_x - track_w / 2 + w / 2, prob_panel_y - 0.05, 0])
            return bar

        def current_readout() -> Text:
            m = int(month_v.get_value()) % 12
            mult = mask[m]
            eff = raw_prob * mult
            return Text(
                f"{MONTH_LABELS[m]} · m̃ = {mult:0.2f}  →  p = {eff:0.3f}",
                font="Monospace", font_size=14, color=COLOR_TEXT,
            ).move_to([prob_panel_x, prob_panel_y - 0.55, 0])

        bar = always_redraw(current_bar)
        readout = always_redraw(current_readout)

        # Pointer line that rotates around the dial.
        def current_pointer() -> Line:
            m = month_v.get_value()
            theta = math.pi / 2 - 2 * math.pi * m / 12
            inner = np.array([DIAL_CX + 0.0 * math.cos(theta),
                              DIAL_CY + 0.0 * math.sin(theta), 0])
            outer = np.array([DIAL_CX + (DIAL_R_OUTER + 0.05) * math.cos(theta),
                              DIAL_CY + (DIAL_R_OUTER + 0.05) * math.sin(theta), 0])
            return Line(inner, outer, stroke_color=COLOR_POST, stroke_width=2.6)

        pointer = always_redraw(current_pointer)

        self.add(bar, readout, pointer)
        # Sweep from Jan to Dec, then nudge to peak (Jul ≈ month 6).
        self.play(month_v.animate.set_value(11.99),
                  rate_func=rate_functions.linear, run_time=6.0)
        self.play(month_v.animate.set_value(6.0),
                  rate_func=rate_functions.ease_in_out_sine, run_time=0.9)
        self.wait(1.5)
