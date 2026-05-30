"""
Scene 2 — Class-Prior Logit Adjustment.

Story arc (≈45 s):
  0:00  Title card.
  0:04  10-species bar chart appears, descending heights matching long-tail.
  0:10  Frequency labels reveal: 51k → 3 examples.
  0:16  Formula z' = z − τ·log π̃_s builds up term by term.
  0:24  Laplace-smoothed prior π̃_s revealed beside.
  0:30  τ slider crawls 0 → 0.25; bars redistribute live; rare bars rise.
  0:38  Callout: "for rare s, log π̃_s is very negative → large boost".
  0:42  Result chip: +0.007 Macro-F1 at τ=0.25 in the paper.
  0:45  Out.

Render with:
    cd paper/animations && make logit
"""

from __future__ import annotations

import math
from typing import List

from manim import (
    AnimationGroup, DOWN, FadeIn, FadeOut, GrowFromCenter,
    LEFT, MathTex, Rectangle, RIGHT, Scene,
    Text, Transform, UP, VGroup, Write, rate_functions,
)

from style import (
    COLOR_DATA, COLOR_DIM, COLOR_MODEL, COLOR_OK,
    COLOR_POST, COLOR_TEXT,
    PLANTCLEF_BG, apply_global_config, title, subtitle, caption, panel,
)


# Same illustrative 10-species long-tail used in the dashboard's
# interactive demo.
SPECIES: List[tuple[str, int]] = [
    ("Trifolium pratense",    51_240),
    ("Plantago lanceolata",   19_870),
    ("Anthriscus sylvestris",  6_120),
    ("Galium verum",           2_440),
    ("Centaurea jacea",          980),
    ("Veronica chamaedrys",      310),
    ("Knautia arvensis",          94),
    ("Hypochaeris radicata",      27),
    ("Salvia pratensis",           8),
    ("Pulsatilla pratensis",       3),
]

TOTAL_IMAGES = sum(n for _, n in SPECIES)
NUM_CLASSES = len(SPECIES)


def laplace_priors() -> List[float]:
    return [(n + 1) / (TOTAL_IMAGES + NUM_CLASSES) for _, n in SPECIES]


def raw_logits() -> List[float]:
    """Pretend a frequency-biased classifier: z_s ∝ log(n_s + 1)."""
    return [math.log(n + 1) for _, n in SPECIES]


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def adjusted_probs(tau: float) -> List[float]:
    """z' = z − τ·log π̃ → softmax."""
    z, p = raw_logits(), laplace_priors()
    zp = [zi - tau * math.log(pi) for zi, pi in zip(z, p)]
    return softmax(zp)


# Bar-chart geometry. We render a fixed grid of N rows; the bar's width is
# proportional to its softmax probability.
N_ROWS = NUM_CLASSES
ROW_HEIGHT = 0.46
BAR_TRACK_W = 5.6   # full width of the bar background
BAR_TRACK_X = 1.3   # left edge of the track in world units
NAME_X = -3.3       # right-aligned end of the species name label
PROB_LABEL_X = 5.7  # left edge of the "%" readout


def bar_for(idx: int, prob: float, max_prob: float, color: str) -> Rectangle:
    width = max(0.02, (prob / max_prob) * BAR_TRACK_W)
    bar = Rectangle(
        width=width, height=ROW_HEIGHT * 0.7,
        stroke_width=0, fill_color=color, fill_opacity=0.85,
    )
    # Anchor left edge to BAR_TRACK_X.
    bar.move_to([BAR_TRACK_X + width / 2, _row_y(idx), 0])
    return bar


def _row_y(idx: int) -> float:
    """Convert row index into a world-y. Top row is positive y."""
    top = (N_ROWS - 1) / 2 * ROW_HEIGHT
    return top - idx * ROW_HEIGHT - 0.45  # slight downward bias for headroom


class LogitAdjustment(Scene):

    def construct(self) -> None:
        apply_global_config()

        # ── 0:00 - 0:04 · Title ─────────────────────────────────────────
        # NARRATION: "Plant ID datasets follow an extreme long-tail
        # distribution. The class-prior logit adjustment is how we make
        # the model emit rare species at inference time."
        t = title("Class-Prior Logit Adjustment", size=42)
        s = subtitle("How we make a long-tail classifier emit rare species", size=20)
        s.next_to(t, DOWN, buff=0.25)
        VGroup(t, s).move_to([0, 2.0, 0])
        self.play(Write(t), run_time=1.2)
        self.play(FadeIn(s, shift=UP * 0.2), run_time=0.9)
        self.wait(1.4)
        self.play(FadeOut(VGroup(t, s)), run_time=0.6)

        # ── 0:04 - 0:10 · Bar chart appears ─────────────────────────────
        # NARRATION: "We start with the raw model posteriors over ten
        # species. The model has been trained on frequency-imbalanced
        # data, so the head dominates by orders of magnitude."
        probs_zero = adjusted_probs(0.0)
        max_p = max(probs_zero)
        name_labels = VGroup(*[
            Text(name, font="Sans", slant="ITALIC",
                 font_size=18,
                 color=COLOR_POST if SPECIES[i][1] < 100 else COLOR_DIM)
              .move_to([NAME_X + 0.0, _row_y(i), 0])
              .align_to([NAME_X, 0, 0], RIGHT)
            for i, (name, _) in enumerate(SPECIES)
        ])
        tracks = VGroup(*[
            Rectangle(
                width=BAR_TRACK_W, height=ROW_HEIGHT * 0.7,
                stroke_color="#FFFFFF22", stroke_width=0.7,
                fill_color="#FFFFFF08", fill_opacity=1.0,
            ).move_to([BAR_TRACK_X + BAR_TRACK_W / 2, _row_y(i), 0])
            for i in range(N_ROWS)
        ])

        # Initial bars (τ = 0): proportional to raw frequency.
        bars = VGroup(*[
            bar_for(i, probs_zero[i], max_p,
                    color=COLOR_POST if SPECIES[i][1] < 100 else COLOR_DATA)
            for i in range(N_ROWS)
        ])

        prob_labels = VGroup(*[
            Text(f"{probs_zero[i]*100:0.2f}%", font="Monospace", font_size=18,
                 color=COLOR_TEXT)
              .move_to([PROB_LABEL_X, _row_y(i), 0])
            for i in range(N_ROWS)
        ])

        self.play(FadeIn(tracks, shift=LEFT * 0.2), Write(name_labels), run_time=1.2)
        self.play(
            AnimationGroup(*[GrowFromCenter(b) for b in bars], lag_ratio=0.04),
            run_time=1.1,
        )
        self.play(FadeIn(prob_labels, shift=LEFT * 0.1), run_time=0.6)
        self.wait(1.4)

        # ── 0:10 - 0:16 · Reveal training-frequency labels ──────────────
        # NARRATION: "Trifolium has fifty-thousand training images.
        # Pulsatilla has three."
        freq_labels = VGroup(*[
            Text(f"n = {SPECIES[i][1]:,}", font="Monospace", font_size=14,
                 color=COLOR_DIM)
              .move_to([BAR_TRACK_X + BAR_TRACK_W + 0.1, _row_y(i), 0])
              .align_to([BAR_TRACK_X + BAR_TRACK_W + 0.1, 0, 0], LEFT)
            for i in range(N_ROWS)
        ])
        # Slide the probability labels further right to make room for n labels.
        new_prob_pos = [
            [BAR_TRACK_X + BAR_TRACK_W + 1.6, _row_y(i), 0]
            for i in range(N_ROWS)
        ]
        self.play(
            AnimationGroup(
                *[prob_labels[i].animate.move_to(new_prob_pos[i]) for i in range(N_ROWS)],
                lag_ratio=0,
            ),
            FadeIn(freq_labels),
            run_time=0.9,
        )
        self.wait(2.0)

        # ── 0:16 - 0:24 · Build the formula ─────────────────────────────
        # NARRATION: "The fix is one line. Subtract τ times the log of
        # the training prior."
        formula = MathTex(
            r"z'_s", r"=", r"z_s", r"\;-\;", r"\tau", r"\,\cdot\,",
            r"\log", r"\tilde{\pi}_s",
            font_size=44,
        )
        for piece, color in zip(
            formula,
            [COLOR_OK, COLOR_TEXT, COLOR_DATA, COLOR_TEXT, COLOR_POST,
             COLOR_TEXT, COLOR_TEXT, COLOR_MODEL],
        ):
            piece.set_color(color)
        formula.move_to([0, -2.65, 0])

        # Reveal piece by piece so the eye follows.
        self.play(Write(formula[2]), run_time=0.4)          # z_s
        self.play(Write(formula[1]), run_time=0.2)          # =
        self.play(Write(formula[0]), run_time=0.3)          # z'_s
        self.wait(0.4)
        self.play(Write(formula[3]), run_time=0.2)          # −
        self.play(Write(VGroup(formula[4], formula[5])), run_time=0.4)  # τ ·
        self.play(Write(VGroup(formula[6], formula[7])), run_time=0.5)  # log π̃_s
        self.wait(1.2)

        # ── 0:24 - 0:30 · Laplace-smoothed prior beside ──────────────────
        # NARRATION: "Where pi-tilde is the Laplace-smoothed training
        # prior — adding one to the count of every class so rare species
        # never explode to negative infinity."
        prior_def = MathTex(
            r"\tilde{\pi}_s", r"=",
            r"\frac{n_s + 1}{N + C}",
            font_size=34,
        )
        prior_def[0].set_color(COLOR_MODEL)
        prior_def[2].set_color(COLOR_DIM)
        prior_def.next_to(formula, RIGHT, buff=0.9)
        self.play(FadeIn(prior_def, shift=LEFT * 0.3), run_time=0.9)
        self.wait(2.2)

        # ── 0:30 - 0:38 · τ slider 0 → 0.25 with bars redistributing ────
        # NARRATION: "Watch what happens as we sweep tau from zero to
        # the headline value, point two five."
        tau_label = MathTex(r"\tau = ", r"0.00", font_size=34)
        tau_label[0].set_color(COLOR_TEXT)
        tau_label[1].set_color(COLOR_POST)
        tau_label.move_to([-5.2, -2.65, 0])
        self.play(FadeIn(tau_label, shift=DOWN * 0.2), run_time=0.5)

        # Sweep τ over 12 frames of state for smooth interpolation.
        sweep_steps = 60
        sweep_run = 5.5
        for step in range(1, sweep_steps + 1):
            tau = 0.25 * (step / sweep_steps)
            probs = adjusted_probs(tau)
            mx = max(probs)
            new_bars = [
                bar_for(i, probs[i], mx,
                        color=COLOR_POST if SPECIES[i][1] < 100 else COLOR_DATA)
                for i in range(N_ROWS)
            ]
            new_labels = [
                Text(f"{probs[i]*100:0.2f}%", font="Monospace", font_size=18,
                     color=COLOR_TEXT)
                  .move_to([BAR_TRACK_X + BAR_TRACK_W + 1.6, _row_y(i), 0])
                for i in range(N_ROWS)
            ]
            new_tau = MathTex(r"\tau = ", f"{tau:0.2f}", font_size=34)
            new_tau[0].set_color(COLOR_TEXT)
            new_tau[1].set_color(COLOR_POST)
            new_tau.move_to(tau_label.get_center())
            self.play(
                *[Transform(bars[i], new_bars[i]) for i in range(N_ROWS)],
                *[Transform(prob_labels[i], new_labels[i]) for i in range(N_ROWS)],
                Transform(tau_label, new_tau),
                run_time=sweep_run / sweep_steps,
                rate_func=rate_functions.linear,
            )
        self.wait(1.0)

        # ── 0:38 - 0:42 · Callout for the tail boost ─────────────────────
        # NARRATION: "Because log of a small probability is a large
        # negative number, rare species get the biggest boost. The
        # head shrinks, the tail rises."
        cal = caption(
            "rare s → log π̃ₛ very negative → −τ·log π̃ₛ very positive → bar rises",
            size=18, color=COLOR_POST,
        )
        cal.move_to([0, -3.3, 0])
        self.play(FadeIn(cal, shift=UP * 0.2), run_time=0.7)
        self.wait(2.0)
        self.play(FadeOut(cal), run_time=0.4)

        # ── 0:42 - 0:45 · Result reveal ──────────────────────────────────
        # NARRATION: "On the PlantCLEF 2026 hidden test set this lifts
        # the Macro F1 by zero point zero zero seven."
        result_pill = panel(width=4.6, height=0.9, fill=PLANTCLEF_BG, fill_opacity=0.7)
        result_pill.move_to([0, -3.3, 0])
        result_pill.set_stroke(color=COLOR_OK, width=1.4)
        result_text = MathTex(
            r"\Delta \text{Macro-F1} \,=\, +0.007", r"\ \text{at}\ \tau=0.25",
            font_size=28,
        )
        result_text[0].set_color(COLOR_OK)
        result_text[1].set_color(COLOR_DIM)
        result_text.move_to(result_pill.get_center())
        self.play(FadeIn(result_pill), Write(result_text), run_time=1.0)
        self.wait(2.0)
