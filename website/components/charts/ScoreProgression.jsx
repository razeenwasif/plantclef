"use client";

import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceDot, Label,
} from "recharts";
import { SCORE_PROGRESSION } from "@/lib/chart_data";
import { COLORS, AXIS_STYLE, LABEL_STYLE, TOOLTIP_STYLE } from "./theme";

export default function ScoreProgression() {
  const data = SCORE_PROGRESSION;
  const bestPoint = data.reduce((b, d) => (d.this > b.this ? d : b), data[0]);

  return (
    <div className="chart-card">
      <div className="chart-head">
        <div>
          <p className="chart-eyebrow">Figure 1</p>
          <h4>Score progression across experiments</h4>
        </div>
        <p className="chart-cap">
          Each dot is the best public Kaggle F1 from one experiment. The
          dashed line tracks best so far. <b>010</b> is the partial unfreeze
          breakthrough; <b>i002</b> is long tail capping with adaptive
          selection.
        </p>
      </div>
      <div style={{ width: "100%", height: 380, minWidth: 0 }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 24, right: 16, left: 0, bottom: 70 }}>
            <CartesianGrid stroke={COLORS.hairlineSoft} strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ ...AXIS_STYLE, angle: -45, textAnchor: "end" }}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
              interval={0}
              height={64}
            />
            <YAxis
              domain={[0, 0.45]}
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
              tickFormatter={(v) => v.toFixed(2)}
              width={36}
            />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(v, n) => [Number(v).toFixed(5), n === "this" ? "this experiment" : "best so far"]}
              labelFormatter={(l) => l}
            />
            <Legend
              verticalAlign="top"
              align="left"
              iconType="line"
              wrapperStyle={{ paddingBottom: 12, fontFamily: "Inter, sans-serif", fontSize: 12 }}
              formatter={(v) => v === "this" ? "per experiment" : "best so far"}
            />
            <Line
              type="linear"
              dataKey="best_so_far"
              stroke={COLORS.mutedSoft}
              strokeWidth={1.3}
              strokeDasharray="4 4"
              dot={false}
              isAnimationActive
              animationDuration={900}
            />
            <Line
              type="linear"
              dataKey="this"
              stroke={COLORS.ink}
              strokeWidth={2}
              dot={{ r: 3.5, fill: COLORS.ink, stroke: "none" }}
              activeDot={{ r: 5, fill: COLORS.ink }}
              isAnimationActive
              animationDuration={1100}
            />
            <ReferenceDot
              x={bestPoint.label}
              y={bestPoint.this}
              r={6}
              fill={COLORS.ink}
              stroke={COLORS.accent}
              strokeWidth={2}
              isFront
            >
              <Label
                value={bestPoint.this.toFixed(3)}
                position="top"
                offset={12}
                style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 11, fontWeight: 700, fill: COLORS.ink }}
              />
            </ReferenceDot>
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
