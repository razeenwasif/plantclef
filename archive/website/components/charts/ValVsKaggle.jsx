"use client";

import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, Cell, Label,
} from "recharts";
import { VAL_VS_KAGGLE } from "@/lib/chart_data";
import { COLORS, AXIS_STYLE, LABEL_STYLE, TOOLTIP_STYLE } from "./theme";

const COLOR_MAP = {
  best:  COLORS.best,
  cap:   COLORS.cap,
  ours:  COLORS.ink,
  sweep: COLORS.sweep,
  inat:  COLORS.inat,
  ssl:   COLORS.ssl,
};

const GROUP_LABEL = {
  best:  "i002 best",
  cap:   "i003 family",
  ours:  "010 anchor",
  sweep: "014b sweep",
  inat:  "015 PC24, iNat",
  ssl:   "012, 013 SSL",
};

function CustomTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div style={TOOLTIP_STYLE.contentStyle}>
      <div style={TOOLTIP_STYLE.labelStyle}>{p.label}</div>
      <div>val top 1: {p.x.toFixed(4)}</div>
      <div>Kaggle F1: {p.y.toFixed(5)}</div>
    </div>
  );
}

export default function ValVsKaggle() {
  const data = VAL_VS_KAGGLE;
  // Pearson r
  const n = data.length;
  const mx = data.reduce((a, b) => a + b.x, 0) / n;
  const my = data.reduce((a, b) => a + b.y, 0) / n;
  let num = 0, dx = 0, dy = 0;
  data.forEach((p) => {
    num += (p.x - mx) * (p.y - my);
    dx += (p.x - mx) ** 2;
    dy += (p.y - my) ** 2;
  });
  const r = num / Math.sqrt(dx * dy);

  // Group by color so the legend renders one entry per group.
  const groups = ["best", "cap", "ours", "sweep", "inat", "ssl"];

  return (
    <div className="chart-card">
      <div className="chart-head">
        <div>
          <p className="chart-eyebrow">Figure 3</p>
          <h4>Validation accuracy is a poor Kaggle predictor</h4>
        </div>
        <p className="chart-cap">
          Each dot is one training run. Validation top 1 (x) and Kaggle public
          F1 (y) are only weakly correlated, <b>Pearson r = {r.toFixed(2)}</b>.
          Within the 014b sweep the relationship inverts: n=5 has the best val
          and the worst Kaggle.
        </p>
      </div>
      <div style={{ width: "100%", height: 420, minWidth: 0 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 24, right: 16, left: 0, bottom: 88 }}>
            <CartesianGrid stroke={COLORS.hairlineSoft} strokeDasharray="2 4" />
            <XAxis
              type="number"
              dataKey="x"
              domain={[0.7, 0.85]}
              ticks={[0.70, 0.73, 0.76, 0.79, 0.82, 0.85]}
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
              tickFormatter={(v) => v.toFixed(2)}
              name="validation top 1"
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0.28, 0.43]}
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
              tickFormatter={(v) => v.toFixed(2)}
              name="Kaggle public F1"
              width={36}
            />
            <Tooltip content={<CustomTip />} cursor={TOOLTIP_STYLE.cursor} />
            <Legend
              verticalAlign="bottom"
              align="center"
              iconType="circle"
              wrapperStyle={{
                paddingTop: 24,
                fontFamily: "Inter, sans-serif",
                fontSize: 12,
                lineHeight: "22px",
              }}
            />
            {groups.map((g) => (
              <Scatter
                key={g}
                name={GROUP_LABEL[g]}
                data={data.filter((d) => d.color === g)}
                fill={COLOR_MAP[g]}
                isAnimationActive
                animationDuration={900}
                shape={g === "best" ? "star" : "circle"}
              >
                {data.filter((d) => d.color === g).map((_, i) => (
                  <Cell key={i} r={g === "best" ? 8 : 5} />
                ))}
              </Scatter>
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
